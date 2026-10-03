from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .human_agentic_data import (
    TOOLS,
    _matched_direct,
    _natural_direct,
    _tool_case,
)
from .paths import project_root
from .server import parse_native_tool_calls
from .tool_schema import KNOWN_TOOLS, schema_conditioned_system_prompt

SCHEMA_VERSION = 1
CATEGORIES = (
    "strict_ast",
    "routing_negative",
    "web_multihop",
    "memory",
    "multi_step",
    "polyglot_java",
    "polyglot_javascript",
    "parallel_multiple",
    "rehearsal",
)
TRAIN_PER_CATEGORY = 390
DEV_PER_CATEGORY = 10
DEV_FOLDS = ("a", "b", "c")
OUTPUT_DIR = project_root() / "downloads" / "generated" / "strict-genetic-v13"
TRAIN_PATH = OUTPUT_DIR / "train-master.jsonl"
PREFERENCE_PATH = OUTPUT_DIR / "preferences.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

POLICY = (
    "You are a precise general tool-calling assistant. Use a declared function only when an "
    "external action is requested and all required values are available. Emit only the native "
    "tool-call AST when acting. Preserve names, keys, scalar types and literal values exactly; "
    "answer directly for hypothetical, negated, explanatory or underspecified requests."
)


def dev_path(fold: str) -> Path:
    if fold not in DEV_FOLDS:
        raise ValueError(f"unknown DEV fold: {fold}")
    return OUTPUT_DIR / f"dev-{fold}.jsonl"


def _system() -> dict[str, str]:
    return {
        "role": "system",
        "content": schema_conditioned_system_prompt(POLICY, KNOWN_TOOLS),
    }


def _native_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    rendered = []
    for name, arguments in calls:
        params = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
        rendered.append(f"{name}({params})")
    return f"<|tool_call_start|>[{', '.join(rendered)}]<|tool_call_end|>"


def _record(
    *,
    split: str,
    case_id: str,
    category: str,
    prompt: list[dict[str, str]],
    completion: str,
    tool: str | None,
    arguments: dict[str, Any] | None = None,
    required_terms: list[str] | None = None,
) -> dict[str, Any]:
    prompt = [dict(message) for message in prompt]
    reference = hashlib.sha256(case_id.encode()).hexdigest()[:12]
    if prompt and prompt[-1].get("role") == "user":
        prompt[-1]["content"] = (
            str(prompt[-1].get("content") or "")
            + f"\nRequest reference: `{reference}`."
        )
    metadata = {
        "case_id": case_id,
        "split": split,
        "condition": "strict-genetic-v13",
        "stratum": category,
        "category": category,
        "decision": "tool" if tool else "direct",
        "tool": tool,
        "expected_arguments": arguments or {},
        "required_terms": required_terms or [],
        "schema_version": SCHEMA_VERSION,
        "benchmark_origin": False,
    }
    key = "training_metadata" if split == "train" else "evaluation_metadata"
    return {
        key: metadata,
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": completion}],
    }


def _strict_ast(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    tool = TOOLS[serial % len(TOOLS)]
    if split == "train":
        local_index = 700 + serial // len(TOOLS)
        source_split = "train"
    else:
        local_index = 80 + fold_index * 40 + serial
        source_split = "dev"
    request, arguments, _family = _tool_case(tool, source_split, local_index)
    return _record(
        split=split,
        case_id=f"{split}-strict-{serial:04d}",
        category="strict_ast",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=_native_calls([(tool, arguments)]),
        tool=tool,
        arguments=arguments,
    )


def _routing_negative(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    tool = TOOLS[serial % len(TOOLS)]
    if split == "train":
        local_index = 200 + serial // len(TOOLS)
        source_split = "train"
    else:
        local_index = 80 + fold_index * 40 + serial
        source_split = "dev"
    request, answer, required, _family = _matched_direct(tool, source_split, local_index)
    return _record(
        split=split,
        case_id=f"{split}-negative-{serial:04d}",
        category="routing_negative",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=answer,
        tool=None,
        required_terms=required,
    )


def _rehearsal(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    base = 2_000 if split == "train" else 4_000 + fold_index * 100
    source_split = "train" if split == "train" else "dev"
    request, answer, required, _family = _natural_direct(source_split, base + serial)
    return _record(
        split=split,
        case_id=f"{split}-rehearsal-{serial:04d}",
        category="rehearsal",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=answer,
        tool=None,
        required_terms=required,
    )


def _web_multihop(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    namespace = (10_000 if split == "train" else 40_000) + fold_index * 1_000 + serial
    marker = f"WEB_{namespace:05d}"
    query = f"release compatibility note {marker} revision {2031 + namespace % 7}"
    url = f"https://docs.example.net/releases/{namespace}/{marker.lower()}"
    search = _native_calls([("websearch", {"query": query})])
    arguments = {"url": url, "prompt": f"Extract only compatibility marker {marker}."}
    prompt = [
        _system(),
        {"role": "user", "content": f"Find the current source for `{query}`."},
        {"role": "assistant", "content": search},
        {"role": "tool", "content": f"Top verified result: {url}"},
        {
            "role": "user",
            "content": f"Open that exact result and extract only compatibility marker `{marker}`.",
        },
    ]
    return _record(
        split=split,
        case_id=f"{split}-web-{serial:04d}",
        category="web_multihop",
        prompt=prompt,
        completion=_native_calls([("webfetch", arguments)]),
        tool="webfetch",
        arguments=arguments,
    )


def _memory(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    namespace = (20_000 if split == "train" else 50_000) + fold_index * 1_000 + serial
    marker = f"MEMORY_{namespace:05d}"
    path = f"workspaces/memory/session-{namespace:05d}/state.txt"
    content = f"session={namespace:05d}\nmarker={marker}\n"
    arguments = {"path": path, "content": content}
    prompt = [
        _system(),
        {
            "role": "user",
            "content": f"Remember the literal `{marker}` for the next request. Do nothing yet.",
        },
        {
            "role": "assistant",
            "content": f"Understood. I will preserve the literal `{marker}` for the next request.",
        },
        {
            "role": "user",
            "content": (
                f"Now create `{path}` with two lines: session={namespace:05d} and the marker "
                "I gave you, followed by a final newline."
            ),
        },
    ]
    return _record(
        split=split,
        case_id=f"{split}-memory-{serial:04d}",
        category="memory",
        prompt=prompt,
        completion=_native_calls([("write", arguments)]),
        tool="write",
        arguments=arguments,
    )


def _multi_step(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    namespace = (30_000 if split == "train" else 60_000) + fold_index * 1_000 + serial
    root = f"projects/chain-{namespace:05d}"
    path = f"{root}/router.py"
    marker = f"CHAIN_{namespace:05d}"
    first = _native_calls([("read", {"path": path, "offset": 0, "limit": 80})])
    arguments = {"pattern": marker, "path": root, "include": "*.py"}
    prompt = [
        _system(),
        {"role": "user", "content": f"Inspect the first 80 lines of `{path}`."},
        {"role": "assistant", "content": first},
        {"role": "tool", "content": f"The file imports routing helpers; marker {marker} is referenced."},
        {
            "role": "user",
            "content": f"Continue by searching Python files under `{root}` for exact marker `{marker}`.",
        },
    ]
    return _record(
        split=split,
        case_id=f"{split}-multi-{serial:04d}",
        category="multi_step",
        prompt=prompt,
        completion=_native_calls([("grep", arguments)]),
        tool="grep",
        arguments=arguments,
    )


def _polyglot_java(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    ns = (70_000 if split == "train" else 80_000) + fold_index * 1_000 + serial
    tool_idx = serial % 10
    if tool_idx == 0:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        arguments = {
            "monitor": f"dbMonitor_{ns:05d}",
            "view": f"EmployeeView_{ns:05d}",
            "source": f"SELECT * FROM Employee_{ns:05d} WHERE status = 'active'",
        }
        request = (
            f"How can I generate the full SQL creation script with a header for a Firebird "
            f"database view named '{arguments['view']}', using a progress monitor "
            f"`{arguments['monitor']}` and the original source '{arguments['source']}'?"
        )
    elif tool_idx == 1:
        tool = "DB2Tablespace.resolveTablespaceReference"
        arguments = {
            "monitor": f"dbMonitor_{ns:05d}",
            "dataSource": f"db2DataSource_{ns:05d}",
            "reference": f"USERSPACE_{ns:05d}",
        }
        request = (
            f"How can I resolve a tablespace reference named '{arguments['reference']}' "
            f"in a DB2 database using data source object `{arguments['dataSource']}` "
            f"and progress monitor `{arguments['monitor']}`?"
        )
    elif tool_idx == 2:
        tool = "PmsProductServiceImpl.updateNewStatus"
        arguments = {
            "ids": [100 + (ns % 800), 200 + (ns % 800)],
            "newStatus": (ns % 4) + 1,
        }
        request = (
            f"How can I update the new status to {arguments['newStatus']} for a list of "
            f"product IDs {arguments['ids']} in the product management system?"
        )
    elif tool_idx == 3:
        tool = "TwoSum.twoSum"
        arguments = {
            "nums": [2 + (ns % 10), 7 + (ns % 10), 11, 15],
            "target": 9 + 2 * (ns % 10),
        }
        request = (
            f"What are the indices of the two numbers in the array {arguments['nums']} "
            f"that add up to the target sum of {arguments['target']}?"
        )
    elif tool_idx == 4:
        tool = "JNIBridge.setLauncherInfo"
        arguments = {
            "launcher": f"/usr/local/bin/launcher_{ns:05d}",
            "name": f"AppLauncher_{ns:05d}",
        }
        request = (
            f"How can I update the launcher information in the JNI Bridge with the launcher "
            f"path '{arguments['launcher']}' and the launcher name '{arguments['name']}'?"
        )
    elif tool_idx == 5:
        tool = "configStorage.dynamicCredentialsScheduledExecutorService"
        arguments = {
            "credentialsFile": f"conf/credentials_{ns:05d}.properties",
            "credentialsRefreshInterval": 30 + (ns % 60),
            "basicCredentials": f"basicAuth_{ns:05d}",
        }
        request = (
            f"How can I create a scheduled executor service that periodically updates credentials "
            f"from a file named '{arguments['credentialsFile']}' every {arguments['credentialsRefreshInterval']} "
            f"seconds, using the basic credentials provided in variable `{arguments['basicCredentials']}`?"
        )
    elif tool_idx == 6:
        tool = "BasePolicyDataProvider.getRegistryPolicyValue"
        arguments = {
            "root": "WinReg.HKEY_LOCAL_MACHINE",
            "property": f"PolicySetting_{ns:05d}",
        }
        request = (
            f"What is the value of the '{arguments['property']}' property in the Windows registry "
            f"`WinReg` object under the {arguments['root']} root when checking system policies?"
        )
    elif tool_idx == 7:
        tool = "ExasolExecutionContext.setCurrentSchema"
        arguments = {
            "monitor": f"progressMonitor_{ns:05d}",
            "schemaName": f"AnalyticsDB_{ns:05d}",
        }
        request = (
            f"How do I change the current schema to '{arguments['schemaName']}' in the Exasol "
            f"execution context while monitoring the progress with a monitor object named `{arguments['monitor']}`?"
        )
    elif tool_idx == 8:
        tool = "DataSerializer.serializePayload"
        arguments = {
            "format": "json_canonical" if (ns % 2 == 0) else "msgpack",
            "payload": {"record_id": ns, "active": True},
        }
        request = (
            f"Please serialize the payload {arguments['payload']} using format '{arguments['format']}'."
        )
    else:
        tool = "AuditLogManager.recordSecurityEvent"
        arguments = {
            "eventType": "AUTH_AUDIT_VERIFY",
            "severity": (ns % 4) + 1,
        }
        request = (
            f"Record security event '{arguments['eventType']}' with severity level {arguments['severity']}."
        )

    return _record(
        split=split,
        case_id=f"{split}-java-{serial:04d}",
        category="polyglot_java",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=_native_calls([(tool, arguments)]),
        tool=tool,
        arguments=arguments,
    )


def _polyglot_javascript(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    ns = (90_000 if split == "train" else 100_000) + fold_index * 1_000 + serial
    tool_idx = serial % 12
    if tool_idx == 0:
        tool = "submitAtCoordinate"
        arguments = {
            "action": "submit",
            "formId": f"loginForm_{ns:05d}",
            "coordinates": [round(0.1 + (ns % 8) * 0.1, 2), round(0.2 + (ns % 7) * 0.1, 2)],
        }
        request = (
            f"How can I send a '{arguments['action']}' action to a React form with ID "
            f"'{arguments['formId']}' at coordinate {arguments['coordinates']}?"
        )
    elif tool_idx == 1:
        tool = "manageReactState"
        arguments = {
            "store": {"initialState": f"initialStateObject_{ns:05d}"},
            "context": f"React.createContext_{ns:05d}()",
            "hooks": {"useStateSelector": f"useStateSelectorHook_{ns:05d}"},
        }
        request = (
            f"Given the manageReactState function, write a line of code to initialize this function "
            f"with store, context `{arguments['context']}`, and custom hooks."
        )
    elif tool_idx == 2:
        tool = "getNextKeyValues"
        arguments = {
            "ctx": f"dataAnalysisContext_{ns:05d}",
            "currentKey": f"userId_{ns:05d}",
        }
        request = (
            f"When analyzing JSON data structures, how can I extract all key-value pairs that follow "
            f"a specific key '{arguments['currentKey']}' within context object `{arguments['ctx']}`?"
        )
    elif tool_idx == 3:
        tool = "doesEmailInputExist"
        arguments = {
            "formElem": f"emailForm_{ns:05d}",
            "inputName": f"emailAddress_{ns:05d}",
        }
        request = (
            f"How can I determine if an email form element referred to as `{arguments['formElem']}` "
            f"includes an input with the name attribute '{arguments['inputName']}'?"
        )
    elif tool_idx == 4:
        tool = "DynamicChartGenerator"
        arguments = {
            "userData": [f"series_{ns}_a", f"series_{ns}_b"],
            "scalingFactor": float((ns % 5) + 1),
            "dashboard": f"dashboardElement_{ns:05d}",
        }
        request = (
            f"How can I generate a dynamic chart with user data `{arguments['userData']}` and "
            f"apply a scaling factor of {arguments['scalingFactor']}, linking it to dashboard `{arguments['dashboard']}`?"
        )
    elif tool_idx == 5:
        tool = "chartDataAccessorFactory"
        arguments = {
            "chart": {"nm": f"BarChart_{ns:05d}", "mn": f"chartModule_{ns:05d}"},
            "library": f"visualizationLibrary_{ns:05d}",
            "configObject": f"config_{ns:05d}",
        }
        request = (
            f"How can I generate a data accessor for a chart component named '{arguments['chart']['nm']}' "
            f"in library `{arguments['library']}` with configuration object named '{arguments['configObject']}'?"
        )
    elif tool_idx == 6:
        tool = "generateNotificationHandler"
        arguments = {
            "app": f"app_{ns:05d}",
            "priorityLevel": (ns % 3) + 1,
            "messagingService": f"messagingSvc_{ns:05d}",
            "notificationType": (ns % 4) + 1,
        }
        request = (
            f"How can I generate a notification handler for application `{arguments['app']}` that filters "
            f"messages based on priority level {arguments['priorityLevel']}, linked to messaging service "
            f"'{arguments['messagingService']}', and categorized under notification type {arguments['notificationType']}?"
        )
    elif tool_idx == 7:
        tool = "createAuthToken"
        arguments = {
            "username": f"johndoe_{ns:05d}",
            "options": {"role": "admin", "algorithm": "HS256"},
        }
        request = (
            f"How can I generate an authorization token for user with username '{arguments['username']}'?"
        )
    elif tool_idx == 8:
        tool = "trackSubmitWithValidation"
        arguments = {
            "obj": f"formHandler_{ns:05d}",
            "validationFlags": ["isRequired", "isValidEmail"],
        }
        request = (
            f"How can I track the 'submitForm' action on `{arguments['obj']}` object when validation flags are set?"
        )
    elif tool_idx == 9:
        tool = "validateReactProp"
        arguments = {
            "obj": f"serviceProvider_{ns:05d}",
            "componentName": f"UserProfile_{ns:05d}",
        }
        request = (
            f"How can I validate an object named `{arguments['obj']}` for React prop-type constraints "
            f"when passed to component '{arguments['componentName']}'?"
        )
    elif tool_idx == 10:
        tool = "transformAllDecoratorsOfDeclaration"
        arguments = {
            "node": f"myNode_{ns:05d}",
            "container": f"myContainer_{ns:05d}",
        }
        request = (
            f"How can I process and transform all decorators of a TypeScript declaration node named `{arguments['node']}` "
            f"within a container named `{arguments['container']}`?"
        )
    else:
        tool = "updateDOMListeners"
        arguments = {
            "oldVnode": f"oldVnode_{ns:05d}",
            "vnode": f"newVnode_{ns:05d}",
        }
        request = (
            f"How can I update DOM event listeners from old virtual node `{arguments['oldVnode']}` to new one `{arguments['vnode']}`?"
        )

    return _record(
        split=split,
        case_id=f"{split}-js-{serial:04d}",
        category="polyglot_javascript",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=_native_calls([(tool, arguments)]),
        tool=tool,
        arguments=arguments,
    )


def _parallel_multiple(split: str, serial: int, fold_index: int = 0) -> dict[str, Any]:
    ns = (110_000 if split == "train" else 120_000) + fold_index * 1_000 + serial
    path1 = f"src/modules/mod_{ns:05d}.py"
    path2 = f"docs/notes/note_{ns:05d}.md"
    pattern = f"PARALLEL_{ns:05d}"
    
    variant = serial % 4
    if variant == 0:
        calls = [
            ("read", {"path": path1, "limit": 40, "offset": 0}),
            ("grep", {"path": "src/", "pattern": pattern}),
        ]
        request = f"Please read the first 40 lines of `{path1}` and concurrently search `src/` for pattern `{pattern}`."
        tool = "read"
        arguments = {"path": path1, "limit": 40, "offset": 0}
    elif variant == 1:
        calls = [
            ("glob", {"path": "src/", "pattern": f"*{ns % 100}*.py"}),
            ("bash", {"command": f"pytest tests/test_{ns % 50}.py"}),
        ]
        request = f"Find files matching pattern `*{ns % 100}*.py` under `src/` and run command `pytest tests/test_{ns % 50}.py`."
        tool = "glob"
        arguments = {"path": "src/", "pattern": f"*{ns % 100}*.py"}
    elif variant == 2:
        calls = [
            ("write", {"path": path2, "content": f"Marker {pattern}\n"}),
            ("bash", {"command": f"git add {path2}"}),
        ]
        request = f"Write 'Marker {pattern}\\n' into `{path2}` and stage it using git."
        tool = "write"
        arguments = {"path": path2, "content": f"Marker {pattern}\n"}
    else:
        calls = [
            ("websearch", {"query": f"python benchmark status {pattern}"}),
            ("websearch", {"query": f"leaderboard release {pattern}"}),
        ]
        request = f"Search for 'python benchmark status {pattern}' and also search for 'leaderboard release {pattern}'."
        tool = "websearch"
        arguments = {"query": f"python benchmark status {pattern}"}

    return _record(
        split=split,
        case_id=f"{split}-parallel-{serial:04d}",
        category="parallel_multiple",
        prompt=[_system(), {"role": "user", "content": request}],
        completion=_native_calls(calls),
        tool=tool,
        arguments=arguments,
    )


_BUILDERS = {
    "strict_ast": _strict_ast,
    "routing_negative": _routing_negative,
    "web_multihop": _web_multihop,
    "memory": _memory,
    "multi_step": _multi_step,
    "polyglot_java": _polyglot_java,
    "polyglot_javascript": _polyglot_javascript,
    "parallel_multiple": _parallel_multiple,
    "rehearsal": _rehearsal,
}


def build_train_rows() -> list[dict[str, Any]]:
    rows = [
        _BUILDERS[category]("train", serial)
        for category in CATEGORIES
        for serial in range(TRAIN_PER_CATEGORY)
    ]
    random.Random(2026082301).shuffle(rows)
    return rows


def build_dev_rows(fold: str) -> list[dict[str, Any]]:
    if fold not in DEV_FOLDS:
        raise ValueError(f"unknown DEV fold: {fold}")
    fold_index = DEV_FOLDS.index(fold)
    rows = [
        _BUILDERS[category](f"dev-{fold}", serial, fold_index)
        for category in CATEGORIES
        for serial in range(DEV_PER_CATEGORY)
    ]
    random.Random(2026082310 + fold_index).shuffle(rows)
    return rows


def _corrupt(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "__near_miss"
    if isinstance(value, list):
        return value[:-1] if value else [{"content": "wrong", "status": "pending"}]
    if isinstance(value, dict):
        result = dict(value)
        result["near_miss"] = True
        return result
    return "wrong-type"


def _rejected_for(row: dict[str, Any], index: int) -> tuple[str, str]:
    metadata = row["training_metadata"]
    tool = metadata["tool"]
    if tool is None:
        prompt_text = str(row["prompt"][-1].get("content") or "")
        return _native_calls([("websearch", {"query": prompt_text[:96]})]), "overtrigger"
    arguments = dict(metadata["expected_arguments"])
    mutation = index % 6
    if mutation == 0:
        if arguments:
            arguments.pop(next(iter(arguments)))
        return _native_calls([(tool, arguments)]), "missing-key"
    if mutation == 1:
        if arguments:
            key = next(iter(arguments))
            arguments[key] = _corrupt(arguments[key])
        return _native_calls([(tool, arguments)]), "wrong-value"
    if mutation == 2:
        arguments["unrequested"] = f"extra-{index:04d}"
        return _native_calls([(tool, arguments)]), "extra-key"
    if mutation == 3:
        if arguments:
            key = next(iter(arguments))
            arguments[key] = {"wrong": arguments[key]}
        return _native_calls([(tool, arguments)]), "wrong-type"
    if mutation == 4:
        other = TOOLS[(TOOLS.index(tool) + 1) % len(TOOLS)] if tool in TOOLS else "read"
        return _native_calls([(other, arguments)]), "wrong-tool"
    exact = _native_calls([(tool, arguments)])
    return f"Certainly.\n{exact}", "extra-prose"


def build_preference_rows(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for index, row in enumerate(train_rows):
        rejected, mutation = _rejected_for(row, index)
        pairs.append(
            {
                "prompt": row["prompt"],
                "chosen": row["completion"],
                "rejected": [{"role": "assistant", "content": rejected}],
                "preference_metadata": {
                    "case_id": row["training_metadata"]["case_id"],
                    "category": row["training_metadata"]["category"],
                    "mutation": mutation,
                    "schema_version": SCHEMA_VERSION,
                },
            }
        )
    return pairs


def _signature(row: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"prompt": row["prompt"], "completion": row["completion"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def audit_corpus(
    train_rows: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
    dev_folds: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    problems: list[str] = []
    train_categories = Counter(row["training_metadata"]["category"] for row in train_rows)
    if set(train_categories) != set(CATEGORIES) or set(train_categories.values()) != {
        TRAIN_PER_CATEGORY
    }:
        problems.append(f"unexpected training category distribution: {dict(train_categories)}")
    signatures: dict[str, set[str]] = {"train": {_signature(row) for row in train_rows}}
    if len(signatures["train"]) != len(train_rows):
        problems.append("training rows are not unique")
    for fold, rows in dev_folds.items():
        counts = Counter(row["evaluation_metadata"]["category"] for row in rows)
        if set(counts) != set(CATEGORIES) or set(counts.values()) != {DEV_PER_CATEGORY}:
            problems.append(f"DEV-{fold} is not 10-per-category: {dict(counts)}")
        signatures[fold] = {_signature(row) for row in rows}
        if len(signatures[fold]) != len(rows):
            problems.append(f"DEV-{fold} rows are not unique")
        if signatures["train"] & signatures[fold]:
            problems.append(f"training overlaps DEV-{fold}")
    for left_index, left in enumerate(DEV_FOLDS):
        for right in DEV_FOLDS[left_index + 1 :]:
            if signatures[left] & signatures[right]:
                problems.append(f"DEV-{left} overlaps DEV-{right}")
    all_rows = train_rows + [row for rows in dev_folds.values() for row in rows]
    menu_counts = [
        str(row["prompt"][0].get("content") or "").count("List of tools: [")
        for row in all_rows
    ]
    if set(menu_counts) != {1}:
        problems.append("every prompt must embed exactly one tool menu")
    unparseable = 0
    for row in all_rows:
        key = "training_metadata" if "training_metadata" in row else "evaluation_metadata"
        if row[key]["tool"] and not parse_native_tool_calls(row["completion"][0]["content"])[1]:
            unparseable += 1
    if unparseable:
        problems.append(f"{unparseable} expected tool calls are not parseable")
    if len(preferences) != len(train_rows):
        problems.append("one preference pair is required per training row")
    if any(pair["chosen"] == pair["rejected"] for pair in preferences):
        problems.append("a preference pair has identical chosen/rejected targets")
    report = {
        "status": "blocked" if problems else "ready",
        "schema_version": SCHEMA_VERSION,
        "benchmark_sources_used": [],
        "sealed_sources_used": [],
        "train_rows": len(train_rows),
        "preference_pairs": len(preferences),
        "train_categories": dict(sorted(train_categories.items())),
        "dev_folds": {
            fold: {
                "rows": len(rows),
                "categories": dict(
                    sorted(Counter(row["evaluation_metadata"]["category"] for row in rows).items())
                ),
            }
            for fold, rows in dev_folds.items()
        },
        "exact_overlap": {
            f"train_dev_{fold}": len(signatures["train"] & signatures[fold])
            for fold in DEV_FOLDS
        },
        "single_tool_menu_per_prompt": set(menu_counts) == {1},
        "problems": problems,
    }
    report["fingerprint"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    if not path.exists() or path.read_text(encoding="utf-8") != payload:
        path.write_text(payload, encoding="utf-8")


def ensure_strict_genetic_corpus() -> dict[str, Any]:
    train_rows = build_train_rows()
    preferences = build_preference_rows(train_rows)
    dev_folds = {fold: build_dev_rows(fold) for fold in DEV_FOLDS}
    audit = audit_corpus(train_rows, preferences, dev_folds)
    if audit["status"] != "ready":
        raise ValueError(f"strict genetic corpus audit failed: {audit['problems']}")
    _write_jsonl(TRAIN_PATH, train_rows)
    _write_jsonl(PREFERENCE_PATH, preferences)
    for fold, rows in dev_folds.items():
        _write_jsonl(dev_path(fold), rows)
    manifest = {
        **audit,
        "paths": {
            "train": str(TRAIN_PATH.resolve()),
            "preferences": str(PREFERENCE_PATH.resolve()),
            **{f"dev_{fold}": str(dev_path(fold).resolve()) for fold in DEV_FOLDS},
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

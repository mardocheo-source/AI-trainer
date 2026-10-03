from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .paths import project_root
from .tool_schema import KNOWN_TOOLS, STANDARD_TOOLS, POLYGLOT_TOOLS, schema_conditioned_system_prompt

SCHEMA_VERSION = 3
OUTPUT_DIR = project_root() / "downloads" / "generated" / "v14_1-hybrid"
TRAIN_PATH = OUTPUT_DIR / "train-master.jsonl"
DEV_FOLDS = ("a", "b", "c")

V14_1_POLICY = (
    "You are an expert autonomous tool-calling assistant. When tools are provided, call "
    "the appropriate function(s) if and only if the user request requires an action supported "
    "by the declared tools. If the request lacks required parameters, ask the user for clarification. "
    "If the request is conversational or no tool is relevant, answer directly in natural language."
)


def dev_path(fold: str) -> Path:
    if fold not in DEV_FOLDS:
        raise ValueError(f"unknown DEV fold: {fold}")
    return OUTPUT_DIR / f"dev-{fold}.jsonl"


def format_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(arguments.items()))
    return f"<|tool_call_start|>[{tool_name}({args_str})]<|tool_call_end|>"


def format_parallel_tool_calls(calls: list[tuple[str, dict[str, Any]]]) -> str:
    rendered = []
    for tool_name, arguments in calls:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(arguments.items()))
        rendered.append(f"{tool_name}({args_str})")
    return f"<|tool_call_start|>[{', '.join(rendered)}]<|tool_call_end|>"


# 1. Conversational Negatives & Irrelevance
def _build_conversational_negative(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (100_000 if fold_idx == 0 else 120_000) + fold_idx * 1_000 + serial
    t = serial % 8
    declared_tools = ["FireBirdUtils.getViewSourceWithHeader", "submitAtCoordinate", "sendEmail", "calculateMortgage"]
    
    if t == 0:
        q = "Can you explain the main difference between TCP and UDP in networking?"
        ans = "TCP (Transmission Control Protocol) is a connection-oriented, reliable protocol that guarantees in-order packet delivery using acknowledgments and retransmissions. In contrast, UDP (User Datagram Protocol) is connectionless and lightweight, providing lower latency without delivery guarantees, making it ideal for real-time applications like video streaming and gaming."
    elif t == 1:
        q = "How do I implement a binary search algorithm in Python?"
        ans = "To implement binary search, maintain two pointers (left and right) on a sorted list, repeatedly check the midpoint element, and halve the search interval until the target is found or pointers cross."
    elif t == 2:
        q = "What is the capital of Australia and what is its population?"
        ans = "The capital of Australia is Canberra. It has a population of approximately 450,000 residents."
    elif t == 3:
        q = "I watched an interesting movie last night about firefighters in California."
        ans = "That sounds fascinating! California wildland and structural firefighting has been the subject of several gripping films, highlighting the bravery and coordination required in emergency operations."
    elif t == 4:
        q = "Do not use any tools or functions. Tell me what HTTP status code 404 indicates."
        ans = "HTTP status code 404 Not Found indicates that the server cannot find the requested resource. The link may be broken or the URL entered incorrectly."
    elif t == 5:
        q = "How does photosynthesis convert sunlight into chemical energy?"
        ans = "Photosynthesis occurs in plant chloroplasts, where chlorophyll captures sunlight to convert water and carbon dioxide into glucose and oxygen through light-dependent and Calvin cycle reactions."
    elif t == 6:
        q = "Can you write a poem about the sea during a thunderstorm?"
        ans = "The dark horizon swells with wrath,\nA lightning flash illumes the path.\nThe roaring tides crash on the shore,\nAs thunder rumbles evermore."
    else:
        q = "Explain why time complexity of merge sort is O(n log n) in all cases."
        ans = "Merge sort divides the array in half at each step (log n division levels) and takes linear O(n) time to merge the sorted sub-arrays at each level, resulting in guaranteed O(n log n) complexity across best, average, and worst cases."

    sys_p = schema_conditioned_system_prompt(V14_1_POLICY, declared_tools)
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ],
        "category": "conversational_negative",
        "expected_tool": None,
    }


# 2. Parallel Multiple Tool Calling
def _build_parallel_multiple_record(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (130_000 if fold_idx == 0 else 140_000) + fold_idx * 1_000 + serial
    t = serial % 4
    if t == 0:
        calls = [
            ("getWeatherForecast", {"city": f"Rome_{ns:05d}", "days": 3}),
            ("findHotels", {"city": f"Rome_{ns:05d}", "stars": 4, "maxPrice": 180}),
        ]
        q = f"Get a 3-day weather forecast for Rome_{ns:05d} and search for 4-star hotels under 180 euros in the same city."
    elif t == 1:
        calls = [
            ("createAuthToken", {"userId": f"usr_{ns:05d}", "expiresInSeconds": 3600}),
            ("sendNotification", {"recipient": f"usr_{ns:05d}", "type": "sms", "message": "Your login token has been generated."}),
        ]
        q = f"Generate an auth token for user '{calls[0][1]['userId']}' valid for 3600 seconds, and send an SMS notification to the user."
    elif t == 2:
        calls = [
            ("DB2Tablespace.resolveTablespaceReference", {"dataSource": f"db_{ns:05d}", "monitor": "mainMon", "reference": "TS_USER"}),
            ("DataSerializer.serializePayload", {"format": "json_canonical", "payload": {"status": "synced", "id": ns}}),
        ]
        q = f"Resolve tablespace reference 'TS_USER' on data source 'db_{ns:05d}' and serialize status payload {calls[1][1]['payload']} to canonical JSON."
    else:
        calls = [
            ("updateDOMListeners", {"elementId": f"btn_{ns:05d}", "event": "click", "handler": "onSubmit"}),
            ("manageReactState", {"key": f"formState_{ns:05d}", "value": {"submitted": True}}),
        ]
        q = f"Update the click listener on element 'btn_{ns:05d}' to 'onSubmit' and update React state key 'formState_{ns:05d}' with submitted true."

    tools_used = [c[0] for c in calls]
    sys_p = schema_conditioned_system_prompt(V14_1_POLICY, tools_used)
    ans = format_parallel_tool_calls(calls)
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ],
        "category": "parallel_multiple",
        "expected_tool": "multiple",
        "expected_arguments": {c[0]: c[1] for c in calls},
    }


# 3. Multi-Turn Clarification Dialogues (Miss Param / Miss Func)
def _build_multi_turn_clarification(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (150_000 if fold_idx == 0 else 160_000) + fold_idx * 1_000 + serial
    t = serial % 3
    if t == 0:
        tool = "bookFlight"
        sys_p = schema_conditioned_system_prompt(V14_1_POLICY, [tool, "findHotels"])
        q1 = f"Please book a flight to Tokyo for passenger Passenger_{ns:05d}."
        a1 = f"To book your flight to Tokyo for Passenger_{ns:05d}, could you please provide your departure city and preferred travel date?"
        q2 = "I will be departing from San Francisco on October 12, 2026."
        call_args = {"passenger": f"Passenger_{ns:05d}", "destination": "Tokyo", "departureCity": "San Francisco", "date": "2026-10-12"}
        a2 = format_tool_call(tool, call_args)
    elif t == 1:
        tool = "transferFunds"
        sys_p = schema_conditioned_system_prompt(V14_1_POLICY, [tool, "checkBalance"])
        q1 = f"Transfer 500 dollars from account ACC_{ns:05d}."
        a1 = f"To complete the transfer of $500 from account ACC_{ns:05d}, please specify the recipient account ID."
        q2 = f"Transfer it to account ACC_{(ns+1):05d}."
        call_args = {"sourceAccount": f"ACC_{ns:05d}", "recipientAccount": f"ACC_{(ns+1):05d}", "amount": 500}
        a2 = format_tool_call(tool, call_args)
    else:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        sys_p = schema_conditioned_system_prompt(V14_1_POLICY, [tool])
        q1 = f"Generate the view script for view 'SalesView_{ns:05d}'."
        a1 = f"To generate the view source script for 'SalesView_{ns:05d}', please provide the progress monitor object and the SQL source definition."
        q2 = f"Use progress monitor 'mon_{ns:05d}' and source 'SELECT * FROM Sales'."
        call_args = {"monitor": f"mon_{ns:05d}", "view": f"SalesView_{ns:05d}", "source": "SELECT * FROM Sales"}
        a2 = format_tool_call(tool, call_args)

    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
            {"role": "assistant", "content": a2},
        ],
        "category": "multi_turn_clarification",
        "expected_tool": tool,
        "expected_arguments": call_args,
    }


# 4. Polyglot Java & JS with Nested Types & Package Dot-Notation
def _build_polyglot_java_v14_1(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (70_000 if fold_idx == 0 else 90_000) + fold_idx * 1_000 + serial
    t = serial % 6
    if t == 0:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        args = {"monitor": f"dbMonitor_{ns:05d}", "view": f"EmployeeView_{ns:05d}", "source": f"SELECT * FROM Employee_{ns:05d} WHERE status = 'active'"}
        q = f"How can I generate the full SQL creation script with a header for a Firebird database view named '{args['view']}', using a progress monitor '{args['monitor']}' and the original source '{args['source']}'?"
    elif t == 1:
        tool = "DB2Tablespace.resolveTablespaceReference"
        args = {"monitor": f"dbMonitor_{ns:05d}", "dataSource": f"db2DataSource_{ns:05d}", "reference": f"USERSPACE_{ns:05d}"}
        q = f"How can I resolve a tablespace reference named '{args['reference']}' in a DB2 database using data source object '{args['dataSource']}' and progress monitor '{args['monitor']}'?"
    elif t == 2:
        tool = "BasePolicyDataProvider.getRegistryPolicyValue"
        args = {"root": "WinReg.HKEY_LOCAL_MACHINE", "property": f"Policy_{ns:05d}"}
        q = f"What is the value of '{args['property']}' in the Windows registry under {args['root']}?"
    elif t == 3:
        tool = "JNIBridge.setLauncherInfo"
        args = {"launcher": f"/usr/local/bin/launcher_{ns:05d}", "name": f"AppLauncher_{ns:05d}"}
        q = f"How can I update the launcher information in the JNI Bridge with launcher path '{args['launcher']}' and name '{args['name']}'?"
    elif t == 4:
        tool = "DataSerializer.serializePayload"
        args = {"format": "json_canonical", "payload": {"id": ns, "active": True, "roles": ["admin", "editor"]}}
        q = f"Serialize payload {args['payload']} using format '{args['format']}'."
    else:
        tool = "ExasolExecutionContext.setCurrentSchema"
        args = {"monitor": f"progressMonitor_{ns:05d}", "schemaName": f"AnalyticsDB_{ns:05d}"}
        q = f"How do I change the current schema to '{args['schemaName']}' in Exasol context with monitor '{args['monitor']}'?"

    sys_p = schema_conditioned_system_prompt(V14_1_POLICY, [tool])
    ans = format_tool_call(tool, args)
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ],
        "expected_tool": tool,
        "expected_arguments": args,
        "category": "polyglot_java",
    }


def _build_polyglot_js_v14_1(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (80_000 if fold_idx == 0 else 95_000) + fold_idx * 1_000 + serial
    t = serial % 5
    if t == 0:
        tool = "submitAtCoordinate"
        args = {"coordinates": [30 + (ns % 200), 60 + (ns % 300)], "formId": f"loginForm_{ns:05d}", "action": "submit"}
        q = f"Trigger a submit action on form '{args['formId']}' at screen coordinates {args['coordinates']}."
    elif t == 1:
        tool = "manageReactState"
        args = {"key": f"userSession_{ns:05d}", "value": {"authenticated": True, "tokenExpiry": 7200}}
        q = f"Update React state key '{args['key']}' with object {args['value']}."
    elif t == 2:
        tool = "getNextKeyValues"
        args = {"ctx": f"dataAnalysisContext_{ns:05d}", "currentKey": f"userKey_{ns:05d}"}
        q = f"Retrieve the next key values for key '{args['currentKey']}' in context '{args['ctx']}'."
    elif t == 3:
        tool = "doesEmailInputExist"
        args = {"formElem": f"emailForm_{ns:05d}", "inputName": "emailAddress"}
        q = f"Check if email input '{args['inputName']}' exists inside form '{args['formElem']}'."
    else:
        tool = "DynamicChartGenerator"
        args = {"dashboard": f"dash_{ns:05d}", "scalingFactor": 3, "userData": [10, 25, 40]}
        q = f"Generate a dynamic chart on dashboard '{args['dashboard']}' with scaling factor {args['scalingFactor']} and data {args['userData']}."

    sys_p = schema_conditioned_system_prompt(V14_1_POLICY, [tool])
    ans = format_tool_call(tool, args)
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ans},
        ],
        "expected_tool": tool,
        "expected_arguments": args,
        "category": "polyglot_javascript",
    }


def _row_content_key(row: dict[str, Any]) -> str:
    msgs = row.get("messages", [])
    if msgs:
        return json.dumps([{"r": m.get("role"), "c": m.get("content")} for m in msgs], sort_keys=True)
    return json.dumps(row, sort_keys=True)


def build_v14_1_corpus(
    seed: int = 20260824,
) -> dict[str, Any]:
    rng = random.Random(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Ingest Glaive v2 records
    from .v14_data_engine import _build_glaive_records
    glaive_all = _build_glaive_records(max_records=1200)
    rng.shuffle(glaive_all)
    glaive_train = glaive_all[:int(len(glaive_all) * 0.85)]
    glaive_dev = glaive_all[int(len(glaive_all) * 0.85):]

    dev_folds = {fold: [] for fold in DEV_FOLDS}
    dev_keys = set()

    for idx, row in enumerate(glaive_dev):
        fold = DEV_FOLDS[idx % len(DEV_FOLDS)]
        dev_folds[fold].append(row)
        dev_keys.add(_row_content_key(row))

    # Dev folds for new categories
    builders_dev = [
        (_build_conversational_negative, 30, "conversational_negative"),
        (_build_parallel_multiple_record, 20, "parallel_multiple"),
        (_build_multi_turn_clarification, 20, "multi_turn_clarification"),
        (_build_polyglot_java_v14_1, 20, "polyglot_java"),
        (_build_polyglot_js_v14_1, 20, "polyglot_javascript"),
    ]
    for b_fn, count, cat in builders_dev:
        for f_idx, fold in enumerate(DEV_FOLDS):
            for s in range(count):
                dev_row = b_fn(s, fold_idx=f_idx + 1)
                dev_folds[fold].append(dev_row)
                dev_keys.add(_row_content_key(dev_row))

    # Standard existing synthetic categories
    from .strict_genetic_data import (
        _strict_ast, _routing_negative, _web_multihop, _memory, _multi_step,
        _parallel_multiple, _rehearsal
    )
    cat_map = {
        "strict_ast": _strict_ast,
        "routing_negative": _routing_negative,
        "web_multihop": _web_multihop,
        "memory": _memory,
        "multi_step": _multi_step,
        "parallel_multiple": _parallel_multiple,
        "rehearsal": _rehearsal,
    }
    for cat, fn in cat_map.items():
        for f_idx, fold in enumerate(DEV_FOLDS):
            for s in range(15):
                dev_row = fn("dev", s, fold_index=f_idx + 1)
                dev_folds[fold].append(dev_row)
                dev_keys.add(_row_content_key(dev_row))

    # Build balanced Train set
    train_rows = []
    
    # 1. Conversational Negatives (900 records)
    for s in range(900):
        r = _build_conversational_negative(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 2. Parallel Multiple (400 records)
    for s in range(400):
        r = _build_parallel_multiple_record(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 3. Multi-Turn Clarification (350 records)
    for s in range(350):
        r = _build_multi_turn_clarification(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 4. Polyglot Java (400 records) & JS (350 records)
    for s in range(400):
        r = _build_polyglot_java_v14_1(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)
    for s in range(350):
        r = _build_polyglot_js_v14_1(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 5. Standard synthetics (250 per cat)
    for cat, fn in cat_map.items():
        for s in range(250):
            r = fn("train", s)
            if _row_content_key(r) not in dev_keys:
                train_rows.append(r)

    # 6. Glaive v2
    for r in glaive_train:
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    rng.shuffle(train_rows)
    for fold in DEV_FOLDS:
        rng.shuffle(dev_folds[fold])

    # Write files
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    dev_manifest = {}
    for fold in DEV_FOLDS:
        p = dev_path(fold)
        with open(p, "w", encoding="utf-8") as f:
            for r in dev_folds[fold]:
                f.write(json.dumps(r) + "\n")
        dev_manifest[fold] = len(dev_folds[fold])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "total_train_rows": len(train_rows),
        "train_categories": dict(Counter(r.get("category", "unknown") for r in train_rows)),
        "dev_folds": dev_manifest,
        "fingerprint": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

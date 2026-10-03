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

SCHEMA_VERSION = 4
OUTPUT_DIR = project_root() / "downloads" / "generated" / "v15-tri-hybrid"
TRAIN_PATH = OUTPUT_DIR / "train-master.jsonl"
DEV_FOLDS = ("a", "b", "c")

V15_POLICY = (
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


# Expanded Conversational Negatives across 12 distinct domains
DOMAINS = [
    ("TCP/UDP Networking", "Can you explain the main difference between TCP and UDP in networking?", "TCP is a connection-oriented protocol providing reliable, ordered delivery with retransmissions, while UDP is connectionless and low-latency without reliability guarantees, optimal for streaming and gaming."),
    ("Binary Search", "How do I implement binary search in Python?", "Maintain two pointers (left and right), check the midpoint element against target, and halve search space repeatedly until found or pointers cross."),
    ("Geography", "What is the capital of Australia and its population?", "The capital of Australia is Canberra, with a population of around 450,000 residents."),
    ("Conversational Story", "I watched a fascinating movie yesterday about forest firefighters.", "That sounds exciting! Forest firefighting involves complex logistics and specialized aviation techniques like air tankers and smokejumpers."),
    ("HTTP Status", "Do not use tools. What does HTTP status code 503 mean?", "HTTP 503 Service Unavailable indicates that the server is currently unable to handle the request due to temporary overloading or maintenance."),
    ("Poetry", "Can you write a short poem about the night sky?", "The velvet night unfolds its grace,\\nWith diamonds scattered through deep space.\\nA crescent moon looks down below,\\nAs silent breezes softly blow."),
    ("Algorithms", "Explain why quicksort average complexity is O(n log n).", "Quicksort partitions the array around a pivot element. On average, balanced partitions yield recursion depth log n with linear partition work per level, totaling O(n log n)."),
    ("Quantum Physics", "What is the Heisenberg uncertainty principle?", "It states that the position and momentum of a quantum particle cannot both be measured with arbitrary precision simultaneously (Δx * Δp >= ħ/2)."),
    ("Software Architecture", "What are the core benefits of microservices over a monolith?", "Microservices offer independent scalability, fault isolation, modular deployment, and tech stack flexibility, at the cost of network complexity and distributed data consistency."),
    ("Database Indexing", "Why are B-Trees preferred over hash maps for database indexes?", "B-Trees maintain sorted order, allowing efficient range scans and prefix matching (O(log n)), whereas hash maps only support O(1) exact equality lookups."),
    ("Economics", "What is the difference between fiscal policy and monetary policy?", "Fiscal policy involves government taxation and spending decisions, while monetary policy is controlled by central banks managing interest rates and money supply."),
]


def _build_conversational_negative_v15(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (200_000 if fold_idx == 0 else 220_000) + fold_idx * 1_000 + serial
    dom = DOMAINS[serial % len(DOMAINS)]
    declared_tools = ["FireBirdUtils.getViewSourceWithHeader", "submitAtCoordinate", "sendEmail", "calculateMortgage", "DB2Tablespace.resolveTablespaceReference"]
    sys_p = schema_conditioned_system_prompt(V15_POLICY, declared_tools)
    return {
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": f"{dom[1]} (ref: {ns})"},
            {"role": "assistant", "content": dom[2]},
        ],
        "category": "conversational_negative",
        "expected_tool": None,
    }


def _build_parallel_multiple_record_v15(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (230_000 if fold_idx == 0 else 240_000) + fold_idx * 1_000 + serial
    t = serial % 5
    if t == 0:
        calls = [
            ("getWeatherForecast", {"city": f"Milan_{ns:05d}", "days": 5}),
            ("findHotels", {"city": f"Milan_{ns:05d}", "stars": 4, "maxPrice": 220}),
        ]
        q = f"Get a 5-day weather forecast for Milan_{ns:05d} and search for 4-star hotels under 220 euros in the same city."
    elif t == 1:
        calls = [
            ("createAuthToken", {"userId": f"usr_{ns:05d}", "expiresInSeconds": 7200}),
            ("sendNotification", {"recipient": f"usr_{ns:05d}", "type": "sms", "message": "Your security token has been generated."}),
        ]
        q = f"Generate an auth token for user '{calls[0][1]['userId']}' valid for 7200 seconds, and send an SMS notification to the user."
    elif t == 2:
        calls = [
            ("DB2Tablespace.resolveTablespaceReference", {"dataSource": f"db2DataSource_{ns:05d}", "monitor": "dbMonitor", "reference": "USERSPACE1"}),
            ("DataSerializer.serializePayload", {"format": "json_canonical", "payload": {"status": "active", "id": ns}}),
        ]
        q = f"Resolve tablespace reference 'USERSPACE1' on data source 'db2DataSource_{ns:05d}' using monitor 'dbMonitor', and serialize payload {calls[1][1]['payload']}."
    elif t == 3:
        calls = [
            ("updateDOMListeners", {"elementId": f"loginBtn_{ns:05d}", "event": "click", "handler": "handleSubmit"}),
            ("manageReactState", {"key": f"authStatus_{ns:05d}", "value": {"authenticated": True}}),
        ]
        q = f"Update the click listener on element 'loginBtn_{ns:05d}' to 'handleSubmit' and set React state '{calls[1][1]['key']}' with authenticated true."
    else:
        calls = [
            ("BasePolicyDataProvider.getRegistryPolicyValue", {"property": "EnableExtensions", "root": "WinReg.HKEY_LOCAL_MACHINE"}),
            ("JNIBridge.setLauncherInfo", {"launcher": "/usr/local/bin/dbeaver", "name": "DBeaverLauncher"}),
        ]
        q = "Check registry policy 'EnableExtensions' under WinReg.HKEY_LOCAL_MACHINE and configure JNI launcher '/usr/local/bin/dbeaver' named 'DBeaverLauncher'."

    tools_used = [c[0] for c in calls]
    sys_p = schema_conditioned_system_prompt(V15_POLICY, tools_used)
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


def _build_multi_turn_clarification_v15(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (250_000 if fold_idx == 0 else 260_000) + fold_idx * 1_000 + serial
    t = serial % 4
    if t == 0:
        tool = "bookFlight"
        sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool, "findHotels"])
        q1 = f"Please book a flight to Paris for passenger Client_{ns:05d}."
        a1 = f"To book your flight to Paris for Client_{ns:05d}, what is your departure city and preferred travel date?"
        q2 = "I am departing from Rome on October 25, 2026."
        call_args = {"passenger": f"Client_{ns:05d}", "destination": "Paris", "departureCity": "Rome", "date": "2026-10-25"}
        a2 = format_tool_call(tool, call_args)
    elif t == 1:
        tool = "transferFunds"
        sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool, "checkBalance"])
        q1 = f"Transfer 1200 dollars from account ACC_{ns:05d}."
        a1 = f"To process the transfer of $1200 from account ACC_{ns:05d}, please provide the recipient account number."
        q2 = f"The recipient account is ACC_{(ns+5):05d}."
        call_args = {"sourceAccount": f"ACC_{ns:05d}", "recipientAccount": f"ACC_{(ns+5):05d}", "amount": 1200}
        a2 = format_tool_call(tool, call_args)
    elif t == 2:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool])
        q1 = f"Generate the view script for view 'CustomReportView_{ns:05d}'."
        a1 = f"To generate the script for 'CustomReportView_{ns:05d}', please provide the progress monitor identifier and the SQL source query."
        q2 = f"Use monitor 'dbMonitor' and query 'SELECT * FROM Reports WHERE year = 2026'."
        call_args = {"monitor": "dbMonitor", "view": f"CustomReportView_{ns:05d}", "source": "SELECT * FROM Reports WHERE year = 2026"}
        a2 = format_tool_call(tool, call_args)
    else:
        tool = "submitAtCoordinate"
        sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool])
        q1 = f"Submit the form 'checkoutForm_{ns:05d}'."
        a1 = f"To submit 'checkoutForm_{ns:05d}', what screen coordinates [X, Y] should be clicked?"
        q2 = "Click at coordinates [120, 340]."
        call_args = {"formId": f"checkoutForm_{ns:05d}", "coordinates": [120, 340], "action": "submit"}
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


def _build_polyglot_java_v15(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (270_000 if fold_idx == 0 else 280_000) + fold_idx * 1_000 + serial
    t = serial % 6
    if t == 0:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        args = {"monitor": "dbMonitor", "view": f"EmployeeView_{ns:05d}", "source": f"SELECT * FROM Employee WHERE id = {ns}"}
        q = f"How can I generate the creation script with a header for Firebird view '{args['view']}' using monitor 'dbMonitor' and source '{args['source']}'?"
    elif t == 1:
        tool = "DB2Tablespace.resolveTablespaceReference"
        args = {"dataSource": "db2DataSource", "monitor": "dbMonitor", "reference": f"USERSPACE_{ns:05d}"}
        q = f"Resolve tablespace reference '{args['reference']}' on DB2 data source 'db2DataSource' with monitor 'dbMonitor'."
    elif t == 2:
        tool = "BasePolicyDataProvider.getRegistryPolicyValue"
        args = {"property": "EnableExtensions", "root": "WinReg.HKEY_LOCAL_MACHINE"}
        q = "What is the value of 'EnableExtensions' under Windows registry root 'WinReg.HKEY_LOCAL_MACHINE'?"
    elif t == 3:
        tool = "JNIBridge.setLauncherInfo"
        args = {"launcher": "/usr/local/bin/dbeaver", "name": "DBeaverLauncher"}
        q = "Set launcher information in JNI Bridge with launcher '/usr/local/bin/dbeaver' and name 'DBeaverLauncher'."
    elif t == 4:
        tool = "DataSerializer.serializePayload"
        args = {"format": "json_canonical", "payload": {"active": True, "id": ns, "roles": ["admin", "editor"]}}
        q = f"Serialize payload {args['payload']} into canonical JSON format."
    else:
        tool = "ExasolExecutionContext.setCurrentSchema"
        args = {"monitor": "dbMonitor", "schemaName": f"ProductionSchema_{ns:05d}"}
        q = f"Set current schema to '{args['schemaName']}' in Exasol context with monitor 'dbMonitor'."

    sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool])
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


def _build_polyglot_js_v15(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (285_000 if fold_idx == 0 else 295_000) + fold_idx * 1_000 + serial
    t = serial % 5
    if t == 0:
        tool = "submitAtCoordinate"
        args = {"action": "submit", "coordinates": [30, 60], "formId": f"loginForm_{ns:05d}"}
        q = f"Trigger a submit action on form '{args['formId']}' at screen coordinates [30, 60]."
    elif t == 1:
        tool = "manageReactState"
        args = {"key": f"userSession_{ns:05d}", "value": {"authenticated": True, "tokenExpiry": 7200}}
        q = f"Update React state key '{args['key']}' with object {args['value']}."
    elif t == 2:
        tool = "getNextKeyValues"
        args = {"ctx": "dataAnalysisContext", "currentKey": f"userKey_{ns:05d}"}
        q = f"Retrieve next key values for key '{args['currentKey']}' in context 'dataAnalysisContext'."
    elif t == 3:
        tool = "doesEmailInputExist"
        args = {"formElem": "emailForm", "inputName": "emailAddress"}
        q = "Check if input 'emailAddress' exists inside form 'emailForm'."
    else:
        tool = "DynamicChartGenerator"
        args = {"dashboard": "mainDashboard", "scalingFactor": 3, "userData": [10, 20, 30, 40]}
        q = "Generate dynamic chart on 'mainDashboard' with scaling factor 3 and data [10, 20, 30, 40]."

    sys_p = schema_conditioned_system_prompt(V15_POLICY, [tool])
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


def build_v15_corpus(seed: int = 20260825) -> dict[str, Any]:
    rng = random.Random(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from .v14_data_engine import _build_glaive_records
    glaive_all = _build_glaive_records(max_records=1500)
    rng.shuffle(glaive_all)
    glaive_train = glaive_all[:int(len(glaive_all) * 0.85)]
    glaive_dev = glaive_all[int(len(glaive_all) * 0.85):]

    dev_folds = {fold: [] for fold in DEV_FOLDS}
    dev_keys = set()

    for idx, row in enumerate(glaive_dev):
        fold = DEV_FOLDS[idx % len(DEV_FOLDS)]
        dev_folds[fold].append(row)
        dev_keys.add(_row_content_key(row))

    builders_dev = [
        (_build_conversational_negative_v15, 40, "conversational_negative"),
        (_build_parallel_multiple_record_v15, 25, "parallel_multiple"),
        (_build_multi_turn_clarification_v15, 25, "multi_turn_clarification"),
        (_build_polyglot_java_v15, 25, "polyglot_java"),
        (_build_polyglot_js_v15, 25, "polyglot_javascript"),
    ]
    for b_fn, count, cat in builders_dev:
        for f_idx, fold in enumerate(DEV_FOLDS):
            for s in range(count):
                dev_row = b_fn(s, fold_idx=f_idx + 1)
                dev_folds[fold].append(dev_row)
                dev_keys.add(_row_content_key(dev_row))

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

    train_rows = []

    # 1. Scaled Conversational Negatives (1,800 records)
    for s in range(1800):
        r = _build_conversational_negative_v15(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 2. Parallel Multiple (600 records)
    for s in range(600):
        r = _build_parallel_multiple_record_v15(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 3. Multi-Turn Clarification (600 records)
    for s in range(600):
        r = _build_multi_turn_clarification_v15(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 4. Polyglot Java (500 records) & JS (450 records)
    for s in range(500):
        r = _build_polyglot_java_v15(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)
    for s in range(450):
        r = _build_polyglot_js_v15(s, fold_idx=0)
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    # 5. Existing synthetics
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

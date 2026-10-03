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

SCHEMA_VERSION = 2
V14_CATEGORIES = (
    "strict_ast",
    "routing_negative",
    "web_multihop",
    "memory",
    "multi_step",
    "polyglot_java",
    "polyglot_javascript",
    "parallel_multiple",
    "glaive_multiturn",
    "glaive_single",
    "rehearsal",
)

OUTPUT_DIR = project_root() / "downloads" / "generated" / "v14-hybrid"
TRAIN_PATH = OUTPUT_DIR / "train-master.jsonl"
DEV_FOLDS = ("a", "b", "c")

V14_POLICY = (
    "You are an expert autonomous tool-calling assistant. When tools are provided, always call "
    "the appropriate function to fulfill the user's intent, including requests phrased as questions "
    "(e.g. 'How can I...', 'How do I...', 'Can you...'). Only answer directly in natural language "
    "when no declared tool is relevant or when the request explicitly forbids tool usage."
)


def dev_path(fold: str) -> Path:
    if fold not in DEV_FOLDS:
        raise ValueError(f"unknown DEV fold: {fold}")
    return OUTPUT_DIR / f"dev-{fold}.jsonl"


def format_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in sorted(arguments.items()))
    return f"<|tool_call_start|>[{tool_name}({args_str})]<|tool_call_end|>"


def _build_polyglot_java_record(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (70_000 if fold_idx == 0 else 90_000) + fold_idx * 1_000 + serial
    t = serial % 8
    if t == 0:
        tool = "FireBirdUtils.getViewSourceWithHeader"
        args = {"monitor": f"dbMonitor_{ns:05d}", "view": f"EmployeeView_{ns:05d}", "source": f"SELECT * FROM Employee_{ns:05d} WHERE status = 'active'"}
        q = f"How can I generate the full SQL creation script with a header for a Firebird database view named '{args['view']}', using a progress monitor '{args['monitor']}' and the original source '{args['source']}'?"
    elif t == 1:
        tool = "DB2Tablespace.resolveTablespaceReference"
        args = {"monitor": f"dbMonitor_{ns:05d}", "dataSource": f"db2DataSource_{ns:05d}", "reference": f"USERSPACE_{ns:05d}"}
        q = f"How can I resolve a tablespace reference named '{args['reference']}' in a DB2 database using data source object '{args['dataSource']}' and progress monitor '{args['monitor']}'?"
    elif t == 2:
        tool = "PmsProductServiceImpl.updateNewStatus"
        args = {"ids": [100 + (ns % 500), 200 + (ns % 500)], "newStatus": (ns % 3) + 1}
        q = f"How can I update the new status to {args['newStatus']} for product IDs {args['ids']} in the system?"
    elif t == 3:
        tool = "TwoSum.twoSum"
        args = {"nums": [2 + (ns % 5), 7, 11, 15], "target": 9 + (ns % 5)}
        q = f"What are the indices of the two numbers in the array {args['nums']} that add up to target sum {args['target']}?"
    elif t == 4:
        tool = "JNIBridge.setLauncherInfo"
        args = {"launcher": f"/usr/local/bin/launcher_{ns:05d}", "name": f"AppLauncher_{ns:05d}"}
        q = f"How can I update the launcher information in the JNI Bridge with launcher path '{args['launcher']}' and name '{args['name']}'?"
    elif t == 5:
        tool = "BasePolicyDataProvider.getRegistryPolicyValue"
        args = {"root": "WinReg.HKEY_LOCAL_MACHINE", "property": f"Policy_{ns:05d}"}
        q = f"What is the value of '{args['property']}' in the Windows registry under {args['root']}?"
    elif t == 6:
        tool = "ExasolExecutionContext.setCurrentSchema"
        args = {"monitor": f"progressMonitor_{ns:05d}", "schemaName": f"AnalyticsDB_{ns:05d}"}
        q = f"How do I change the current schema to '{args['schemaName']}' in Exasol context with monitor '{args['monitor']}'?"
    else:
        tool = "DataSerializer.serializePayload"
        args = {"format": "json_canonical" if ns % 2 == 0 else "msgpack", "payload": {"id": ns, "valid": True}}
        q = f"Serialize payload {args['payload']} using format '{args['format']}'."

    sys_p = schema_conditioned_system_prompt(V14_POLICY, {tool})
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


def _build_polyglot_js_record(serial: int, fold_idx: int = 0) -> dict[str, Any]:
    ns = (80_000 if fold_idx == 0 else 95_000) + fold_idx * 1_000 + serial
    t = serial % 7
    if t == 0:
        tool = "submitAtCoordinate"
        args = {"x": 100 + (ns % 800), "y": 200 + (ns % 600), "button": "left"}
        q = f"How can I trigger a submit action at screen coordinates x={args['x']}, y={args['y']} with {args['button']} button?"
    elif t == 1:
        tool = "manageReactState"
        args = {"key": f"userSession_{ns:05d}", "value": {"authenticated": True, "role": "admin"}}
        q = f"Can you update the React state key '{args['key']}' with value {args['value']}?"
    elif t == 2:
        tool = "getNextKeyValues"
        args = {"iteratorId": f"iter_{ns:05d}", "batchSize": 10 + (ns % 20)}
        q = f"Retrieve the next {args['batchSize']} key values from iterator '{args['iteratorId']}'."
    elif t == 3:
        tool = "DynamicChartGenerator"
        args = {"chartType": "bar", "data": [10 + (ns % 50), 20, 30], "title": f"Revenue_{ns:05d}"}
        q = f"How do I generate a dynamic {args['chartType']} chart with title '{args['title']}' for data {args['data']}?"
    elif t == 4:
        tool = "createAuthToken"
        args = {"userId": f"usr_{ns:05d}", "expiresInSeconds": 3600}
        q = f"Create an authentication token for user '{args['userId']}' expiring in {args['expiresInSeconds']} seconds."
    elif t == 5:
        tool = "validateReactProp"
        args = {"component": f"ButtonComponent_{ns:05d}", "propName": "onClick", "expectedType": "function"}
        q = f"Validate that prop '{args['propName']}' on component '{args['component']}' is of type '{args['expectedType']}'."
    else:
        tool = "updateDOMListeners"
        args = {"elementId": f"mainButton_{ns:05d}", "event": "click", "handler": "handleFormSubmit"}
        q = f"Update DOM listener on element '{args['elementId']}' for event '{args['event']}' with handler '{args['handler']}'."

    sys_p = schema_conditioned_system_prompt(V14_POLICY, {tool})
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


def _build_glaive_records(max_records: int = 1500) -> list[dict[str, Any]]:
    os.environ['HF_HOME'] = str(project_root() / '.cache' / 'huggingface')
    os.environ['HF_DATASETS_CACHE'] = str(project_root() / '.cache' / 'huggingface' / 'datasets')
    from datasets import load_dataset
    try:
        ds = load_dataset('glaiveai/glaive-function-calling-v2', split='train')
    except Exception:
        return []

    records = []
    seen_prompts = set()
    for item in ds:
        if len(records) >= max_records:
            break
        sys_text = item.get('system', '')
        chat_text = item.get('chat', '')
        
        funcs = re.findall(r'(\{\s*"name":\s*"[^"]+".*?\n\})', sys_text, re.DOTALL)
        if not funcs and 'You are a helpful assistant' not in sys_text:
            continue
            
        turns = re.split(r'\n\n(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):)', chat_text.strip())
        messages = []
        has_tool = False
        valid = True
        
        clean_sys = re.sub(r'SYSTEM:\s*', '', sys_text).strip()
        messages.append({"role": "system", "content": f"{V14_POLICY}\n\nTools available:\n{clean_sys}"})
        
        for turn in turns:
            turn = turn.strip()
            if turn.startswith('USER:'):
                c = turn.replace('USER:', '', 1).strip()
                messages.append({"role": "user", "content": c})
            elif turn.startswith('ASSISTANT:'):
                c = turn.replace('ASSISTANT:', '', 1).replace('<|endoftext|>', '').strip()
                fn_matches = re.findall(r'<functioncall>\s*(\{.*?\})', c, re.DOTALL)
                if fn_matches:
                    has_tool = True
                    tool_calls_text = []
                    for fn_m in fn_matches:
                        try:
                            parsed = json.loads(fn_m)
                            name = parsed.get('name')
                            args = parsed.get('arguments', {})
                            if isinstance(args, str):
                                try: args = json.loads(args)
                                except Exception: args = {'raw': args}
                            tool_calls_text.append(format_tool_call(name, args))
                        except Exception:
                            valid = False
                    if valid and tool_calls_text:
                        messages.append({"role": "assistant", "content": " ".join(tool_calls_text)})
                else:
                    messages.append({"role": "assistant", "content": c})
            elif turn.startswith('FUNCTION RESPONSE:'):
                c = turn.replace('FUNCTION RESPONSE:', '', 1).strip()
                messages.append({"role": "user", "content": f"[Tool Output: {c}]"})

        if valid and len(messages) >= 3:
            first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
            if first_user in seen_prompts:
                continue
            seen_prompts.add(first_user)
            records.append({
                "messages": messages,
                "category": "glaive_multiturn" if len(messages) > 3 else ("glaive_single" if has_tool else "rehearsal"),
                "expected_tool": "external_tool" if has_tool else None,
            })

    return records


def _row_content_key(row: dict[str, Any]) -> str:
    msgs = row.get("messages", [])
    if msgs:
        return json.dumps([{"r": m.get("role"), "c": m.get("content")} for m in msgs], sort_keys=True)
    return json.dumps(row, sort_keys=True)


def build_v14_corpus(
    train_per_cat: int = 350,
    dev_per_cat: int = 15,
    glaive_records: int = 1200,
    seed: int = 20260823,
) -> dict[str, Any]:
    rng = random.Random(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Ingest Glaive Open-Source Records
    glaive_all = _build_glaive_records(max_records=glaive_records)
    rng.shuffle(glaive_all)
    split_idx = int(len(glaive_all) * 0.85)
    glaive_train = glaive_all[:split_idx]
    glaive_dev = glaive_all[split_idx:]

    dev_folds = {fold: [] for fold in DEV_FOLDS}
    dev_keys = set()
    
    # Distribute Glaive dev
    for idx, row in enumerate(glaive_dev):
        fold = DEV_FOLDS[idx % len(DEV_FOLDS)]
        dev_folds[fold].append(row)
        dev_keys.add(_row_content_key(row))

    # 2. Synthetics
    for cat in ["polyglot_java", "polyglot_javascript"]:
        builder = _build_polyglot_java_record if cat == "polyglot_java" else _build_polyglot_js_record
        for f_idx, fold in enumerate(DEV_FOLDS):
            for s in range(dev_per_cat):
                dev_row = builder(s, fold_idx=f_idx + 1)
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
            for s in range(dev_per_cat):
                dev_row = fn("dev", s, fold_index=f_idx + 1)
                dev_folds[fold].append(dev_row)
                dev_keys.add(_row_content_key(dev_row))

    # 3. Build Training set ensuring 0% overlap with ANY dev sample
    train_rows = []
    for r in glaive_train:
        if _row_content_key(r) not in dev_keys:
            train_rows.append(r)

    for cat in ["polyglot_java", "polyglot_javascript"]:
        builder = _build_polyglot_java_record if cat == "polyglot_java" else _build_polyglot_js_record
        for s in range(train_per_cat):
            r = builder(s, fold_idx=0)
            if _row_content_key(r) not in dev_keys:
                train_rows.append(r)

    for cat, fn in cat_map.items():
        for s in range(train_per_cat):
            r = fn("train", s)
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
        "glaive_ingested": len(glaive_all),
        "fingerprint": hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest(),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

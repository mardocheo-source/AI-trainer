from __future__ import annotations

import json
from typing import Any

STANDARD_TOOLS = {
    "apply_patch",
    "bash",
    "edit",
    "glob",
    "grep",
    "lsp",
    "read",
    "skill",
    "task",
    "todowrite",
    "webfetch",
    "websearch",
    "write",
}

POLYGLOT_TOOLS = {
    # Polyglot Java
    "FireBirdUtils.getViewSourceWithHeader",
    "DB2Tablespace.resolveTablespaceReference",
    "PmsProductServiceImpl.updateNewStatus",
    "TwoSum.twoSum",
    "JNIBridge.setLauncherInfo",
    "configStorage.dynamicCredentialsScheduledExecutorService",
    "BasePolicyDataProvider.getRegistryPolicyValue",
    "ExasolExecutionContext.setCurrentSchema",
    "DataSerializer.serializePayload",
    "AuditLogManager.recordSecurityEvent",
    # Polyglot JavaScript
    "submitAtCoordinate",
    "manageReactState",
    "getNextKeyValues",
    "doesEmailInputExist",
    "DynamicChartGenerator",
    "chartDataAccessorFactory",
    "generateNotificationHandler",
    "createAuthToken",
    "trackSubmitWithValidation",
    "validateReactProp",
    "transformAllDecoratorsOfDeclaration",
    "updateDOMListeners",
}

KNOWN_TOOLS = STANDARD_TOOLS | POLYGLOT_TOOLS

REQUIRED_ARGUMENTS = {
    "read": {"path"},
    "write": {"path", "content"},
    "edit": {"path", "oldText", "newText"},
    "bash": {"command"},
    "apply_patch": {"patchText"},
    "glob": {"pattern"},
    "grep": {"path", "pattern"},
    "lsp": {"operation"},
    "skill": {"name"},
    "task": {"description", "prompt"},
    "todowrite": {"todos"},
    "webfetch": {"url"},
    "websearch": {"query"},
    # Polyglot Java
    "FireBirdUtils.getViewSourceWithHeader": {"monitor", "view", "source"},
    "DB2Tablespace.resolveTablespaceReference": {"monitor", "dataSource", "reference"},
    "PmsProductServiceImpl.updateNewStatus": {"ids", "newStatus"},
    "TwoSum.twoSum": {"nums", "target"},
    "JNIBridge.setLauncherInfo": {"launcher", "name"},
    "configStorage.dynamicCredentialsScheduledExecutorService": {"credentialsFile", "credentialsRefreshInterval", "basicCredentials"},
    "BasePolicyDataProvider.getRegistryPolicyValue": {"root", "property"},
    "ExasolExecutionContext.setCurrentSchema": {"monitor", "schemaName"},
    "DataSerializer.serializePayload": {"format", "payload"},
    "AuditLogManager.recordSecurityEvent": {"eventType", "severity"},
    # Polyglot JavaScript
    "submitAtCoordinate": {"action", "formId", "coordinates"},
    "manageReactState": {"store", "context", "hooks"},
    "getNextKeyValues": {"ctx", "currentKey"},
    "doesEmailInputExist": {"formElem", "inputName"},
    "DynamicChartGenerator": {"userData", "scalingFactor", "dashboard"},
    "chartDataAccessorFactory": {"chart", "library", "configObject"},
    "generateNotificationHandler": {"app", "priorityLevel", "messagingService", "notificationType"},
    "createAuthToken": {"username", "options"},
    "trackSubmitWithValidation": {"obj", "validationFlags"},
    "validateReactProp": {"obj", "componentName"},
    "transformAllDecoratorsOfDeclaration": {"node", "container"},
    "updateDOMListeners": {"oldVnode", "vnode"},
}

ARGUMENT_TYPES = {
    "read": {"path": str, "limit": int, "offset": int},
    "write": {"path": str, "content": str},
    "edit": {"path": str, "oldText": str, "newText": str, "replaceAll": bool},
    "bash": {"command": str, "description": str, "timeout": int},
    "apply_patch": {"patchText": str},
    "glob": {"pattern": str, "path": str},
    "grep": {"pattern": str, "path": str, "include": str},
    "lsp": {
        "operation": str,
        "filePath": str,
        "line": int,
        "character": int,
        "symbol": str,
    },
    "skill": {"name": str},
    "task": {"description": str, "prompt": str, "subagent_type": str},
    "todowrite": {"todos": list},
    "webfetch": {"url": str, "prompt": str},
    "websearch": {"query": str},
    # Polyglot Java
    "FireBirdUtils.getViewSourceWithHeader": {"monitor": str, "view": str, "source": str},
    "DB2Tablespace.resolveTablespaceReference": {"monitor": str, "dataSource": str, "reference": str},
    "PmsProductServiceImpl.updateNewStatus": {"ids": list, "newStatus": int},
    "TwoSum.twoSum": {"nums": list, "target": int},
    "JNIBridge.setLauncherInfo": {"launcher": str, "name": str},
    "configStorage.dynamicCredentialsScheduledExecutorService": {"credentialsFile": str, "credentialsRefreshInterval": int, "basicCredentials": str},
    "BasePolicyDataProvider.getRegistryPolicyValue": {"root": str, "property": str},
    "ExasolExecutionContext.setCurrentSchema": {"monitor": str, "schemaName": str},
    "DataSerializer.serializePayload": {"format": str, "payload": dict, "compress": bool},
    "AuditLogManager.recordSecurityEvent": {"eventType": str, "severity": int, "details": dict},
    # Polyglot JavaScript
    "submitAtCoordinate": {"action": str, "formId": str, "coordinates": list},
    "manageReactState": {"store": dict, "context": str, "hooks": dict},
    "getNextKeyValues": {"ctx": str, "currentKey": str},
    "doesEmailInputExist": {"formElem": str, "inputName": str},
    "DynamicChartGenerator": {"userData": list, "scalingFactor": float, "dashboard": str},
    "chartDataAccessorFactory": {"chart": dict, "library": str, "configObject": str},
    "generateNotificationHandler": {"app": str, "priorityLevel": int, "messagingService": str, "notificationType": int},
    "createAuthToken": {"username": str, "validity": int, "options": dict},
    "trackSubmitWithValidation": {"obj": str, "validationFlags": list},
    "validateReactProp": {"obj": str, "componentName": str},
    "transformAllDecoratorsOfDeclaration": {"node": str, "container": str},
    "updateDOMListeners": {"oldVnode": str, "vnode": str},
}

TOOL_DESCRIPTIONS = {
    "read": "Read a file from the repository.",
    "write": "Create or overwrite a file with exact content.",
    "edit": "Replace exact text in an existing file.",
    "bash": "Run a shell command in the repository.",
    "apply_patch": "Apply one exact unified patch to repository files.",
    "glob": "Find repository paths matching a glob pattern.",
    "grep": "Search repository files for a text or regular-expression pattern.",
    "lsp": "Query language-server symbols, definitions, references or diagnostics.",
    "skill": "Load a named specialist skill.",
    "task": "Delegate one bounded task to a subagent.",
    "todowrite": "Create or update the structured task list.",
    "webfetch": "Fetch a specific web page and optionally extract requested information.",
    "websearch": "Search the web for current information.",
    # Polyglot Java
    "FireBirdUtils.getViewSourceWithHeader": "Generates the SQL script to create or alter a Firebird database view.",
    "DB2Tablespace.resolveTablespaceReference": "Resolves a tablespace reference to a DB2Tablespace object using the data source.",
    "PmsProductServiceImpl.updateNewStatus": "Updates the status for a list of product IDs in the product management system.",
    "TwoSum.twoSum": "Finds two numbers in the array that add up to target sum and returns their indices.",
    "JNIBridge.setLauncherInfo": "Sets the launcher information in the JNI Bridge with launcher path and name.",
    "configStorage.dynamicCredentialsScheduledExecutorService": "Creates a ScheduledExecutorService that periodically loads credentials from a file.",
    "BasePolicyDataProvider.getRegistryPolicyValue": "Retrieves the value of a specified property from the registry policy node.",
    "ExasolExecutionContext.setCurrentSchema": "Sets the current schema for the Exasol execution context.",
    "DataSerializer.serializePayload": "Serializes structured payload dictionary to the target wire format.",
    "AuditLogManager.recordSecurityEvent": "Records a structured security event with severity and metadata.",
    # Polyglot JavaScript
    "submitAtCoordinate": "Sends a submit action to a React form element at specific position coordinates.",
    "manageReactState": "Encapsulates state management logic for React applications.",
    "getNextKeyValues": "Extracts key-value pairs in a JSON object that follow a specified key.",
    "doesEmailInputExist": "Verifies whether a given email form contains an input with specific name attribute.",
    "DynamicChartGenerator": "Creates a dynamic chart based on user data, applies scaling factor and links to dashboard.",
    "chartDataAccessorFactory": "Generates a data accessor for a specific chart component.",
    "generateNotificationHandler": "Generates a notification handler filtering messages by priority level.",
    "createAuthToken": "Generates an authorization token with user details, validity, and configuration options.",
    "trackSubmitWithValidation": "Tracks submit action on a given object when validation flags are satisfied.",
    "validateReactProp": "Validates an object to ensure it complies with React component prop constraints.",
    "transformAllDecoratorsOfDeclaration": "Processes and transforms all decorators of a TypeScript declaration node.",
    "updateDOMListeners": "Updates DOM event listeners from an old virtual node to a new one.",
}


def _json_type(value_type: type[Any]) -> dict[str, Any]:
    if value_type is str:
        return {"type": "string"}
    if value_type is int:
        return {"type": "integer"}
    if value_type is float:
        return {"type": "number"}
    if value_type is bool:
        return {"type": "boolean"}
    if value_type is dict:
        return {"type": "object"}
    if value_type is list:
        return {
            "type": "array",
            "items": {"type": "string"},
        }
    return {}


def build_tool_menu(tool_names: list[str] | set[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Build the single tool contract shared by training, serving and evaluation."""
    menu = []
    for name in sorted(set(tool_names)):
        expected_types = ARGUMENT_TYPES.get(name, {})
        menu.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS.get(name, f"Call the {name} tool."),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            key: _json_type(expected_type)
                            for key, expected_type in expected_types.items()
                        },
                        "required": sorted(REQUIRED_ARGUMENTS.get(name, set())),
                        "additionalProperties": not bool(expected_types),
                    },
                },
            }
        )
    return menu


def schema_conditioned_system_prompt(policy: str, tool_names: list[str] | set[str]) -> str:
    """Render the exact tool-list text emitted by LFM2.5's chat template.

    TRL formats prompt/completion datasets without forwarding a separate ``tools``
    argument. Embedding this text into the system message therefore makes SFT see
    the same menu representation that inference receives through
    ``apply_chat_template(..., tools=...)``.
    """
    menu = build_tool_menu(tool_names)
    rendered = ", ".join(json.dumps(tool, ensure_ascii=False) for tool in menu)
    prefix = policy.strip()
    return f"{prefix}\nList of tools: [{rendered}]" if prefix else f"List of tools: [{rendered}]"

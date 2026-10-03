from __future__ import annotations

import ast
import gc
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

TOOL_BLOCK = re.compile(
    r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", flags=re.DOTALL
)
COMMON_TOOL_BLOCKS = (
    re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL | re.IGNORECASE),
    re.compile(r"```(?:json)?\s*(.*?)```", flags=re.DOTALL | re.IGNORECASE),
)
THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", flags=re.DOTALL)
SURROGATE = re.compile(r"[\ud800-\udfff]")


def _sanitize_json_value(value: Any) -> Any:
    """Remove lone UTF-16 surrogates before FastAPI/uvicorn serializes a response."""

    if isinstance(value, str):
        return SURROGATE.sub("", value)
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_json_value(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_json_value(key): _sanitize_json_value(item)
            for key, item in value.items()
        }
    return value


def _use_kv_cache(input_tokens: int, long_prompt_cache_threshold: int) -> bool:
    """Disable the growing KV allocation after the configured safe prompt limit."""
    return input_tokens <= long_prompt_cache_threshold


def _eval_tool_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_tool_node(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_tool_node(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _eval_tool_node(key): _eval_tool_node(value)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_tool_node(node.operand)
    raise ValueError(f"Unsupported tool argument: {ast.dump(node)}")


def _tool_function_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_tool_function_name(node.value)}.{node.attr}"
    raise ValueError(f"Unsupported tool function: {ast.dump(node)}")


def parse_native_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    text = SURROGATE.sub("", THINK_BLOCK.sub("", text or ""))
    match = TOOL_BLOCK.search(text)
    if not match:
        return _clean_text(text), []
    try:
        expression = ast.parse(match.group(1).strip(), mode="eval").body
        nodes = expression.elts if isinstance(expression, ast.List) else [expression]
        calls = []
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            arguments = {}
            for keyword in node.keywords:
                if keyword.arg is not None:
                    arguments[keyword.arg] = _eval_tool_node(keyword.value)
            function_name = _tool_function_name(node.func)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        content = _clean_text(text[: match.start()] + text[match.end() :])
        return content, calls
    except (SyntaxError, ValueError, TypeError):
        return _clean_text(text), []


def _openai_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return {
        "id": f"call_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _calls_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    values = value if isinstance(value, list) else [value]
    calls: list[dict[str, Any]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        candidate = item.get("function") if isinstance(item.get("function"), dict) else item
        name = candidate.get("name") or candidate.get("tool") or candidate.get("tool_name")
        arguments = (
            candidate.get("arguments")
            if "arguments" in candidate
            else candidate.get("parameters", candidate.get("args", {}))
        )
        if name:
            calls.append(_openai_tool_call(str(name), arguments))
    return calls


def parse_generic_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse common JSON tool-call envelopes emitted by native-tools tokenizers.

    A fully vendor-specific syntax should still be exposed through an OpenAI-compatible
    endpoint.  This fallback deliberately accepts only unambiguous JSON/Python call forms;
    normal prose containing JSON must not silently become a tool invocation.
    """

    liquid_content, liquid_calls = parse_native_tool_calls(text)
    if liquid_calls:
        return liquid_content, liquid_calls

    candidates: list[tuple[str, int, int]] = []
    stripped = text.strip()
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        candidates.append((stripped, 0, len(text)))
    for pattern in COMMON_TOOL_BLOCKS:
        for match in pattern.finditer(text):
            candidates.append((match.group(1).strip(), match.start(), match.end()))
    for payload, start, end in candidates:
        try:
            calls = _calls_from_json(json.loads(payload))
        except json.JSONDecodeError:
            calls = []
        if calls:
            return _clean_text(text[:start] + text[end:]), calls

    # Several open-weight models emit a bare Python-style call without Liquid markers.
    try:
        expression = ast.parse(stripped, mode="eval").body
        nodes = expression.elts if isinstance(expression, ast.List) else [expression]
        calls = []
        for node in nodes:
            if not isinstance(node, ast.Call):
                calls = []
                break
            arguments = {
                keyword.arg: _eval_tool_node(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            calls.append(_openai_tool_call(_tool_function_name(node.func), arguments))
        if calls:
            return "", calls
    except (SyntaxError, ValueError, TypeError):
        pass
    return _clean_text(text), []


def _clean_text(text: str) -> str:
    text = SURROGATE.sub("", text or "")
    for token in ("<|im_end|>", "<|endoftext|>", "<|pad|>"):
        text = text.replace(token, "")
    return text.strip()


class LocalModel:
    def __init__(
        self,
        model_path: str,
        adapter_path: str | None,
        device: str,
        handler: str = "liquid-lfm2",
        generation_timeout_seconds: float = 180.0,
        long_prompt_cache_threshold: int = 2048,
        xpu_memory_fraction: float = 0.88,
        default_max_new_tokens: int = 1024,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = Path(adapter_path or model_path).name
        if handler not in {"liquid-lfm2", "generic-openai-tools"}:
            raise ValueError(f"Unsupported gateway handler: {handler}")
        self.handler = handler
        selected_device = device
        if selected_device == "auto":
            selected_device = "xpu" if torch.xpu.is_available() else "cpu"
        if selected_device == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("XPU requested but unavailable")
        self.device = selected_device
        self.generation_timeout_seconds = max(1.0, float(generation_timeout_seconds))
        self.long_prompt_cache_threshold = max(0, int(long_prompt_cache_threshold))
        self.default_max_new_tokens = max(1, int(default_max_new_tokens))
        self.xpu_memory_fraction = float(xpu_memory_fraction)
        if selected_device == "xpu":
            if not 0.1 <= self.xpu_memory_fraction <= 1.0:
                raise ValueError("XPU memory fraction must be between 0.1 and 1.0")
            # A hard allocator ceiling prevents one pathological BFCL multi-turn case from
            # consuming the last few hundred MiB of the 12 GiB B580 and destabilising the
            # desktop. Long prompts also disable KV cache below, so normal cases retain speed.
            torch.xpu.set_per_process_memory_fraction(self.xpu_memory_fraction)
        dtype = torch.bfloat16 if selected_device == "xpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, trust_remote_code=False, low_cpu_mem_usage=True
        )
        if adapter_path:
            from peft import PeftModel

            candidate = Path(adapter_path).expanduser().resolve()
            if (candidate / "adapter-final").is_dir():
                candidate = candidate / "adapter-final"
            model = PeftModel.from_pretrained(model, candidate)
        self.model = model.to(selected_device).eval()
        self.model.config.use_cache = True

    def complete(self, request: dict[str, Any]) -> tuple[str, list[dict[str, Any]], int, int]:
        raw_messages = request.get("messages") or []
        tools = request.get("tools") or None

        # Normalize multi-turn assistant tool_calls for Liquid chat template
        messages = []
        for m in raw_messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_str = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                        call_args = ", ".join(f"{k}={v!r}" for k, v in args_dict.items())
                        calls.append(f"{name}({call_args})")
                    except Exception:
                        calls.append(f"{name}()")
                formatted_content = f"<|tool_call_start|>[{', '.join(calls)}]<|tool_call_end|>"
                messages.append({"role": "assistant", "content": formatted_content})
            else:
                messages.append(m)

        template_kwargs: dict[str, Any] = {
            "conversation": messages,
            "add_generation_prompt": True,
            "tokenize": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        encoded = self.tokenizer.apply_chat_template(**template_kwargs).to(self.device)
        input_tokens = int(encoded["input_ids"].shape[-1])
        max_new_tokens = min(
            int(request.get("max_tokens") or self.default_max_new_tokens),
            self.default_max_new_tokens,
        )
        temperature = float(request.get("temperature") or 0.0)
        use_cache = _use_kv_cache(input_tokens, self.long_prompt_cache_threshold)
        try:
            with self.torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    max_time=self.generation_timeout_seconds,
                    use_cache=use_cache,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except Exception:
            if self.device == "xpu" and hasattr(self.torch, "xpu"):
                self.torch.xpu.empty_cache()
            elif self.device == "cuda" and hasattr(self.torch, "cuda"):
                self.torch.cuda.empty_cache()
            gc.collect()
            raise
        new_tokens = output[0, input_tokens:].detach().cpu()
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        parser = parse_native_tool_calls if self.handler == "liquid-lfm2" else parse_generic_tool_calls
        content, tool_calls = parser(raw)

        # Explicitly release GPU/XPU memory after every generation to prevent fragmentation
        del output, encoded
        if self.device == "xpu" and hasattr(self.torch, "xpu"):
            self.torch.xpu.empty_cache()
        elif self.device == "cuda" and hasattr(self.torch, "cuda"):
            self.torch.cuda.empty_cache()
        gc.collect()

        return (
            _sanitize_json_value(content),
            _sanitize_json_value(tool_calls),
            input_tokens,
            int(new_tokens.shape[-1]),
        )


def create_app(runtime: LocalModel) -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title="Exotic Agent Trainer Gateway")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": runtime.model_id, "device": runtime.device}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": runtime.model_id, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: dict[str, Any]) -> Any:
        try:
            content, tool_calls, prompt_tokens, completion_tokens = runtime.complete(request)
        except Exception as error:
            raise HTTPException(status_code=500, detail=f"{type(error).__name__}: {error}") from error
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        finish_reason = "tool_calls" if tool_calls else "stop"
        if request.get("stream"):

            def events() -> Any:
                delta: dict[str, Any] = {"role": "assistant"}
                if content:
                    delta["content"] = content
                if tool_calls:
                    delta["tool_calls"] = [
                        {"index": index, **call} for index, call in enumerate(tool_calls)
                    ]
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": runtime.model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": runtime.model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runtime.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def serve(
    model_path: str,
    adapter_path: str | None,
    device: str,
    host: str,
    port: int,
    handler: str = "liquid-lfm2",
    generation_timeout_seconds: float = 180.0,
    long_prompt_cache_threshold: int = 2048,
    xpu_memory_fraction: float = 0.88,
    default_max_new_tokens: int = 1024,
) -> None:
    import uvicorn

    runtime = LocalModel(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        handler=handler,
        generation_timeout_seconds=generation_timeout_seconds,
        long_prompt_cache_threshold=long_prompt_cache_threshold,
        xpu_memory_fraction=xpu_memory_fraction,
        default_max_new_tokens=default_max_new_tokens,
    )
    uvicorn.run(create_app(runtime), host=host, port=port, log_level="info")

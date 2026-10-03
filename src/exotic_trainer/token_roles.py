from __future__ import annotations

import ast
import io
import re
import tokenize
from enum import IntEnum
from typing import Any


class TokenRole(IntEnum):
    ORDINARY = 0
    DELIMITER = 1
    TOOL_NAME = 2
    ARGUMENT_KEY = 3
    ARGUMENT_VALUE = 4
    CLOSING_DELIMITER = 5


TOOL_BLOCK = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
TOOL_NAME = re.compile(r"(?:^|\[|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
ARGUMENT_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=")
CHAT_TEMPLATE_SUFFIX = re.compile(r"(?:<\|[A-Za-z0-9_:-]+\|>\s*)+$")


def _value_end(text: str, start: int, block_end: int) -> int:
    quote: str | None = None
    escaped = False
    depth = 0
    index = start
    while index < block_end:
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0 or (char == ")" and depth == 0):
                break
            depth -= 1
        elif char == "," and depth == 0:
            break
        index += 1
    return index


def _native_role_spans(text: str) -> list[tuple[int, int, TokenRole]]:
    """Return structural spans for native LFM tool calls.

    Spans are deliberately syntax-based: they work for tools not known in
    advance and therefore do not hard-code the current read/write/edit/bash
    vocabulary.
    """

    spans: list[tuple[int, int, TokenRole]] = []
    for marker in ("<|tool_call_start|>", "<|tool_call_end|>"):
        offset = 0
        while (position := text.find(marker, offset)) >= 0:
            role = (
                TokenRole.CLOSING_DELIMITER
                if marker == "<|tool_call_end|>"
                else TokenRole.DELIMITER
            )
            spans.append((position, position + len(marker), role))
            offset = position + len(marker)
    for block in TOOL_BLOCK.finditer(text):
        block_start, block_end = block.span(1)
        for match in TOOL_NAME.finditer(text, block_start, block_end):
            spans.append((*match.span(1), TokenRole.TOOL_NAME))
        for match in ARGUMENT_KEY.finditer(text, block_start, block_end):
            spans.append((*match.span(1), TokenRole.ARGUMENT_KEY))
            equals = text.find("=", match.end(1), block_end)
            if equals >= 0:
                value_start = equals + 1
                while value_start < block_end and text[value_start].isspace():
                    value_start += 1
                value_end = _value_end(text, value_start, block_end)
                while value_end > value_start and text[value_end - 1].isspace():
                    value_end -= 1
                if value_end > value_start:
                    spans.append((value_start, value_end, TokenRole.ARGUMENT_VALUE))
        for index in range(block_start, block_end):
            if text[index] in "[](),=":
                spans.append((index, index + 1, TokenRole.DELIMITER))
    return spans


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    return starts


def _absolute_offset(text: str, starts: list[int], line: int, byte_column: int) -> int:
    """Translate the UTF-8 byte columns used by ``ast`` into character offsets."""
    line_start = starts[line - 1]
    line_end = text.find("\n", line_start)
    if line_end < 0:
        line_end = len(text)
    line_text = text[line_start:line_end]
    prefix = line_text.encode("utf-8")[:byte_column].decode("utf-8", errors="ignore")
    return line_start + len(prefix)


def _node_span(
    text: str, starts: list[int], node: ast.AST
) -> tuple[int, int] | None:
    required = ("lineno", "col_offset", "end_lineno", "end_col_offset")
    if not all(hasattr(node, name) for name in required):
        return None
    return (
        _absolute_offset(text, starts, node.lineno, node.col_offset),
        _absolute_offset(text, starts, node.end_lineno, node.end_col_offset),
    )


def python_role_spans(text: str) -> list[tuple[int, int, TokenRole]]:
    """Return role spans for executable Python-State call expressions.

    Assignment targets are marked as values as well.  This makes a producer
    variable and its later consumer visible to the State Binding objective.
    Plain conversational text either fails Python parsing or contains no call,
    and therefore produces no structural spans.
    """
    # Completion-only labels may include the chat template's terminal token.
    # Strip only trailing special-token markers; all Python offsets before the
    # suffix remain unchanged and therefore still map to the original tokens.
    suffix = CHAT_TEMPLATE_SUFFIX.search(text)
    source = text[: suffix.start()] if suffix is not None else text
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    if not any(isinstance(node, ast.Call) for node in ast.walk(tree)):
        return []

    starts = _line_starts(source)
    spans: list[tuple[int, int, TokenRole]] = []
    if suffix is not None:
        for marker in re.finditer(r"<\|[A-Za-z0-9_:-]+\|>", suffix.group(0)):
            spans.append(
                (
                    suffix.start() + marker.start(),
                    suffix.start() + marker.end(),
                    TokenRole.CLOSING_DELIMITER,
                )
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_span = _node_span(source, starts, node.func)
            if function_span is not None:
                spans.append((*function_span, TokenRole.TOOL_NAME))
            for argument in node.args:
                value_span = _node_span(source, starts, argument)
                if value_span is not None:
                    spans.append((*value_span, TokenRole.ARGUMENT_VALUE))
            for keyword in node.keywords:
                value_span = _node_span(source, starts, keyword.value)
                if value_span is not None:
                    spans.append((*value_span, TokenRole.ARGUMENT_VALUE))
                if keyword.arg is not None:
                    key_start = _absolute_offset(
                        source, starts, keyword.lineno, keyword.col_offset
                    )
                    key_end = key_start + len(keyword.arg)
                    if source[key_start:key_end] == keyword.arg:
                        spans.append((key_start, key_end, TokenRole.ARGUMENT_KEY))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Call
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                target_span = _node_span(source, starts, target)
                if target_span is not None:
                    spans.append((*target_span, TokenRole.ARGUMENT_VALUE))

    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.OP or token.string not in "=(),[]{}:.":
                continue
            start = _absolute_offset(source, starts, token.start[0], token.start[1])
            end = _absolute_offset(source, starts, token.end[0], token.end[1])
            spans.append((start, end, TokenRole.DELIMITER))
    except (IndentationError, tokenize.TokenError):
        return []
    return spans


def role_spans(text: str) -> list[tuple[int, int, TokenRole]]:
    """Recognize native Trinity calls and Python-State calls."""
    native = _native_role_spans(text)
    if native:
        suffix = CHAT_TEMPLATE_SUFFIX.search(text)
        if suffix is not None:
            for marker in re.finditer(r"<\|[A-Za-z0-9_:-]+\|>", suffix.group(0)):
                native.append(
                    (
                        suffix.start() + marker.start(),
                        suffix.start() + marker.end(),
                        TokenRole.CLOSING_DELIMITER,
                    )
                )
    return native if native else python_role_spans(text)


def _decode_pieces(tokenizer: Any, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token_id in token_ids:
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(piece)
        offsets.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    return "".join(pieces), offsets


def literal_mask_for_token_ids(tokenizer: Any, token_ids: list[int]) -> list[bool]:
    """Mark tokens inside Markdown inline/fenced code spans.

    LiteralLock datasets put every execution-critical value in one of these
    spans.  The mask protects those prompt embeddings while π still modulates
    noise over the surrounding instruction and tool menu.
    """
    text, offsets = _decode_pieces(tokenizer, token_ids)
    spans: list[tuple[int, int]] = []
    fenced = list(re.finditer(r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)```", text, re.DOTALL))
    spans.extend(match.span(1) for match in fenced)
    fenced_ranges = [match.span() for match in fenced]
    for match in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
        if not any(left <= match.start() < right for left, right in fenced_ranges):
            spans.append(match.span(1))
    return [
        any(start < right and end > left for left, right in spans)
        for start, end in offsets
    ]


def batch_prompt_literal_mask(tokenizer: Any, input_ids: Any, torch: Any) -> Any:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row in range(input_ids.shape[0]):
        ids = input_ids[row].detach().cpu().tolist()
        values = torch.tensor(
            literal_mask_for_token_ids(tokenizer, ids),
            device=input_ids.device,
            dtype=torch.bool,
        )
        mask[row].copy_(values)
    return mask


def roles_for_token_ids(tokenizer: Any, token_ids: list[int]) -> list[TokenRole]:
    text, offsets = _decode_pieces(tokenizer, token_ids)
    spans = role_spans(text)
    roles = [TokenRole.ORDINARY for _ in token_ids]
    # Later/higher-valued roles take precedence over generic delimiters.
    for token_index, (start, end) in enumerate(offsets):
        matches = [role for left, right, role in spans if start < right and end > left]
        if matches:
            roles[token_index] = max(matches)
    return roles


def batch_token_roles(tokenizer: Any, input_ids: Any, labels: Any, torch: Any) -> Any:
    roles = torch.full_like(labels, int(TokenRole.ORDINARY), dtype=torch.long)
    for row in range(labels.shape[0]):
        positions = labels[row].ne(-100).nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        ids = input_ids[row].index_select(0, positions).detach().cpu().tolist()
        values = torch.tensor(
            [int(role) for role in roles_for_token_ids(tokenizer, ids)],
            device=labels.device,
            dtype=torch.long,
        )
        roles[row].index_copy_(0, positions, values)
    return roles

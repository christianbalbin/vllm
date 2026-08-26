# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.rust_tool_parser import RustToolParser

_TOOL_CALL_START = "]<]minimax[>[<tool_call>"
_TOOL_CALL_END = "]<]minimax[>[</tool_call>"


def _build_union_structural_tag(
    schema: dict | None,
    *,
    require_tool_call: bool,
) -> str:
    """Structural tag allowing EITHER the user's response schema OR an M3
    tool-call block (namespace-delimited XML markup parsed by the Rust
    parser post-hoc; the block content is deliberately permissive - the
    grammar only needs to make tool markup *emittable*, not validate it).

    With ``require_tool_call`` (tool_choice "required"/named) the JSON
    branch is dropped so the model must produce a tool call.

    A leading whitespace run is allowed before either branch: the model
    typically emits a newline after the reasoning block, and
    JSONSchemaFormat rejects leading whitespace on its own.

    A trailing whitespace run is allowed after either branch: the model
    often emits a newline before EOS, and without it the grammar
    terminates at the closing brace/end tag, so the spec-decode bitmask
    path probes an already-terminated matcher on every trailing
    whitespace draft (log spam: "Grammar is terminated, cannot fill
    bitmask" / rejected token id 10).
    """
    tool_block: dict = {
        "type": "tag",
        "begin": _TOOL_CALL_START,
        "content": {"type": "any_text"},
        "end": _TOOL_CALL_END,
    }
    branches: list[dict] = []
    if not require_tool_call:
        branches.append(
            {
                "type": "json_schema",
                "json_schema": schema if schema is not None else {"type": "object"},
            }
        )
    branches.append(tool_block)
    fmt: dict = (
        branches[0]
        if len(branches) == 1
        else {
            "type": "or",
            "elements": branches,
        }
    )
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "regex", "pattern": "[\\s]*"},
                    fmt,
                    {"type": "regex", "pattern": "[\\s]*"},
                ],
            },
        }
    )


class MinimaxM3ToolParser(RustToolParser):
    """Adapter from the Rust MiniMax M3 parser to vLLM ToolParser.

    The real M3 grammar lives in the Rust tool-parser crate. This class only
    configures the generic Rust bridge with the MiniMax M3 parser name.

    M3 is not M2 with renamed tags: it prefixes each structural tag with the
    MiniMax namespace marker, allows multiple ``<invoke>`` tags in one wrapper,
    and represents nested arguments with parameter-name XML tags.
    """

    rust_parser_name = "MinimaxM3ToolParser"
    tool_call_start_token = "]<]minimax[>[<tool_call>"

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = super().adjust_request(request)

        # Tools + response_format coexistence: vLLM applies ONE grammar per
        # request, so a bare json_schema response_format makes M3's XML tool
        # markup grammar-illegal - tools become silently unemittable (and
        # tool_choice "required" invites fabricated tool results). Replace
        # the schema grammar with a union structural tag: the user's schema
        # OR a tool-call block. The Rust parser still owns markup parsing;
        # requests without both features are untouched.
        if not isinstance(request, ChatCompletionRequest):
            return request
        if not request.tools or request.tool_choice == "none":
            return request
        if request.structured_outputs is not None:
            # An explicit structured_outputs param is caller-owned.
            return request
        response_format = request.response_format
        if response_format is None or response_format.type not in (
            "json_schema",
            "json_object",
        ):
            return request

        schema: dict | None = None
        if response_format.type == "json_schema":
            json_schema = response_format.json_schema
            if json_schema is None or json_schema.json_schema is None:
                return request
            schema = json_schema.json_schema

        require_tool_call = request.tool_choice == "required" or isinstance(
            request.tool_choice, ChatCompletionNamedToolChoiceParam
        )
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=_build_union_structural_tag(
                schema, require_tool_call=require_tool_call
            )
        )
        request.response_format = None
        return request

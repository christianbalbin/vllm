# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import string
from collections.abc import Sequence

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.minimax_m3_reasoning_parser import MiniMaxM3ReasoningParser

pytestmark = pytest.mark.skip_global_cleanup


class MiniMaxM3Tokenizer:
    """Small tokenizer with MiniMax M3 reasoning tags as special tokens."""

    special_tokens = ("<mm:think>", "</mm:think>")

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        for token in self.special_tokens:
            self._add_token(token)
        for char in string.printable:
            self._add_token(char)

    def _add_token(self, token: str) -> int:
        token_id = self._token_to_id.get(token)
        if token_id is None:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._token_to_id)

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> list[int]:
        return [self._add_token(token) for token in self.tokenize(text)]

    def decode(
        self, ids: Sequence[int] | int, skip_special_tokens: bool = False
    ) -> str:
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self._id_to_token[token_id] for token_id in ids)

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            for special_token in self.special_tokens:
                if text.startswith(special_token, pos):
                    tokens.append(special_token)
                    pos += len(special_token)
                    break
            else:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def convert_ids_to_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self._add_token(tokens)
        return [self._add_token(token) for token in tokens]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)


class SplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
    """Tokenizer that exposes marker vocab entries but encodes them as text."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class RuntimeSplitMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
    """Tokenizer whose runtime output splits markers despite atomic encodes."""

    def encode_runtime(self, text: str) -> list[int]:
        return [self._add_token(token) for token in list(text)]


def make_parser(
    chat_template_kwargs: dict[str, str] | None = None,
) -> tuple[MiniMaxM3ReasoningParser, MiniMaxM3Tokenizer]:
    tokenizer = MiniMaxM3Tokenizer()
    return (
        MiniMaxM3ReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs),
        tokenizer,
    )


def run_streaming(
    parser: MiniMaxM3ReasoningParser,
    tokenizer: MiniMaxM3Tokenizer,
    chunks: list[str],
) -> tuple[str | None, str | None, list[bool]]:
    previous_text = ""
    previous_token_ids: list[int] = []
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    reasoning_end_states: list[bool] = []

    for chunk in chunks:
        encode_runtime = getattr(tokenizer, "encode_runtime", tokenizer.encode)
        delta_token_ids = encode_runtime(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        reasoning_end_states.append(
            parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
        )

        if delta is not None:
            if delta.reasoning is not None:
                reasoning_parts.append(delta.reasoning)
            if delta.content is not None:
                content_parts.append(delta.content)

        previous_text = current_text
        previous_token_ids = current_token_ids

    return (
        "".join(reasoning_parts) or None,
        "".join(content_parts) or None,
        reasoning_end_states,
    )


def test_parser_registration():
    parser_cls = ReasoningParserManager.get_reasoning_parser("minimax_m3")

    assert parser_cls is MiniMaxM3ReasoningParser


def test_nonstreaming_extracts_explicit_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "<mm:think>plan</mm:think>answer", request
    )

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_without_start_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plain answer", request)

    assert reasoning is None
    assert content == "plain answer"


def test_nonstreaming_drops_leading_end_tag():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("</mm:think>answer", request)

    assert reasoning is None
    assert content == "answer"


def test_nonstreaming_non_leading_end_tag_is_content():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("XXX</mm:think>YYY", request)

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"


def test_nonstreaming_enabled_mode_starts_in_reasoning():
    parser, _ = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("plan</mm:think>answer", request)

    assert reasoning == "plan"
    assert content == "answer"


def test_nonstreaming_open_reasoning_block():
    parser, _ = make_parser()
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("<mm:think>still thinking", request)

    assert reasoning == "still thinking"
    assert content is None


def test_streaming_reasoning_tags_are_not_returned():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, False, True, True]


def test_streaming_boundary_can_emit_reasoning_and_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>plan</mm:think>answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [True]


def test_streaming_drops_leading_end_tag():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["</mm:think>", "answer"],
    )

    assert reasoning is None
    assert content == "answer"
    assert end_states == [True, True]


def test_streaming_non_leading_end_tag_is_content():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["XXX</mm:think>YYY"],
    )

    assert reasoning is None
    assert content == "XXX</mm:think>YYY"
    assert end_states == [True]


def test_streaming_enabled_mode_starts_in_reasoning():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plan", "</mm:think>", "answer"],
    )

    assert reasoning == "plan"
    assert content == "answer"
    assert end_states == [False, True, True]


def test_streaming_plain_content_ends_reasoning_phase():
    parser, tokenizer = make_parser()

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["plain ", "answer"],
    )

    assert reasoning is None
    assert content == "plain answer"
    assert end_states == [True, True]


def test_streaming_split_marker_tokens_are_not_returned():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:think>", "Reasoning", " content", "</mm:think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, True, True]


def test_streaming_split_marker_text_drives_end_state():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    previous_text = ""
    previous_token_ids: list[int] = []

    for chunk in ["<mm:think>", "Reasoning", " content", "</mm:think>"]:
        delta_token_ids = tokenizer.encode_runtime(chunk)
        current_text = previous_text + chunk
        current_token_ids = previous_token_ids + delta_token_ids
        parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=previous_token_ids,
            current_token_ids=current_token_ids,
            delta_token_ids=delta_token_ids,
        )
        previous_text = current_text
        previous_token_ids = current_token_ids

    assert parser.is_reasoning_end_streaming(previous_token_ids, []) is True


def test_streaming_split_end_marker_content_ids_are_stripped():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    previous_text = "<mm:think>Reasoning"
    previous_token_ids = tokenizer.encode_runtime(previous_text)
    delta_text = "</mm:think>content"
    delta_token_ids = tokenizer.encode_runtime(delta_text)
    current_token_ids = previous_token_ids + delta_token_ids

    parser.extract_reasoning_streaming(
        previous_text=previous_text,
        current_text=previous_text + delta_text,
        delta_text=delta_text,
        previous_token_ids=previous_token_ids,
        current_token_ids=current_token_ids,
        delta_token_ids=delta_token_ids,
    )

    assert parser.is_reasoning_end_streaming(current_token_ids, delta_token_ids)
    assert tokenizer.decode(parser.extract_content_ids(delta_token_ids)) == "content"


def test_streaming_split_marker_tokens_enabled_mode():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(
        tokenizer, chat_template_kwargs={"thinking_mode": "enabled"}
    )

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["Reasoning", " content", "</mm:think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, True, True]


def test_streaming_split_marker_text_across_deltas():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["<mm:", "think>", "Reasoning", " content", "</mm:", "think>", "content"],
    )

    assert reasoning == "Reasoning content"
    assert content == "content"
    assert end_states == [False, False, False, False, False, True, True]


def test_streaming_split_leading_end_marker_text_across_deltas():
    tokenizer = RuntimeSplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["</mm:", "think>", "content"],
    )

    assert reasoning is None
    assert content == "content"
    assert end_states == [False, True, True]


def test_token_id_helpers_with_split_marker_tokens():
    tokenizer = SplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(tokenizer)
    output_ids = tokenizer.encode(
        "<mm:think>abc</mm:think>def", add_special_tokens=False
    )
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


def test_token_id_helpers():
    parser, tokenizer = make_parser()
    output_ids = tokenizer.encode(
        "<mm:think>abc</mm:think>def", add_special_tokens=False
    )
    open_reasoning_ids = tokenizer.encode("<mm:think>abc", add_special_tokens=False)
    content_ids = tokenizer.encode("plain", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert not parser.is_reasoning_end(content_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.extract_content_ids(content_ids) == content_ids
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))


def test_token_id_helpers_enabled_mode():
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    output_ids = tokenizer.encode("abc</mm:think>def", add_special_tokens=False)
    open_reasoning_ids = tokenizer.encode("abc", add_special_tokens=False)

    assert parser.is_reasoning_end(output_ids)
    assert not parser.is_reasoning_end(open_reasoning_ids)
    assert tokenizer.decode(parser.extract_content_ids(output_ids)) == "def"
    assert parser.extract_content_ids(open_reasoning_ids) == []
    assert parser.count_reasoning_tokens(output_ids) == len(tokenizer.encode("abc"))
    assert parser.count_reasoning_tokens(open_reasoning_ids) == len(
        tokenizer.encode("abc")
    )


def test_reasoning_end_streaming_scheduler_pattern_enabled_mode():
    """The scheduler's structured-output gate calls is_reasoning_end_streaming
    on a request-local parser without ever calling
    extract_reasoning_streaming (which owns the frontend streaming-state
    flags). With thinking_mode="enabled" the parser starts inside a reasoning
    block; detection must still fire once </mm:think> is generated, or grammar
    constraints never engage and structured output is silently ignored.
    """
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    token_ids = tokenizer.encode(
        'some reasoning</mm:think>{"a": 1}', add_special_tokens=False
    )
    end_marker_index = token_ids.index(tokenizer.convert_tokens_to_ids("</mm:think>"))

    all_ids: list[int] = []
    fired_at = None
    for i, token in enumerate(token_ids):
        all_ids.append(token)
        if parser.is_reasoning_end_streaming(all_ids, [token]) and fired_at is None:
            fired_at = i

    assert fired_at == end_marker_index
    # Once the marker is in the history, detection must keep returning True.
    assert parser.is_reasoning_end_streaming(all_ids, [])


def test_reasoning_end_streaming_scheduler_pattern_spec_decode_window():
    """Same scheduler pattern, but with multi-token deltas as produced by
    speculative decoding: the end marker may land mid-window."""
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    token_ids = tokenizer.encode(
        'thinking</mm:think>{"a": 1}', add_special_tokens=False
    )
    end_marker_index = token_ids.index(tokenizer.convert_tokens_to_ids("</mm:think>"))

    window = 4
    all_ids: list[int] = []
    fired_in_window = None
    for start in range(0, len(token_ids), window):
        delta = token_ids[start : start + window]
        all_ids.extend(delta)
        if (
            parser.is_reasoning_end_streaming(all_ids, delta)
            and fired_in_window is None
        ):
            fired_in_window = start

    assert fired_in_window is not None
    assert fired_in_window <= end_marker_index < fired_in_window + window


def test_reasoning_end_streaming_ignores_prompt_scaffolding_markers():
    """The real chat template embeds literal marker text inside the prompt (a
    <thinking_instructions> block mentions <mm:think> and </mm:think>) and then
    prefills a trailing <mm:think>. The scheduler passes prompt+output token
    ids; naive containment fires at step 0, suppressing thinking entirely.
    Detection must stay False until the model itself emits </mm:think>."""
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    prompt_ids = tokenizer.encode(
        "instructions: wrap thinking in <mm:think> and </mm:think> tags.\n"
        "user: hi\nai: <mm:think>",
        add_special_tokens=False,
    )
    generated = tokenizer.encode(
        'thinking here</mm:think>{"a": 1}', add_special_tokens=False
    )
    end_offset = generated.index(tokenizer.convert_tokens_to_ids("</mm:think>"))

    all_ids = list(prompt_ids)
    # Step 0 sanity: prompt scaffolding alone must not end reasoning.
    assert parser.is_reasoning_end_streaming(all_ids, []) is False

    fired_at = None
    for i, token in enumerate(generated):
        all_ids.append(token)
        if parser.is_reasoning_end_streaming(all_ids, [token]) and fired_at is None:
            fired_at = i

    assert fired_at == end_offset


def test_reasoning_end_streaming_incremental_scan_split_marker():
    """Scheduler pattern with a tokenizer that splits markers into many
    tokens: the incremental history scan must still detect a marker that
    straddles successive scan windows, and must not fire early."""
    tokenizer = SplitMiniMaxM3Tokenizer()
    parser = MiniMaxM3ReasoningParser(
        tokenizer, chat_template_kwargs={"thinking_mode": "enabled"}
    )
    generated = tokenizer.encode('r</mm:think>{"a": 1}', add_special_tokens=False)
    # Marker is split into one token per character; it completes at the
    # index of its final '>' character.
    end_offset = len("r</mm:think>") - 1

    all_ids: list[int] = []
    fired_at = None
    for i, token in enumerate(generated):
        all_ids.append(token)
        if parser.is_reasoning_end_streaming(all_ids, [token]) and fired_at is None:
            fired_at = i
    assert fired_at == end_offset


def test_reasoning_end_streaming_scan_cache_resets_on_new_stream():
    """Reusing a parser on a shorter, fresh token stream must reset the
    incremental scan cache rather than reuse stale marker positions."""
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    first = tokenizer.encode("abc</mm:think>x", add_special_tokens=False)
    assert parser.is_reasoning_end_streaming(first, first[-1:]) is True

    fresh = tokenizer.encode("ab", add_special_tokens=False)
    parser._reasoning_ended_streaming = False
    assert parser.is_reasoning_end_streaming(fresh, fresh[-1:]) is False


def test_reasoning_end_streaming_rejected_draft_marker_not_cached():
    """The scheduler's bitmask path scans simulated sequences that include
    scheduled speculative draft tokens. A draft proposing </mm:think> can be
    rejected by the verifier while a later real sequence reaches the same
    length with different tokens (accept-all-but-last + bonus). The scan
    cache must not treat the rejected draft's marker as real - that engages
    the grammar mid-reasoning and empties the response content."""
    parser, tokenizer = make_parser(chat_template_kwargs={"thinking_mode": "enabled"})
    end_id = tokenizer.convert_tokens_to_ids("</mm:think>")

    history = tokenizer.encode("thinking a", add_special_tokens=False)
    drafts = [
        tokenizer.convert_tokens_to_ids("b"),
        end_id,
        tokenizer.convert_tokens_to_ids("c"),
    ]
    simulated = history + drafts
    # Bitmask-path probe over the simulated (draft-containing) sequence.
    assert parser.is_reasoning_end_streaming(simulated, drafts) is True

    # Verifier rejected the marker draft; the real sequence reaches the SAME
    # length with different tokens. Detection must NOT fire from the cache.
    real = history + tokenizer.encode("bde", add_special_tokens=False)
    assert len(real) == len(simulated)
    assert parser.is_reasoning_end_streaming(real, real[-1:]) is False

    # And the genuine marker must still be detected when it arrives.
    real2 = real + [end_id]
    assert parser.is_reasoning_end_streaming(real2, [end_id]) is True


# ── Implicit reasoning end: tool call opened inside the think block ────────
#
# MiniMax M3 sometimes drafts its answer inside <mm:think> and emits a
# complete native tool call there without ever sampling </mm:think>. Without
# recovery the round ends `stop` with empty content and no tool calls.

TOOL_CALL_START = "]<]minimax[>[<tool_call>"
TOOL_BLOCK = (
    TOOL_CALL_START + '\n]<]minimax[>[<invoke name="render_chart">'
    "]<]minimax[>[<version>1]<]minimax[>[</version>"
    "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
)


class ToolMarkupMiniMaxM3Tokenizer(MiniMaxM3Tokenizer):
    """Adds M3's tool-markup special tokens (namespace marker + tag tokens)."""

    special_tokens = MiniMaxM3Tokenizer.special_tokens + (
        "]<]minimax[>[",
        "<tool_call>",
        "</tool_call>",
    )


def make_tool_markup_parser(
    chat_template_kwargs: dict[str, str] | None = None,
) -> tuple[MiniMaxM3ReasoningParser, MiniMaxM3Tokenizer]:
    tokenizer = ToolMarkupMiniMaxM3Tokenizer()
    return (
        MiniMaxM3ReasoningParser(tokenizer, chat_template_kwargs=chat_template_kwargs),
        tokenizer,
    )


def test_nonstreaming_unclosed_block_recovers_tool_call():
    parser, _ = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "let me draft:\n\n## Snapshot\nprose" + TOOL_BLOCK, request
    )

    # The sentinel and everything after it become content, so the tool parser
    # sees the block exactly as it would after an explicit close.
    assert reasoning == "let me draft:\n\n## Snapshot\nprose"
    assert content == TOOL_BLOCK


def test_nonstreaming_explicit_close_with_tool_call_unchanged():
    parser, _ = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning(
        "plan</mm:think>## Snapshot" + TOOL_BLOCK, request
    )

    assert reasoning == "plan"
    assert content == "## Snapshot" + TOOL_BLOCK


def test_nonstreaming_unclosed_block_without_sentinel_stays_reasoning():
    parser, _ = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    request = ChatCompletionRequest(messages=[], model="test-model")

    reasoning, content = parser.extract_reasoning("thinking, no tool call", request)

    assert reasoning == "thinking, no tool call"
    assert content is None


def test_streaming_unclosed_block_routes_tool_call_to_content():
    parser, tokenizer = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )

    reasoning, content, end_states = run_streaming(
        parser,
        tokenizer,
        ["drafting:", "\n## Snapshot", TOOL_CALL_START, '\n]<]minimax[>[<invoke n="c">'],
    )

    assert reasoning == "drafting:\n## Snapshot"
    assert content == TOOL_CALL_START + '\n]<]minimax[>[<invoke n="c">'
    # Reasoning ends on the sentinel delta, which is what flips the frontend
    # into its tool-call phase.
    assert end_states == [False, False, True, True]


def test_streaming_split_sentinel_is_withheld_from_reasoning():
    """A sentinel straddling deltas must never be emitted as reasoning."""
    parser, tokenizer = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )

    reasoning, content, _ = run_streaming(
        parser, tokenizer, ["think", *list(TOOL_CALL_START), "rest"]
    )

    assert reasoning == "think"
    assert content == TOOL_CALL_START + "rest"


def test_streaming_implicit_end_yields_content_ids_from_sentinel():
    parser, tokenizer = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    delta_ids = tokenizer.encode(TOOL_CALL_START, add_special_tokens=False)

    parser.extract_reasoning_streaming(
        previous_text="think",
        current_text="think" + TOOL_CALL_START,
        delta_text=TOOL_CALL_START,
        previous_token_ids=tokenizer.encode("think", add_special_tokens=False),
        current_token_ids=tokenizer.encode(
            "think" + TOOL_CALL_START, add_special_tokens=False
        ),
        delta_token_ids=delta_ids,
    )

    # Content ids include the block's own opening tag.
    assert parser.extract_content_ids(delta_ids) == delta_ids


def test_implicit_reasoning_end_disabled_for_grammar_gate():
    """The structured-output manager clears the flag on its own instance.

    An implicit end fires on content the model has *already* emitted, so
    advancing the FSM there would leave the grammar a step behind the visible
    text (trim_reasoning_for_advance drops the marker as non-content).
    """
    parser, tokenizer = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    parser.allow_implicit_reasoning_end = False

    _, _, end_states = run_streaming(parser, tokenizer, ["think", TOOL_CALL_START])

    assert end_states == [False, False]
    # An explicit marker still ends reasoning for the grammar gate.
    end_id = tokenizer.convert_tokens_to_ids("</mm:think>")
    assert parser.is_reasoning_end_streaming([end_id], [end_id]) is True


def test_implicit_end_not_inferred_from_prompt_tool_markup():
    """is_reasoning_end must ignore tool markup replayed by the prompt.

    A multi-turn prompt contains earlier turns' tool calls; inferring an end
    from those would engage the grammar before the model has thought at all.
    """
    parser, tokenizer = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    prompt_ids = tokenizer.encode(
        "earlier turn " + TOOL_BLOCK, add_special_tokens=False
    )

    assert parser.is_reasoning_end(prompt_ids) is False
    assert parser.is_reasoning_end_streaming(prompt_ids, prompt_ids) is False


def test_nonstreaming_truncated_tool_call_stays_in_reasoning():
    """A block cut off by finish_reason=length must not become the answer.

    The tool parser raises on an incomplete M3 block and falls back to
    returning the raw markup as content, which is worse than an empty answer
    the caller can retry. The streaming parser swallows an incomplete block on
    its own, so both paths agree: a truncated call yields no content.
    """
    parser, _ = make_tool_markup_parser(
        chat_template_kwargs={"thinking_mode": "enabled"}
    )
    request = ChatCompletionRequest(messages=[], model="test-model")

    truncated = 'draft' + TOOL_CALL_START + '\n]<]minimax[>[<invoke name="render_ch'
    reasoning, content = parser.extract_reasoning(truncated, request)

    assert content is None
    assert reasoning == truncated

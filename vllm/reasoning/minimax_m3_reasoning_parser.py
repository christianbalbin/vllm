# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Opening sentinel of an M3 native tool call. Keep in sync with
# ``vllm.tool_parsers.minimax_m3_tool_parser._TOOL_CALL_START`` (and the Rust
# parser's own token): this parser only needs to recognize the marker, never
# to parse the block, which stays the tool parser's job.
_TOOL_CALL_START = "]<]minimax[>[<tool_call>"


class MiniMaxM3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for MiniMax M3 explicit thinking blocks.

    MiniMax M3 emits reasoning as:

        <mm:think>reasoning text</mm:think>assistant content

    The M3 tokenizer exposes both markers as complete vocabulary entries, but
    generated marker text may be tokenized into smaller pieces. The streaming
    parser therefore uses text markers for extraction instead of relying on the
    single vocabulary IDs. The chat template may also prefill the start marker
    when ``thinking_mode="enabled"``, so generated text can begin directly
    inside a reasoning block without emitting ``<mm:think>`` again.
    """

    @property
    def start_token(self) -> str:
        return "<mm:think>"

    @property
    def end_token(self) -> str:
        return "</mm:think>"

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._start_token_ids = self._encode_marker(self.start_token)
        self._end_token_ids = self._encode_marker(self.end_token)
        self._tool_call_start_token_ids = self._encode_marker(_TOOL_CALL_START)
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self._initial_in_reasoning = chat_kwargs.get("thinking_mode") == "enabled"
        # Implicit reasoning end: see _segments_without_explicit_end. Owners of
        # an instance that drives grammar advancement (the structured-output
        # manager) turn this off - an implicit end must never move the FSM,
        # because the sentinel it fires on is already-emitted grammar content.
        self.allow_implicit_reasoning_end = True
        self._implicit_content_started = False
        self._reasoning_ended_streaming = False
        self._reasoning_active_streaming = self._initial_in_reasoning
        self._pending_marker_streaming = False
        self._last_streaming_delta_token_ids: tuple[int, ...] | None = None
        self._last_streaming_content_token_ids: list[int] | None = None
        # Incremental marker-scan cache for _reasoning_end_in_history.
        # The parser is request-local and its token history is append-only,
        # so each call only needs to scan tokens it has not seen yet.
        self._history_scan_pos = 0
        self._history_last_start_pos = -1
        self._history_last_end_pos = -1

    def _encode_text(self, text: str) -> list[int]:
        try:
            return list(self.model_tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.model_tokenizer.encode(text))

    def _encode_marker(self, marker: str) -> tuple[int, ...]:
        return tuple(self._encode_text(marker))

    def _decode_text(self, token_ids: Sequence[int]) -> str:
        try:
            return self.model_tokenizer.decode(
                list(token_ids), skip_special_tokens=False
            )
        except TypeError:
            return self.model_tokenizer.decode(list(token_ids))

    def _content_suffix_token_ids(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        content: str | None,
    ) -> list[int]:
        if content is None:
            return []
        if content == delta_text:
            return list(delta_token_ids)
        if delta_text.endswith(content):
            prefix_text = delta_text[: len(delta_text) - len(content)]
            for index in range(len(delta_token_ids) + 1):
                if self._decode_text(delta_token_ids[:index]) == prefix_text:
                    return list(delta_token_ids[index:])
        return self._encode_text(content)

    @staticmethod
    def _contains_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return False
        marker_len = len(marker_ids)
        return any(
            tuple(token_ids[i : i + marker_len]) == tuple(marker_ids)
            for i in range(len(token_ids) - marker_len + 1)
        )

    @staticmethod
    def _rfind_token_sequence(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> int:
        if not marker_ids or len(marker_ids) > len(token_ids):
            return -1
        marker_len = len(marker_ids)
        for i in range(len(token_ids) - marker_len, -1, -1):
            if tuple(token_ids[i : i + marker_len]) == tuple(marker_ids):
                return i
        return -1

    @staticmethod
    def _ends_with_token_sequence_prefix(
        token_ids: Sequence[int], marker_ids: Sequence[int]
    ) -> bool:
        if not marker_ids:
            return False
        max_len = min(len(token_ids), len(marker_ids) - 1)
        for prefix_len in range(max_len, 0, -1):
            if tuple(token_ids[-prefix_len:]) == tuple(marker_ids[:prefix_len]):
                return True
        return False

    @staticmethod
    def _strip_partial_marker_suffix(text: str, marker: str) -> str:
        max_len = min(len(text), len(marker) - 1)
        for suffix_len in range(max_len, 0, -1):
            if marker.startswith(text[-suffix_len:]):
                return text[:-suffix_len]
        return text

    @staticmethod
    def _split_at_tool_call_sentinel(text: str) -> tuple[str, str] | None:
        """Split ``text`` at the first tool-call sentinel, or None if absent."""
        index = text.find(_TOOL_CALL_START)
        if index < 0:
            return None
        return text[:index], text[index:]

    def _segments_without_explicit_end(
        self, reasoning: str
    ) -> tuple[str | None, str | None, bool]:
        """Segment an unclosed reasoning block, recovering a tool call in it.

        MiniMax M3 sometimes drafts its answer *inside* the think block and
        emits a complete native tool call there without ever sampling
        ``</mm:think>`` (~1 in 3 long-context tool rounds as of 2026-08). The
        round then ends with all of the answer and the tool call attributed to
        reasoning, so the client sees ``finish_reason: stop`` with empty
        content and no tool calls.

        The tool-call sentinel is special-token markup that can only belong to
        content, so its first occurrence implicitly ends reasoning: text
        before it stays reasoning, the sentinel onward becomes content and
        reaches the tool parser exactly as it would after an explicit close.
        The drafted prose is deliberately left in reasoning - the model
        regenerates it in the round after the tool result, as it does for any
        normal tool-first round.

        Returns ``(reasoning, content, implicit_end)``.
        """
        split = self._split_at_tool_call_sentinel(reasoning)
        if split is not None:
            head, tail = split
            return head or None, tail, True
        # Withhold a partial marker so a sentinel straddling two deltas is
        # never emitted as reasoning and then retracted.
        head = self._strip_partial_marker_suffix(reasoning, self.end_token)
        head = self._strip_partial_marker_suffix(head, _TOOL_CALL_START)
        return head or None, None, False

    @staticmethod
    def _visible_delta(previous: str | None, current: str | None) -> str | None:
        if not current:
            return None
        if not previous:
            return current
        if current.startswith(previous):
            delta = current[len(previous) :]
            return delta or None
        return current

    def _visible_segments(self, text: str) -> tuple[str | None, str | None, bool]:
        """Return ``(reasoning, content, implicit_end)`` for cumulative text."""
        if not text:
            return None, None, False

        if not self._initial_in_reasoning:
            if self.end_token.startswith(text) and len(text) < len(self.end_token):
                return None, None, False
            if text.startswith(self.end_token):
                text = text[len(self.end_token) :]
                if not text:
                    return None, None, False

        if self._initial_in_reasoning and self.start_token not in text:
            reasoning, end, content = text.partition(self.end_token)
            if end:
                return reasoning or None, content or None, False
            return self._segments_without_explicit_end(reasoning)

        if self.start_token not in text:
            content = self._strip_partial_marker_suffix(text, self.start_token)
            return None, content or None, False

        content_before, _, after_start = text.partition(self.start_token)
        reasoning, end, content_after = after_start.partition(self.end_token)
        if end:
            return reasoning or None, (content_before + content_after) or None, False

        head, tail, implicit_end = self._segments_without_explicit_end(reasoning)
        if implicit_end:
            return head, (content_before + (tail or "")) or None, True
        return head, content_before or None, False

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        # MiniMax M3 can start a response with a stray closer. Drop that first
        # token only; later unmatched closers stay visible as content.
        if not self._initial_in_reasoning and model_output.startswith(self.end_token):
            content = model_output[len(self.end_token) :]
            return None, content or None

        if self._initial_in_reasoning and self.start_token not in model_output:
            reasoning, end, content = model_output.partition(self.end_token)
            if not end:
                # Unclosed think block: recover a tool call drafted inside it
                # (see _segments_without_explicit_end).
                split = self._split_at_tool_call_sentinel(reasoning)
                if split is None:
                    return model_output, None
                return split[0] or None, split[1]
            return reasoning, content or None

        if self.start_token not in model_output:
            return None, model_output

        content_before, _, after_start = model_output.partition(self.start_token)
        reasoning, end, content_after = after_start.partition(self.end_token)
        if not end:
            split = self._split_at_tool_call_sentinel(reasoning)
            if split is None:
                return reasoning, content_before or None
            return split[0] or None, (content_before + split[1]) or None

        return reasoning, (content_before + content_after) or None

    def _reasoning_end_in_history(self, input_ids: Sequence[int]) -> bool:
        """Incremental equivalent of ``is_reasoning_end(input_ids)``.

        Tracks the last-seen start/end marker positions across calls,
        scanning only tokens beyond the previous scan position (plus a
        marker-length overlap for markers straddling the line). If the
        sequence shrank since the last call - a new stream reusing the
        parser - the cache resets and the scan restarts from zero.

        Cached positions are re-verified against the current tokens before
        being trusted: the scheduler's bitmask path scans *simulated*
        sequences that include scheduled speculative draft tokens, and a
        draft that proposes a marker can be rejected by the verifier while
        the same sequence length is later reached by different tokens
        (accept-all-but-last plus bonus). A stale cached marker would then
        end reasoning that never ended - engaging the grammar mid-thought
        and leaving the response with empty content. Verification is
        O(marker length); a mismatch triggers a full rescan.
        """
        n = len(input_ids)
        if n < self._history_scan_pos:
            self._history_scan_pos = 0
            self._history_last_start_pos = -1
            self._history_last_end_pos = -1
        else:
            for pos, marker in (
                (self._history_last_start_pos, self._start_token_ids),
                (self._history_last_end_pos, self._end_token_ids),
            ):
                if pos >= 0 and (
                    pos + len(marker) > n
                    or tuple(input_ids[pos : pos + len(marker)]) != marker
                ):
                    self._history_scan_pos = 0
                    self._history_last_start_pos = -1
                    self._history_last_end_pos = -1
                    break
        overlap = max(len(self._start_token_ids), len(self._end_token_ids)) - 1
        begin = max(0, self._history_scan_pos - overlap)
        window = input_ids[begin:n] if begin else input_ids
        start_pos = self._rfind_token_sequence(window, self._start_token_ids)
        if start_pos >= 0:
            self._history_last_start_pos = max(
                self._history_last_start_pos, begin + start_pos
            )
        end_pos = self._rfind_token_sequence(window, self._end_token_ids)
        if end_pos >= 0:
            self._history_last_end_pos = max(
                self._history_last_end_pos, begin + end_pos
            )
        self._history_scan_pos = n
        return (
            self._history_last_end_pos >= 0
            and self._history_last_end_pos > self._history_last_start_pos
        )

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if self._reasoning_ended_streaming:
            return True

        # An implicit end (tool-call sentinel emitted inside the think block)
        # is derived from the streaming text state, which only
        # extract_reasoning_streaming maintains. The structured-output
        # manager's request-local parser never calls that method, so this
        # never fires on the grammar-advance path even before that owner
        # clears ``allow_implicit_reasoning_end`` - the prompt it scans also
        # replays earlier turns' tool markup, which must never end reasoning.
        if self.allow_implicit_reasoning_end and self._implicit_content_started:
            return True

        # Check the token stream for the end marker before consulting the
        # frontend streaming-state flags: the scheduler's structured-output
        # gate (StructuredOutputManager.should_advance) calls this method on a
        # request-local parser without ever calling
        # extract_reasoning_streaming, which is the only place those flags are
        # updated. With thinking_mode="enabled",
        # ``_reasoning_active_streaming`` starts True and would otherwise gate
        # detection off forever, so grammar constraints never engage
        # (structured output is silently ignored).
        #
        # Ordering matters, not mere containment: ``input_ids`` spans prompt
        # and output, and the chat template's <thinking_instructions> block
        # embeds literal marker text in the prompt. Reasoning has ended only
        # when the last end marker follows the last start marker (the
        # template's trailing ``<mm:think>`` prefill keeps this False until
        # the model really closes the block) - the same predicate as
        # ``is_reasoning_end``, evaluated incrementally so per-token calls
        # from the scheduler stay O(new tokens) instead of O(history).
        if self._reasoning_end_in_history(input_ids):
            return True

        delta_ids = tuple(delta_ids)
        if self._contains_token_sequence(delta_ids, self._end_token_ids):
            return True

        if self._reasoning_active_streaming or self._pending_marker_streaming:
            return False

        if self._initial_in_reasoning:
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._start_token_ids):
            return False
        if self._ends_with_token_sequence_prefix(input_ids, self._end_token_ids):
            return False
        if not self._contains_token_sequence(input_ids, self._start_token_ids):
            return bool(input_ids)
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if (
            self._last_streaming_delta_token_ids == tuple(input_ids)
            and self._last_streaming_content_token_ids is not None
        ):
            content_ids = self._last_streaming_content_token_ids
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
            return list(content_ids)

        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index >= 0:
            return input_ids[end_index + len(self._end_token_ids) :]

        # No explicit close: if a tool call was opened inside the think block,
        # content starts at (and includes) the sentinel, so the tool parser
        # receives the block's own opening tag.
        if self.allow_implicit_reasoning_end and self._implicit_content_started:
            sentinel_index = self._rfind_token_sequence(
                input_ids, self._tool_call_start_token_ids
            )
            if sentinel_index >= 0:
                return input_ids[sentinel_index:]

        has_start = self._contains_token_sequence(input_ids, self._start_token_ids)
        if self._initial_in_reasoning and not has_start:
            return []

        if not has_start:
            return input_ids
        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not delta_text:
            return None

        if not previous_text:
            self._reasoning_ended_streaming = False
            self._reasoning_active_streaming = self._initial_in_reasoning
            self._pending_marker_streaming = False
            self._implicit_content_started = False
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
        previous_reasoning, previous_content, _ = self._visible_segments(previous_text)
        current_reasoning, current_content, implicit_end = self._visible_segments(
            current_text
        )
        if implicit_end:
            self._implicit_content_started = True
        # An implicit end is tracked separately: ``_reasoning_ended_streaming``
        # means the model really closed the block, and flipping it here would
        # claim a reasoning end this parser cannot substantiate from the token
        # stream alone (the marker is simply absent).
        if self.end_token in current_text or (
            current_content is not None and not implicit_end
        ):
            self._reasoning_ended_streaming = True
            self._reasoning_active_streaming = False
            self._pending_marker_streaming = False
        else:
            self._last_streaming_delta_token_ids = None
            self._last_streaming_content_token_ids = None
            self._reasoning_active_streaming = (
                self._initial_in_reasoning
                or self.start_token in current_text
                or current_reasoning is not None
            )
            self._pending_marker_streaming = not self._reasoning_active_streaming and (
                self.start_token.startswith(current_text)
                or self.end_token.startswith(current_text)
            )
        reasoning = self._visible_delta(previous_reasoning, current_reasoning)
        content = self._visible_delta(previous_content, current_content)
        if self._reasoning_ended_streaming or self._implicit_content_started:
            self._last_streaming_delta_token_ids = tuple(delta_token_ids)
            self._last_streaming_content_token_ids = self._content_suffix_token_ids(
                delta_text, delta_token_ids, content
            )
        if reasoning is None and content is None:
            return None
        return DeltaMessage(reasoning=reasoning, content=content)

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        count = 0
        depth = 1 if self._initial_in_reasoning else 0
        i = 0
        while i < len(token_ids):
            if tuple(token_ids[i : i + len(self._start_token_ids)]) == (
                self._start_token_ids
            ):
                depth += 1
                i += len(self._start_token_ids)
                continue
            if tuple(token_ids[i : i + len(self._end_token_ids)]) == (
                self._end_token_ids
            ):
                if depth > 0:
                    depth -= 1
                i += len(self._end_token_ids)
                continue
            if depth > 0:
                count += 1
            i += 1
        return count

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_index = self._rfind_token_sequence(input_ids, self._start_token_ids)
        end_index = self._rfind_token_sequence(input_ids, self._end_token_ids)
        if end_index < 0:
            return False
        if start_index < 0:
            return True
        return end_index > start_index

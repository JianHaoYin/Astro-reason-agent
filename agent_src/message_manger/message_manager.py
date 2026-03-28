"""Message data model and persistence helpers for the refactored agent stack."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

MessageRole = Literal["system", "user", "assistant", "tool"]
MessageDirection = Literal["request", "response"]
MessageSource = Literal["human", "model", "tool", "system", "memory"]
FragmentKind = Literal["prompt", "prior_result", "memory", "context", "tool_result", "note"]


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class MessageFragment:
    """A human-editable piece of text that can be assembled into one model message.

    Args:
        kind: Semantic type of the fragment for downstream formatting.
        text: The raw text included in the final message.
        title: Optional human-readable label shown when multiple fragments are merged.
        source: Where this fragment originally came from.
        origin_message_id: ID of the message this fragment was copied from, if any.
        origin_session_id: Session that produced the source message, if any.
        metadata: Optional structured metadata for auditing or debugging.
    """

    kind: FragmentKind
    text: str
    title: str | None = None
    source: MessageSource = "human"
    origin_message_id: str | None = None
    origin_session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """Render the fragment into readable text for the model."""
        body = self.text.strip()
        if not self.title:
            return body

        header = self.title.strip()
        return f"[{header}]\n{body}" if body else f"[{header}]"


@dataclass(slots=True)
class ManagedMessage:
    """Tracked message exchanged with or around a model session.

    Args:
        message_id: Stable message identifier.
        role: Chat role expected by the model API.
        direction: Whether the message was sent to the model or returned by it.
        source: Who produced the message content.
        session_id: Session responsible for this message in the refactor framework.
        created_at: UTC creation timestamp.
        fragments: Human-friendly fragments making up the content.
        content: Cached plain-text content used for model calls.
        exported_from_session: Original session when the message was imported from elsewhere.
        tool_call_id: Tool call identifier for `tool` role messages.
        tool_calls: Structured tool calls returned by the model.
        raw_payload: Original raw request/response payload for debugging.
        metadata: Extra structured metadata attached by callers.
    """

    message_id: str
    role: MessageRole
    direction: MessageDirection
    source: MessageSource
    session_id: str | None
    created_at: str
    fragments: list[MessageFragment]
    content: str
    exported_from_session: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the internal message object into a JSON-friendly dictionary."""
        return asdict(self)


MessageComponent = str | MessageFragment | ManagedMessage | Mapping[str, Any]


class MessageManager:
    """Central registry for request/response messages across model sessions.

    The manager keeps a structured in-memory history and can optionally append the
    same records to a JSONL file for offline inspection.
    """

    def __init__(self, store_path: str | Path | None = None) -> None:
        """Initialize the manager and optionally persist every record to JSONL."""
        self.store_path = Path(store_path) if store_path is not None else None
        self._messages: list[ManagedMessage] = []
        self._by_session: dict[str, list[str]] = {}

    def create_fragment(
        self,
        text: str,
        *,
        kind: FragmentKind = "prompt",
        title: str | None = None,
        source: MessageSource = "human",
        origin_message_id: str | None = None,
        origin_session_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MessageFragment:
        """Create one reusable fragment from raw text."""
        return MessageFragment(
            kind=kind,
            text=text,
            title=title,
            source=source,
            origin_message_id=origin_message_id,
            origin_session_id=origin_session_id,
            metadata=dict(metadata or {}),
        )

    def fragment_from_message(
        self,
        message: ManagedMessage,
        *,
        kind: FragmentKind = "prior_result",
        title: str | None = None,
    ) -> MessageFragment:
        """Wrap a previous message as a reusable fragment in a later prompt."""
        default_title = title or f"{message.role} from session {message.session_id or 'unknown'}"
        return self.create_fragment(
            message.content,
            kind=kind,
            title=default_title,
            source=message.source,
            origin_message_id=message.message_id,
            origin_session_id=message.exported_from_session or message.session_id,
            metadata={"referenced_message_direction": message.direction},
        )

    def compose_message(
        self,
        *,
        role: MessageRole,
        direction: MessageDirection,
        source: MessageSource,
        session_id: str | None,
        components: Sequence[MessageComponent],
        exported_from_session: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        record: bool = True,
    ) -> ManagedMessage:
        """Assemble a tracked message from simple text, fragments, or old messages."""
        fragments = [self._normalize_component(component) for component in components]
        content = self.render_fragments(fragments)
        message = ManagedMessage(
            message_id=str(uuid.uuid4()),
            role=role,
            direction=direction,
            source=source,
            session_id=session_id,
            created_at=_utc_timestamp(),
            fragments=fragments,
            content=content,
            exported_from_session=exported_from_session,
            tool_call_id=tool_call_id,
            tool_calls=[dict(tool_call) for tool_call in (tool_calls or [])],
            raw_payload=dict(raw_payload) if raw_payload is not None else None,
            metadata=dict(metadata or {}),
        )
        if record:
            self.record_message(message)
        return message

    def record_message(self, message: ManagedMessage) -> None:
        """Store one message in memory and optionally append it to JSONL."""
        self._messages.append(message)
        if message.session_id:
            self._by_session.setdefault(message.session_id, []).append(message.message_id)
        if self.store_path is not None:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with self.store_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(message.to_dict(), ensure_ascii=False) + "\n")

    def build_request(
        self,
        *,
        session_id: str | None,
        role: MessageRole,
        components: Sequence[MessageComponent],
        source: MessageSource = "human",
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Build and record one outbound request message."""
        return self.compose_message(
            role=role,
            direction="request",
            source=source,
            session_id=session_id,
            components=components,
            metadata=metadata,
            record=True,
        )

    def start_session(
        self,
        *,
        session_id: str,
        system_prompt: MessageComponent | None = None,
        user_prompt: MessageComponent | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[ManagedMessage]:
        """Create the initial message list for a session and record it.

        This method replaces ad-hoc code such as:
        `messages = [{"role": "system", ...}, {"role": "user", ...}]`.
        """
        created: list[ManagedMessage] = []
        if system_prompt is not None:
            created.append(
                self.append_message(
                    session_id=session_id,
                    role="system",
                    components=[system_prompt],
                    direction="request",
                    source="system",
                    metadata=metadata,
                )
            )
        if user_prompt is not None:
            created.append(
                self.append_message(
                    session_id=session_id,
                    role="user",
                    components=[user_prompt],
                    direction="request",
                    source="human",
                    metadata=metadata,
                )
            )
        return created

    def append_message(
        self,
        *,
        session_id: str | None,
        role: MessageRole,
        components: Sequence[MessageComponent],
        direction: MessageDirection,
        source: MessageSource,
        exported_from_session: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Append one message into the tracked session history."""
        return self.compose_message(
            role=role,
            direction=direction,
            source=source,
            session_id=session_id,
            components=components,
            exported_from_session=exported_from_session,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            raw_payload=raw_payload,
            metadata=metadata,
            record=True,
        )

    def append_user_message(
        self,
        *,
        session_id: str | None,
        components: Sequence[MessageComponent],
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Append one outbound user message to a session."""
        return self.append_message(
            session_id=session_id,
            role="user",
            components=components,
            direction="request",
            source="human",
            metadata=metadata,
        )

    def append_system_message(
        self,
        *,
        session_id: str | None,
        components: Sequence[MessageComponent],
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Append one outbound system message to a session."""
        return self.append_message(
            session_id=session_id,
            role="system",
            components=components,
            direction="request",
            source="system",
            metadata=metadata,
        )

    def append_assistant_message(
        self,
        *,
        session_id: str | None,
        components: Sequence[MessageComponent],
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        exported_from_session: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Append one assistant response message to a session."""
        return self.append_message(
            session_id=session_id,
            role="assistant",
            components=components,
            direction="response",
            source="model",
            exported_from_session=exported_from_session,
            tool_calls=tool_calls,
            raw_payload=raw_payload,
            metadata=metadata,
        )

    def append_tool_result(
        self,
        *,
        session_id: str | None,
        tool_call_id: str,
        result: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Append one tool execution result as a `tool` role message."""
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return self.append_message(
            session_id=session_id,
            role="tool",
            components=[content],
            direction="response",
            source="tool",
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    def build_response(
        self,
        *,
        session_id: str | None,
        role: MessageRole,
        components: Sequence[MessageComponent],
        source: MessageSource = "model",
        exported_from_session: str | None = None,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        raw_payload: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ManagedMessage:
        """Build and record one inbound response message."""
        return self.compose_message(
            role=role,
            direction="response",
            source=source,
            session_id=session_id,
            components=components,
            exported_from_session=exported_from_session,
            tool_calls=tool_calls,
            raw_payload=raw_payload,
            metadata=metadata,
            record=True,
        )

    def list_messages(self, session_id: str | None = None) -> list[ManagedMessage]:
        """Return all messages, optionally filtered by session."""
        if session_id is None:
            return list(self._messages)
        return [message for message in self._messages if message.session_id == session_id]

    def get_session_messages(self, session_id: str) -> list[ManagedMessage]:
        """Return the ordered tracked messages for one session."""
        return self.list_messages(session_id=session_id)

    def render_fragments(self, fragments: Sequence[MessageFragment]) -> str:
        """Render fragments into a readable message body for the model.

        A single unlabeled fragment stays as raw text. Multiple fragments are joined
        with blank lines and keep explicit labels so humans can curate prompt pieces.
        """
        if not fragments:
            return ""
        if len(fragments) == 1 and not fragments[0].title:
            return fragments[0].text.strip()
        rendered = [fragment.render() for fragment in fragments if fragment.text.strip() or fragment.title]
        return "\n\n".join(chunk for chunk in rendered if chunk.strip())

    def _normalize_component(self, component: MessageComponent) -> MessageFragment:
        """Convert supported message components into a fragment instance."""
        if isinstance(component, MessageFragment):
            return component
        if isinstance(component, ManagedMessage):
            return self.fragment_from_message(component)
        if isinstance(component, str):
            return self.create_fragment(component)
        text = str(component.get("text") or component.get("content") or "")
        kind = component.get("kind", "context")
        title = component.get("title")
        source = component.get("source", "memory")
        origin_message_id = component.get("origin_message_id")
        origin_session_id = component.get("origin_session_id")
        metadata = component.get("metadata") or {}
        return self.create_fragment(
            text,
            kind=kind,
            title=title,
            source=source,
            origin_message_id=origin_message_id,
            origin_session_id=origin_session_id,
            metadata=metadata,
        )

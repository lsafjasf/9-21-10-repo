"""Collaborative plain-text merging with an operation-log CRDT.

The model uses immutable, totally ordered insertion positions. Concurrent
insertions into the same gap are ordered by the operation id of their causing
insert. Deletions identify characters by globally unique character ids and
use tombstones until a snapshot compaction reclaims them.

Only the Python standard library is used.
"""

from __future__ import annotations

from functools import cmp_to_key
from typing import Any, Dict, Iterable, List, Optional, Tuple

MIN_POS: Tuple[Any, ...] = ()
MAX_POS: Tuple[Any, ...] = (None,)

Tag = Tuple[int, str]
Position = Tuple[Tuple[int, Tag], ...]
CharId = Tuple[str, int, int]
OpId = Tuple[int, str]


def _tuplify(value: Any) -> Any:
    """Convert decoded JSON lists to tuples used as identity/order keys."""
    if isinstance(value, dict):
        return {key: _tuplify(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value


def _op_id(op: Dict[str, Any]) -> OpId:
    return (int(op["seq"]), str(op["site"]))


def compare_positions(left: Position, right: Position) -> int:
    """Lexicographic total order over immutable dense positions."""
    if left == right:
        return 0
    if len(left) < len(right) and right[: len(left)] == left:
        return -1
    if len(right) < len(left) and left[: len(right)] == right:
        return 1
    if right == MAX_POS:
        return -1
    if left == MAX_POS:
        return 1
    if left < right:
        return -1
    return 1


def allocate_between(left: Position, right: Position, tag: Tag) -> Position:
    """Return a position strictly between two known positions.

    ``tag`` is an operation id ``(seq, site)``. It makes positions selected by
    concurrent inserts in the same gap distinct and totally ordered.
    """
    if left == MIN_POS and right == MAX_POS:
        return ((0, tag),)
    if left == MIN_POS:
        return ((right[0][0] - 1, tag),)
    if right == MAX_POS:
        return left[:-1] + ((left[-1][0] + 1, tag),)

    index = 0
    limit = min(len(left), len(right))
    while index < limit and left[index] == right[index]:
        index += 1

    if index == len(left):
        return left + ((right[index][0] - 1, tag),)
    if index == len(right):
        return right + ((left[index][0] + 1, tag),)

    left_digit = left[index][0]
    right_digit = right[index][0]
    if right_digit - left_digit >= 2:
        middle = (left_digit + right_digit) // 2
        return left[:index] + ((middle, tag),)

    return left + ((0, tag),)


POSITION_KEY = cmp_to_key(compare_positions)


def _format_op_id(op_id: OpId) -> str:
    seq, site = op_id
    return f"{site}#{seq}"


class SyncDiagnostics:
    """Optional per-replica recorder that opens up the causal sync buffer.

    Recording is a pure function of the sequence of ``apply`` calls a replica
    receives, so replaying the same delivery order (e.g. one produced from a
    fixed random seed) yields byte-identical reports. Enabling or disabling
    diagnostics never changes merge results.
    """

    def __init__(self, site: str) -> None:
        self.site = site
        self.received = 0
        self.events: List[Dict[str, Any]] = []
        self._applied_ops: List[Dict[str, Any]] = []
        self._pending_wait: Dict[OpId, Dict[str, int]] = {}

    def record_received(self) -> None:
        self.received += 1

    def record_applied(self, op: Dict[str, Any], via: str) -> None:
        op_id = _op_id(op)
        self._applied_ops.append(op)
        self._pending_wait.pop(op_id, None)
        self.events.append({"event": "applied", "op": op_id, "via": via})

    def record_pending(self, op: Dict[str, Any], waiting_for: Dict[str, int]) -> None:
        op_id = _op_id(op)
        redelivered = op_id in self._pending_wait
        self._pending_wait[op_id] = dict(waiting_for)
        self.events.append(
            {
                "event": "pending",
                "op": op_id,
                "waiting_for": dict(waiting_for),
                "redelivered": redelivered,
            }
        )

    def record_skipped(self, op: Dict[str, Any], basis: Dict[str, Any]) -> None:
        self.events.append({"event": "skipped", "op": _op_id(op), "basis": basis})

    def record_dropped(self, op: Dict[str, Any], basis: Dict[str, Any]) -> None:
        op_id = _op_id(op)
        self._pending_wait.pop(op_id, None)
        self.events.append({"event": "dropped", "op": op_id, "basis": basis})

    @property
    def applied_count(self) -> int:
        return len(self._applied_ops)

    @property
    def skipped_count(self) -> int:
        return sum(1 for event in self.events if event["event"] == "skipped")

    @property
    def pending_count(self) -> int:
        return len(self._pending_wait)

    def currently_pending(self) -> Dict[OpId, Dict[str, int]]:
        """Buffered operations and the predecessors each is still waiting for."""
        return {op_id: dict(wait) for op_id, wait in self._pending_wait.items()}

    def replay_text(self) -> str:
        """Replay the recorded applied operations into a fresh replica."""
        replica = Document(f"{self.site}-diag-replay")
        for op in self._applied_ops:
            status = replica.apply(op)
            if status != "applied":
                raise AssertionError(
                    f"diagnostics replay failed for {_format_op_id(_op_id(op))}: {status}"
                )
        return replica.text

    def verify(self, document: "Document") -> None:
        """Assert the recorded diagnostics agree with the document state."""
        replayed = self.replay_text()
        assert replayed == document.text, (
            f"replica {self.site}: diagnostics replay {replayed!r} "
            f"!= document text {document.text!r}"
        )
        buffered = {_op_id(op) for op in document.pending_operations}
        assert set(self._pending_wait) == buffered, (
            f"replica {self.site}: diagnostics pending "
            f"{sorted(self._pending_wait)} != document pending {sorted(buffered)}"
        )
        arrivals = sum(
            1
            for event in self.events
            if event["event"] in ("skipped", "pending")
            or (event["event"] == "applied" and event["via"] == "direct")
        )
        assert arrivals == self.received, (
            f"replica {self.site}: {arrivals} arrival events "
            f"!= {self.received} received operations"
        )

    def report(self) -> str:
        """Deterministic human-readable summary of the recorded sync trace."""
        lines = [
            f"replica {self.site}: received={self.received} "
            f"applied={self.applied_count} skipped={self.skipped_count} "
            f"pending={self.pending_count}"
        ]
        for index, event in enumerate(self.events, 1):
            op_id = _format_op_id(event["op"])
            kind = event["event"]
            if kind == "applied":
                detail = f"via={event['via']}"
            elif kind == "pending":
                waiting = event["waiting_for"]
                detail = f"waiting_for={waiting or '{}'}"
                if event["redelivered"]:
                    detail += " redelivered"
            else:
                detail = f"basis={event['basis']}"
            lines.append(f"  {index:3d}. {kind:<8} {op_id:<10} {detail}")
        if self._pending_wait:
            lines.append("  still waiting:")
            for op_id in sorted(self._pending_wait):
                lines.append(
                    f"    {_format_op_id(op_id)} waits for {self._pending_wait[op_id]}"
                )
        return "\n".join(lines)


class Document:
    """A replica of a collaborative text document."""

    def __init__(self, site: str, diagnostics: bool = False) -> None:
        if not isinstance(site, str) or not site:
            raise ValueError("site must be a non-empty string")
        self.site = site
        self._version_vector: Dict[str, int] = {}
        self._chars: Dict[CharId, Dict[str, Any]] = {}
        self._tombstones: set[CharId] = set()
        self._applied_log: List[Dict[str, Any]] = []
        self._pending: Dict[OpId, Dict[str, Any]] = {}
        self._lamport = 0
        self.diagnostics: Optional[SyncDiagnostics] = (
            SyncDiagnostics(site) if diagnostics else None
        )

    @property
    def text(self) -> str:
        return "".join(char["ch"] for char in self._ordered_chars())

    @property
    def version_vector(self) -> Dict[str, int]:
        return dict(self._version_vector)

    @property
    def applied_log(self) -> List[Dict[str, Any]]:
        return list(self._applied_log)

    @property
    def pending_operations(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(self._pending[key] for key in sorted(self._pending))

    @property
    def tombstones(self) -> Tuple[CharId, ...]:
        """Uncompacted delete markers; compact() reclaims this set."""
        return tuple(sorted(self._tombstones))

    def __len__(self) -> int:
        return len(self._chars)

    def _ordered_chars(self) -> List[Dict[str, Any]]:
        return sorted(
            self._chars.values(),
            key=lambda char: POSITION_KEY(char["pos"]),
        )

    def _is_ready(self, op: Dict[str, Any]) -> bool:
        prev = op.get("prev", {})
        return all(
            seq <= self._version_vector.get(site, 0)
            for site, seq in prev.items()
        )

    def _missing_predecessors(self, op: Dict[str, Any]) -> Dict[str, int]:
        """Predecessor entries of ``op`` not yet covered by the version vector."""
        return {
            site: seq
            for site, seq in op.get("prev", {}).items()
            if seq > self._version_vector.get(site, 0)
        }

    def _apply_effect(self, op: Dict[str, Any]) -> None:
        kind = op["type"]
        if kind == "ins":
            for char in op["chars"]:
                char_id = char["id"]
                if char_id not in self._chars:
                    self._chars[char_id] = {
                        "id": char_id,
                        "pos": char["pos"],
                        "ch": char["ch"],
                    }
        elif kind == "del":
            for char_id in op["ids"]:
                self._chars.pop(char_id, None)
                self._tombstones.add(char_id)
        else:
            raise ValueError(f"unknown operation type: {kind!r}")

        self._version_vector[op["site"]] = op["seq"]
        self._lamport = max(self._lamport, int(op.get("ts", 0)))
        self._applied_log.append(op)

    def _drain_pending(self) -> int:
        applied = 0
        diag = self.diagnostics
        while True:
            progressed = False
            for op_id in sorted(self._pending):
                op = self._pending[op_id]
                if op["seq"] <= self._version_vector.get(op["site"], 0):
                    del self._pending[op_id]
                    if diag is not None:
                        diag.record_dropped(
                            op,
                            {
                                "kind": "already_applied",
                                "site": op["site"],
                                "op_seq": op["seq"],
                                "vv_seq": self._version_vector.get(op["site"], 0),
                            },
                        )
                    progressed = True
                    break
                if self._is_ready(op):
                    del self._pending[op_id]
                    self._apply_effect(op)
                    if diag is not None:
                        diag.record_applied(op, via="drain")
                    applied += 1
                    progressed = True
                    break
            if not progressed:
                return applied

    def apply(self, raw_op: Dict[str, Any]) -> str:
        """Apply one operation and return applied, pending, or skipped."""
        op = _tuplify(raw_op)
        op_id = _op_id(op)
        diag = self.diagnostics
        if diag is not None:
            diag.record_received()
        site_seq = self._version_vector.get(op["site"], 0)
        if op["seq"] <= site_seq:
            if diag is not None:
                diag.record_skipped(
                    op,
                    {
                        "kind": "already_applied",
                        "site": op["site"],
                        "op_seq": op["seq"],
                        "vv_seq": site_seq,
                    },
                )
            return "skipped"
        if not self._is_ready(op):
            if diag is not None:
                diag.record_pending(op, self._missing_predecessors(op))
            self._pending[op_id] = op
            return "pending"

        self._apply_effect(op)
        if diag is not None:
            diag.record_applied(op, via="direct")
        self._drain_pending()
        return "applied"

    def merge(self, operations: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Merge an arbitrary batch of operations in its supplied order."""
        result = {"applied": 0, "pending": 0, "skipped": 0}
        for raw_op in operations:
            result[self.apply(raw_op)] += 1
        return result

    def _make_op(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        seq = self._version_vector.get(self.site, 0) + 1
        self._lamport += 1
        op = {
            "type": kind,
            "site": self.site,
            "seq": seq,
            "ts": self._lamport,
            "prev": dict(self._version_vector),
        }
        op.update(payload)
        return op

    def local_insert(self, index: int, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("cannot insert empty text")
        visible = self._ordered_chars()
        if index < 0 or index > len(visible):
            raise IndexError("insert index out of range")

        left = visible[index - 1]["pos"] if index > 0 else MIN_POS
        right = visible[index]["pos"] if index < len(visible) else MAX_POS
        seq_hint = self._version_vector.get(self.site, 0) + 1
        tag = (seq_hint, self.site)

        chars = []
        for offset, value in enumerate(text):
            position = allocate_between(left, right, tag)
            chars.append(
                {
                    "id": (self.site, seq_hint, offset),
                    "pos": position,
                    "ch": value,
                }
            )
            left = position

        op = self._make_op("ins", {"chars": chars})
        self.apply(op)
        return op

    def local_delete(self, start: int, end: int) -> Dict[str, Any]:
        visible = self._ordered_chars()
        if start < 0 or end > len(visible) or start >= end:
            raise IndexError("invalid deletion interval")
        ids = [char["id"] for char in visible[start:end]]
        op = self._make_op("del", {"ids": ids})
        self.apply(op)
        return op

    def snapshot(self) -> Dict[str, Any]:
        """Return a compact snapshot containing only live character state."""
        return {
            "format": 1,
            "site": self.site,
            "vv": dict(self._version_vector),
            "chars": [
                {"id": char["id"], "pos": char["pos"], "ch": char["ch"]}
                for char in self._ordered_chars()
            ],
        }

    def compact(self) -> Dict[str, Any]:
        """Discard the applied operation log and return the snapshot."""
        snap = self.snapshot()
        self._applied_log = []
        self._tombstones.clear()
        return snap

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Dict[str, Any],
        site: Optional[str] = None,
        diagnostics: bool = False,
    ) -> "Document":
        snap = _tuplify(snapshot)
        doc = cls(site or snap["site"], diagnostics=diagnostics)
        doc._version_vector = dict(snap["vv"])
        doc._chars = {
            char["id"]: {"id": char["id"], "pos": char["pos"], "ch": char["ch"]}
            for char in snap["chars"]
        }
        doc._tombstones = set()
        doc._applied_log = []
        return doc

    def install_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Install a snapshot whose version dominates this replica's version."""
        snap = _tuplify(snapshot)
        new_vv = dict(snap["vv"])
        for site, seq in self._version_vector.items():
            if new_vv.get(site, 0) < seq:
                raise RuntimeError("cannot install a concurrent or older snapshot")

        self._version_vector = new_vv
        self._chars = {
            char["id"]: {"id": char["id"], "pos": char["pos"], "ch": char["ch"]}
            for char in snap["chars"]
        }
        self._tombstones = set()
        self._applied_log = []
        self._drain_pending()

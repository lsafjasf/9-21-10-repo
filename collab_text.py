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


class Document:
    """A replica of a collaborative text document."""

    def __init__(self, site: str) -> None:
        if not isinstance(site, str) or not site:
            raise ValueError("site must be a non-empty string")
        self.site = site
        self._version_vector: Dict[str, int] = {}
        self._chars: Dict[CharId, Dict[str, Any]] = {}
        self._tombstones: set[CharId] = set()
        self._applied_log: List[Dict[str, Any]] = []
        self._pending: Dict[OpId, Dict[str, Any]] = {}
        self._lamport = 0

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
        while True:
            progressed = False
            for op_id in sorted(self._pending):
                op = self._pending[op_id]
                if op["seq"] <= self._version_vector.get(op["site"], 0):
                    del self._pending[op_id]
                    progressed = True
                    break
                if self._is_ready(op):
                    del self._pending[op_id]
                    self._apply_effect(op)
                    applied += 1
                    progressed = True
                    break
            if not progressed:
                return applied

    def apply(self, raw_op: Dict[str, Any]) -> str:
        """Apply one operation and return applied, pending, or skipped."""
        op = _tuplify(raw_op)
        op_id = _op_id(op)
        site_seq = self._version_vector.get(op["site"], 0)
        if op["seq"] <= site_seq:
            return "skipped"
        if not self._is_ready(op):
            self._pending[op_id] = op
            return "pending"

        self._apply_effect(op)
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
    ) -> "Document":
        snap = _tuplify(snapshot)
        doc = cls(site or snap["site"])
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
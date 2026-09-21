"""Deterministic self-tests for collab_text.py.

Run: python test_collab_text.py
"""

import unittest
from typing import Tuple

import json
from collab_text import (
    MIN_POS,
    MAX_POS,
    Document,
    allocate_between,
    compare_positions,
)


def make_base() -> Tuple[Document, Document, Document, dict]:
    alpha = Document("a")
    base = alpha.local_insert(0, "abc")
    bravo = Document("b")
    charlie = Document("c")
    bravo.merge([base])
    charlie.merge([base])
    return alpha, bravo, charlie, base


class PositionOrderTests(unittest.TestCase):
    def test_concurrent_insertion_positions_are_total_ordered(self) -> None:
        tag_a = (1, "a")
        tag_b = (1, "b")
        left = allocate_between(MIN_POS, MAX_POS, tag_a)
        right = allocate_between(MIN_POS, MAX_POS, tag_b)
        self.assertNotEqual(left, right)
        self.assertEqual(compare_positions(left, right), -1)

    def test_repeated_prepend_and_append_remain_between_sentinels(self) -> None:
        first = ((0, (1, "a")),)
        left = allocate_between(MIN_POS, first, (2, "a"))
        right = allocate_between(first, MAX_POS, (3, "a"))
        self.assertEqual(compare_positions(MIN_POS, left), -1)
        self.assertEqual(compare_positions(left, first), -1)
        self.assertEqual(compare_positions(first, right), -1)
        self.assertEqual(compare_positions(right, MAX_POS), -1)


class MergeConvergenceTests(unittest.TestCase):
    def test_concurrent_inserts_at_same_position(self) -> None:
        alpha, bravo, charlie, base = make_base()
        op_a = alpha.local_insert(1, "A")
        op_b = bravo.local_insert(1, "B")
        op_c = charlie.local_insert(1, "C")

        first = Document("first")
        first.merge([op_c, base, op_b, op_a])
        second = Document("second")
        second.merge([op_a, base, op_c])
        second.merge([op_b])
        third = Document("third")
        third.merge([op_b])
        third.merge([op_a, base])
        third.merge([op_c])

        self.assertEqual(first.text, second.text)
        self.assertEqual(second.text, third.text)
        self.assertEqual(len(first.text), 6)
        self.assertEqual(first.version_vector, third.version_vector)

    def test_delete_and_insert_interleave(self) -> None:
        alpha, bravo, _, base = make_base()
        insert_op = bravo.local_insert(1, "X")
        delete_op = alpha.local_delete(1, 2)

        one = Document("one")
        one.merge([base, delete_op, insert_op])
        two = Document("two")
        two.merge([insert_op, base, delete_op])

        self.assertEqual(one.text, two.text)
        self.assertEqual(one.text, "aXc")
        self.assertEqual(len(one), 3)

    def test_concurrent_deletes_of_same_interval_are_idempotent(self) -> None:
        alpha, bravo, charlie, base = make_base()
        delete_a = alpha.local_delete(1, 3)
        delete_b = bravo.local_delete(0, 3)

        one = Document("one")
        one.merge([delete_a, delete_b])
        two = Document("two")
        two.merge([delete_b, delete_a])
        charlie.merge([delete_a, delete_b])

        self.assertEqual(one.text, "")
        self.assertEqual(two.text, "")
        self.assertEqual(charlie.text, "")
        self.assertEqual(len(one), 0)
        self.assertEqual(one.version_vector, two.version_vector)

    def test_json_roundtrip_preserves_state(self) -> None:
        alpha, bravo, _, base = make_base()
        alpha.local_delete(2, 3)
        bravo.local_insert(1, "Q")
        wire = json.loads(json.dumps(base))
        bravo = Document("d")
        bravo.merge([wire])
        snapshot = json.loads(json.dumps(alpha.snapshot()))
        loaded = Document.from_snapshot(snapshot, "e")
        self.assertEqual(loaded.text, alpha.text)

    def test_arbitrary_batches_and_duplicates_converge(self) -> None:
        alpha, bravo, charlie, base = make_base()
        operations = [
            alpha.local_insert(3, "!"),
            bravo.local_insert(0, "<"),
            charlie.local_delete(1, 2),
            alpha.local_insert(0, "["),
        ]

        left = Document("left")
        left.merge(operations)
        left.merge(reversed(operations))
        right = Document("right")
        right.merge([operations[3], operations[1]])
        right.merge([operations[0], operations[2], operations[3]])

        self.assertEqual(left.text, right.text)
        self.assertEqual(left.version_vector, right.version_vector)


class CausalBufferTests(unittest.TestCase):
    def test_missing_predecessor_is_buffered_and_replayed(self) -> None:
        alpha, bravo, charlie, base = make_base()
        first_insert = bravo.local_insert(3, "X")
        second_insert = bravo.local_insert(4, "Y")

        status = charlie.apply(second_insert)
        self.assertEqual(status, "pending")
        self.assertEqual(charlie.text, "abc")
        self.assertEqual(len(charlie.pending_operations), 1)

        charlie.apply(first_insert)
        self.assertEqual(charlie.text, "abcXY")
        self.assertEqual(charlie.pending_operations, ())
        self.assertEqual(charlie.version_vector["b"], 2)

    def test_causal_chain_replays_in_dependency_order(self) -> None:
        alpha = Document("a")
        ops = [alpha.local_insert(0, "a")]
        for value in "bcdef":
            ops.append(alpha.local_insert(len(alpha.text), value))

        bravo = Document("b")
        for op in reversed(ops):
            bravo.apply(op)
        self.assertEqual(bravo.text, "abcdef")
        self.assertEqual(len(bravo.pending_operations), 0)


class SnapshotTests(unittest.TestCase):
    def test_snapshot_equals_replay_and_accepts_new_and_old_ops(self) -> None:
        alpha, bravo, _, base = make_base()
        old_insert = bravo.local_insert(2, "X")
        alpha.merge([old_insert])
        history = [base, old_insert]
        snapshot = alpha.snapshot()
        replayed = Document("replay")
        replayed.merge(history)
        loaded = Document.from_snapshot(snapshot, "loaded")
        self.assertEqual(replayed.text, loaded.text)
        self.assertEqual(replayed.version_vector, loaded.version_vector)
        self.assertEqual(alpha.compact(), snapshot)
        self.assertEqual(alpha.applied_log, [])

        new_op = bravo.local_insert(0, "Y")
        old_duplicate = old_insert
        loaded.merge([new_op, old_insert])
        self.assertEqual(loaded.text, "YabXc")
        self.assertEqual(len(loaded.applied_log), 1)

    def test_install_snapshot_drains_old_and_new_pending_ops(self) -> None:
        alpha = Document("a")
        first = alpha.local_insert(0, "abc")
        second = alpha.local_insert(3, "d")
        third = alpha.local_insert(4, "e")

        recipient = Document("r")
        recipient.apply(third)
        self.assertEqual(recipient.pending_operations[0]["seq"], 3)

        snapshot = alpha.snapshot()
        snapshot["site"] = "snapshot"
        recipient.install_snapshot(snapshot)
        self.assertEqual(recipient.text, "abcde")
        self.assertEqual(len(recipient.pending_operations), 0)
        self.assertEqual(recipient.applied_log, [])

        newer = Document("a")
        newer._version_vector = alpha.version_vector
        newer._chars = dict(alpha._chars)
        next_op = newer.local_insert(5, "f")
        self.assertEqual(recipient.apply(next_op), "applied")
        self.assertEqual(recipient.text, "abcdef")

    def test_compaction_has_no_tombstones_or_unbounded_log(self) -> None:
        alpha = Document("a")
        alpha.local_insert(0, "abc")
        for _ in range(20):
            alpha.local_insert(0, "Z")
            alpha.local_delete(0, 1)
        self.assertEqual(alpha.text, "abc")
        self.assertGreater(len(alpha.tombstones), 0)
        snapshot = alpha.compact()
        self.assertEqual(alpha.applied_log, [])
        self.assertEqual(alpha.tombstones, ())
        self.assertEqual(len(snapshot["chars"]), 3)
        self.assertEqual(len(alpha), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
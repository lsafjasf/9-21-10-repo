"""test_crdt_text.py - self-tests for crdt_text (stdlib unittest only).

Run all tests:           python -m unittest test_crdt_text -v
Run one random seed:     python test_crdt_text.py 7
Override random seeds:   set CRDT_SEED=7 && python -m unittest test_crdt_text

On a random-test failure the failing seed is printed and the full op log is
dumped to repro_<seed>.json; re-running with that seed replays the exact
same scenario (the simulation is fully determined by the seed).
"""

import json
import os
import random
import sys
import unittest

from crdt_text import Doc


def _P(raw):
    return tuple((int(d), s, int(c)) for d, s, c in raw)


def full_sync(docs, rounds=5):
    """Merge every replica's op log into every other replica, repeatedly."""
    for _ in range(rounds):
        logs = [list(d.log) for d in docs]
        for d in docs:
            for lg in logs:
                d.merge(lg)


def reference_text(ops):
    """Independent declarative reference: live chars sorted by position."""
    ins, dels = {}, set()
    for op in ops:
        if op["kind"] == "ins":
            ins[_P(op["pos"])] = op["ch"]
        else:
            dels.add(_P(op["target"]))
    return "".join(ch for pos, ch in sorted(ins.items()) if pos not in dels)


def simulate(seed, n_replicas=4, n_steps=300):
    """Deterministic random simulation; returns (texts, reference, all_ops)."""
    rng = random.Random(seed)
    docs = [Doc("r%d" % i) for i in range(n_replicas)]
    all_ops = []
    for _ in range(n_steps):
        d = rng.choice(docs)
        t = d.text()
        if not t or rng.random() < 0.55:
            idx = rng.randint(0, len(t))
            s = "".join(rng.choice("abcde") for _ in range(rng.randint(1, 3)))
            all_ops += d.insert(idx, s)
        else:
            idx = rng.randrange(len(t))
            all_ops += d.delete(idx, rng.randint(1, min(3, len(t) - idx)))
        if rng.random() < 0.4:                      # random partial sync
            src, dst = rng.sample(docs, 2)
            batch = list(src.log)
            rng.shuffle(batch)
            dst.merge(batch[: rng.randint(0, len(batch))])
    full_sync(docs)
    return [d.text() for d in docs], reference_text(all_ops), all_ops


class EdgeCases(unittest.TestCase):
    def test_concurrent_insert_same_position(self):
        a, b, c = Doc("a"), Doc("b"), Doc("c")
        oa = a.insert(0, "A")
        ob = b.insert(0, "B")
        oc = c.insert(0, "C")
        for d in (a, b, c):
            d.merge(oa + ob + oc)
            d.merge(oa + ob + oc)          # duplicates must be harmless
        texts = {a.text(), b.text(), c.text()}
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts.pop(), "ABC")   # site id breaks the digit tie

    def test_delete_insert_crossing(self):
        a, b = Doc("a"), Doc("b")
        a.insert(0, "abc")
        full_sync([a, b])
        o_del = a.delete(1, 1)               # delete 'b'
        o_ins = b.insert(2, "X")             # concurrently insert after 'b'
        a.merge(o_ins)
        b.merge(o_del)
        self.assertEqual(a.text(), b.text())
        self.assertEqual(a.text(), "aXc")    # inserted char survives the delete

    def test_overlapping_concurrent_deletes(self):
        docs = [Doc("r%d" % i) for i in range(3)]
        docs[0].insert(0, "abcdef")
        full_sync(docs)
        o0 = docs[0].delete(0, 3)            # remove abc
        o1 = docs[1].delete(2, 3)            # remove cde (overlaps on c)
        o2 = docs[2].delete(1, 1)            # remove b (already covered by o0)
        all_ops = o0 + o1 + o2
        for d in docs:
            d.merge(all_ops)
            d.merge(all_ops)                 # idempotent re-merge
        texts = {d.text() for d in docs}
        self.assertEqual(texts, {"f"})       # union of ranges, no double delete

    def test_delete_then_insert_same_place(self):
        a, b = Doc("a"), Doc("b")
        a.insert(0, "xy")
        full_sync([a, b])
        o_del = a.delete(0, 2)               # a clears the document
        o_ins = b.insert(1, "Z")             # b inserts in the middle of it
        a.merge(o_ins)
        b.merge(o_del)
        self.assertEqual(a.text(), b.text())
        self.assertEqual(a.text(), "Z")


class CausalDependencies(unittest.TestCase):
    def test_seq_gap_is_buffered_then_replayed(self):
        a = Doc("a")
        ops = a.insert(0, "abc")             # seq 1,2,3 from site a
        b = Doc("b")
        b.apply(ops[2])                      # gap: seq 1,2 missing
        b.apply(ops[1])
        self.assertEqual(b.text(), "")       # still buffered
        b.apply(ops[0])                      # gap closed -> replay in order
        self.assertEqual(b.text(), "abc")

    def test_delete_before_insert_is_buffered(self):
        a, b, c = Doc("a"), Doc("b"), Doc("c")
        o_ins = a.insert(0, "x")
        b.merge(o_ins)
        o_del = b.delete(0, 1)               # b deletes a's character
        c.merge(o_del)                       # c has not seen the insert yet
        self.assertEqual(c.text(), "")
        c.merge(o_ins)                       # dependency arrives -> replay
        self.assertEqual(c.text(), "")
        self.assertEqual(len(c.tombs), 1)    # delete was applied, not dropped

    def test_shuffled_delivery_converges(self):
        a, b = Doc("a"), Doc("b")
        a.insert(0, "hello ")
        full_sync([a, b])
        b.insert(6, "world")
        a.delete(0, 3)
        b.delete(6, 2)
        ops = list(a.log) + list(b.log)
        rng = random.Random(42)
        c = Doc("c")
        rng.shuffle(ops)
        for op in ops:                       # one op at a time, random order
            c.apply(op)
        full_sync([a, b, c])
        self.assertEqual(a.text(), b.text())
        self.assertEqual(a.text(), c.text())
        self.assertEqual(a.text(), reference_text(ops))

class SnapshotAndCompaction(unittest.TestCase):
    def _make_state(self):
        a, b = Doc("a"), Doc("b")
        a.insert(0, "the quick brown fox")
        full_sync([a, b])
        b.insert(9, "red ")
        a.delete(0, 4)
        b.delete(10, 1)
        full_sync([a, b])
        return a, b

    def test_snapshot_roundtrip(self):
        a, _ = self._make_state()
        snap = a.snapshot()
        json.dumps(snap)                     # snapshot must be serialisable
        r = Doc.load(snap, "r")
        self.assertEqual(r.text(), a.text())
        self.assertEqual(r.vv, a.vv)
        self.assertEqual(set(r.tombs), set(a.tombs))

    def test_compaction_matches_from_scratch_replay(self):
        a, b = self._make_state()
        frontier = dict(a.vv)
        snap = a.compact(frontier)
        self.assertEqual(len(a.tombs), 0)    # all tombstones reclaimed
        self.assertEqual(a.log, [])          # log compressed into snapshot
        fresh = Doc("f")                     # from-scratch replay of full log
        fresh.merge(list(b.log))
        self.assertEqual(fresh.text(), a.text())
        r = Doc.load(snap, "r")
        self.assertEqual(r.text(), fresh.text())

    def test_compacted_replica_accepts_old_and_new_ops(self):
        a, b = self._make_state()
        old_ops = list(b.log)
        a.compact(dict(a.vv))
        # old ops (already covered by the version vector) are ignored
        a.merge(old_ops)
        self.assertEqual(a.text(), b.text())
        # new ops flow in both directions and both replicas converge
        o1 = b.insert(3, "NEW ")
        o2 = a.insert(0, "head ")
        a.merge(o1)
        b.merge(o2)
        self.assertEqual(a.text(), b.text())
        self.assertTrue(a.text().startswith("head "))
        self.assertIn("NEW", a.text())

    def test_snapshot_joiner_syncs_with_peers(self):
        a, b = self._make_state()
        snap = a.compact(dict(a.vv))
        j = Doc.load(snap, "j")              # new replica joins from snapshot
        o1 = b.insert(5, "b-was-here ")
        o2 = j.insert(0, "j:")
        for d in (a, b, j):
            d.merge(o1 + o2)
        self.assertEqual(a.text(), b.text())
        self.assertEqual(b.text(), j.text())

    def test_tombstones_do_not_grow_unboundedly(self):
        docs = [Doc("r%d" % i) for i in range(3)]
        rng = random.Random(7)
        for _ in range(60):
            for d in docs:
                t = d.text()
                if not t or rng.random() < 0.5:
                    d.insert(rng.randint(0, len(t)), "xyz"[: rng.randint(1, 3)])
                else:
                    i = rng.randrange(len(t))
                    d.delete(i, rng.randint(1, min(2, len(t) - i)))
            full_sync(docs)
            frontier = dict(docs[0].vv)
            for d in docs:
                d.compact(frontier)
            for d in docs:
                self.assertEqual(len(d.tombs), 0)            # bounded
                self.assertEqual(len(d.poss), len(d.text()) + 2)
        texts = {d.text() for d in docs}
        self.assertEqual(len(texts), 1)

class RandomDifferential(unittest.TestCase):
    """Randomised differential test: N replicas edit concurrently, sync in
    random shuffled batches, and must converge to the declarative reference
    (live chars sorted by position) computed from the union of all ops."""

    def test_random_seeds(self):
        env = os.environ.get("CRDT_SEED")
        seeds = [int(env)] if env else range(1, 26)
        for seed in seeds:
            with self.subTest(seed=seed):
                texts, ref, ops = simulate(seed)
                if len(set(texts)) != 1 or texts[0] != ref:
                    fname = "repro_%d.json" % seed
                    with open(fname, "w") as fh:
                        json.dump({"seed": seed, "ops": ops}, fh)
                    self.fail(
                        "seed=%d diverged (reproduce: python test_crdt_text.py %d; "
                        "op log dumped to %s)" % (seed, seed, fname))

    def test_random_with_compaction(self):
        """Same as above, but replicas compact at random stable points."""
        rng = random.Random(99)
        docs = [Doc("r%d" % i) for i in range(3)]
        all_ops = []
        for _ in range(150):
            d = rng.choice(docs)
            t = d.text()
            if not t or rng.random() < 0.5:
                all_ops += d.insert(rng.randint(0, len(t)), "ab"[rng.randrange(2)])
            else:
                i = rng.randrange(len(t))
                all_ops += d.delete(i, 1)
            if rng.random() < 0.3:
                src, dst = rng.sample(docs, 2)
                batch = list(src.log)
                rng.shuffle(batch)
                dst.merge(batch)
            if rng.random() < 0.1:           # full sync + compact everyone
                full_sync(docs)
                frontier = dict(docs[0].vv)
                for d in docs:
                    d.compact(frontier)
        full_sync(docs)
        texts = [d.text() for d in docs]
        self.assertEqual(len(set(texts)), 1)
        self.assertEqual(texts[0], reference_text(all_ops))
        frontier = dict(docs[0].vv)          # final compaction reclaims all
        for d in docs:
            d.compact(frontier)
        self.assertTrue(all(len(d.tombs) == 0 for d in docs))
        self.assertTrue(all(d.text() == texts[0] for d in docs))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seed = int(sys.argv[1])
        texts, ref, ops = simulate(seed)
        print("seed=%d ops=%d" % (seed, len(ops)))
        print("replica texts equal:", len(set(texts)) == 1)
        print("matches reference :", texts[0] == ref)
        print("text:", repr(texts[0]))
        sys.exit(0 if (len(set(texts)) == 1 and texts[0] == ref) else 1)
    unittest.main()

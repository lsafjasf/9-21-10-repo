# crdt_text - collaborative plain-text merge library

Pure Python 3 standard library, no dependencies. Multiple replicas edit the
same document offline and later sync their operation logs in any order and
any batching; all replicas converge to identical text.

## Model

- Operation log: every edit is an op `{kind, site, seq, ...}` (`ins` / `del`).
- Position identifiers: each character owns a Logoot-style position
  `((digit, site, clk), ...)` with a deterministic total order. Concurrent
  inserts at the same place share the deepest digit and are ordered by
  `(site, clk)`, so every replica computes the same character order.
  `(site, clk)` never repeats, so positions are never reused.
- Deletes are tombstones. Positions are absolute (not relative pointers), so
  `compact()` can reclaim tombstones below a causal stability frontier
  without disturbing any order - tombstones cannot grow without bound.
- Causal delivery: ops are applied in per-site sequence order. Sequence gaps
  and deletes whose target insert has not arrived are buffered and replayed
  automatically once their dependencies land.

## API

```python
from crdt_text import Doc

a, b = Doc("alice"), Doc("bob")      # site ids must be globally unique
ops = a.insert(0, "hello")           # local edit -> ops to broadcast
b.merge(ops)                         # any order, any batching, duplicates ok
b.apply(ops[0])                      # single-op variant
a.merge(b.delete(0, 1))              # concurrent edits converge
assert a.text() == b.text()

snap = a.snapshot()                  # JSON-serialisable snapshot
r = Doc.load(snap, "new-replica")    # join from snapshot
a.compact(frontier)                  # reclaim tombstones <= frontier,
                                     # compress the op log into a snapshot
```

`compact(frontier)` requires `frontier` to be causally stable (every
replica's version vector >= frontier, e.g. the common `vv` right after a
full sync). Old ops are then dropped by the version vector; new ops are
accepted as usual. `doc.log` holds the ops applied since the last
compaction and is all you need to ship for incremental sync.

## Tests

```
python -m unittest test_crdt_text -v     # full suite (14 tests)
python test_crdt_text.py 7               # replay random simulation seed 7
set CRDT_SEED=7 && python -m unittest test_crdt_text   # pin random seed
```

Randomised differential test: seeds 1-25 by default (4 replicas, 300 random
edits each, random shuffled partial syncs). The reference is an independent
declarative model (live chars sorted by position) computed from the union
of all ops. A compaction variant (seed 99) interleaves random compactions.

## Failure reproduction

The random simulation is fully determined by its seed. On failure the test
prints the seed and dumps the complete op log to `repro_<seed>.json`;
`python test_crdt_text.py <seed>` replays the exact scenario. To minimise,
trim ops from the end of the dumped log while the divergence persists.

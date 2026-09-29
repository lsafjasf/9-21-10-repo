# Collaborative Text Merge

A small Python 3 collaborative plain-text library using an operation-log CRDT.
Only the Python standard library is required.

## Files

- `collab_text.py` — library implementation and public API.
- `test_collab_text.py` — deterministic edge-case self-tests.
- `random_differential_test.py` — randomized differential tests.
- `sync_diagnostics.py` — reproducible sync-diagnostics demo and verification.

## Model

Each inserted character has:

- a globally unique character id `(site, seq, offset)`;
- an immutable dense position in a lexicographic tuple order;
- one Unicode character.

Each position node is `(digit, (seq, site))`. Concurrent insertions into the
same gap receive the same integer digit but distinct operation tags, so their
relative order is the deterministic total order `(seq, site)`. Applying the
same operations in different batches and orders therefore yields the same
text.

Deletions are idempotent operations containing character ids. A delete can be
applied before or after an unrelated concurrent insert without ambiguity.
Repeated concurrent deletes of the same interval are naturally idempotent.

A delete adds character ids to a compact tombstone set and removes those
characters from live state. `compact()` discards the entire applied operation
log and clears tombstones, keeping only live characters plus the version
vector. Missing remote operations stay in a separate causal staging buffer
until their predecessors arrive.

## API

```python
from collab_text import Document

a = Document("site-a")
op = a.local_insert(0, "hello")
op2 = a.local_delete(0, 2)

b = Document("site-b")
b.apply(op)
b.merge([op2])
assert b.text == a.text

snapshot = a.compact()
c = Document.from_snapshot(snapshot, "site-c")
```

Operations and snapshots are plain JSON-compatible dictionaries and lists.
Use `json.dumps` / `json.loads` for transport or persistence.

`Document.install_snapshot(snapshot)` accepts a snapshot whose version vector
dominates the current version. It refuses concurrent snapshots because a
GC'd snapshot does not contain enough deleted-history information to perform
a three-way merge.

## Sync Diagnostics

The causal buffer is observable. Construct a replica with
`Document(site, diagnostics=True)` and it records a `SyncDiagnostics` trace
on `doc.diagnostics`:

- `received` — how many operations this replica was offered;
- `pending` events — each buffered operation plus the exact predecessors it
  waits for (`waiting_for`), with `redelivered` marking duplicates that
  arrived while already buffered;
- `skipped` events — duplicate deliveries with the identification basis:
  the version-vector entry (`vv_seq`) that already covers the operation's
  `op_seq`;
- `applied` events — `via=direct` or `via=drain` (replayed from the buffer
  once predecessors arrived).

`diag.verify(doc)` asserts the diagnostics agree with the final state: the
recorded applied operations replay to exactly `doc.text`, and the recorded
pending set matches the live causal buffer. `diag.report()` renders a
deterministic text trace; recording is a pure function of the delivery
order, so replaying the same order (fixed seed) yields identical reports.
Diagnostics default to off and never change merge results.

```powershell
python sync_diagnostics.py --seed 7 --steps 40
```

The script asserts report reproducibility across runs, diagnostics/text
consistency per replica, and identical convergence with diagnostics off.

## Self-tests

```powershell
python test_collab_text.py
python random_differential_test.py
python sync_diagnostics.py
```

Run several reproducible randomized cases:

```powershell
python random_differential_test.py --seed 1 2 3 4 5 7 19 42 100 31337 --iterations 250
```

Default differential seed: `20260922`, iterations: `400`.

The randomized test independently computes a Kahn causal ordering of all
generated operations, replays that canonical order, sorts live characters by
immutable position, and compares it with every replica after arbitrary
batches, reordering, duplicates, delayed causal dependencies, dynamic replica
creation from snapshots, and repeated compaction.

## Failure Reproduction

Every assertion includes the seed when relevant. A failure like

```text
AssertionError: seed=12345 replica=r2: 'xy' != 'yx'
```

is reproduced with:

```powershell
python random_differential_test.py --seed 12345 --iterations 400
```

For minimization, repeatedly halve `--iterations` while retaining the same
seed until the smallest failing step count is found. The fixed tests provide
hand-written minimal scenarios for the three required edge classes:
concurrent same-position insertion, delete/insert interleaving, and repeated
concurrent deletion.

## State Bounds

- Live state is proportional to visible characters.
- Tombstones and the applied log are fully removable by `compact()`.
- Pending operations are retained only while causal predecessors are absent;
  they disappear as dependencies arrive.
- Position identifiers grow with the depth of pathological same-gap editing,
  not with the number of deletions. Deleting and recreating text does not
  create tombstone or log accumulation after compaction.

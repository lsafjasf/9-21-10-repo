"""crdt_text.py - collaborative plain-text merge library (stdlib only, Python 3.8+)

Model: operation log + Logoot-style position identifiers
--------------------------------------------------------
* Every character carries a globally unique, totally ordered position id
  pos = ((digit, site, clk), ...). Concurrent inserts at the same place
  land between the same two neighbours and are ordered deterministically
  by (digit, site, clk), so all replicas compute the same relative order.
* Insert ops carry their new position (no predecessor needed); delete ops
  reference the target position and are recorded as tombstones.
* Positions are absolute identifiers, not relative pointers, so reclaiming
  tombstones never changes the order of surviving characters. compact()
  reclaims tombstones below a causal stability frontier, which keeps
  tombstone growth bounded.
* Each replica applies ops in causal (site, seq) order: sequence gaps or
  deletes whose target has not arrived yet are buffered and replayed
  automatically once their dependencies land.

API
---
Doc(site)                 create a replica (site must be globally unique)
doc.insert(index, s)      local insert, returns the generated ops
doc.delete(index, length) local delete, returns the generated ops
doc.apply(op)             apply one remote op (any order, duplicates ok)
doc.merge(ops)            merge a batch of remote ops (any order/batching)
doc.text() / len(doc)     current text / length
doc.log                   ops applied since last compaction (for syncing)
doc.snapshot()            serialisable snapshot (includes tombstones)
Doc.load(snap, site)      restore a replica from a snapshot
doc.compact(frontier)     reclaim tombstones <= frontier and compress log
"""

from bisect import bisect_left

_BASE = 1 << 31
_START = ((0, "", 0),)                     # document head sentinel (undeletable)
_END = ((_BASE - 1, "\U0010FFFF", 0),)  # document tail sentinel (undeletable)


def _pos(raw):
    """Convert a JSON-ified position [[digit, site, clk], ...] to tuple form."""
    return tuple((int(d), s, int(c)) for d, s, c in raw)


def _alloc(p, q, site, clk):
    """Allocate a fresh position strictly between p < q (deterministic).

    Every allocated digit carries (value, site, clk) where clk is the insert
    op's per-site sequence number. Concurrent allocations in the same
    interval share the value and are ordered by (site, clk); because
    (site, clk) never repeats, a position is never reused - not even when a
    site inserts again between the same two neighbours after a delete.
    """
    prefix = []
    i = 0
    while True:
        pd = p[i][0] if i < len(p) else 0            # pad p with -inf digits
        qd = q[i][0] if i < len(q) else _BASE - 1    # pad q with +inf digits
        if qd - pd > 1:
            return tuple(prefix + [((pd + 1 + qd) // 2, site, clk)])
        # No room at this level: carry p's digit and descend one level.
        prefix.append(p[i] if i < len(p) else (0, "", 0))
        i += 1


class Doc:
    def __init__(self, site):
        if not site:
            raise ValueError("site must be a non-empty, globally unique id")
        self.site = site
        self.vv = {}                    # site -> highest contiguous seq applied
        self.poss = [_START, _END]      # sorted positions (incl. sentinels)
        self.chs = ["", ""]
        self.present = {_START, _END}   # live positions
        self.tombs = {}                 # pos -> (del_seq, del_site)
        self.log = []                   # ops applied since last compaction
        self.pending = {}               # site -> {seq: op} causal buffer
        self.base = None                # snapshot produced by last compact()

    # ---------------- queries ----------------
    def __len__(self):
        return len(self.poss) - 2

    def text(self):
        return "".join(self.chs[1:-1])

    # ---------------- local edits ----------------
    def insert(self, index, s):
        if not 0 <= index <= len(self):
            raise IndexError(index)
        ops = []
        for ch in s:
            seq = self.vv.get(self.site, 0) + 1
            pos = _alloc(self.poss[index], self.poss[index + 1], self.site, seq)
            op = {"kind": "ins", "site": self.site, "seq": seq,
                  "pos": [[d, st, c] for d, st, c in pos], "ch": ch}
            self.merge([op])
            ops.append(op)
            index += 1
        return ops

    def delete(self, index, length=1):
        if not 0 <= index < len(self) or index + length > len(self):
            raise IndexError(index)
        ops = []
        for _ in range(length):
            target = self.poss[index + 1]   # visible index -> position
            op = {"kind": "del", "site": self.site,
                  "seq": self.vv.get(self.site, 0) + 1,
                  "target": [[d, st, c] for d, st, c in target]}
            self.merge([op])
            ops.append(op)
        return ops

    # ---------------- remote merge ----------------
    def apply(self, op):
        self.merge([op])

    def merge(self, ops):
        for op in ops:
            site, seq = op["site"], op["seq"]
            if seq <= self.vv.get(site, 0):
                continue                  # duplicate or pre-compaction op
            self.pending.setdefault(site, {})[seq] = op
        self._drain()

    def _drain(self):
        """Apply buffered ops while possible; any progress may unblock more."""
        progressed = True
        while progressed:
            progressed = False
            for site in list(self.pending):
                seq = self.vv.get(site, 0) + 1
                op = self.pending[site].pop(seq, None)
                if op is None:
                    continue
                if self._try_apply(op):
                    self.vv[site] = seq
                    self.log.append(op)
                    progressed = True
                else:
                    self.pending[site][seq] = op    # deps missing: re-buffer
                if not self.pending[site]:
                    del self.pending[site]

    def _try_apply(self, op):
        if op["kind"] == "ins":
            pos = _pos(op["pos"])
            if pos in self.present or pos in self.tombs:
                return True               # defensive idempotence
            i = bisect_left(self.poss, pos)
            self.poss.insert(i, pos)
            self.chs.insert(i, op["ch"])
            self.present.add(pos)
            return True
        target = _pos(op["target"])
        if target in self.tombs:
            return True                   # concurrent duplicate delete
        if target not in self.present:
            return False                  # insert not arrived yet: buffer
        if target in (_START, _END):
            return True                   # sentinels are undeletable
        i = bisect_left(self.poss, target)
        self.poss.pop(i)
        self.chs.pop(i)
        self.present.discard(target)
        self.tombs[target] = (op["seq"], op["site"])
        return True

    # ---------------- snapshot & compaction ----------------
    def snapshot(self):
        return {
            "vv": dict(self.vv),
            "items": [[[[d, s, c] for d, s, c in pos], ch]
                      for pos, ch in zip(self.poss, self.chs)
                      if pos not in (_START, _END)],
            "tombs": [[[[d, s, c] for d, s, c in pos], ds, dt]
                      for pos, (ds, dt) in self.tombs.items()],
        }

    @classmethod
    def load(cls, snap, site):
        doc = cls(site)
        doc.vv = {s: int(v) for s, v in snap["vv"].items()}
        for raw_pos, ch in snap["items"]:
            pos = _pos(raw_pos)
            i = bisect_left(doc.poss, pos)
            doc.poss.insert(i, pos)
            doc.chs.insert(i, ch)
            doc.present.add(pos)
        for raw_pos, ds, dt in snap["tombs"]:
            doc.tombs[_pos(raw_pos)] = (ds, dt)
        return doc

    def compact(self, frontier=None):
        """Reclaim causally stable tombstones and compress the op log.

        The caller certifies that `frontier` is causally stable: every
        replica's version vector is at least `frontier` (typically the
        common vv right after a full synchronisation). Because positions
        are absolute identifiers, reclaiming tombstones cannot reorder
        surviving characters. Old ops are dropped by the version vector,
        new ops are accepted as usual. Defaults to the local vv.
        """
        frontier = frontier or self.vv
        self.tombs = {p: d for p, d in self.tombs.items()
                      if d[0] > frontier.get(d[1], 0)}
        self.base = self.snapshot()
        self.log = []
        return self.base

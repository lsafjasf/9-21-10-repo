"""Randomized differential tests for collab_text.py.

Default command:
    python random_differential_test.py

Reproduce one failing/shrinkable run:
    python random_differential_test.py --seed 12345 --iterations 400
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from typing import Any, Dict, List

from collab_text import Document

ALPHABET = "xyzXYZ01"


def ready(op: Dict[str, Any], vv: Dict[str, int]) -> bool:
    return all(seq <= vv.get(site, 0) for site, seq in op["prev"].items())


def canonical_order(operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Independent Kahn linearization; ties use the global operation id."""
    remaining = deepcopy(operations)
    vv: Dict[str, int] = {}
    ordered: List[Dict[str, Any]] = []
    while remaining:
        available = [op for op in remaining if ready(op, vv)]
        if not available:
            raise AssertionError("operation graph is not causally consistent")
        chosen = min(available, key=lambda op: (op["seq"], op["site"]))
        ordered.append(chosen)
        remaining.remove(chosen)
        vv[chosen["site"]] = chosen["seq"]
    return ordered


def reference_state(operations: List[Dict[str, Any]]) -> str:
    """Reference replay in one canonical causal order, then sort by position."""
    live: Dict[tuple, tuple] = {}
    for op in canonical_order(operations):
        if op["type"] == "ins":
            for char in op["chars"]:
                live.setdefault(
                    tuple(char["id"]),
                    (tuple(char["pos"]), char["ch"]),
                )
        else:
            for char_id in op["ids"]:
                live.pop(tuple(char_id), None)
    ordered = sorted(live.values(), key=lambda item: item[0])
    return "".join(value for _, value in ordered)


def replay_from_zero(snapshot: Dict[str, Any], operations: List[Dict[str, Any]]) -> Document:
    allowed_vv = snapshot["vv"]
    needed = [
        op
        for op in operations
        if op["seq"] <= allowed_vv.get(op["site"], 0)
    ]
    replica = Document("replay")
    for op in canonical_order(needed):
        replica.apply(op)
    return replica


def run_one(seed: int, iterations: int, verbose: bool = False) -> None:
    rng = random.Random(seed)
    replicas = {
        "r0": Document("r0"),
        "r1": Document("r1"),
    }
    queues = {name: [] for name in replicas}
    all_ops: List[Dict[str, Any]] = []
    base = replicas["r0"].local_insert(0, "abc")
    all_ops.append(base)
    queues["r1"].append(base)

    for step in range(iterations):
        if verbose and step % 50 == 0:
            print(f"seed={seed} step={step}")

        if rng.random() < 0.55:
            site = rng.choice(list(replicas))
            replica = replicas[site]
            if rng.random() < 0.65:
                index = rng.randrange(len(replica.text) + 1)
                text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 3)))
                op = replica.local_insert(index, text)
            else:
                if len(replica.text) == 0:
                    op = replica.local_insert(0, rng.choice(ALPHABET))
                else:
                    start = rng.randrange(len(replica.text))
                    end = rng.randint(start + 1, min(len(replica.text), start + 3))
                    op = replica.local_delete(start, end)
            all_ops.append(op)
            for target in replicas:
                if target != site and op["seq"] > replicas[target].version_vector.get(site, 0):
                    queues[target].append(op)

        for _ in range(rng.randint(1, 3)):
            target = rng.choice(list(queues))
            queue = queues[target]
            if queue:
                count = rng.randint(1, len(queue))
                batch = [queue.pop(rng.randrange(len(queue))) for _ in range(count)]
                replicas[target].merge(batch)
                if rng.random() < 0.15:
                    replicas[target].merge(deepcopy(batch[:1]))

        if rng.random() < 0.08 and len(replicas) < 5:
            source_name = rng.choice(list(replicas))
            source = replicas[source_name]
            new_name = f"r{len(replicas)}"
            snapshot = source.snapshot()
            new_replica = Document.from_snapshot(snapshot, new_name)
            replicas[new_name] = new_replica
            queues[new_name] = []
            for op in all_ops:
                if op["seq"] > new_replica.version_vector.get(op["site"], 0):
                    queues[new_name].append(op)

        if rng.random() < 0.10:
            replica = rng.choice(list(replicas.values()))
            snapshot = replica.compact()
            assert replica.applied_log == []
            assert replica.tombstones == ()
            replayed = replay_from_zero(snapshot, all_ops)
            assert replayed.text == replica.text, (
                f"seed={seed} step={step}: compact/replay mismatch: "
                f"{replayed.text!r} != {replica.text!r}"
            )
            assert replayed.version_vector == replica.version_vector

    for target, queue in list(queues.items()):
        while queue:
            count = min(len(queue), rng.randint(1, 5))
            batch = [queue.pop(rng.randrange(len(queue))) for _ in range(count)]
            replicas[target].merge(batch)

    expected = reference_state(all_ops)
    texts = {name: replica.text for name, replica in replicas.items()}
    vectors = {name: replica.version_vector for name, replica in replicas.items()}

    for name, replica in replicas.items():
        assert replica.pending_operations == (), (seed, name, replica.pending_operations)
        assert replica.text == expected, (
            f"seed={seed} replica={name}: {replica.text!r} != {expected!r}"
        )
        assert len(replica) == len(expected)

    assert len(set(texts.values())) == 1, (seed, texts)
    assert all(vector == vectors["r0"] for vector in vectors.values()), (seed, vectors)


def run_bounded_growth(seed: int = 999) -> None:
    rng = random.Random(seed)
    alpha = Document("a")
    bravo = Document("b")
    ops = []
    initial = alpha.local_insert(0, "base")
    bravo.apply(initial)
    ops.append(initial)
    for index in range(100):
        op = alpha.local_insert(0, "T")
        ops.append(op)
        bravo.apply(op)
        op = alpha.local_delete(0, 1)
        ops.append(op)
        bravo.apply(op)
        if index % 10 == 0:
            alpha.compact()
            bravo.compact()
            assert len(alpha.applied_log) == 0
            assert alpha.tombstones == ()
            assert bravo.tombstones == ()
            assert len(alpha) <= 4
            assert len(bravo) <= 4
    assert alpha.text == "base" == bravo.text
    assert len(alpha.snapshot()["chars"]) == 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=[20260922])
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for seed in args.seed:
        run_one(seed, args.iterations, args.verbose)
        print(f"PASS randomized convergence seed={seed} iterations={args.iterations}")
    run_bounded_growth()
    print("PASS bounded tombstone/log growth")


if __name__ == "__main__":
    main()
"""Reproducible sync-diagnostics demo and verification for collab_text.py.

Run:
    python sync_diagnostics.py [--seed 7] [--steps 40]

For a fixed seed the delivery schedule (batches, reordering, duplicates,
delayed causal predecessors) is fully determined, so two runs emit identical
diagnostic reports. The script asserts that:

- every replica's diagnostics replay equals its final text;
- the diagnostics view of the causal buffer matches the live buffer;
- a second run with the same seed produces byte-identical reports;
- disabling diagnostics yields the same converged text.
"""

from __future__ import annotations

import argparse
import random
from typing import Any, Dict, List, Tuple

from collab_text import Document

SITES = ("r0", "r1", "r2")
ALPHABET = "abXY01"


def run_scenario(seed: int, steps: int, diagnostics: bool) -> Dict[str, Document]:
    """One deterministic sync session; the seed fixes every delivery choice."""
    rng = random.Random(seed)
    replicas = {site: Document(site, diagnostics=diagnostics) for site in SITES}
    queues: Dict[str, List[Dict[str, Any]]] = {site: [] for site in SITES}

    base = replicas["r0"].local_insert(0, "base")
    for site in SITES[1:]:
        queues[site].append(base)

    for _ in range(steps):
        site = rng.choice(SITES)
        replica = replicas[site]
        if replica.text and rng.random() < 0.35:
            start = rng.randrange(len(replica.text))
            end = rng.randint(start + 1, min(len(replica.text), start + 2))
            op = replica.local_delete(start, end)
        else:
            index = rng.randrange(len(replica.text) + 1)
            text = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(1, 2)))
            op = replica.local_insert(index, text)
        for target in SITES:
            if target != site:
                queues[target].append(op)

        for _ in range(rng.randint(0, 2)):
            target = rng.choice(SITES)
            queue = queues[target]
            if not queue:
                continue
            count = rng.randint(1, len(queue))
            batch = [queue.pop(rng.randrange(len(queue))) for _ in range(count)]
            if rng.random() < 0.3:
                batch = batch + batch[:1]  # duplicate delivery
            if rng.random() < 0.5:
                batch.reverse()  # reorder, possibly before predecessors
            replicas[target].merge(batch)

    for target in SITES:
        while queues[target]:
            count = rng.randint(1, len(queues[target]))
            batch = [queues[target].pop(rng.randrange(len(queues[target]))) for _ in range(count)]
            replicas[target].merge(batch)

    return replicas


def verify(seed: int, steps: int) -> Tuple[Dict[str, Document], Dict[str, str]]:
    replicas = run_scenario(seed, steps, diagnostics=True)

    texts = {site: doc.text for site, doc in replicas.items()}
    assert len(set(texts.values())) == 1, f"seed={seed}: divergence {texts}"

    reports: Dict[str, str] = {}
    for site, doc in replicas.items():
        diag = doc.diagnostics
        assert diag is not None
        diag.verify(doc)  # diagnostics consistent with final text and buffer
        reports[site] = diag.report()

    rerun = run_scenario(seed, steps, diagnostics=True)
    for site, doc in rerun.items():
        assert doc.diagnostics is not None
        assert doc.diagnostics.report() == reports[site], (
            f"seed={seed} replica={site}: report not reproducible"
        )

    silent = run_scenario(seed, steps, diagnostics=False)
    for site, doc in silent.items():
        assert doc.diagnostics is None
        assert doc.text == texts[site], (
            f"seed={seed} replica={site}: disabling diagnostics changed the result"
        )

    return replicas, reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()

    replicas, reports = verify(args.seed, args.steps)
    for site in SITES:
        print(reports[site])
        print(f"  final text: {replicas[site].text!r}")
        print()
    print(
        f"PASS sync diagnostics seed={args.seed} steps={args.steps}: "
        "reports reproducible, diagnostics match final text, "
        "diagnostics-off convergence identical"
    )


if __name__ == "__main__":
    main()

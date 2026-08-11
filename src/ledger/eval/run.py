"""CLI: `python -m ledger.eval.run --config <slug|all> --out RESULTS.md`.

Computes eval results for one or more retrieval configurations and folds
them into RESULTS.md via results_table.update_results_table. Each
configuration's actual pipeline (retriever + generator + latency capture) is
registered separately as the corresponding chunking/indexing/retrieval
stories ship (LEDGER-024 through LEDGER-027) -- none are registered yet, so
running this today leaves RESULTS.md unchanged rather than inventing numbers.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from ledger.eval.results_table import ConfigResult, update_results_table

SLUG_TO_DISPLAY_NAME: dict[str, str] = {
    "naive_dense": "Naive 512 + dense",
    "structural_dense": "Structural + dense",
    "structural_hybrid_rrf": "Structural + hybrid RRF",
    "contextual_hybrid_rrf": "Contextual + hybrid RRF",
    "contextual_hybrid_rrf_reranked": "+ cross-encoder rerank",
}

# slug -> zero-arg callable producing that configuration's ConfigResult.
# Populated by each retrieval story as it ships; empty until then.
PIPELINES: dict[str, Callable[[], ConfigResult]] = {}


def register_pipeline(slug: str, runner: Callable[[], ConfigResult]) -> None:
    """Register the eval runner for one configuration slug."""
    if slug not in SLUG_TO_DISPLAY_NAME:
        raise ValueError(
            f"unknown configuration slug {slug!r}; expected one of "
            f"{sorted(SLUG_TO_DISPLAY_NAME)}"
        )
    PIPELINES[slug] = runner


def run(config: str, out: Path) -> None:
    if config == "all":
        slugs = list(SLUG_TO_DISPLAY_NAME)
    elif config in SLUG_TO_DISPLAY_NAME:
        slugs = [config]
    else:
        raise SystemExit(
            f"unknown --config {config!r}; expected 'all' or one of "
            f"{sorted(SLUG_TO_DISPLAY_NAME)}"
        )

    results: list[ConfigResult] = []
    skipped: list[str] = []
    for slug in slugs:
        runner = PIPELINES.get(slug)
        if runner is None:
            skipped.append(slug)
            continue
        result = runner()
        expected_name = SLUG_TO_DISPLAY_NAME[slug]
        if result.configuration != expected_name:
            raise ValueError(
                f"pipeline for {slug!r} returned a ConfigResult for "
                f"{result.configuration!r}, expected {expected_name!r}"
            )
        results.append(result)

    if results:
        update_results_table(out, results)
        names = ", ".join(r.configuration for r in results)
        print(f"updated {out} with {len(results)} configuration(s): {names}")
    if skipped:
        slugs_str = ", ".join(skipped)
        print(
            f"no pipeline registered yet for: {slugs_str} -- left unchanged in {out} "
            "(see LEDGER-024..027)"
        )
    if not results and not skipped:
        print("nothing to do")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="all", help="'all' or a specific configuration slug"
    )
    parser.add_argument("--out", default="RESULTS.md", type=Path, help="path to RESULTS.md")
    args = parser.parse_args(argv)
    run(args.config, args.out)


if __name__ == "__main__":
    main()

"""Auto-generates the before/after configuration table in RESULTS.md.

Keeps RESULTS.md as the single source of truth for benchmark numbers: this
module only ever rewrites the table between its header and its last
contiguous row, in a fixed configuration order, merging newly given results
on top of whatever numbers are already recorded there. A configuration that
isn't part of this run keeps its previously recorded row untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TABLE_HEADER = "| Configuration | recall@10 | MRR | Faithfulness | p95 latency |"
TABLE_DIVIDER = "|---|---|---|---|---|"

CONFIG_ORDER = [
    "Naive 512 + dense",
    "Structural + dense",
    "Structural + hybrid RRF",
    "Contextual + hybrid RRF",
    "+ cross-encoder rerank",
]

_EMPTY_CELL = "—"


@dataclass
class ConfigResult:
    """One row's worth of eval numbers for a single retrieval configuration."""

    configuration: str
    recall_at_10: float | None = None
    mrr: float | None = None
    faithfulness: float | None = None
    p95_latency_ms: float | None = None

    def __post_init__(self) -> None:
        if self.configuration not in CONFIG_ORDER:
            raise ValueError(
                f"unknown configuration {self.configuration!r}; must be one of {CONFIG_ORDER}"
            )

    def as_row(self) -> str:
        def fmt_metric(value: float | None) -> str:
            return _EMPTY_CELL if value is None else f"{value:.3f}"

        latency = (
            _EMPTY_CELL if self.p95_latency_ms is None else f"{self.p95_latency_ms:.0f} ms"
        )
        return (
            f"| {self.configuration} | {fmt_metric(self.recall_at_10)} | "
            f"{fmt_metric(self.mrr)} | {fmt_metric(self.faithfulness)} | {latency} |"
        )


def _parse_metric(cell: str) -> float | None:
    return None if cell in (_EMPTY_CELL, "-", "") else float(cell)


def _parse_latency(cell: str) -> float | None:
    if cell in (_EMPTY_CELL, "-", ""):
        return None
    return float(cell.removesuffix("ms").strip())


def _parse_existing_rows(row_lines: list[str]) -> list[ConfigResult]:
    parsed: list[ConfigResult] = []
    for line in row_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in CONFIG_ORDER:
            continue
        name, recall, mrr, faithfulness, latency = cells
        parsed.append(
            ConfigResult(
                configuration=name,
                recall_at_10=_parse_metric(recall),
                mrr=_parse_metric(mrr),
                faithfulness=_parse_metric(faithfulness),
                p95_latency_ms=_parse_latency(latency),
            )
        )
    return parsed


def render_table(results: dict[str, ConfigResult]) -> str:
    """Render the full table: header, divider, one row per CONFIG_ORDER entry."""
    lines = [TABLE_HEADER, TABLE_DIVIDER]
    for name in CONFIG_ORDER:
        if name in results:
            lines.append(results[name].as_row())
        else:
            empty = _EMPTY_CELL
            lines.append(f"| {name} | {empty} | {empty} | {empty} | {empty} |")
    return "\n".join(lines)


def update_results_table(path: str | Path, results: list[ConfigResult]) -> None:
    """Merge `results` into the table in `path`; every other row is left untouched."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()

    try:
        header_idx = lines.index(TABLE_HEADER)
    except ValueError as exc:
        raise ValueError(f"{path} has no results table to update (missing header row)") from exc
    if header_idx + 1 >= len(lines) or lines[header_idx + 1] != TABLE_DIVIDER:
        raise ValueError(f"{path} table header is not followed by the expected divider row")

    end_idx = header_idx + 2
    while end_idx < len(lines) and lines[end_idx].lstrip().startswith("|"):
        end_idx += 1

    existing_rows = _parse_existing_rows(lines[header_idx + 2 : end_idx])
    existing = {row.configuration: row for row in existing_rows}
    merged = {**existing, **{r.configuration: r for r in results}}

    new_table_lines = render_table(merged).splitlines()
    updated = lines[:header_idx] + new_table_lines + lines[end_idx:]
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")

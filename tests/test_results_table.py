import pytest

from ledger.eval.results_table import (
    CONFIG_ORDER,
    TABLE_DIVIDER,
    TABLE_HEADER,
    ConfigResult,
    update_results_table,
)

FRESH_TABLE = (
    "# Results\n"
    "\n"
    "## Expected shape\n"
    "\n"
    f"{TABLE_HEADER}\n"
    f"{TABLE_DIVIDER}\n"
    "| Naive 512 + dense | — | — | — | — |\n"
    "| Structural + dense | — | — | — | — |\n"
    "| Structural + hybrid RRF | — | — | — | — |\n"
    "| Contextual + hybrid RRF | — | — | — | — |\n"
    "| + cross-encoder rerank | — | — | — | — |\n"
    "\n"
    "Numbers above will be filled in as each configuration ships.\n"
)


def test_config_result_rejects_unknown_configuration():
    with pytest.raises(ValueError, match="unknown configuration"):
        ConfigResult(configuration="Not a real config")


def test_config_result_formats_row_with_missing_values_as_em_dash():
    result = ConfigResult(configuration="Naive 512 + dense")
    assert result.as_row() == "| Naive 512 + dense | — | — | — | — |"


def test_config_result_formats_metrics_and_latency():
    result = ConfigResult(
        configuration="Naive 512 + dense",
        recall_at_10=0.8123,
        mrr=0.71,
        faithfulness=0.9006,
        p95_latency_ms=812.4,
    )
    assert result.as_row() == "| Naive 512 + dense | 0.812 | 0.710 | 0.901 | 812 ms |"


def test_update_results_table_fills_in_one_row(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")

    update_results_table(
        path,
        [ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.62, mrr=0.55)],
    )

    text = path.read_text(encoding="utf-8")
    assert "| Naive 512 + dense | 0.620 | 0.550 | — | — |" in text
    assert "| Structural + dense | — | — | — | — |" in text
    assert "Numbers above will be filled in as each configuration ships." in text
    assert "# Results" in text


def test_update_results_table_preserves_earlier_rows_across_calls(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")

    update_results_table(
        path, [ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.62)]
    )
    update_results_table(
        path, [ConfigResult(configuration="Structural + dense", recall_at_10=0.74)]
    )

    text = path.read_text(encoding="utf-8")
    assert "| Naive 512 + dense | 0.620 | — | — | — |" in text
    assert "| Structural + dense | 0.740 | — | — | — |" in text


def test_update_results_table_overwrites_same_configuration_on_rerun(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")

    update_results_table(
        path, [ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.62)]
    )
    update_results_table(
        path, [ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.70)]
    )

    text = path.read_text(encoding="utf-8")
    assert "| Naive 512 + dense | 0.700 | — | — | — |" in text
    assert "0.620" not in text


def test_update_results_table_keeps_rows_in_config_order(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")

    update_results_table(
        path,
        [
            ConfigResult(configuration="+ cross-encoder rerank", recall_at_10=0.9),
            ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.6),
        ],
    )

    text = path.read_text(encoding="utf-8")
    row_order = [
        line for line in text.splitlines() if line.startswith("| ") and "recall" not in line
    ]
    names_in_order = [line.split("|")[1].strip() for line in row_order]
    assert names_in_order == CONFIG_ORDER


def test_update_results_table_requires_a_table(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text("# Results\n\nNo table yet.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no results table"):
        update_results_table(path, [ConfigResult(configuration="Naive 512 + dense")])

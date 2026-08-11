import pytest

from ledger.eval import run as eval_run
from ledger.eval.results_table import ConfigResult

FRESH_TABLE = (
    "# Results\n"
    "\n"
    "| Configuration | recall@10 | MRR | Faithfulness | p95 latency |\n"
    "|---|---|---|---|---|\n"
    "| Naive 512 + dense | — | — | — | — |\n"
    "| Structural + dense | — | — | — | — |\n"
    "| Structural + hybrid RRF | — | — | — | — |\n"
    "| Contextual + hybrid RRF | — | — | — | — |\n"
    "| + cross-encoder rerank | — | — | — | — |\n"
)


@pytest.fixture(autouse=True)
def _clean_pipeline_registry():
    """PIPELINES is process-global; keep tests from leaking registrations into each other."""
    original = dict(eval_run.PIPELINES)
    eval_run.PIPELINES.clear()
    yield
    eval_run.PIPELINES.clear()
    eval_run.PIPELINES.update(original)


def test_register_pipeline_rejects_unknown_slug():
    with pytest.raises(ValueError, match="unknown configuration slug"):
        eval_run.register_pipeline("not_a_real_slug", lambda: None)


def test_run_with_no_pipelines_registered_leaves_file_unchanged(tmp_path, capsys):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    eval_run.run("all", path)

    assert path.read_text(encoding="utf-8") == before
    assert "no pipeline registered" in capsys.readouterr().out


def test_run_updates_only_the_registered_configuration(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")
    eval_run.register_pipeline(
        "naive_dense",
        lambda: ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.62),
    )

    eval_run.run("all", path)

    text = path.read_text(encoding="utf-8")
    assert "| Naive 512 + dense | 0.620 | — | — | — |" in text
    assert "| Structural + dense | — | — | — | — |" in text


def test_run_with_specific_config_slug_ignores_others(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")
    calls = []
    eval_run.register_pipeline(
        "naive_dense",
        lambda: calls.append("naive_dense")
        or ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.5),
    )
    eval_run.register_pipeline(
        "structural_dense",
        lambda: calls.append("structural_dense")
        or ConfigResult(configuration="Structural + dense", recall_at_10=0.5),
    )

    eval_run.run("naive_dense", path)

    assert calls == ["naive_dense"]


def test_run_rejects_unknown_config():
    with pytest.raises(SystemExit, match="unknown --config"):
        eval_run.run("nonsense", "RESULTS.md")


def test_run_rejects_pipeline_that_returns_mismatched_configuration(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")
    eval_run.register_pipeline(
        "naive_dense",
        lambda: ConfigResult(configuration="Structural + dense", recall_at_10=0.5),
    )

    with pytest.raises(ValueError, match="expected 'Naive 512 \\+ dense'"):
        eval_run.run("naive_dense", path)


def test_main_parses_config_and_out_arguments(tmp_path):
    path = tmp_path / "RESULTS.md"
    path.write_text(FRESH_TABLE, encoding="utf-8")
    eval_run.register_pipeline(
        "naive_dense",
        lambda: ConfigResult(configuration="Naive 512 + dense", recall_at_10=0.9),
    )

    eval_run.main(["--config", "naive_dense", "--out", str(path)])

    assert "| Naive 512 + dense | 0.900 | — | — | — |" in path.read_text(encoding="utf-8")

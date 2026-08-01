"""Load/save the gold question set as JSON Lines, one GoldQuestion per line."""

from __future__ import annotations

from pathlib import Path

from ledger.eval.schema import GoldQuestion


def load_gold_set(path: str | Path) -> list[GoldQuestion]:
    path = Path(path)
    if not path.exists():
        return []
    questions: list[GoldQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(GoldQuestion.model_validate_json(line))
    return questions


def save_gold_set(questions: list[GoldQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json())
            f.write("\n")


def load_gold_dir(directory: str | Path) -> list[GoldQuestion]:
    """Load and concatenate every *.jsonl file in a directory (one per band)."""
    directory = Path(directory)
    questions: list[GoldQuestion] = []
    for path in sorted(directory.glob("*.jsonl")):
        questions.extend(load_gold_set(path))
    return questions

# Contributing

This is a solo portfolio build, but it follows real engineering conventions
throughout — that consistency is part of the point.

## Workflow

- `main` is always green (CI passing).
- Work in short-lived branches: `feat/<slug>`, `fix/<slug>`, `chore/<slug>`.
- Open a PR even solo — it's the changelog and the review trail.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(chunking): add structural section-aware chunker
fix(eval): correct recall@k off-by-one
docs(decisions): record ADR-001 rationale
chore(ci): add ruff lint step
```

## Before pushing

```bash
make lint
make test
```

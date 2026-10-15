# Contributing

Thanks for your interest! This is a research and learning project, so small, well-measured
changes are the most welcome kind.

## Scope

The project analyses **recorded** video offline. Out of scope, and not accepted:
hardware integration (cameras, flight controllers, gimbals, sensors), control, guidance,
targeting or any actuation logic, real-time or on-board deployment, identification of specific
military models, manufacturers or units, and geolocation of footage or people.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -r requirements.txt -r requirements-dev.txt
pytest
ruff check . && ruff format --check .
```

## Conventions

- Every script is an `argparse` CLI with `--help` and a `main()`; use `logging`, not `print`, and
  `tqdm` for long loops.
- `pathlib.Path` everywhere, no absolute paths; defaults resolve against the repo root.
- Shared code goes into `src/fpvdd/` with type hints and short docstrings.
- Tests for data-prep logic (splitting, dedup, label parsing) in `tests/`.
- Code, comments and commit messages in English; commit messages in the
  [Conventional Commits](https://www.conventionalcommits.org/) style (`feat:`, `fix:`, `docs:`).

## Data

- Split by video, never by frame.
- Never commit data, frames, videos or weights. The only exception is the README showcase in
  `docs/media/`.
- Every new data source gets a row in `sources/sources.csv` with its license. No download
  scrapers for YouTube, Telegram or similar platforms.
- Footage showing identifiable people or casualties is excluded.

## Experiments

New experiments get a write-up in `docs/experiments/` (setup, exact commands, results, findings,
limits) and, if they change the reference numbers, a row in the README results table. Results
that do not help are just as worth writing up.

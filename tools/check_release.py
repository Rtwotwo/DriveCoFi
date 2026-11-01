"""Non-destructive structural checks for a proposed public CoFiDrive release."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "weights",
    "exp",
    "dataset",
    "viz/output",
    "download/download_private_test_e2e.sh",
    "navsim/planning/script/config/common/train_test_split/scene_filter/competition_private_tokens.yaml",
)
MAX_FILE_BYTES = 100 * 1024 * 1024


def main():
    problems = []
    for name in FORBIDDEN:
        if (ROOT / name).exists():
            problems.append(f"Unexpected private/generated path: {name}")

    parsed = 0
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            problems.append(f"File exceeds 100 MiB: {rel}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
                parsed += 1
            except (SyntaxError, UnicodeError) as exc:
                problems.append(f"Python source cannot be parsed: {rel}: {exc}")

    if problems:
        raise SystemExit("Release check failed:\n" + "\n".join(problems))
    print(f"OK: {parsed} Python files parsed; no forbidden paths or >100 MiB files")


if __name__ == "__main__":
    main()

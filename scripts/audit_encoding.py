"""Fail when project text files are not UTF-8 or contain common mojibake."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".example",
    ".gitignore",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRECTORIES = {".git", "__pycache__", "venv"}
MOJIBAKE_MARKERS = (
    "\u00c6\u00b0",  # often appears where Vietnamese ư was double-decoded
    "\u00c6\u00a1",
    "\u00c4\u2018",
    "\u00e1\u00bb",
    "\u00e1\u00ba",
    "\u00e2\u20ac",
    "\u00f0\u0178",
)


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".editorconfig"}:
            yield path


def main() -> int:
    failures = []
    checked = 0
    for path in iter_text_files():
        checked += 1
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            failures.append(f"{path.relative_to(ROOT)}: invalid UTF-8 at byte {exc.start}")
            continue

        markers = [marker for marker in MOJIBAKE_MARKERS if marker in text]
        if markers:
            readable = ", ".join(ascii(marker) for marker in markers)
            failures.append(f"{path.relative_to(ROOT)}: mojibake marker(s) {readable}")

    if failures:
        print("Encoding audit FAILED:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"Encoding audit OK: {checked} UTF-8 text files, no mojibake markers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

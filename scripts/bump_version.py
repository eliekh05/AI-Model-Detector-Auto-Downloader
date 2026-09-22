#!/usr/bin/env python3
"""Bump the package version in src/ai_model_detector/__init__.py.

pyproject.toml reads the version dynamically — do not edit version there.

Usage:
    python scripts/bump_version.py patch   # 1.6.0 -> 1.6.1
    python scripts/bump_version.py minor   # 1.6.0 -> 1.7.0
    python scripts/bump_version.py major   # 1.6.0 -> 2.0.0
    python scripts/bump_version.py 1.7.0   # set exact version
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "src" / "ai_model_detector" / "__init__.py"
VERSION_RE = re.compile(r'^(__version__\s*=\s*)(["\'])([^"\']+)\2', re.M)


def read_version() -> str:
    text = INIT.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit(f"Could not find __version__ in {INIT}")
    return match.group(3)


def write_version(new: str) -> None:
    text = INIT.read_text(encoding="utf-8")
    updated, n = VERSION_RE.subn(rf'\1"{new}"', text, count=1)
    if n != 1:
        raise SystemExit(f"Failed to update __version__ in {INIT}")
    INIT.write_text(updated, encoding="utf-8")


def bump(part: str, current: str) -> str:
    try:
        major, minor, patch = (int(x) for x in current.split("."))
    except ValueError as exc:
        raise SystemExit(f"Version must be MAJOR.MINOR.PATCH, got {current!r}") from exc

    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise SystemExit(f"Unknown bump part: {part}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="patch | minor | major | or an exact X.Y.Z version",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the new version without writing the file",
    )
    args = parser.parse_args()

    current = read_version()
    if args.target in {"major", "minor", "patch"}:
        new = bump(args.target, current)
    elif re.fullmatch(r"\d+\.\d+\.\d+", args.target):
        new = args.target
    else:
        parser.error("target must be patch, minor, major, or X.Y.Z")

    if args.dry_run:
        print(f"{current} -> {new}")
        return

    write_version(new)
    print(f"Bumped {current} -> {new}")
    print(f"Updated {INIT.relative_to(ROOT)}")
    print("pyproject.toml picks this up automatically (dynamic version).")
    print(f"Optional: git tag v{new} && git push origin v{new}")


if __name__ == "__main__":
    main()
    sys.exit(0)

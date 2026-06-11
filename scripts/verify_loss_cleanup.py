from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import sys


ROOT = Path(__file__).resolve().parents[1]

BAD_PATTERNS = [
    r"\bcount_ord_loss\b",
    r"\bcount_ord\b",
    r"\bordinal_count\b",
    r"\bcount_ordinal\b",
    r"\bcount_rank\b",
    r"\bgcs_count_rank\b",
    r"\brank_gap_diagnostics\b",
    r"\brank_margin_loss\b",
    r"\brank_margin\b",
    r"\bmargin_loss\b",
]

SKIP_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    "archive",
    "datasets",
    "GCS-YOLO-Lane_LSA",
    "outputs",
    "runs",
    "wandb",
    "logs",
    "weights",
    "checkpoints",
    "dist",
    "build",
}

SKIP_FILES = {
    Path("scripts/verify_loss_cleanup.py"),
    Path("AGENTS.md"),
    Path("docs/agent-context/implementation-manual.md"),
}

SOURCE_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".md",
    ".txt",
}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in SKIP_DIRS for part in rel.parts):
        return True
    if rel in SKIP_FILES:
        return True
    return path.suffix not in SOURCE_SUFFIXES


def iter_source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        current = Path(dirpath)
        rel = current.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = current / filename
            if not should_skip(path):
                yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guard the default GCS mainline against accidental legacy loss tokens.")
    parser.add_argument(
        "--allow-legacy-loss-tokens",
        action="store_true",
        help="Bypass this default-mainline guard for a documented official-ACC restoration experiment.",
    )
    return parser.parse_args()


def legacy_loss_tokens_allowed(args: argparse.Namespace) -> bool:
    env_value = os.environ.get("GCS_ALLOW_LEGACY_LOSS_TOKENS", "").strip().lower()
    return args.allow_legacy_loss_tokens or env_value in {"1", "true", "yes", "on"}


def main() -> int:
    args = parse_args()
    if legacy_loss_tokens_allowed(args):
        print("OK: legacy loss token guard bypassed for an explicit restoration experiment.")
        return 0

    failed = False
    for path in iter_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in BAD_PATTERNS:
            for match in re.finditer(pattern, text):
                line_no = text.count("\n", 0, match.start()) + 1
                print(f"[BAD] {path.relative_to(ROOT)}:{line_no}: matched {pattern}")
                failed = True

    if failed:
        print("\nDefault-mainline loss cleanup verification failed.")
        print("Legacy loss tokens were found outside an approved restoration experiment.")
        print("Remove them from the default mainline, or use --allow-legacy-loss-tokens only for a documented official-ACC restoration branch.")
        return 1

    print("OK: no accidental legacy loss tokens found in the default mainline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

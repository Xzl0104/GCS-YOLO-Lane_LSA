from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK_SUFFIXES = {".py", ".yaml", ".yml", ".md"}
SKIP_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "archive",
    "datasets",
    "outputs",
    "runs",
    "GCS-YOLO-Lane_LSA",
}
SKIP_FILES = {
    Path("docs/agent-context/implementation-manual.md"),
}

LEGACY_LOSS_TOKENS = (
    "valid_continuity_loss",
    "count_over3_loss",
    "count_under4_loss",
    "count_over4_loss",
    "count_under5_loss",
    "duplicate_loss",
    "pred_count_over3",
    "pred_count_under4",
    "pred_count_over4",
    "pred_count_under5",
)


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


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CHECK_SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.relative_to(root) in SKIP_FILES:
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def main() -> None:
    args = parse_args()
    if legacy_loss_tokens_allowed(args):
        print("check_removed_losses: OK, legacy loss token guard bypassed for an explicit restoration experiment")
        return

    hits: list[str] = []
    for path in iter_files(ROOT):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            found = [token for token in LEGACY_LOSS_TOKENS if token in line]
            if found:
                rel = path.relative_to(ROOT)
                hits.append(f"{rel}:{lineno}: {', '.join(found)}")

    if hits:
        joined = "\n".join(hits)
        raise SystemExit(
            "Default-mainline legacy loss tokens are present outside an approved restoration experiment:\n"
            f"{joined}\n"
            "Remove them from the default mainline, or rerun with --allow-legacy-loss-tokens only for a documented official-ACC restoration branch."
        )

    print("check_removed_losses: OK, no accidental legacy loss tokens found in the default mainline")


if __name__ == "__main__":
    main()

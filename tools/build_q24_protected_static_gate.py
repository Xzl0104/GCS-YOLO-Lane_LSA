from __future__ import annotations

from pathlib import Path

from build_q20_dualbank_static_gate import DEFAULT_Q24_PUBLISH_BANK, DEFAULT_Q24_SAVE_DIR, main


if __name__ == "__main__":
    main(
        default_num_queries=24,
        default_allocations="6:6,7:5,8:4",
        default_save_dir=Path(DEFAULT_Q24_SAVE_DIR),
        default_publish_bank=Path(DEFAULT_Q24_PUBLISH_BANK),
    )

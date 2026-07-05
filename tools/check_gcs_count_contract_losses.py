from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_gcs import parse_args  # noqa: E402
from tools.diagnose_gcs_count_contract import (  # noqa: E402
    anchor_query_arrays,
    best_raw_query_for_gt,
    query_role_flags,
    selected_shortside_for_gt,
    side_ranks_by_lower_x,
    sorted_query_arrays,
)
from ultralytics.cfg import CFG_BOOL_KEYS, CFG_FLOAT_KEYS, CFG_INT_KEYS  # noqa: E402
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer  # noqa: E402
from ultralytics.models.yolo.gcs_lane.val import DEFAULT_LOSS_GAINS, LOSS_GAIN_ARGS, LOSS_NAMES  # noqa: E402
from ultralytics.utils.gcs_count_contract import shortside_image_visible_median  # noqa: E402
from ultralytics.utils.gcs_loss import GCSLoss  # noqa: E402


def _logit(p: float) -> float:
    t = torch.tensor(float(p)).clamp(1e-6, 1.0 - 1e-6)
    return float(torch.log(t / (1.0 - t)))


def _make_gt(k: int = 56, visible_counts: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    visible_counts = visible_counts or [6, 6, 6, 6]
    xs = [0.10, 0.30, 0.60, 0.82] if len(visible_counts) == 4 else torch.linspace(0.10, 0.90, len(visible_counts)).tolist()
    gt = torch.zeros(len(visible_counts), k, 2)
    valid = torch.zeros(len(visible_counts), k)
    for lane_i, (x, visible) in enumerate(zip(xs, visible_counts)):
        gt[lane_i, :, 0] = x
        gt[lane_i, :, 1] = torch.linspace(0.95, 0.20, k)
        valid[lane_i, :visible] = 1.0
    return gt, valid


def _make_preds(gt: torch.Tensor, valid: torch.Tensor, q: int = 7) -> dict[str, torch.Tensor]:
    num_lanes, k = gt.shape[:2]
    pred_points = torch.zeros(1, q, k, 2)
    pred_logits = torch.full((1, q), _logit(0.02))
    pred_valid_logits = torch.full((1, q, k), -6.0)
    for lane_i in range(num_lanes):
        pred_points[0, lane_i] = gt[lane_i]
        pred_logits[0, lane_i] = _logit(0.20)
        pred_valid_logits[0, lane_i, valid[lane_i] > 0.5] = 6.0

    # Clear far-spurious candidate, high score, enough valid anchors, far from right GT.
    pred_points[0, 4, :, 0] = 0.98
    pred_points[0, 4, :, 1] = gt[0, :, 1]
    pred_logits[0, 4] = _logit(0.80)
    pred_valid_logits[0, 4, :3] = 6.0

    # Duplicate-like unmatched query near a non-side matched GT.
    pred_points[0, 5] = gt[2]
    pred_points[0, 5, :, 0] += 0.01
    pred_logits[0, 5] = _logit(0.70)
    pred_valid_logits[0, 5, valid[2] > 0.5] = 6.0

    # Low-score filler query.
    pred_points[0, 6, :, 0] = 0.50
    pred_points[0, 6, :, 1] = gt[0, :, 1]
    return {"pred_points": pred_points, "pred_logits": pred_logits, "pred_valid_logits": pred_valid_logits}


def _loss_args(**overrides) -> dict:
    args = {"gcs_imgsz": [544, 960]}
    args.update(overrides)
    return args


def check_loss_names_and_gains() -> None:
    assert GCSLoss.loss_names == GCSLaneTrainer.loss_names
    assert GCSLoss.loss_names == LOSS_NAMES
    assert len(GCSLoss.loss_names) == len(GCSLaneTrainer.progress_loss_names) == 164
    assert len(LOSS_NAMES) == len(LOSS_GAIN_ARGS) == len(DEFAULT_LOSS_GAINS) == 164
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("farspur_if_loss")] == "gcs_farspur_weight"
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("rank_topk_loss")] == "gcs_rank_topk_weight"
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("shortside_rescue_exist_loss")] == "gcs_shortside_rescue_exist_gain"
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("shortside_rescue_point_loss")] == "gcs_shortside_rescue_point_gain"
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("shortside_rescue_valid_loss")] == "gcs_shortside_rescue_valid_gain"
    assert DEFAULT_LOSS_GAINS[LOSS_NAMES.index("shortside_score_floor_loss")] == 1.0
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("shortside_ultra_point_loss")] == "gcs_shortside_ultra_point_gain"
    assert LOSS_GAIN_ARGS[LOSS_NAMES.index("shortside_ultra_valid_loss")] == "gcs_shortside_ultra_valid_gain"
    for name in (
        "shortside_selected_hungarian_rawmatch",
        "shortside_selected_unmatched_rescue",
        "shortside_rescue_conflict",
        "shortside_missing_no_rawmatch",
        "shortside_nearest_was_duplicate_but_hungarian_boosted",
        "shortside_hungarian_rawmatch_candidate_count",
        "shortside_unmatched_raw_rescue_candidate_count",
        "shortside_rawmatch_candidate_total_count",
        "short_raw_boost_exist_target_mean",
        "short_raw_boost_exist_target_min",
        "short_raw_boost_exist_target_p25",
        "short_raw_boost_exist_target_p50",
        "short_raw_boost_exist_target_p75",
        "short_raw_boost_target_below_05_count",
        "short_raw_boost_target_below_07_count",
        "shortside_rawmatch_target_mean_before",
        "shortside_rawmatch_target_min_before",
        "shortside_rawmatch_target_p25_before",
        "shortside_rawmatch_target_p50_before",
        "shortside_rawmatch_target_below_05_count",
        "shortside_rawmatch_target_below_07_count",
        "shortside_target_floor_applied_count",
        "shortside_score_grad_down_count",
        "shortside_score_grad_up_count",
        "shortside_boost_only_mode",
        "shortside_protect_mode",
        "shortside_reliable_count",
        "shortside_reliable_selected_count",
        "shortside_ultra_seen_count",
        "shortside_ultra_enabled_count",
        "shortside_ultra_score_floor_count",
        "shortside_ultra_valid_count",
        "shortside_visible_lt2_skipped_count",
        "shortside_ultra_hungarian_base_boost_count",
        "shortside_ultra_unmatched_rescue_seen_count",
        "shortside_ultra_unmatched_rescue_enabled_count",
        "shortside_ultra_unmatched_rescue_skipped_count",
        "shortside_ultra_in_gt4_4to3_count",
        "shortside_ultra_in_gt5_5to4_count",
        "clear_far_boundary_count",
        "rank_neg_side_duplicate_like_count",
        "rank_neg_normal_duplicate_like_count",
        "rank_side_ambiguous_ignored_count",
        "rank_side_duplicate_rejected_better_than_true_count",
        "rank_pos_scope",
        "rank_pos_shortside_reliable",
        "rank_pos_shortside_ultra",
        "rank_pos_gt4gt5_matched",
        "rank_pos_all_matched",
        "rank_pos_hungarian_count",
        "rank_pos_unmatched_rescue_excluded_count",
        "rank_pos_unmatched_rescue_included_count",
        "rank_pos_conflict_excluded_count",
        "rank_loss_noop_images",
        "rank_noop_because_no_pos",
        "rank_noop_because_no_neg",
        "base_exist_ignore_raw_rescue_count",
        "base_exist_ignore_rank_near_count",
        "base_exist_ignore_rank_side_count",
        "base_exist_ignore_farspur_near_count",
        "base_exist_ignore_farspur_side_count",
        "base_exist_ignore_duplicate_rank_only_count",
        "base_exist_ignore_near_count",
        "base_exist_ignore_side_ambiguous_count",
        "base_exist_ignore_duplicate_like_count",
        "base_ignore_farspur_near_count",
        "base_ignore_farspur_side_count",
        "base_ignore_duplicate_like_count",
        "farspur_aux_only_mode",
        "farspur_full_ignore_first_mode",
        "farspur_near_still_base_negative_count",
        "farspur_side_still_base_negative_count",
        "rank_pair_only_mode",
        "rank_full_duplicate_contract_mode",
        "duplicate_like_still_base_negative_count",
        "base_exist_negative_kept_clear_far_count",
        "base_exist_negative_kept_other_count",
        "rank_neg_duplicate_like_count",
        "rank_neg_clear_far_count",
        "rank_pos_shortside_reliable_count",
        "rank_pos_gt4gt5_matched_count",
        "rank_loss_noop_image_count",
        "rank_noop_because_no_pos_count",
        "rank_noop_because_no_neg_count",
    ):
        idx = LOSS_NAMES.index(name)
        assert LOSS_GAIN_ARGS[idx] is None
        assert DEFAULT_LOSS_GAINS[idx] == 0.0


def check_k56_lower_half_uses_bottom_anchors() -> None:
    k = 56
    gt = torch.zeros(4, k, 2)
    valid = torch.ones(4, k)
    bottom_x = torch.tensor([0.10, 0.30, 0.60, 0.82])
    top_x = torch.tensor([0.90, 0.70, 0.40, 0.20])
    for lane_i in range(4):
        gt[lane_i, : k // 2, 0] = bottom_x[lane_i]
        gt[lane_i, k // 2 :, 0] = top_x[lane_i]
        gt[lane_i, :, 1] = torch.linspace(0.95, 0.20, k)

    crit = GCSLoss(_loss_args())
    loss_ranks = crit._gt_side_ranks_by_lower_x(gt, valid)
    diag_ranks = side_ranks_by_lower_x(gt.numpy(), valid.numpy())
    assert loss_ranks[0] == 0 and loss_ranks[3] == 3, loss_ranks
    assert diag_ranks[0] == 0 and diag_ranks[3] == 3, diag_ranks
    _, side_rank_values, eligible, _ = crit._eligible_shortside_gt_ids(gt, valid, gt_count=4)
    assert side_rank_values == {0, 3}
    assert eligible == [], eligible


def check_cli_and_config_defaults() -> None:
    expected = {
        "gcs_shortside_rawmatch_boost": 0.0,
        "gcs_shortside_raw_rescue": False,
        "gcs_shortside_min_gt_lanes": 4,
        "gcs_shortside_min_valid_points": 6,
        "gcs_shortside_ultra_min_valid_points": 2,
        "gcs_shortside_use_median": True,
        "gcs_shortside_median_margin": 0.0,
        "gcs_shortside_exist_target_floor": 0.0,
        "gcs_shortside_score_floor_gain": 0.0,
        "gcs_shortside_score_target": 0.8,
        "gcs_shortside_ultra_enable": False,
        "gcs_shortside_ultra_score_floor_gain": 0.0,
        "gcs_shortside_ultra_point_gain": 0.0,
        "gcs_shortside_ultra_valid_gain": 0.0,
        "gcs_shortside_ultra_rank_pos": False,
        "gcs_shortside_rescue_exist_gain": 0.0,
        "gcs_shortside_rescue_point_gain": 0.0,
        "gcs_shortside_rescue_valid_gain": 0.0,
        "gcs_shortside_debug": False,
        "gcs_allow_legacy_spurious_with_new_contract": False,
        "gcs_farspur_ignore_first": False,
        "gcs_farspur_weight": 0.0,
        "gcs_farspur_score_thr": 0.05,
        "gcs_farspur_near_dist_px": 40.0,
        "gcs_farspur_side_ignore_dist_px": 60.0,
        "gcs_farspur_clear_dist_px": 80.0,
        "gcs_farspur_min_valid_points": 2,
        "gcs_rank_topk_weight": 0.0,
        "gcs_rank_margin": 0.05,
        "gcs_rank_max_negs": 3,
        "gcs_rank_gt_min_lanes": 4,
        "gcs_rank_focus_shortside": True,
        "gcs_rank_pos_scope": "shortside_reliable",
        "gcs_rank_include_unmatched_rescue_pos": False,
        "gcs_rank_dup_close_px": 30.0,
        "gcs_rank_dup_min_overlap": 6,
        "gcs_rank_side_duplicate_enable": False,
        "gcs_rank_side_dup_margin_px": 5.0,
        "gcs_rank_side_dup_min_overlap": 6,
        "gcs_rank_near_gt_ignore_px": 40.0,
        "gcs_rank_side_ignore_px": 60.0,
        "gcs_rank_pair_reduction": "global_pair_mean",
        "gcs_base_ignore_raw_rescue": False,
        "gcs_base_ignore_rank_near": False,
        "gcs_base_ignore_farspur_near": False,
        "gcs_base_ignore_duplicate_like": False,
        "gcs_allow_gt3_count_contract_ablation": False,
    }
    args = parse_args([])
    defaults = yaml.safe_load((ROOT / "ultralytics/cfg/default.yaml").read_text(encoding="utf-8"))
    for name, value in expected.items():
        assert getattr(args, name) == value, (name, getattr(args, name), value)
        assert defaults[name] == value, (name, defaults[name], value)
    assert args.gcs_shortside_rawmatch_dist_px is None
    assert args.gcs_shortside_side_dist_px is None
    assert args.gcs_shortside_max_valid_points is None
    assert args.gcs_shortside_rawmatch_px is None
    assert args.gcs_shortside_side_rawmatch_px is None
    assert args.gcs_shortside_visible_max is None
    assert defaults["gcs_shortside_rawmatch_dist_px"] == 30.0
    assert defaults["gcs_shortside_side_dist_px"] == 40.0
    assert defaults["gcs_shortside_max_valid_points"] == 42
    assert defaults["gcs_shortside_rawmatch_px"] == 30.0
    assert defaults["gcs_shortside_side_rawmatch_px"] == 40.0
    assert defaults["gcs_shortside_visible_max"] == 42
    assert args.gcs_shortside_reliable_min_valid_points is None
    assert defaults["gcs_shortside_reliable_min_valid_points"] == 6
    assert GCSLoss(_loss_args()).shortside_visible_max == 42
    assert GCSLoss(_loss_args()).shortside_ultra_min_valid_points == 2
    assert GCSLoss(_loss_args()).shortside_reliable_min_valid_points == 6
    assert not GCSLoss(_loss_args()).shortside_ultra_enable
    assert GCSLoss(_loss_args()).rank_pos_scope == "shortside_reliable"
    assert not GCSLoss(_loss_args()).rank_include_unmatched_rescue_pos
    assert not GCSLoss(_loss_args()).rank_side_duplicate_enable
    assert GCSLoss(_loss_args()).rank_pair_reduction == "global_pair_mean"
    assert not GCSLoss(_loss_args()).base_ignore_raw_rescue
    assert not GCSLoss(_loss_args()).base_ignore_rank_near
    assert not GCSLoss(_loss_args()).base_ignore_farspur_near
    assert not GCSLoss(_loss_args()).base_ignore_duplicate_like
    assert not GCSLoss(_loss_args()).allow_gt3_count_contract_ablation

    for name in (
        "gcs_shortside_rawmatch_boost",
        "gcs_shortside_rawmatch_dist_px",
        "gcs_shortside_side_dist_px",
        "gcs_shortside_rawmatch_px",
        "gcs_shortside_side_rawmatch_px",
        "gcs_shortside_median_margin",
        "gcs_shortside_exist_target_floor",
        "gcs_shortside_score_floor_gain",
        "gcs_shortside_score_target",
        "gcs_shortside_ultra_score_floor_gain",
        "gcs_shortside_ultra_point_gain",
        "gcs_shortside_ultra_valid_gain",
        "gcs_shortside_rescue_exist_gain",
        "gcs_shortside_rescue_point_gain",
        "gcs_shortside_rescue_valid_gain",
        "gcs_farspur_weight",
        "gcs_farspur_score_thr",
        "gcs_farspur_near_dist_px",
        "gcs_farspur_side_ignore_dist_px",
        "gcs_farspur_clear_dist_px",
        "gcs_rank_topk_weight",
        "gcs_rank_margin",
        "gcs_rank_dup_close_px",
        "gcs_rank_side_dup_margin_px",
        "gcs_rank_near_gt_ignore_px",
        "gcs_rank_side_ignore_px",
    ):
        assert name in CFG_FLOAT_KEYS, name
    for name in (
        "gcs_shortside_max_valid_points",
        "gcs_shortside_min_gt_lanes",
        "gcs_shortside_min_valid_points",
        "gcs_shortside_ultra_min_valid_points",
        "gcs_shortside_reliable_min_valid_points",
        "gcs_shortside_visible_max",
        "gcs_farspur_min_valid_points",
        "gcs_rank_max_negs",
        "gcs_rank_gt_min_lanes",
        "gcs_rank_dup_min_overlap",
        "gcs_rank_side_dup_min_overlap",
    ):
        assert name in CFG_INT_KEYS, name
    for name in (
        "gcs_shortside_raw_rescue",
        "gcs_shortside_ultra_enable",
        "gcs_shortside_ultra_rank_pos",
        "gcs_shortside_use_median",
        "gcs_shortside_debug",
        "gcs_allow_legacy_spurious_with_new_contract",
        "gcs_farspur_ignore_first",
        "gcs_rank_focus_shortside",
        "gcs_rank_include_unmatched_rescue_pos",
        "gcs_rank_side_duplicate_enable",
        "gcs_base_ignore_raw_rescue",
        "gcs_base_ignore_rank_near",
        "gcs_base_ignore_farspur_near",
        "gcs_base_ignore_duplicate_like",
        "gcs_allow_gt3_count_contract_ablation",
    ):
        assert name in CFG_BOOL_KEYS, name

    canonical = GCSLoss(
        _loss_args(
            gcs_shortside_rawmatch_px=31.0,
            gcs_shortside_rawmatch_dist_px=20.0,
            gcs_shortside_side_rawmatch_px=41.0,
            gcs_shortside_side_dist_px=21.0,
            gcs_shortside_visible_max=44,
            gcs_shortside_max_valid_points=30,
        )
    )
    assert canonical.shortside_rawmatch_px == 31.0
    assert canonical.shortside_side_rawmatch_px == 41.0
    assert canonical.shortside_visible_max == 44

    alias = GCSLoss(
        _loss_args(
            gcs_shortside_rawmatch_dist_px=22.0,
            gcs_shortside_side_dist_px=33.0,
            gcs_shortside_max_valid_points=40,
        )
    )
    assert alias.shortside_rawmatch_px == 22.0
    assert alias.shortside_side_rawmatch_px == 33.0
    assert alias.shortside_visible_max == 40


def check_shortside_rawmatch_boost() -> None:
    gt, valid = _make_gt()
    preds = _make_preds(gt, valid)
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    crit = GCSLoss(_loss_args(gcs_shortside_rawmatch_boost=0.25))
    query_w, valid_w, count, gt4_count, gt5_count, missing = crit.shortside_rawmatch_boost_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        torch.tensor([4.0]),
    )
    assert (float(count), float(gt4_count), float(gt5_count), float(missing)) == (2.0, 2.0, 0.0, 0.0)
    assert abs(float(query_w[0, 0]) - 1.25) < 1e-6
    assert abs(float(query_w[0, 3]) - 1.25) < 1e-6
    assert abs(float(valid_w[0, 0, 0]) - 1.20) < 1e-6
    assert abs(float(valid_w[0, 3, 0]) - 1.20) < 1e-6
    assert abs(float(valid_w[0, 1, 0]) - 1.00) < 1e-6

    partial_indices = [(torch.arange(3, dtype=torch.long), torch.arange(3, dtype=torch.long))]
    crit_rescue = GCSLoss(_loss_args(gcs_shortside_raw_rescue=True))
    masks = crit_rescue.build_count_contract_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        partial_indices,
        torch.tensor([4.0]),
    )
    assert float(masks["shortside_hungarian_rawmatch_candidate_count"]) == 1.0
    assert float(masks["shortside_unmatched_raw_rescue_candidate_count"]) == 1.0
    assert float(masks["shortside_rawmatch_candidate_total_count"]) == 2.0
    assert float(masks["raw_rescue_candidate_count"]) == 1.0
    assert float(masks["raw_rescue_final_count"]) == 1.0
    assert not bool(masks["exist_ignore"][0, 3])
    assert bool(masks["shortside_rescue_pos"][0, 3])
    assert bool(masks["shortside_unmatched_rescue_pos"][0, 3])
    assert bool(masks["shortside_rawmatch_pos"][0, 3])
    assert not bool(masks["shortside_hungarian_rawmatch_pos"][0, 3])
    assert not bool(masks["rank_pos"][0, 3])
    assert float(masks["rank_pos_unmatched_rescue_excluded_count"]) == 1.0
    assert float(masks["rank_pos_unmatched_rescue_included_count"]) == 0.0
    assert float(masks["rank_pos_conflict_excluded_count"]) == 0.0
    assert float(masks["base_exist_ignore_raw_rescue_count"]) == 0.0
    exist_aux, point_aux, valid_aux = crit_rescue.shortside_rescue_aux_loss(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], masks
    )
    assert float(exist_aux) > 0.0
    assert float(point_aux) == 0.0
    assert float(valid_aux) > 0.0

    crit_rescue_base_ignore = GCSLoss(
        _loss_args(gcs_shortside_raw_rescue=True, gcs_base_ignore_raw_rescue=True)
    )
    masks_rescue_base_ignore = crit_rescue_base_ignore.build_count_contract_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        partial_indices,
        torch.tensor([4.0]),
    )
    assert bool(masks_rescue_base_ignore["exist_ignore"][0, 3])
    assert float(masks_rescue_base_ignore["base_exist_ignore_raw_rescue_count"]) == 1.0

    crit_rescue_rank = GCSLoss(
        _loss_args(gcs_shortside_raw_rescue=True, gcs_rank_include_unmatched_rescue_pos=True)
    )
    masks_rescue_rank = crit_rescue_rank.build_count_contract_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        partial_indices,
        torch.tensor([4.0]),
    )
    assert bool(masks_rescue_rank["rank_pos"][0, 3])
    assert float(masks_rescue_rank["rank_pos_unmatched_rescue_included_count"]) == 1.0
    assert float(masks_rescue_rank["rank_pos_unmatched_rescue_excluded_count"]) == 0.0


def check_shortside_hungarian_rawmatch_beats_nearest_duplicate() -> None:
    gt, valid = _make_gt(visible_counts=[6, 55, 55, 55])
    q, k = 6, gt.shape[1]
    preds = {
        "pred_points": torch.zeros(1, q, k, 2),
        "pred_logits": torch.full((1, q), _logit(0.02)),
        "pred_valid_logits": torch.full((1, q, k), -6.0),
    }
    width = 960.0
    preds["pred_points"][0, 0] = gt[0]
    preds["pred_points"][0, 0, :, 0] = gt[0, :, 0] + 10.0 / width
    preds["pred_logits"][0, 0] = _logit(0.20)
    preds["pred_valid_logits"][0, 0, valid[0] > 0.5] = 6.0

    preds["pred_points"][0, 1] = gt[0]
    preds["pred_points"][0, 1, :, 0] = gt[0, :, 0] + 5.0 / width
    preds["pred_logits"][0, 1] = _logit(0.40)
    preds["pred_valid_logits"][0, 1, valid[0] > 0.5] = 6.0

    for lane_i, query_i in enumerate([2, 3, 4], start=1):
        preds["pred_points"][0, query_i] = gt[lane_i]
        preds["pred_logits"][0, query_i] = _logit(0.20)
        preds["pred_valid_logits"][0, query_i, valid[lane_i] > 0.5] = 6.0
    preds["pred_points"][0, 5, :, 0] = 0.50
    preds["pred_points"][0, 5, :, 1] = gt[0, :, 1]

    indices = [(torch.tensor([0, 2, 3, 4]), torch.tensor([0, 1, 2, 3]))]
    gt_lanes = torch.tensor([4.0])
    crit = GCSLoss(_loss_args(gcs_shortside_rawmatch_boost=0.25))
    masks = crit.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks["shortside_boost_pos"][0, 0])
    assert bool(masks["shortside_hungarian_rawmatch_pos"][0, 0])
    assert bool(masks["shortside_rawmatch_pos"][0, 0])
    assert not bool(masks["shortside_rescue_pos"][0, 0])
    assert not bool(masks["shortside_boost_pos"][0, 1])
    assert not bool(masks["raw_rescue_pos"][0, 1])
    assert float(masks["shortside_selected_hungarian_rawmatch_count"]) == 1.0
    assert float(masks["shortside_selected_unmatched_rescue_count"]) == 0.0
    assert float(masks["shortside_nearest_was_duplicate_but_hungarian_boosted_count"]) == 1.0
    assert float(masks["shortside_missing_no_rawmatch_count"]) == 0.0
    assert int(masks["shortside_selected_reason_code"][0, 0].item()) == 1
    assert int(masks["shortside_selected_gt_idx"][0, 0].item()) == 0
    assert selected_shortside_for_gt(0, masks) == (0, "hungarian_rawmatch")
    roles = query_role_flags(0, masks)
    assert roles["selected_shortside_reason"] == "hungarian_rawmatch"

    query_w, _, count, _, _, missing = crit.shortside_rawmatch_boost_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        gt_lanes,
        contract_masks=masks,
    )
    assert float(count) == 1.0
    assert float(missing) == 0.0
    assert float(query_w[0, 0]) > 1.0
    assert float(query_w[0, 1]) == 1.0


def check_shortside_visible_boundaries() -> None:
    crit = GCSLoss(_loss_args())
    assert shortside_image_visible_median([30, 40, 50, 56], min_valid_points=2) == 45.0

    gt_low_high, valid_low_high = _make_gt(visible_counts=[5, 20, 20, 43])
    _, _, eligible_low_high, stats_low_high = crit._eligible_shortside_gt_ids(gt_low_high, valid_low_high, gt_count=4)
    assert eligible_low_high == [0], eligible_low_high
    assert stats_low_high == {"abs": 1, "median": 0, "total": 1, "reliable": 0, "ultra": 1, "lt2": 0}

    gt_too_short, valid_too_short = _make_gt(visible_counts=[1, 20, 20, 43])
    _, _, eligible_too_short, stats_too_short = crit._eligible_shortside_gt_ids(gt_too_short, valid_too_short, gt_count=4)
    assert eligible_too_short == [], eligible_too_short
    assert stats_too_short == {"abs": 0, "median": 0, "total": 0, "reliable": 0, "ultra": 0, "lt2": 1}

    gt_edges, valid_edges = _make_gt(visible_counts=[6, 20, 20, 42])
    _, _, eligible_edges, stats_edges = crit._eligible_shortside_gt_ids(gt_edges, valid_edges, gt_count=4)
    assert eligible_edges == [0, 3], eligible_edges
    assert stats_edges == {"abs": 2, "median": 0, "total": 2, "reliable": 2, "ultra": 0, "lt2": 0}

    gt_median, valid_median = _make_gt(visible_counts=[44, 55, 52, 46, 56])
    _, _, eligible_median, stats_median = crit._eligible_shortside_gt_ids(gt_median, valid_median, gt_count=5)
    assert eligible_median == [0], eligible_median
    assert stats_median == {"abs": 0, "median": 1, "total": 1, "reliable": 1, "ultra": 0, "lt2": 0}


def check_shortside_target_floor_and_ultra_tiers() -> None:
    gt, valid = _make_gt(visible_counts=[6, 20, 20, 6])
    preds = _make_preds(gt, valid)
    preds["pred_valid_logits"][0, 0] = -6.0
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])
    crit = GCSLoss(_loss_args(gcs_shortside_rawmatch_boost=0.25))
    masks = crit.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks["shortside_target_floor_pos"][0, 0])
    _, target = crit.exist_loss(
        preds["pred_logits"],
        preds["pred_points"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        target_floor_mask=masks["shortside_target_floor_pos"],
        target_floor=crit.shortside_exist_target_floor,
        return_target=True,
    )
    assert float(crit.shortside_exist_target_floor) == 0.0
    assert float(target[0, 0]) < 0.7
    _, boost_items = crit(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(boost_items[GCSLoss.loss_names.index("shortside_boost_only_mode")]) == 1.0
    assert float(boost_items[GCSLoss.loss_names.index("shortside_protect_mode")]) == 0.0
    assert float(boost_items[GCSLoss.loss_names.index("shortside_score_grad_down_count")]) > 0.0

    crit_floor = GCSLoss(
        _loss_args(gcs_shortside_rawmatch_boost=0.25, gcs_shortside_exist_target_floor=0.7)
    )
    masks_floor = crit_floor.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    _, target = crit_floor.exist_loss(
        preds["pred_logits"],
        preds["pred_points"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        target_floor_mask=masks_floor["shortside_target_floor_pos"],
        target_floor=crit_floor.shortside_exist_target_floor,
        return_target=True,
    )
    assert float(target[0, 0]) >= 0.699
    stats = crit_floor._selected_target_stats(target, masks_floor["shortside_target_floor_pos"])
    assert float(stats[5]) == 0.0
    assert float(stats[6]) == 0.0
    _, floor_items = crit_floor(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(floor_items[GCSLoss.loss_names.index("shortside_rawmatch_target_below_07_count")]) > 0.0
    assert float(floor_items[GCSLoss.loss_names.index("shortside_target_floor_applied_count")]) > 0.0
    assert float(floor_items[GCSLoss.loss_names.index("shortside_boost_only_mode")]) == 0.0
    assert float(floor_items[GCSLoss.loss_names.index("shortside_protect_mode")]) == 1.0
    assert float(floor_items[GCSLoss.loss_names.index("shortside_score_grad_up_count")]) > 0.0

    ultra_gt, ultra_valid = _make_gt(visible_counts=[5, 20, 20, 43])
    ultra_preds = _make_preds(ultra_gt, ultra_valid)
    ultra_preds["pred_logits"].fill_(_logit(0.001))
    for q in (0, 1, 2):
        ultra_preds["pred_logits"][0, q] = _logit(0.99)
    ultra_masks = crit.build_count_contract_masks(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        [ultra_gt],
        [ultra_valid],
        indices,
        gt_lanes,
    )
    assert float(ultra_masks["shortside_reliable_count"]) == 0.0
    assert float(ultra_masks["shortside_ultra_seen_count"]) == 1.0
    assert float(ultra_masks["shortside_ultra_enabled_count"]) == 0.0
    assert float(ultra_masks["shortside_ultra_score_floor_count"]) == 0.0
    assert float(ultra_masks["shortside_ultra_valid_count"]) == 0.0
    assert not bool(ultra_masks["shortside_ultra_score_floor_pos"][0, 0])
    assert not bool(ultra_masks["shortside_ultra_valid_pos"][0, 0])
    assert bool(ultra_masks["shortside_boost_pos"][0, 0])
    assert bool(ultra_masks["shortside_target_floor_pos"][0, 0])
    assert float(ultra_masks["shortside_ultra_hungarian_base_boost_count"]) == 1.0
    assert float(ultra_masks["shortside_ultra_unmatched_rescue_seen_count"]) == 0.0
    assert float(ultra_masks["rank_pos_shortside_matched_count"]) == 0.0
    assert float(ultra_masks["shortside_ultra_in_gt4_4to3_count"]) == 1.0

    query_w, _, ultra_boost_count, _, _, _ = crit.shortside_rawmatch_boost_masks(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        [ultra_gt],
        [ultra_valid],
        indices,
        gt_lanes,
        contract_masks=ultra_masks,
    )
    assert float(ultra_boost_count) == 1.0
    assert float(query_w[0, 0]) > 1.0

    ultra_enabled = GCSLoss(
        _loss_args(
            gcs_shortside_ultra_enable=True,
            gcs_shortside_ultra_score_floor_gain=0.005,
            gcs_shortside_ultra_valid_gain=0.005,
        )
    )
    ultra_masks_enabled = ultra_enabled.build_count_contract_masks(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        [ultra_gt],
        [ultra_valid],
        indices,
        gt_lanes,
    )
    assert float(ultra_masks_enabled["shortside_ultra_seen_count"]) == 1.0
    assert float(ultra_masks_enabled["shortside_ultra_hungarian_base_boost_count"]) == 1.0
    assert float(ultra_masks_enabled["shortside_ultra_enabled_count"]) == 0.0
    assert float(ultra_masks_enabled["shortside_ultra_score_floor_count"]) == 0.0
    assert float(ultra_masks_enabled["shortside_ultra_valid_count"]) == 0.0
    assert not bool(ultra_masks_enabled["shortside_ultra_score_floor_pos"][0, 0])
    assert not bool(ultra_masks_enabled["shortside_ultra_valid_pos"][0, 0])

    ultra_partial_indices = [(torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([1, 2, 3], dtype=torch.long))]
    ultra_unmatched_masks = crit.build_count_contract_masks(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        [ultra_gt],
        [ultra_valid],
        ultra_partial_indices,
        gt_lanes,
    )
    assert float(ultra_unmatched_masks["shortside_ultra_unmatched_rescue_seen_count"]) == 1.0
    assert float(ultra_unmatched_masks["shortside_ultra_unmatched_rescue_enabled_count"]) == 0.0
    assert float(ultra_unmatched_masks["shortside_ultra_unmatched_rescue_skipped_count"]) == 1.0
    assert not bool(ultra_unmatched_masks["shortside_boost_pos"][0, 0])

    ultra_unmatched_enabled = ultra_enabled.build_count_contract_masks(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        [ultra_gt],
        [ultra_valid],
        ultra_partial_indices,
        gt_lanes,
    )
    assert float(ultra_unmatched_enabled["shortside_ultra_unmatched_rescue_seen_count"]) == 1.0
    assert float(ultra_unmatched_enabled["shortside_ultra_unmatched_rescue_enabled_count"]) == 1.0
    assert float(ultra_unmatched_enabled["shortside_ultra_unmatched_rescue_skipped_count"]) == 0.0
    assert bool(ultra_unmatched_enabled["shortside_ultra_score_floor_pos"][0, 0])
    assert bool(ultra_unmatched_enabled["shortside_ultra_valid_pos"][0, 0])
    ultra_valid_loss = ultra_enabled.shortside_ultra_valid_loss(
        ultra_preds["pred_valid_logits"], [ultra_valid], ultra_unmatched_enabled
    )
    assert float(ultra_valid_loss) > 0.0


def check_farspur_ignore_first_and_ranking() -> None:
    gt, valid = _make_gt()
    preds = _make_preds(gt, valid)
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])

    crit_far = GCSLoss(_loss_args(gcs_farspur_weight=0.005, gcs_farspur_ignore_first=True))
    loss, samples, pos, ignore, clear, near, side = crit_far.farspur_ignore_first_loss(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], indices, gt_lanes, [gt], [valid]
    )
    assert float(loss) > 0.0
    assert (float(samples), float(pos), float(clear)) == (1.0, 4.0, 1.0)
    assert float(ignore) >= 1.0
    assert float(near) >= 1.0 or float(side) >= 1.0

    crit_rank = GCSLoss(_loss_args(gcs_rank_topk_weight=0.02))
    rank_loss, rank_samples, rank_pos, rank_neg, rank_clear, rank_dup, rank_noop, rank_no_pos, rank_no_neg = (
        crit_rank.rank_topk_loss(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], indices, gt_lanes, [gt], [valid]
        )
    )
    assert float(rank_loss) > 0.0
    assert (float(rank_samples), float(rank_pos)) == (1.0, 2.0)
    assert float(rank_neg) == 2.0
    assert float(rank_clear) == 1.0
    assert float(rank_dup) == 1.0
    assert (float(rank_noop), float(rank_no_pos), float(rank_no_neg)) == (0.0, 0.0, 0.0)
    masks_focus = crit_rank.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert float(masks_focus["rank_pos_scope_id"]) == 0.0
    assert float(masks_focus["rank_pos_hungarian_all_count"]) == 4.0
    assert float(masks_focus["rank_pos_shortside_matched_count"]) == 2.0
    assert float(masks_focus["rank_pos_shortside_rescue_count"]) == 0.0
    assert float(masks_focus["rank_pos_shortside_reliable_count"]) == 2.0
    assert float(masks_focus["rank_pos_shortside_ultra_count"]) == 0.0
    assert float(masks_focus["rank_pos_gt4gt5_matched_count"]) == 4.0
    assert float(masks_focus["rank_pos_all_matched_count"]) == 4.0
    assert float(masks_focus["rank_pos_total_count"]) == (
        float(masks_focus["rank_pos_shortside_matched_count"]) + float(masks_focus["rank_pos_shortside_rescue_count"])
    )

    crit_rank_all = GCSLoss(_loss_args(gcs_rank_topk_weight=0.02, gcs_rank_focus_shortside=False))
    _, _, rank_pos_all, rank_neg_all, _, _, _, _, _ = crit_rank_all.rank_topk_loss(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], indices, gt_lanes, [gt], [valid]
    )
    assert float(rank_pos_all) == 4.0
    assert float(rank_neg_all) == 2.0

    crit_rank_gt4 = GCSLoss(_loss_args(gcs_rank_topk_weight=0.02, gcs_rank_pos_scope="gt4gt5_matched"))
    _, _, rank_pos_gt4, rank_neg_gt4, _, _, _, _, _ = crit_rank_gt4.rank_topk_loss(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], indices, gt_lanes, [gt], [valid]
    )
    assert float(rank_pos_gt4) == 4.0
    assert float(rank_neg_gt4) == 2.0

    total, items = crit_rank(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    idx = GCSLoss.loss_names.index("rank_topk_loss")
    assert int(items.numel()) == len(GCSLoss.loss_names)
    assert float(items[idx]) > 0.0
    assert float(items[GCSLoss.loss_names.index("rank_pos_total")]) == 2.0
    assert float(items[GCSLoss.loss_names.index("rank_pos_hungarian_all")]) == 4.0
    assert float(items[GCSLoss.loss_names.index("rank_pos_shortside_matched")]) == 2.0
    assert float(items[GCSLoss.loss_names.index("rank_pos_shortside_reliable")]) == 2.0
    assert float(items[GCSLoss.loss_names.index("rank_neg_duplicate_like")]) == 1.0
    assert float(items[GCSLoss.loss_names.index("rank_neg_clear_far")]) == 1.0
    assert float(items[GCSLoss.loss_names.index("rank_loss_noop_images")]) == 0.0
    assert torch.isfinite(total)

    no_neg_preds = _make_preds(gt, valid)
    no_neg_preds["pred_valid_logits"][0, 4:] = -6.0
    _, no_neg_samples, _, _, _, _, no_neg_noop, no_neg_no_pos, no_neg_no_neg = crit_rank.rank_topk_loss(
        no_neg_preds["pred_points"],
        no_neg_preds["pred_logits"],
        no_neg_preds["pred_valid_logits"],
        indices,
        gt_lanes,
        [gt],
        [valid],
    )
    assert float(no_neg_samples) == 0.0
    assert (float(no_neg_noop), float(no_neg_no_pos), float(no_neg_no_neg)) == (1.0, 0.0, 1.0)

    ultra_gt, ultra_valid = _make_gt(visible_counts=[5, 20, 20, 43])
    ultra_preds = _make_preds(ultra_gt, ultra_valid)
    ultra_indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    _, _, _, _, _, _, ultra_noop, ultra_no_pos, ultra_no_neg = crit_rank.rank_topk_loss(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        ultra_indices,
        gt_lanes,
        [ultra_gt],
        [ultra_valid],
    )
    assert (float(ultra_noop), float(ultra_no_pos), float(ultra_no_neg)) == (1.0, 1.0, 0.0)

    _, _, ultra_gt4_pos, ultra_gt4_neg, _, _, _, _, _ = crit_rank_gt4.rank_topk_loss(
        ultra_preds["pred_points"],
        ultra_preds["pred_logits"],
        ultra_preds["pred_valid_logits"],
        ultra_indices,
        gt_lanes,
        [ultra_gt],
        [ultra_valid],
    )
    assert float(ultra_gt4_pos) == 4.0
    assert float(ultra_gt4_neg) == 2.0


def check_rank_pair_reduction() -> None:
    pred_points = torch.zeros(2, 4, 56, 2)
    pred_valid_logits = torch.zeros(2, 4, 56)
    pred_logits = torch.tensor(
        [
            [_logit(0.10), _logit(0.90), _logit(0.01), _logit(0.01)],
            [_logit(0.80), _logit(0.70), _logit(0.10), _logit(0.20)],
        ]
    )
    rank_pos = torch.tensor([[True, False, False, False], [True, True, False, False]])
    rank_neg = torch.tensor([[False, True, False, False], [False, False, True, True]])
    clear_far = rank_neg.clone()
    duplicate = torch.zeros_like(rank_neg)
    masks = {
        "rank_pos": rank_pos,
        "rank_neg": rank_neg,
        "clear_far_spurious": clear_far,
        "duplicate_like": duplicate,
    }
    gt_lanes = torch.tensor([4.0, 4.0])
    gt_points = [torch.zeros(4, 56, 2), torch.zeros(4, 56, 2)]
    gt_valid = [torch.ones(4, 56), torch.ones(4, 56)]
    indices = [
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
    ]
    crit_global = GCSLoss(_loss_args(gcs_rank_topk_weight=0.02, gcs_rank_max_negs=2))
    global_loss, *_ = crit_global.rank_topk_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        indices,
        gt_lanes,
        gt_points,
        gt_valid,
        contract_masks=masks,
    )
    crit_image = GCSLoss(
        _loss_args(gcs_rank_topk_weight=0.02, gcs_rank_max_negs=2, gcs_rank_pair_reduction="image_mean")
    )
    image_loss, *_ = crit_image.rank_topk_loss(
        pred_points,
        pred_logits,
        pred_valid_logits,
        indices,
        gt_lanes,
        gt_points,
        gt_valid,
        contract_masks=masks,
    )
    assert abs(float(global_loss) - 0.17) < 1e-5, float(global_loss)
    assert abs(float(image_loss) - 0.425) < 1e-5, float(image_loss)
    assert float(global_loss) < float(image_loss)


def check_near_only_is_not_rank_negative() -> None:
    gt, valid = _make_gt()
    preds = _make_preds(gt, valid)
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])

    near_q = 6
    preds["pred_points"][0, near_q] = gt[2]
    preds["pred_points"][0, near_q, :, 0] = gt[2, :, 0] + 55.0 / 960.0
    preds["pred_logits"][0, near_q] = _logit(0.90)
    preds["pred_valid_logits"][0, near_q, valid[2] > 0.5] = 6.0

    crit = GCSLoss(_loss_args(gcs_rank_near_gt_ignore_px=80.0, gcs_farspur_clear_dist_px=80.0))
    masks = crit.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks["near_gt_corridor"][0, near_q])
    assert not bool(masks["duplicate_like"][0, near_q])
    assert not bool(masks["clear_far_spurious"][0, near_q])
    assert not bool(masks["rank_neg"][0, near_q])


def check_base_exist_ignore_contract() -> None:
    gt, valid = _make_gt(visible_counts=[6, 20, 20, 6])
    preds = _make_preds(gt, valid)
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])

    side_q = 5
    preds["pred_points"][0, side_q] = gt[3]
    preds["pred_points"][0, side_q, :, 0] = gt[3, :, 0] + 12.0 / 960.0
    preds["pred_logits"][0, side_q] = _logit(0.70)
    preds["pred_valid_logits"][0, side_q] = -6.0
    preds["pred_valid_logits"][0, side_q, valid[3] > 0.5] = 6.0

    dup_q = 6
    preds["pred_points"][0, dup_q] = gt[2]
    preds["pred_points"][0, dup_q, :, 0] = gt[2, :, 0] + 8.0 / 960.0
    preds["pred_logits"][0, dup_q] = _logit(0.70)
    preds["pred_valid_logits"][0, dup_q] = -6.0
    preds["pred_valid_logits"][0, dup_q, valid[2] > 0.5] = 6.0

    crit_rank_only = GCSLoss(_loss_args(gcs_rank_topk_weight=0.02))
    masks_rank_only = crit_rank_only.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_rank_only["clear_far_spurious"][0, 4])
    assert not bool(masks_rank_only["exist_ignore"][0, 4])
    assert not bool(masks_rank_only["exist_ignore"][0, side_q])
    assert not bool(masks_rank_only["exist_ignore"][0, dup_q])
    assert masks_rank_only["point_valid_ignore"] is not None
    assert not bool(masks_rank_only["point_valid_ignore"][0, side_q].any())
    assert not bool(masks_rank_only["point_valid_ignore"][0, dup_q].any())
    assert float(masks_rank_only["base_exist_ignore_rank_near_count"]) == 0.0
    assert float(masks_rank_only["base_exist_ignore_rank_side_count"]) == 0.0
    assert float(masks_rank_only["base_exist_ignore_duplicate_rank_only_count"]) == 0.0
    assert float(masks_rank_only["duplicate_like_still_base_negative_count"]) >= 1.0
    _, rank_pair_items = crit_rank_only(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(rank_pair_items[GCSLoss.loss_names.index("rank_pair_only_mode")]) == 1.0
    assert float(rank_pair_items[GCSLoss.loss_names.index("rank_full_duplicate_contract_mode")]) == 0.0
    assert float(rank_pair_items[GCSLoss.loss_names.index("duplicate_like_still_base_negative_count")]) >= 1.0

    crit_dup_only = GCSLoss(
        _loss_args(
            gcs_rank_topk_weight=0.02,
            gcs_base_ignore_duplicate_like=True,
        )
    )
    masks_dup_only = crit_dup_only.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_dup_only["duplicate_like"][0, dup_q])
    assert bool(masks_dup_only["exist_ignore"][0, dup_q])
    assert not bool(masks_dup_only["duplicate_like"][0, side_q])
    assert bool(masks_dup_only["ambiguous_side_region"][0, side_q])
    assert not bool(masks_dup_only["exist_ignore"][0, side_q])
    assert masks_dup_only["point_valid_ignore"] is not None
    assert bool(masks_dup_only["point_valid_ignore"][0, dup_q].all())
    assert not bool(masks_dup_only["point_valid_ignore"][0, side_q].any())
    assert float(masks_dup_only["base_exist_ignore_rank_side_count"]) == 0.0
    assert float(masks_dup_only["base_exist_ignore_duplicate_rank_only_count"]) >= 1.0
    _, rank_dup_only_items = crit_dup_only(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(rank_dup_only_items[GCSLoss.loss_names.index("rank_pair_only_mode")]) == 0.0
    assert float(rank_dup_only_items[GCSLoss.loss_names.index("rank_full_duplicate_contract_mode")]) == 1.0
    assert float(rank_dup_only_items[GCSLoss.loss_names.index("base_ignore_duplicate_like_count")]) >= 1.0
    assert float(rank_dup_only_items[GCSLoss.loss_names.index("duplicate_like_still_base_negative_count")]) == 0.0

    crit = GCSLoss(
        _loss_args(
            gcs_rank_topk_weight=0.02,
            gcs_base_ignore_rank_near=True,
            gcs_base_ignore_duplicate_like=True,
        )
    )
    masks = crit.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert not bool(masks["exist_ignore"][0, 4])
    assert bool(masks["exist_ignore"][0, side_q])
    assert bool(masks["exist_ignore"][0, dup_q])
    assert masks["point_valid_ignore"] is not None
    assert bool(masks["point_valid_ignore"][0, side_q].all())
    assert bool(masks["point_valid_ignore"][0, dup_q].all())
    assert float(masks["base_exist_ignore_near_count"]) >= 1.0
    assert float(masks["base_exist_ignore_side_ambiguous_count"]) >= 1.0
    assert float(masks["base_exist_ignore_duplicate_like_count"]) >= 1.0
    assert float(masks["base_exist_ignore_rank_near_count"]) >= 1.0
    assert float(masks["base_exist_ignore_rank_side_count"]) >= 1.0
    assert float(masks["base_exist_ignore_duplicate_rank_only_count"]) >= 1.0
    assert float(masks["base_exist_negative_kept_clear_far_count"]) >= 1.0
    assert float(masks["duplicate_like_still_base_negative_count"]) == 0.0
    _, rank_full_items = crit(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(rank_full_items[GCSLoss.loss_names.index("rank_pair_only_mode")]) == 0.0
    assert float(rank_full_items[GCSLoss.loss_names.index("rank_full_duplicate_contract_mode")]) == 1.0
    assert float(rank_full_items[GCSLoss.loss_names.index("base_ignore_duplicate_like_count")]) >= 1.0
    assert float(rank_full_items[GCSLoss.loss_names.index("duplicate_like_still_base_negative_count")]) == 0.0

    gt3_masks = crit.build_count_contract_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        torch.tensor([3.0]),
    )
    assert not bool(gt3_masks["exist_ignore"][0, side_q])
    assert not bool(gt3_masks["exist_ignore"][0, dup_q])
    assert float(gt3_masks["base_exist_ignore_rank_side_count"]) == 0.0
    assert float(gt3_masks["base_exist_ignore_duplicate_rank_only_count"]) == 0.0


def check_side_ambiguous_duplicate_split() -> None:
    gt, valid = _make_gt(visible_counts=[6, 20, 20, 6])
    preds = _make_preds(gt, valid)
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])
    side_dup_q = 5
    preds["pred_points"][0, side_dup_q] = gt[3]
    preds["pred_points"][0, side_dup_q, :, 0] = gt[3, :, 0] + 12.0 / 960.0
    preds["pred_logits"][0, side_dup_q] = _logit(0.70)
    preds["pred_valid_logits"][0, side_dup_q] = -6.0
    preds["pred_valid_logits"][0, side_dup_q, valid[3] > 0.5] = 6.0

    crit_default = GCSLoss(_loss_args())
    masks_default = crit_default.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_default["ambiguous_side_region"][0, side_dup_q])
    assert not bool(masks_default["normal_duplicate_like"][0, side_dup_q])
    assert not bool(masks_default["side_duplicate_like"][0, side_dup_q])
    assert not bool(masks_default["rank_neg"][0, side_dup_q])
    assert float(masks_default["rank_side_ambiguous_ignored_count"]) == 1.0

    crit_side = GCSLoss(_loss_args(gcs_rank_side_duplicate_enable=True, gcs_rank_side_dup_margin_px=5.0))
    masks_side = crit_side.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_side["side_duplicate_like"][0, side_dup_q])
    assert bool(masks_side["duplicate_like"][0, side_dup_q])
    assert bool(masks_side["rank_neg"][0, side_dup_q])
    assert not bool(masks_side["normal_duplicate_like"][0, side_dup_q])
    assert float(masks_side["rank_neg_side_duplicate_like_count"]) == 1.0
    assert float(masks_side["rank_neg_normal_duplicate_like_count"]) == 0.0
    assert float(masks_side["rank_side_ambiguous_ignored_count"]) == 0.0

    better_preds = _make_preds(gt, valid)
    better_preds["pred_points"][0, side_dup_q] = gt[3]
    better_preds["pred_points"][0, side_dup_q, :, 0] = gt[3, :, 0] + 2.0 / 960.0
    better_preds["pred_logits"][0, side_dup_q] = _logit(0.70)
    better_preds["pred_valid_logits"][0, side_dup_q] = -6.0
    better_preds["pred_valid_logits"][0, side_dup_q, valid[3] > 0.5] = 6.0
    masks_better = crit_side.build_count_contract_masks(
        better_preds["pred_points"],
        better_preds["pred_logits"],
        better_preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        gt_lanes,
    )
    assert not bool(masks_better["side_duplicate_like"][0, side_dup_q])
    assert not bool(masks_better["rank_neg"][0, side_dup_q])
    assert float(masks_better["rank_side_duplicate_rejected_better_than_true_count"]) == 1.0


def check_clear_far_boundary_is_strictly_greater() -> None:
    gt, valid = _make_gt(visible_counts=[56, 56, 56, 56])
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])
    crit = GCSLoss(
        _loss_args(
            gcs_farspur_clear_dist_px=80.0,
            gcs_farspur_score_thr=0.05,
            gcs_rank_near_gt_ignore_px=40.0,
            gcs_farspur_near_dist_px=40.0,
        )
    )

    def masks_for_offset(offset_px: float) -> dict[str, torch.Tensor | None]:
        preds = _make_preds(gt, valid)
        q = 4
        preds["pred_points"][0, q, :, 0] = gt[3, :, 0] + offset_px / 960.0
        preds["pred_points"][0, q, :, 1] = gt[3, :, 1]
        preds["pred_logits"][0, q] = _logit(0.80)
        preds["pred_valid_logits"][0, q] = -6.0
        preds["pred_valid_logits"][0, q, :4] = 6.0
        return crit.build_count_contract_masks(
            preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
        )

    boundary = masks_for_offset(80.0)
    assert not bool(boundary["clear_far_spurious"][0, 4])
    assert not bool(boundary["farspur_clear_far_spurious"][0, 4])
    assert float(boundary["clear_far_boundary_count"]) == 1.0

    beyond = masks_for_offset(80.5)
    assert bool(beyond["clear_far_spurious"][0, 4])
    assert bool(beyond["farspur_clear_far_spurious"][0, 4])
    assert float(beyond["clear_far_boundary_count"]) == 0.0


def check_legacy_spurious_conflict() -> None:
    try:
        GCSLoss(_loss_args(gcs_spurious_neg=0.1, gcs_rank_topk_weight=0.02))
    except ValueError as exc:
        assert "gcs_spurious_neg conflicts with the new ignore-first/ranking contract" in str(exc)
    else:
        raise AssertionError("Expected gcs_spurious_neg + ranking contract to raise.")

    try:
        GCSLoss(_loss_args(gcs_far_spurious_neg=0.1, gcs_farspur_ignore_first=True, gcs_farspur_weight=0.005))
    except ValueError as exc:
        assert "gcs_far_spurious_neg conflicts with the new ignore-first/ranking contract" in str(exc)
    else:
        raise AssertionError("Expected gcs_far_spurious_neg + ignore-first contract to raise.")

    allowed = GCSLoss(
        _loss_args(
            gcs_spurious_neg=0.1,
            gcs_far_spurious_neg=0.1,
            gcs_rank_topk_weight=0.02,
            gcs_allow_legacy_spurious_with_new_contract=True,
        )
    )
    assert allowed.spurious_neg_gain == 0.1
    assert allowed.far_spurious_neg_gain == 0.1
    assert bool(allowed.allow_legacy_spurious_with_new_contract)


def check_gt3_count_contract_guard() -> None:
    message = "GT4/GT5 count-contract losses require min_gt_lanes >= 4"
    for kwargs in (
        {"gcs_rank_topk_weight": 0.02, "gcs_rank_gt_min_lanes": 3},
        {"gcs_shortside_rawmatch_boost": 0.25, "gcs_shortside_min_gt_lanes": 3},
    ):
        try:
            GCSLoss(_loss_args(**kwargs))
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"Expected GT3 count-contract guard for {kwargs}.")

    allowed = GCSLoss(
        _loss_args(
            gcs_rank_topk_weight=0.02,
            gcs_rank_gt_min_lanes=3,
            gcs_allow_gt3_count_contract_ablation=True,
        )
    )
    assert allowed.rank_gt_min_lanes == 3
    assert bool(allowed.allow_gt3_count_contract_ablation)


def check_default_off_objective_unchanged() -> None:
    gt, valid = _make_gt()
    preds = _make_preds(gt, valid)
    batch = {"lanes": [gt], "lane_valid": [valid], "num_lanes": torch.tensor([4.0])}
    crit = GCSLoss(_loss_args())

    indices = crit.matcher(preds["pred_points"], preds["pred_logits"], [gt], [valid])
    gt_lanes = crit.target_lane_count(preds["pred_logits"], batch, [valid])
    masks = crit.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert not bool(masks["exist_ignore"].any())
    assert masks["point_valid_ignore"] is not None
    assert not bool(masks["point_valid_ignore"].any())
    query_w, valid_w, *_ = crit.shortside_rawmatch_boost_masks(
        preds["pred_points"],
        preds["pred_logits"],
        preds["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        gt_lanes,
        contract_masks=masks,
    )
    assert torch.allclose(query_w, torch.ones_like(query_w))
    assert valid_w is not None and torch.allclose(valid_w, torch.ones_like(valid_w))

    total, _ = crit(preds, batch)
    point_valid_loss, *_ = crit.point_valid_loss(
        preds["pred_valid_logits"],
        preds["pred_points"],
        [valid],
        indices,
        gt_lanes=gt_lanes,
        return_details=True,
    )
    count_loss, count_under5_loss, count_boundary_loss, *_ = crit.count_losses(
        preds["pred_logits"], batch, [valid], target=gt_lanes
    )
    manual = (
        crit.exist_gain
        * crit.exist_loss(preds["pred_logits"], preds["pred_points"], preds["pred_valid_logits"], [gt], [valid], indices)
        + crit.point_gain * crit.point_loss(preds["pred_points"], [gt], [valid], indices)
        + crit.point_valid_gain * point_valid_loss
        + crit.smooth_gain * crit.smooth_loss(preds["pred_points"], [valid], indices)
        + crit.curve_gain * crit.curve_loss(preds["pred_points"], [gt], [valid], indices)
        + crit.count_gain * count_loss
        + crit.count_under5_gain * count_under5_loss
    )
    if crit.count_boundary_gain != 0.0:
        manual = manual + crit.count_boundary_gain * count_boundary_loss
    assert torch.allclose(total, manual, atol=1e-6), (float(total), float(manual))


def check_farspur_and_rank_ignore_knobs_are_separate() -> None:
    gt, valid = _make_gt()
    indices = [(torch.arange(4, dtype=torch.long), torch.arange(4, dtype=torch.long))]
    gt_lanes = torch.tensor([4.0])

    preds = _make_preds(gt, valid)
    preds["pred_points"][0, 6] = gt[2]
    preds["pred_points"][0, 6, :, 0] = gt[2, :, 0] + 55.0 / 960.0
    crit_far_near_classify_only = GCSLoss(
        _loss_args(gcs_farspur_ignore_first=True, gcs_farspur_near_dist_px=80.0, gcs_rank_near_gt_ignore_px=10.0)
    )
    masks_far_near_classify_only = crit_far_near_classify_only.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_far_near_classify_only["farspur_near_gt_corridor"][0, 6])
    assert not bool(masks_far_near_classify_only["near_gt_corridor"][0, 6])
    assert not bool(masks_far_near_classify_only["exist_ignore"][0, 6])
    assert float(masks_far_near_classify_only["base_exist_ignore_farspur_near_count"]) == 0.0

    crit_far_weight0_base_flag = GCSLoss(
        _loss_args(
            gcs_farspur_ignore_first=True,
            gcs_farspur_weight=0.0,
            gcs_farspur_near_dist_px=80.0,
            gcs_rank_near_gt_ignore_px=10.0,
            gcs_base_ignore_farspur_near=True,
        )
    )
    masks_far_weight0_base_flag = crit_far_weight0_base_flag.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_far_weight0_base_flag["farspur_near_gt_corridor"][0, 6])
    assert not bool(masks_far_weight0_base_flag["exist_ignore"][0, 6])
    assert float(masks_far_weight0_base_flag["base_exist_ignore_farspur_near_count"]) == 0.0
    assert float(masks_far_weight0_base_flag["farspur_near_still_base_negative_count"]) >= 1.0

    crit_far_aux = GCSLoss(
        _loss_args(
            gcs_farspur_ignore_first=True,
            gcs_farspur_weight=0.005,
            gcs_farspur_near_dist_px=80.0,
            gcs_rank_near_gt_ignore_px=10.0,
        )
    )
    masks_far_aux = crit_far_aux.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_far_aux["farspur_near_gt_corridor"][0, 6])
    assert not bool(masks_far_aux["exist_ignore"][0, 6])
    assert float(masks_far_aux["farspur_near_still_base_negative_count"]) >= 1.0
    _, far_aux_items = crit_far_aux(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(far_aux_items[GCSLoss.loss_names.index("farspur_aux_only_mode")]) == 1.0
    assert float(far_aux_items[GCSLoss.loss_names.index("farspur_full_ignore_first_mode")]) == 0.0
    assert float(far_aux_items[GCSLoss.loss_names.index("farspur_near_still_base_negative_count")]) >= 1.0

    crit_far_near = GCSLoss(
        _loss_args(
            gcs_farspur_ignore_first=True,
            gcs_farspur_weight=0.005,
            gcs_farspur_near_dist_px=80.0,
            gcs_rank_near_gt_ignore_px=10.0,
            gcs_base_ignore_farspur_near=True,
        )
    )
    masks_far_near = crit_far_near.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert bool(masks_far_near["exist_ignore"][0, 6])
    assert masks_far_near["point_valid_ignore"] is not None
    assert bool(masks_far_near["point_valid_ignore"][0, 6].all())
    assert float(masks_far_near["base_exist_ignore_farspur_near_count"]) >= 1.0
    assert float(masks_far_near["farspur_near_still_base_negative_count"]) == 0.0
    _, far_full_items = crit_far_near(preds, {"lanes": [gt], "lane_valid": [valid], "num_lanes": gt_lanes})
    assert float(far_full_items[GCSLoss.loss_names.index("farspur_aux_only_mode")]) == 0.0
    assert float(far_full_items[GCSLoss.loss_names.index("farspur_full_ignore_first_mode")]) == 1.0
    assert float(far_full_items[GCSLoss.loss_names.index("base_ignore_farspur_near_count")]) >= 1.0
    assert float(far_full_items[GCSLoss.loss_names.index("farspur_near_still_base_negative_count")]) == 0.0

    crit_rank_near = GCSLoss(_loss_args(gcs_farspur_near_dist_px=10.0, gcs_rank_near_gt_ignore_px=80.0))
    masks_rank_near = crit_rank_near.build_count_contract_masks(
        preds["pred_points"], preds["pred_logits"], preds["pred_valid_logits"], [gt], [valid], indices, gt_lanes
    )
    assert not bool(masks_rank_near["farspur_near_gt_corridor"][0, 6])
    assert bool(masks_rank_near["near_gt_corridor"][0, 6])

    preds_side = _make_preds(gt, valid)
    preds_side["pred_points"][0, 6] = gt[3]
    preds_side["pred_points"][0, 6, :, 0] = gt[3, :, 0] + 50.0 / 960.0
    crit_far_side = GCSLoss(
        _loss_args(
            gcs_farspur_ignore_first=True,
            gcs_farspur_weight=0.005,
            gcs_farspur_near_dist_px=10.0,
            gcs_rank_near_gt_ignore_px=10.0,
            gcs_farspur_side_ignore_dist_px=80.0,
            gcs_rank_side_ignore_px=10.0,
            gcs_base_ignore_farspur_near=True,
        )
    )
    masks_far_side = crit_far_side.build_count_contract_masks(
        preds_side["pred_points"],
        preds_side["pred_logits"],
        preds_side["pred_valid_logits"],
        [gt],
        [valid],
        indices,
        gt_lanes,
    )
    assert bool(masks_far_side["farspur_ambiguous_side_region"][0, 6])
    assert not bool(masks_far_side["ambiguous_side_region"][0, 6])
    assert bool(masks_far_side["exist_ignore"][0, 6])
    assert masks_far_side["point_valid_ignore"] is not None
    assert bool(masks_far_side["point_valid_ignore"][0, 6].all())
    assert float(masks_far_side["base_exist_ignore_farspur_side_count"]) >= 1.0


def check_diagnostic_rawmatch_uses_anchor_order() -> None:
    gt, valid = _make_gt()
    preds = _make_preds(gt, valid)
    gt[0, :, 0] = torch.linspace(0.10, 0.30, gt.shape[1])
    preds["pred_points"][0, 0] = gt[0]
    preds["pred_points"][0, 0, :, 1] = torch.flip(gt[0, :, 1], dims=[0])
    preds["pred_valid_logits"][0, 0, valid[0] > 0.5] = 6.0

    anchor_points, _ = anchor_query_arrays(preds["pred_points"][0], preds["pred_valid_logits"][0])
    sorted_points, _ = sorted_query_arrays(preds["pred_points"][0], preds["pred_valid_logits"][0])
    _, anchor_dist = best_raw_query_for_gt(anchor_points, gt.numpy(), valid.numpy(), 0, width=960)
    _, sorted_dist = best_raw_query_for_gt(sorted_points, gt.numpy(), valid.numpy(), 0, width=960)
    assert anchor_dist < 1e-5, anchor_dist
    assert sorted_dist > 30.0, sorted_dist


def main() -> None:
    check_loss_names_and_gains()
    check_cli_and_config_defaults()
    check_k56_lower_half_uses_bottom_anchors()
    check_shortside_rawmatch_boost()
    check_shortside_hungarian_rawmatch_beats_nearest_duplicate()
    check_shortside_visible_boundaries()
    check_shortside_target_floor_and_ultra_tiers()
    check_farspur_ignore_first_and_ranking()
    check_rank_pair_reduction()
    check_near_only_is_not_rank_negative()
    check_base_exist_ignore_contract()
    check_side_ambiguous_duplicate_split()
    check_clear_far_boundary_is_strictly_greater()
    check_legacy_spurious_conflict()
    check_gt3_count_contract_guard()
    check_default_off_objective_unchanged()
    check_farspur_and_rank_ignore_knobs_are_separate()
    check_diagnostic_rawmatch_uses_anchor_order()
    print("GCS count-contract loss checks passed.")


if __name__ == "__main__":
    main()

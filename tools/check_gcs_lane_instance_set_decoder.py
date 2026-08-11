"""Synthetic contract checks for the default-off lane-instance-set decoder."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import GCSLaneHead
from ultralytics.nn.tasks import _assert_current_gcs_checkpoint_model
from ultralytics.models.gcs.decode_summary import (
    LANE_INSTANCE_SET_DECODE_SCHEMA,
    build_official_best_decode_cfg,
    lane_instance_set_decode_cfg,
    lane_instance_set_sweep_summary,
    load_decode_yaml,
    validate_decode_yaml_for_model,
)
from ultralytics.models.yolo.gcs_lane.train import GCSLaneTrainer
from ultralytics.models.yolo.gcs_lane.val import GCSLaneValidator
from ultralytics.utils.gcs_loss import GCSLoss
from ultralytics.cfg import get_cfg
from tools.eval_tusimple_official import resolve_pred_json_decode_contract
from tools.sweep_tusimple_official import (
    _combo_key as direct_sweep_combo_key,
    _row_sort_key as direct_sweep_row_sort_key,
    apply_lane_instance_decode_yaml,
    build_combos,
)
from tools.train_gcs import parse_args as parse_train_args
from ultralytics.utils.gcs_lane_instance_set import (
    LANE_INSTANCE_PREDICTION_KEYS,
    decode_lane_instance_set_predictions,
    interval_valid_logits_from_start_end,
    resolve_lane_instance_decode_mode,
    select_lane_instance_survivors,
)
from tools.sweep_tusimple_official_cached import (
    CACHE_FILE,
    CACHE_SCHEMA,
    PREDICTION_KEYS,
    _batched_pred_dict,
    _tensor_for_cache,
    load_prediction_cache,
    validate_prediction_cache_entries,
)

EXPERIMENT_CFG = ROOT / "ultralytics/cfg/models/gcs/gcs-yolo-lane-s-q12-k56-lane-instance-set-decoder.yaml"


def _assert(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _fixed_y_anchors(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.arange(710.0, 150.0, -10.0, device=device, dtype=dtype) / 720.0


def _assert_contiguous(mask: torch.Tensor, context: str) -> None:
    flat = mask.detach().bool().reshape(-1, mask.shape[-1]).cpu()
    for row_i, row in enumerate(flat):
        indices = torch.nonzero(row, as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        expected = torch.arange(int(indices[0]), int(indices[-1]) + 1)
        _assert(torch.equal(indices, expected), f"{context}: row {row_i} has an internal visibility hole.")


def _assert_all_finite(tensor: torch.Tensor, context: str) -> None:
    _assert(torch.isfinite(tensor.detach()).all().item(), f"{context} contains NaN or Inf.")


def _make_head(enabled: bool) -> GCSLaneHead:
    head = GCSLaneHead(
        c1=32,
        num_queries=12,
        num_points=56,
        num_decoder_layers=1,
        nhead=4,
        aux=False,
        point_mode="fixed_y",
        lane_instance_set_decoder_head=enabled,
        lane_instance_identity_dim=8,
    )
    head.min_spatial_tokens = 0
    return head


def _make_features(batch: int = 2, requires_grad: bool = False, device=None) -> list[torch.Tensor]:
    return [
        torch.randn(batch, 32, 32, 64, requires_grad=requires_grad, device=device),
        torch.randn(batch, 32, 16, 32, requires_grad=requires_grad, device=device),
        torch.randn(batch, 32, 8, 16, requires_grad=requires_grad, device=device),
        torch.randn(batch, 32, 4, 8, requires_grad=requires_grad, device=device),
    ]


def _check_head_shapes_and_gradients() -> None:
    torch.manual_seed(7)
    head = _make_head(enabled=True)
    head.train()
    xs = _make_features(requires_grad=True)
    out = head(xs)

    expected_shapes = {
        "pred_points": (2, 12, 56, 2),
        "pred_logits": (2, 12),
        "pred_valid_logits": (2, 12, 56),
        "pred_lane_instance_points": (2, 12, 56, 2),
        "pred_lane_instance_start_logits": (2, 12, 56),
        "pred_lane_instance_end_logits": (2, 12, 56),
        "pred_lane_instance_valid_logits": (2, 12, 56),
        "pred_lane_instance_identity": (2, 12, 8),
        "pred_lane_instance_pair_duplicate_logits": (2, 12, 12),
        "pred_lane_instance_novelty_logits": (2, 12),
        "pred_lane_instance_left_logits": (2, 12, 12),
        "pred_lane_instance_right_logits": (2, 12, 12),
        "pred_lane_instance_geometry_quality_logits": (2, 12),
        "pred_lane_instance_survival_logits": (2, 12),
        "pred_lane_instance_empty_logit": (2,),
    }
    for key, shape in expected_shapes.items():
        _assert(key in out, f"missing output key {key}.")
        _assert(tuple(out[key].shape) == shape, f"{key} shape mismatch: got {tuple(out[key].shape)}, expected {shape}.")
        _assert_all_finite(out[key], key)
    _assert("pred_count_logits" not in out, "lane-instance-set experiment must not emit pred_count_logits.")
    _assert(torch.equal(out["pred_points"], out["pred_lane_instance_points"]), "main pred_points must be lane-instance candidates.")
    _assert(torch.equal(out["pred_logits"], out["pred_lane_instance_survival_logits"]), "main pred_logits must be survival logits.")
    _assert(
        torch.equal(out["pred_valid_logits"], out["pred_lane_instance_valid_logits"]),
        "main pred_valid_logits must be derived interval logits.",
    )

    expected_y = _fixed_y_anchors(dtype=out["pred_points"].dtype).view(1, 1, 56)
    max_y_err = (out["pred_points"][..., 1].detach().cpu() - expected_y).abs().max().item()
    _assert(max_y_err <= 1e-6, f"fixed-y anchors mismatch; max y error={max_y_err}.")
    _assert_contiguous(out["pred_valid_logits"] >= 0.0, "head-derived pred_valid_logits")

    loss = (
        out["pred_points"].sum()
        + out["pred_logits"].sum()
        + out["pred_valid_logits"].sum()
        + out["pred_lane_instance_identity"].sum()
        + out["pred_lane_instance_pair_duplicate_logits"].sum()
        + out["pred_lane_instance_novelty_logits"].sum()
        + out["pred_lane_instance_left_logits"].sum()
        + out["pred_lane_instance_right_logits"].sum()
        + out["pred_lane_instance_geometry_quality_logits"].sum()
        + out["pred_lane_instance_empty_logit"].sum()
    )
    loss.backward()
    _assert(xs[0].grad is not None and torch.isfinite(xs[0].grad).all().item(), "P2 gradient is missing or non-finite.")
    _assert(xs[1].grad is not None and torch.isfinite(xs[1].grad).all().item(), "P3 gradient is missing or non-finite.")
    lane_instance_params = [(name, param) for name, param in head.named_parameters() if "lane_instance" in name]
    _assert(lane_instance_params, "lane-instance-set parameters were not registered.")
    for name, param in lane_instance_params:
        _assert(param.grad is not None, f"{name} did not receive a gradient.")
        _assert(torch.isfinite(param.grad).all().item(), f"{name} gradient contains NaN or Inf.")


def _check_interval_reverse_normalization() -> None:
    start_logits = torch.zeros(1, 1, 56)
    end_logits = torch.zeros(1, 1, 56)
    start_logits[..., 40] = 30.0
    end_logits[..., 10] = 30.0
    valid_logits, start_index, end_index = interval_valid_logits_from_start_end(start_logits, end_logits)
    _assert(float(start_index.item()) < 10.01, f"reversed start/end did not normalize start: {start_index.item()}.")
    _assert(float(end_index.item()) > 39.99, f"reversed start/end did not normalize end: {end_index.item()}.")
    _assert(float(start_index.item()) <= float(end_index.item()), "ordered start/end violated start <= end.")
    mask = valid_logits >= 0.0
    _assert_contiguous(mask, "reverse-normalized interval")
    visible = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    _assert(int(visible[0]) == 10 and int(visible[-1]) == 40, "reverse-normalized interval has wrong endpoints.")


def _check_lane_instance_independent_from_base_query_outputs() -> None:
    torch.manual_seed(19)
    head = _make_head(enabled=True).eval()
    for name in ("query_embed", "decoder", "point_mlp", "point_valid_mlp", "point_valid_refine_mlp", "exist_mlp"):
        _assert(not hasattr(head, name), f"lane-instance head still registers legacy query module {name}.")
    legacy_names = {
        name
        for name, _ in head.named_parameters()
        if name.startswith(("query_embed", "decoder", "point_mlp", "point_valid", "exist_mlp"))
    }
    _assert(not legacy_names, f"lane-instance head still registers legacy query parameters: {sorted(legacy_names)}")
    features = _make_features(batch=1, requires_grad=False)
    with torch.no_grad():
        before = head(features)
        after = head(features)
    for key in LANE_INSTANCE_PREDICTION_KEYS:
        _assert(torch.equal(before[key], after[key]), f"{key} still depends on legacy base-query outputs.")
    for key in ("pred_points", "pred_logits", "pred_valid_logits"):
        _assert(torch.equal(before[key], after[key]), f"lane-instance main output {key} depends on the legacy query head.")


def _check_identity_coupling_and_real_loss() -> None:
    torch.manual_seed(31)
    head = _make_head(enabled=True).train()
    features = _make_features(batch=1, requires_grad=False)
    for key in ("pred_lane_instance_survival_logits", "pred_lane_instance_novelty_logits"):
        head.zero_grad(set_to_none=True)
        head(features)[key].sum().backward()
        grads = [p.grad for n, p in head.named_parameters() if "lane_instance_identity_mlp" in n]
        _assert(grads and all(g is not None and torch.isfinite(g).all() for g in grads), f"{key} identity gradients missing/non-finite.")
        _assert(sum(float(g.abs().sum()) for g in grads) > 0.0, f"{key} is not coupled to identity MLP.")
    head.zero_grad(set_to_none=True)
    topo = head(features)
    (topo["pred_lane_instance_left_logits"].sum() + topo["pred_lane_instance_right_logits"].sum()).backward()
    grads = [p.grad for n, p in head.named_parameters() if "lane_instance_identity_mlp" in n]
    _assert(grads and all(g is not None and torch.isfinite(g).all() for g in grads), "topology identity gradients missing/non-finite.")
    _assert(sum(float(g.abs().sum()) for g in grads) > 0.0, "topology is not coupled to identity MLP.")

    head.zero_grad(set_to_none=True)
    preds = head(features, orig_size=(544, 960))
    y = _fixed_y_anchors().view(1, 56)
    x = torch.stack((torch.full((56,), 0.3), torch.full((56,), 0.7)))
    gt_points = [torch.stack((x, y.expand(2, -1)), dim=-1)]
    gt_valid = [torch.ones(2, 56)]
    criterion = GCSLoss({
        "gcs_imgsz": [544, 960], "gcs_lane_instance_set": 1.0,
        "gcs_mask": 0.0, "gcs_edge": 0.0,
    })
    total, items = criterion(preds, {"img": torch.zeros(1, 3, 544, 960), "lanes": gt_points, "lane_valid": gt_valid})
    _assert(torch.isfinite(total).item() and torch.isfinite(items).all().item(), "real GCSLoss lane-instance path is non-finite.")
    _assert(int(items.numel()) == len(GCSLoss.loss_names) + len(GCSLoss.lane_instance_loss_names), "lane-instance log item count mismatch.")
    total.backward()
    grads = [p.grad for n, p in head.named_parameters() if "lane_instance" in n and p.grad is not None]
    _assert(grads and all(torch.isfinite(g).all() for g in grads), "real lane-instance loss gradients missing/non-finite.")
    _assert(sum(float(g.abs().sum()) for g in grads) > 0.0, "real lane-instance loss produced only zero gradients.")
    identity_grads = [p.grad for n, p in head.named_parameters() if "lane_instance_identity_mlp" in n]
    _assert(all(g is not None and torch.isfinite(g).all() for g in identity_grads), "real loss did not train identity MLP.")


def _check_loss_name_contracts() -> None:
    args = SimpleNamespace(gcs_mode="query", gcs_lane_instance_set=1.0)
    trainer = object.__new__(GCSLaneTrainer)
    trainer.args = args
    trainer._gcs_mode = lambda: "query"
    trainer._set_loss_names_for_mode()
    validator = object.__new__(GCSLaneValidator)
    validator.args = args
    validator._gcs_mode = lambda: "query"
    expected = GCSLoss.active_loss_names(args)
    _assert(tuple(trainer.loss_names) == expected, "trainer lane-instance loss names diverged from GCSLoss.")
    _assert(tuple(validator._loss_names()) == expected, "validator lane-instance loss names diverged from GCSLoss.")
    _assert(int(validator._loss_gains(torch.device("cpu")).numel()) == len(expected), "validator loss gains length diverged.")

    head = _make_head(enabled=True).train()
    preds = head(_make_features(batch=1), orig_size=(544, 960))
    empty_long = torch.empty(0, dtype=torch.long)
    criterion = GCSLoss({
        "gcs_imgsz": [544, 960], "gcs_lane_instance_set": 1.0,
        "gcs_lane_instance_visibility_weight": 0.0, "gcs_lane_instance_endpoint_weight": 0.0,
        "gcs_lane_instance_order_weight": 0.0, "gcs_lane_instance_contiguity_weight": 0.0,
        "gcs_lane_instance_positive_span_weight": 0.0, "gcs_lane_instance_empty_weight": 1.0,
        "gcs_lane_instance_geometry_quality_weight": 0.0, "gcs_lane_instance_survival_weight": 0.0,
        "gcs_lane_instance_duplicate_weight": 0.0, "gcs_lane_instance_novelty_weight": 0.0,
        "gcs_lane_instance_topology_weight": 0.0, "gcs_lane_instance_identity_weight": 0.0,
        "gcs_lane_instance_set_noop_weight": 1.0,
    })
    values = criterion.lane_instance_set_loss(
        preds, [torch.empty(0, 56, 2)], [torch.empty(0, 56)], [(empty_long, empty_long)]
    )
    _assert(torch.allclose(values[0], values[7] + values[14]), "empty loss was duplicated inside set_noop.")


def _check_decode_schema_and_cli_contracts() -> None:
    row = {
        "decode_mode": "lane_instance_set", "conf": 0.2, "point_valid_thr": 0.45,
        "max_det": 5, "min_points": 5, "duplicate_thr": 0.6,
        "allow_empty": False, "empty_thr": 0.8, "min_survivors": 2,
    }
    cfg = lane_instance_set_decode_cfg(row)
    _assert(cfg["schema"] == LANE_INSTANCE_SET_DECODE_SCHEMA, "wrong lane-instance decode schema.")
    _assert(build_official_best_decode_cfg(row, "lane_instance_set") == cfg, "official-best schema builder diverged.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decode.yaml"
        path.write_text(yaml.safe_dump({"decode": cfg}, sort_keys=False), encoding="utf-8")
        _, loaded = load_decode_yaml(path)
        pred_args = SimpleNamespace(pred_json="predictions.json", decode_yaml=str(path), decode_mode="auto")
        pred_mode, pred_cfg = resolve_pred_json_decode_contract(pred_args)
        _assert(pred_mode == "lane_instance_set" and pred_cfg == cfg, "pred-json did not preserve lane-instance contract.")
    validate_decode_yaml_for_model(loaded, "lane_instance_set")
    _assert(loaded == cfg, "lane-instance decode YAML round-trip changed fields.")
    applied = SimpleNamespace()
    apply_lane_instance_decode_yaml(applied, loaded)
    _assert(applied.max_dets == [5] and applied.lane_instance_duplicate_thrs == [0.6], "decode YAML apply failed.")
    sweep_args = SimpleNamespace(
        decode_mode="lane_instance_set", confs=[0.2], point_valid_thrs=[0.45], max_dets=[5], min_points=[5],
        lane_instance_duplicate_thrs=[0.6], lane_instance_allow_empty=False,
        lane_instance_empty_thr=0.8, lane_instance_min_survivors=2, count_aware_topk=False,
    )
    combos = build_combos(sweep_args)
    _assert(len(combos) == 1 and combos[0]["duplicate_thr"] == 0.6, "direct sweep schema reproduction failed.")
    direct_sweep_row_sort_key(combos[0])
    direct_sweep_combo_key(combos[0])
    summary = lane_instance_set_sweep_summary(sweep_args)
    cached_args = SimpleNamespace(**vars(sweep_args), lane_instance_max_dets=[5])
    _assert(summary == lane_instance_set_sweep_summary(cached_args), "direct/cached sweep summaries diverged.")

    parsed = parse_train_args([
        "--gcs-lane-instance-set", "1.0",
        "--gcs-lane-instance-duplicate-weight", "0.75",
        "--gcs-lane-instance-decode-duplicate-thr", "0.6",
    ])
    _assert(parsed.gcs_lane_instance_set == 1.0, "train_gcs silently reset lane-instance gain to zero.")
    _assert(parsed.gcs_lane_instance_duplicate_weight == 0.75, "train_gcs loss parameter parse failed.")
    _assert(parsed.gcs_lane_instance_decode_duplicate_thr == 0.6, "train_gcs decode parameter parse failed.")
    cfg_args = get_cfg(overrides={
        "gcs_lane_instance_set": parsed.gcs_lane_instance_set,
        "gcs_lane_instance_duplicate_weight": parsed.gcs_lane_instance_duplicate_weight,
        "gcs_lane_instance_decode_duplicate_thr": parsed.gcs_lane_instance_decode_duplicate_thr,
    })
    _assert(float(cfg_args.gcs_lane_instance_set) == 1.0, "cfg validation silently reset lane-instance gain.")

    trainer = object.__new__(GCSLaneTrainer)
    trainer.model = torch.nn.Sequential(_make_head(enabled=True))
    trainer.args = SimpleNamespace(
        gcs_mode="query", gcs_official_confs=[0.2], gcs_official_point_valid_thrs=[0.45],
        gcs_official_min_points=[5], gcs_official_count_aware_topk=True,
        gcs_lane_instance_decode_duplicate_thr=0.6, gcs_lane_instance_decode_allow_empty=False,
        gcs_lane_instance_decode_empty_thr=0.8, gcs_lane_instance_decode_min_survivors=2,
        device="cpu",
    )
    trainer.device = torch.device("cpu")
    trainer.last = Path("synthetic.pt")
    trainer._resolve_gcs_imgsz = lambda: (544, 960)
    official = trainer._official_sweep_args(Path("synthetic_sweep"))
    _assert(official.decode_mode == "lane_instance_set", "official_best did not resolve model capability.")
    _assert(not official.count_aware_topk and official.max_dets == [5], "lane-instance official_best entered query count path.")


def _check_cuda_amp_smoke() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda:0")
    head = _make_head(enabled=True).to(device).train()
    criterion = GCSLoss({"gcs_imgsz": [544, 960], "gcs_lane_instance_set": 1.0, "gcs_mask": 0.0, "gcs_edge": 0.0}).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    y = _fixed_y_anchors(device=device).view(1, 56)
    x = torch.stack((torch.full((56,), 0.3, device=device), torch.full((56,), 0.7, device=device)))
    batch = {"img": torch.zeros(1, 3, 544, 960, device=device), "lanes": [torch.stack((x, y.expand(2, -1)), dim=-1)], "lane_valid": [torch.ones(2, 56, device=device)]}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            total, items = criterion(head(_make_features(1, device=device), orig_size=(544, 960)), batch)
        _assert(torch.isfinite(total).item() and torch.isfinite(items).all().item(), "AMP lane-instance loss is non-finite.")
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f"CUDA AMP lane-instance smoke passed: device={torch.cuda.get_device_name(0)}, steps=3")


def _synthetic_preds(
    x_positions: list[float],
    survival_logits: list[float] | None = None,
    duplicate_pairs: list[tuple[int, int]] | None = None,
    empty_logit: float = -8.0,
) -> dict[str, torch.Tensor]:
    q = len(x_positions)
    k = 56
    y = _fixed_y_anchors().view(1, 1, k).expand(1, q, -1)
    x = torch.tensor(x_positions, dtype=torch.float32).view(1, q, 1).expand(-1, -1, k)
    points = torch.stack((x, y), dim=-1)
    valid_logits = torch.full((1, q, k), 6.0)
    survival = torch.tensor(survival_logits or [6.0] * q, dtype=torch.float32).view(1, q)
    quality = torch.full((1, q), 6.0)
    novelty = torch.full((1, q), 6.0)
    duplicate = torch.full((1, q, q), -8.0)
    duplicate[:, torch.arange(q), torch.arange(q)] = 20.0
    for i, j in duplicate_pairs or []:
        duplicate[0, i, j] = 8.0
        duplicate[0, j, i] = 8.0
    mean_x = torch.tensor(x_positions, dtype=torch.float32)
    signed = mean_x.view(q, 1) - mean_x.view(1, q)
    left = (-16.0 * signed).view(1, q, q)
    right = (16.0 * signed).view(1, q, q)
    left[:, torch.arange(q), torch.arange(q)] = -20.0
    right[:, torch.arange(q), torch.arange(q)] = -20.0
    start_logits = torch.full((1, q, k), -8.0)
    end_logits = torch.full((1, q, k), -8.0)
    start_logits[..., 0] = 8.0
    end_logits[..., -1] = 8.0
    return {
        "pred_lane_instance_points": points,
        "pred_lane_instance_start_logits": start_logits,
        "pred_lane_instance_end_logits": end_logits,
        "pred_lane_instance_valid_logits": valid_logits,
        "pred_lane_instance_interval_start": torch.zeros(1, q),
        "pred_lane_instance_interval_end": torch.full((1, q), float(k - 1)),
        "pred_lane_instance_identity": torch.nn.functional.normalize(torch.randn(1, q, 8), dim=-1),
        "pred_lane_instance_survival_logits": survival,
        "pred_lane_instance_geometry_quality_logits": quality,
        "pred_lane_instance_pair_duplicate_logits": duplicate,
        "pred_lane_instance_novelty_logits": novelty,
        "pred_lane_instance_left_logits": left,
        "pred_lane_instance_right_logits": right,
        "pred_lane_instance_empty_logit": torch.tensor([empty_logit], dtype=torch.float32),
    }


def _lane_signature(lanes: list[dict]) -> list[tuple]:
    return [(lane["query"], lane["decoded_count"], tuple(lane["point_valid"].tolist())) for lane in lanes]


def _check_cache_round_trip() -> None:
    _assert(set(LANE_INSTANCE_PREDICTION_KEYS).issubset(PREDICTION_KEYS), "cache PREDICTION_KEYS is incomplete.")
    preds = _synthetic_preds([0.25, 0.55, 0.8], duplicate_pairs=[(0, 1)])
    direct = decode_lane_instance_set_predictions(preds, score_thr=0.25, duplicate_thr=0.65)
    cached = {key: _tensor_for_cache(preds[key]) for key in LANE_INSTANCE_PREDICTION_KEYS}
    _assert(set(cached) == set(LANE_INSTANCE_PREDICTION_KEYS), "synthetic cache omitted required tensors.")
    entry = {"index": 0, "raw_file": "clips/1/1.jpg", "h_samples": list(range(160, 720, 10)), "image_shape": [720, 1280], "predictions": cached}
    record = {"raw_file": entry["raw_file"], "h_samples": entry["h_samples"], "lanes": []}
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        torch.save([entry], cache_dir / CACHE_FILE)
        (cache_dir / "manifest.json").write_text(json.dumps({"schema": CACHE_SCHEMA, "prediction_keys": sorted(cached)}), encoding="utf-8")
        _, loaded = load_prediction_cache(cache_dir)
        validate_prediction_cache_entries(loaded, [record], decode_mode="lane_instance_set")
        replay = decode_lane_instance_set_predictions(_batched_pred_dict(loaded[0]["predictions"]), batch_index=0, score_thr=0.25, duplicate_thr=0.65)
        incomplete = [{**loaded[0], "predictions": dict(loaded[0]["predictions"])}]
        incomplete[0]["predictions"].pop(LANE_INSTANCE_PREDICTION_KEYS[0])
        validate_prediction_cache_entries(incomplete, [record], decode_mode="query")
        try:
            validate_prediction_cache_entries(incomplete, [record], decode_mode="lane_instance_set")
        except RuntimeError as error:
            _assert("missing tensors" in str(error), "incomplete lane-instance cache error is not actionable.")
        else:
            raise AssertionError("incomplete lane-instance cache was accepted.")
    _assert(_lane_signature(replay) == _lane_signature(direct), "cache round-trip decode differs from direct decode.")


def _check_yaml_model_and_unified_decode_smoke() -> None:
    model = YOLO(str(EXPERIMENT_CFG)).model.eval()
    _assert_current_gcs_checkpoint_model(model, "synthetic_lane_instance.pt")
    _assert(resolve_lane_instance_decode_mode("auto", model) == "lane_instance_set", "experiment YAML did not enable lane-instance decode.")
    with torch.no_grad():
        preds = model(torch.randn(1, 3, 544, 960))
    lanes, diagnostics = decode_lane_instance_set_predictions(
        preds, batch_index=0, image_shape=(544, 960), max_det=5, allow_empty=False, return_diagnostics=True
    )
    _assert(isinstance(lanes, list) and diagnostics["decoded_count"] == len(lanes), "unified random-model decode smoke failed.")
    _assert("pred_count_logits" not in preds, "experiment YAML unexpectedly emitted pred_count_logits.")


def _check_decoder_contracts() -> None:
    empty_preds = _synthetic_preds([0.25, 0.5, 0.75], survival_logits=[-8.0, -8.0, -8.0], empty_logit=8.0)
    empty_lanes = decode_lane_instance_set_predictions(empty_preds, score_thr=0.25, allow_empty=True)
    _assert(empty_lanes == [], "empty/no-lane case should decode to zero survivors.")

    dup_preds = _synthetic_preds([0.30, 0.305, 0.72], duplicate_pairs=[(0, 1)])
    dup_lanes = decode_lane_instance_set_predictions(dup_preds, score_thr=0.25, duplicate_thr=0.65, min_survivors=2)
    dup_queries = {lane["query"] for lane in dup_lanes}
    _assert(len(dup_lanes) == 2, f"duplicate suppression should keep two survivors, got {len(dup_lanes)}.")
    _assert(2 in dup_queries and len(dup_queries & {0, 1}) == 1, f"wrong duplicate survivor set: {dup_queries}.")
    _assert(all(lane["decoded_count"] == len(dup_lanes) for lane in dup_lanes), "decoded_count must equal len(survivors).")
    _assert(all(lane["count_source"] == "survivors" for lane in dup_lanes), "decoder count source must be survivors.")

    adjacent_preds = _synthetic_preds([0.30, 0.37])
    adjacent_lanes = decode_lane_instance_set_predictions(
        adjacent_preds,
        score_thr=0.25,
        duplicate_thr=0.65,
        min_survivors=2,
    )
    _assert(len(adjacent_lanes) == 2, "adjacent real lanes must not be merged when duplicate logits are low.")
    _assert({lane["query"] for lane in adjacent_lanes} == {0, 1}, "adjacent lane survivor identity changed.")

    selection = select_lane_instance_survivors(dup_preds, score_thr=0.25, duplicate_thr=0.65, min_survivors=2)
    _assert(int(selection["decoded_count"]) == len(dup_lanes), "selection decoded_count must be survivor-derived.")
    _assert(
        selection["selection_strategy"] == "exact_global_survival_duplicate",
        "lane-instance selection did not use exact global set optimization.",
    )
    global_preds = _synthetic_preds(
        [0.20, 0.50, 0.80],
        survival_logits=[4.0, 3.0, 3.0],
        duplicate_pairs=[(0, 1), (0, 2)],
    )
    global_lanes = decode_lane_instance_set_predictions(global_preds, score_thr=0.25, duplicate_thr=0.65)
    _assert(
        [lane["query"] for lane in global_lanes] == [1, 2],
        "global selector failed to replace one high-score conflicting lane with the better two-lane set.",
    )
    signature = inspect.signature(decode_lane_instance_set_predictions)
    banned = {"gt", "evaluator", "oracle_count"}
    _assert(not (set(signature.parameters) & banned), "decoder signature contains a forbidden GT/evaluator/oracle parameter.")
    _assert("pred_count_logits" not in dup_preds, "synthetic decoder path must not use pred_count_logits.")
    low_preds = _synthetic_preds([0.2, 0.5, 0.8], survival_logits=[-8.0, -8.0, 8.0])
    low_lanes, low_diag = decode_lane_instance_set_predictions(low_preds, score_thr=0.25, min_survivors=2, return_diagnostics=True)
    _assert(len(low_lanes) == 1 and low_diag["under_min"], "low-score lanes must not be fabricated to satisfy min_survivors.")
    invalid_preds = _synthetic_preds([0.2, 0.5])
    invalid_preds["pred_lane_instance_valid_logits"][0, 0] = -8.0
    invalid_lanes = decode_lane_instance_set_predictions(invalid_preds, score_thr=0.25, min_points=6)
    _assert(len(invalid_lanes) == 1 and invalid_lanes[0]["decoded_count"] == 1, "decoded_count must follow post-min_points survivors.")
    empty_forbidden = decode_lane_instance_set_predictions(empty_preds, score_thr=0.25, allow_empty=False)
    _assert(empty_forbidden == [], "TuSimple allow_empty=False must not fabricate lanes when evidence is absent.")

    utility_preds = _synthetic_preds([0.25, 0.75], survival_logits=[8.0, 1.0])
    utility_preds["pred_lane_instance_geometry_quality_logits"] = torch.tensor([[-20.0, 20.0]])
    utility_preds["pred_lane_instance_novelty_logits"] = torch.tensor([[-20.0, 20.0]])
    utility_preds["pred_lane_instance_left_logits"].zero_()
    utility_preds["pred_lane_instance_right_logits"].zero_()
    utility_lanes = decode_lane_instance_set_predictions(utility_preds, score_thr=0.25)
    _assert(
        [lane["query"] for lane in utility_lanes] == [0, 1],
        "decode must rank only by unified survival utility, not auxiliary quality/novelty/topology outputs.",
    )
    minimal_utility_preds = {
        key: value
        for key, value in utility_preds.items()
        if key
        not in {
            "pred_lane_instance_geometry_quality_logits",
            "pred_lane_instance_novelty_logits",
            "pred_lane_instance_left_logits",
            "pred_lane_instance_right_logits",
        }
    }
    minimal_utility_lanes = decode_lane_instance_set_predictions(minimal_utility_preds, score_thr=0.25)
    _assert(
        _lane_signature(minimal_utility_lanes) == _lane_signature(utility_lanes),
        "auxiliary lane-instance heads became parallel decode decisions.",
    )


def _check_default_state_transfer_unchanged() -> None:
    torch.manual_seed(11)
    source = _make_head(enabled=False)
    torch.manual_seed(23)
    target = _make_head(enabled=False)
    target.load_state_dict(source.state_dict(), strict=True)
    source.eval()
    target.eval()
    xs = _make_features(batch=1, requires_grad=False)
    with torch.no_grad():
        out_source = source(xs)
        out_target = target(xs)
    for key in ("pred_points", "pred_logits", "pred_valid_logits"):
        _assert(torch.equal(out_source[key], out_target[key]), f"default branch state-transfer output changed for {key}.")
    _assert("pred_count_logits" not in out_source, "default query head must not emit pred_count_logits.")
    _assert(
        not any(key.startswith("pred_lane_instance_") for key in out_source),
        "default query head must not emit lane-instance-set tensors.",
    )
    _assert(
        not any("lane_instance" in name for name, _ in source.named_parameters()),
        "default-off lane-instance-set parameters should not be registered in the default head.",
    )


def main() -> None:
    _check_head_shapes_and_gradients()
    _check_interval_reverse_normalization()
    _check_lane_instance_independent_from_base_query_outputs()
    _check_identity_coupling_and_real_loss()
    _check_loss_name_contracts()
    _check_decode_schema_and_cli_contracts()
    _check_cuda_amp_smoke()
    _check_decoder_contracts()
    _check_cache_round_trip()
    _check_yaml_model_and_unified_decode_smoke()
    _check_default_state_transfer_unchanged()
    print("GCS lane-instance-set decoder synthetic checks passed.")


if __name__ == "__main__":
    main()

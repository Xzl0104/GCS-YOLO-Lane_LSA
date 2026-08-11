# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""GCS-YOLO-Lane neural network modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.gcs_lane_instance_set import interval_valid_logits_from_start_end

__all__ = (
    "CoordReweight",
    "LineStripAttention",
    "LSEM",
    "WeightedFusion",
    "ConvBNAct",
    "LaneBiFPN",
    "build_2d_sincos_position_embedding",
    "GCSLaneHead",
)


class CoordReweight(nn.Module):
    """Coordinate-aware reweighting used inside Line-Strip Attention."""

    def __init__(self, c, reduction=32):
        """Initialize coordinate-sensitive height and width reweighting."""
        super().__init__()
        mid = max(8, c // reduction)

        self.conv1 = nn.Conv2d(c, mid, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mid, c, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid, c, kernel_size=1, bias=True)

    def forward(self, x):
        """Apply separate height-aware and width-aware coordinate weights."""
        _, _, h, w = x.shape

        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        return x * a_h * a_w


class LineStripAttention(nn.Module):
    """Line-Strip Attention, the core directional context module of LSEM."""

    def __init__(self, c, k=9, reduction=32):
        """Initialize horizontal/vertical strip branches, direction gate, and coordinate reweighting."""
        super().__init__()
        if k % 2 == 0:
            raise ValueError(f"LineStripAttention requires an odd strip kernel size, got k={k}.")

        p = k // 2
        hidden = max(8, c // 4)

        self.strip_h = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, k), padding=(0, p), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.strip_v = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(k, 1), padding=(p, 0), groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        self.direction_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 2, hidden, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.coord_reweight = CoordReweight(c, reduction=reduction)

    def forward(self, x):
        """Fuse horizontal and vertical strip responses with learned direction weights."""
        h_feat = self.strip_h(x)
        v_feat = self.strip_v(x)

        gate = self.direction_gate(torch.cat([h_feat, v_feat], dim=1))
        gate = torch.softmax(gate, dim=1)

        out = gate[:, 0:1] * h_feat + gate[:, 1:2] * v_feat
        out = self.fuse(out)
        return self.coord_reweight(out)


class LSEM(nn.Module):
    """Lane Structure Enhancement Module: LSA + dilated context + residual enhancement."""

    def __init__(self, c1, k=9, dilation=2):
        """Initialize the LSEM block while preserving input/output channel count."""
        super().__init__()
        self.lsa = LineStripAttention(c1, k=k)

        self.dilated_context = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=dilation, dilation=dilation, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        """Apply line-structure enhancement and residual preservation."""
        identity = x
        x = self.lsa(x)
        x = self.dilated_context(x)
        x = self.out_conv(x)
        return self.act(x + identity)


class WeightedFusion(nn.Module):
    """Learnable normalized feature fusion used by Lane-BiFPN."""

    def __init__(self, n, eps=1e-4):
        """Initialize non-negative normalized fusion weights for n inputs."""
        super().__init__()
        if n < 2:
            raise ValueError(f"WeightedFusion expects at least 2 inputs, got n={n}.")
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = eps

    def forward(self, xs):
        """Fuse input features with learned positive weights."""
        if len(xs) != self.w.numel():
            raise ValueError(f"WeightedFusion expected {self.w.numel()} inputs, got {len(xs)}.")

        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = xs[0] * w[0]
        for i in range(1, len(xs)):
            out = out + xs[i] * w[i]
        return out


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and SiLU activation block."""

    def __init__(self, c1, c2, k=3, s=1, p=None):
        """Initialize a standard Conv-BN-SiLU block."""
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else tuple(x // 2 for x in k)
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Apply convolution, normalization, and activation."""
        return self.act(self.bn(self.conv(x)))


def build_2d_sincos_position_embedding(h, w, dim, device):
    """Build 2D sine-cosine position embeddings with shape [H*W, dim]."""
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sin-cos position embedding, got dim={dim}.")

    y_embed = torch.linspace(0, 1, steps=h, device=device)
    x_embed = torch.linspace(0, 1, steps=w, device=device)
    yy, xx = torch.meshgrid(y_embed, x_embed, indexing="ij")

    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    out_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    out_y = yy.reshape(-1, 1) * omega.reshape(1, -1)

    return torch.cat((torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)), dim=1)


class GCSLaneHead(nn.Module):
    """Query-based structured lane head for GCS-YOLO-Lane.

    The head preserves spatial feature tokens from P2-P5, adds 2D position
    encoding and level embeddings, then lets learnable lane queries attend to
    those tokens with a Transformer decoder.
    """

    def __init__(
        self,
        c1=128,
        num_queries=12,
        num_points=56,
        num_decoder_layers=3,
        nhead=8,
        aux=True,
        point_mode="free",
        fixed_y_start=0.9861111111111112,
        fixed_y_end=0.2222222222222222,
        gcs_mode="query",
        num_slots=5,
        min_lanes=2,
        max_lanes=5,
        count_classes=None,
        query_count_head: bool = False,
        dense_instance_head: bool = False,
        dense_instance_embed_dim: int = 8,
        dense_candidate_head: bool = False,
        dense_endpoint_offset_head: bool = False,
        residual_proposal_head: bool = False,
        residual_proposal_count: int = 8,
        query_survival_head: bool = False,
        query_survival_geometry_adapter: bool = False,
        residual_replace_listwise_head: bool = False,
        query_valid_local_adapter: bool = False,
        query_valid_interval_adapter: bool = False,
        query_valid_interval_base_thr: float = 0.6,
        query_valid_interval_max_shift: float = 8.0,
        lane_instance_set_decoder_head: bool = False,
        lane_instance_identity_dim: int = 16,
        lane_instance_interval_sharpness: float = 4.0,
    ):
        """Initialize the GCS lane query decoder and training-only auxiliary heads."""
        super().__init__()
        if isinstance(c1, (list, tuple)):
            if len(c1) == 0:
                raise ValueError("GCSLaneHead received an empty channel list.")
            if any(c != c1[0] for c in c1):
                raise ValueError(f"GCSLaneHead expects equal P2-P5 channels after LaneBiFPN, got {c1}.")
            c1 = c1[0]
        if c1 % 4 != 0:
            raise ValueError(f"GCSLaneHead channel count must be divisible by 4 for 2D position encoding, got {c1}.")
        if c1 % nhead != 0:
            raise ValueError(f"GCSLaneHead channel count {c1} must be divisible by nhead={nhead}.")

        self.c1 = c1
        self.gcs_mode = str(gcs_mode).lower()
        if self.gcs_mode in {"ordered-slot", "orderedslot"}:
            self.gcs_mode = "ordered_slot"
        if self.gcs_mode not in {"query", "ordered_slot"}:
            raise ValueError(f"GCSLaneHead gcs_mode must be 'query' or 'ordered_slot', got {gcs_mode!r}.")
        self.lane_instance_set_decoder_head = self.gcs_mode == "query" and bool(lane_instance_set_decoder_head)
        if self.gcs_mode == "ordered_slot":
            self.num_slots = int(num_slots)
            if self.num_slots <= 0:
                raise ValueError(f"GCSLaneHead num_slots must be positive, got {num_slots}.")
            self.min_lanes = int(min_lanes)
            self.max_lanes = int(max_lanes)
            if self.min_lanes <= 0 or self.max_lanes < self.min_lanes:
                raise ValueError(f"GCSLaneHead lane bounds must satisfy 0 < min_lanes <= max_lanes, got {min_lanes}/{max_lanes}.")
            if self.max_lanes > self.num_slots:
                raise ValueError(f"GCSLaneHead max_lanes={self.max_lanes} exceeds num_slots={self.num_slots}.")
            expected_count_classes = self.max_lanes - self.min_lanes + 1
            self.count_classes = int(count_classes) if count_classes is not None else expected_count_classes
            if self.count_classes != expected_count_classes:
                raise ValueError(
                    f"GCSLaneHead count_classes must be max_lanes-min_lanes+1={expected_count_classes}, "
                    f"got {self.count_classes}."
                )
            if self.num_slots != 5:
                raise ValueError(f"GCSLaneHead ordered_slot requires num_slots=5, got {self.num_slots}.")
            if int(num_queries) != 5:
                raise ValueError(f"GCSLaneHead ordered_slot requires num_queries=5, got {num_queries}.")
        else:
            self.num_slots = 5
            self.min_lanes = 2
            self.max_lanes = 5
            self.count_classes = 4
        self.num_queries = int(num_queries)
        self.num_points = int(num_points)
        self.aux = aux
        self.point_mode = str(point_mode).lower()
        if self.point_mode in {"fixed-y", "fixedy"}:
            self.point_mode = "fixed_y"
        if self.point_mode not in {"free", "fixed_y"}:
            raise ValueError(f"GCSLaneHead point_mode must be 'free' or 'fixed_y', got {point_mode!r}.")
        self.fixed_y_start = float(fixed_y_start)
        self.fixed_y_end = float(fixed_y_end)
        if not (0.0 <= self.fixed_y_end < self.fixed_y_start <= 1.0):
            raise ValueError(
                f"Expected 0 <= fixed_y_end < fixed_y_start <= 1, got "
                f"{self.fixed_y_end} < {self.fixed_y_start}."
            )
        if self.point_mode == "fixed_y":
            self._validate_fixed_y_anchors()
        self.point_dims = 1 if self.point_mode == "fixed_y" else 2
        self.return_aux = False
        self.min_spatial_tokens = 1024
        self._last_spatial_debug = None

        if not self.lane_instance_set_decoder_head:
            self.query_embed = nn.Embedding(num_queries, c1)
            self.level_embed = nn.Parameter(torch.empty(4, c1))
            nn.init.normal_(self.level_embed)

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=c1,
                nhead=nhead,
                dim_feedforward=c1 * 4,
                dropout=0.0,
                batch_first=True,
                activation="gelu",
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

            self.point_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, num_points * self.point_dims),
            )
            self.point_valid_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, num_points),
            )
            self.point_embed = nn.Embedding(num_points, c1)
            self.point_coord_mlp = nn.Sequential(
                nn.Linear(2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, c1),
            )
            self.point_refine_norm = nn.LayerNorm(c1)
            self.point_image_norm = nn.LayerNorm(c1)
            self.point_refine_mlp = nn.Sequential(
                nn.Linear(c1 * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.point_valid_refine_mlp = nn.Sequential(
                nn.Linear(c1 * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.exist_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
        self.query_count_head = self.gcs_mode == "query" and bool(query_count_head)
        if self.query_count_head:
            self.query_count_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.count_classes),
            )
        self.dense_instance_head = self.gcs_mode == "query" and bool(dense_instance_head)
        self.dense_candidate_head = self.dense_instance_head and bool(dense_candidate_head)
        self.dense_endpoint_offset_head = self.dense_instance_head and bool(dense_endpoint_offset_head)
        if self.dense_instance_head:
            self.dense_instance_embed_dim = int(dense_instance_embed_dim)
            if self.dense_instance_embed_dim < 2:
                raise ValueError(
                    "GCSLaneHead dense_instance_embed_dim must be >= 2, "
                    f"got {dense_instance_embed_dim}."
                )
            self.dense_instance_stem = ConvBNAct(c1, c1, k=3)
            self.dense_instance_centerline = nn.Conv2d(c1, 1, kernel_size=1)
            self.dense_instance_endpoint = nn.Conv2d(c1, 2, kernel_size=1)
            self.dense_instance_embed = nn.Conv2d(c1, self.dense_instance_embed_dim, kernel_size=1)
            if self.dense_endpoint_offset_head:
                self.dense_instance_endpoint_offset = nn.Conv2d(c1, 4, kernel_size=1)
            if self.dense_candidate_head:
                self.dense_instance_candidate_quality = nn.Conv2d(c1, 1, kernel_size=1)
                self.dense_instance_candidate_replace = nn.Conv2d(c1, 1, kernel_size=1)
        self.residual_proposal_head = self.gcs_mode == "query" and bool(residual_proposal_head)
        self.residual_proposal_count = int(residual_proposal_count)
        self.residual_replace_listwise_head = self.residual_proposal_head and bool(residual_replace_listwise_head)
        if self.residual_proposal_head:
            if self.point_mode != "fixed_y":
                raise ValueError("GCSLaneHead residual proposals require point_mode=fixed_y.")
            if self.residual_proposal_count <= 0:
                raise ValueError("GCSLaneHead residual_proposal_count must be positive.")
            self.residual_proposal_stem = ConvBNAct(c1, c1, k=3)
            self.residual_proposal_mask = nn.Conv2d(c1, self.residual_proposal_count, kernel_size=1)
            self.residual_proposal_valid_mlp = nn.Sequential(
                nn.Linear(self.num_points, self.num_points),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_points, self.num_points),
            )
            self.residual_proposal_start_mlp = nn.Linear(self.num_points, self.num_points)
            self.residual_proposal_end_mlp = nn.Linear(self.num_points, self.num_points)
            self.residual_proposal_exist_mlp = nn.Sequential(
                nn.Linear(self.num_points, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.residual_relation_dim = 64
            self.residual_relation_proposal_mlp = nn.Sequential(
                nn.Linear(self.num_points * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_relation_base_mlp = nn.Sequential(
                nn.Linear(self.num_points * 2 + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_relation_base_pair_mlp = nn.Sequential(
                nn.Linear(self.num_points * 3 + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_relation_proposal_pair_mlp = nn.Sequential(
                nn.Linear(self.num_points * 3, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_relation_norm = nn.LayerNorm(self.residual_relation_dim)
            self.residual_topology_base_pair_mlp = nn.Sequential(
                nn.Linear(self.num_points * 3 + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_topology_proposal_pair_mlp = nn.Sequential(
                nn.Linear(self.num_points * 3, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_topology_norm = nn.LayerNorm(self.residual_relation_dim)
            self.residual_proposal_identity_mlp = nn.Sequential(
                nn.Linear(self.residual_relation_dim, self.residual_relation_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.residual_relation_dim, 16),
            )
            self.residual_proposal_quality_mlp = nn.Sequential(
                nn.Linear(self.residual_relation_dim * 3 + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.residual_topology_quality_mlp = nn.Sequential(
                nn.Linear(self.residual_relation_dim * 3 + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.residual_replace_pair_mlp = nn.Sequential(
                nn.Linear(self.residual_relation_dim * 2 + c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.residual_visual_token_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.residual_relation_dim),
            )
            self.residual_replace_visual_pair_mlp = nn.Sequential(
                nn.Linear(self.residual_relation_dim * 3 + c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            if self.residual_replace_listwise_head:
                self.residual_noop_mlp = nn.Sequential(
                    nn.Linear(c1 + self.residual_relation_dim * 2 + 6, c1),
                    nn.ReLU(inplace=True),
                    nn.Linear(c1, 1),
                )
            nn.init.zeros_(self.residual_topology_quality_mlp[-1].weight)
            nn.init.zeros_(self.residual_topology_quality_mlp[-1].bias)
            nn.init.zeros_(self.residual_replace_pair_mlp[-1].weight)
            nn.init.zeros_(self.residual_replace_pair_mlp[-1].bias)
            nn.init.zeros_(self.residual_replace_visual_pair_mlp[-1].weight)
            nn.init.zeros_(self.residual_replace_visual_pair_mlp[-1].bias)
            if self.residual_replace_listwise_head:
                nn.init.zeros_(self.residual_noop_mlp[-1].weight)
                nn.init.zeros_(self.residual_noop_mlp[-1].bias)
        self.query_survival_head = self.gcs_mode == "query" and bool(query_survival_head)
        self.query_survival_geometry_adapter = self.query_survival_head and bool(query_survival_geometry_adapter)
        self.query_valid_local_adapter = self.gcs_mode == "query" and bool(query_valid_local_adapter)
        self.query_valid_interval_adapter = self.gcs_mode == "query" and bool(query_valid_interval_adapter)
        self.query_valid_interval_base_thr = float(query_valid_interval_base_thr)
        self.query_valid_interval_max_shift = float(query_valid_interval_max_shift)
        self.lane_instance_identity_dim = int(lane_instance_identity_dim)
        self.lane_instance_interval_sharpness = float(lane_instance_interval_sharpness)
        if self.query_valid_local_adapter and self.query_valid_interval_adapter:
            raise ValueError("query_valid_local_adapter and query_valid_interval_adapter are mutually exclusive.")
        if not 0.0 < self.query_valid_interval_base_thr < 1.0:
            raise ValueError(
                "query_valid_interval_base_thr must be in (0, 1), "
                f"got {self.query_valid_interval_base_thr}."
            )
        if self.query_valid_interval_max_shift <= 0.0:
            raise ValueError(
                "query_valid_interval_max_shift must be > 0, "
                f"got {self.query_valid_interval_max_shift}."
            )
        if self.query_survival_head:
            survival_input_dim = c1 + self.num_points * 2 + 1
            self.query_survival_state_mlp = nn.Sequential(
                nn.Linear(survival_input_dim, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, c1),
            )
            self.query_survival_attention = nn.MultiheadAttention(c1, nhead, batch_first=True)
            self.query_survival_norm = nn.LayerNorm(c1)
            self.query_survival_delta_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            nn.init.zeros_(self.query_survival_delta_mlp[-1].weight)
            nn.init.zeros_(self.query_survival_delta_mlp[-1].bias)
            if self.query_survival_geometry_adapter:
                self.query_survival_point_delta_mlp = nn.Sequential(
                    nn.Linear(c1, c1),
                    nn.ReLU(inplace=True),
                    nn.Linear(c1, self.num_points),
                )
                self.query_survival_valid_delta_mlp = nn.Sequential(
                    nn.Linear(c1, c1),
                    nn.ReLU(inplace=True),
                    nn.Linear(c1, self.num_points),
                )
                nn.init.zeros_(self.query_survival_point_delta_mlp[-1].weight)
                nn.init.zeros_(self.query_survival_point_delta_mlp[-1].bias)
                nn.init.zeros_(self.query_survival_valid_delta_mlp[-1].weight)
                nn.init.zeros_(self.query_survival_valid_delta_mlp[-1].bias)
        if self.query_valid_local_adapter:
            self.query_valid_local_delta_mlp = nn.Sequential(
                nn.Linear(c1 * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            nn.init.zeros_(self.query_valid_local_delta_mlp[-1].weight)
            nn.init.zeros_(self.query_valid_local_delta_mlp[-1].bias)
        if self.query_valid_interval_adapter:
            self.query_valid_interval_offset_mlp = nn.Sequential(
                nn.Linear(c1 * 4, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 2),
            )
            nn.init.zeros_(self.query_valid_interval_offset_mlp[-1].weight)
            nn.init.zeros_(self.query_valid_interval_offset_mlp[-1].bias)
        if self.lane_instance_set_decoder_head:
            if self.point_mode != "fixed_y":
                raise ValueError("GCSLaneHead lane-instance-set decoder requires point_mode=fixed_y.")
            if self.lane_instance_identity_dim <= 0:
                raise ValueError(
                    "lane_instance_identity_dim must be positive, "
                    f"got {self.lane_instance_identity_dim}."
                )
            if self.lane_instance_interval_sharpness <= 0.0:
                raise ValueError(
                    "lane_instance_interval_sharpness must be positive, "
                    f"got {self.lane_instance_interval_sharpness}."
                )
            self.lane_instance_p2_stem = ConvBNAct(c1, c1, k=3)
            self.lane_instance_p3_stem = ConvBNAct(c1, c1, k=3)
            self.lane_instance_fuse = ConvBNAct(c1 * 2, c1, k=1, p=0)
            self.lane_instance_query_embed = nn.Embedding(self.num_queries, c1)
            self.lane_instance_row_embed = nn.Embedding(self.num_points, c1)
            self.lane_instance_row_kernel_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, c1),
            )
            self.lane_instance_point_norm = nn.LayerNorm(c1)
            self.lane_instance_candidate_norm = nn.LayerNorm(c1)
            self.lane_instance_x_refine_mlp = nn.Sequential(
                nn.Linear(c1 * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_start_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_end_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_survival_mlp = nn.Sequential(
                nn.Linear(c1 + 2 + self.lane_instance_identity_dim + 2 + 5, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_quality_mlp = nn.Sequential(
                nn.Linear(c1 + 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_novelty_mlp = nn.Sequential(
                nn.Linear(c1 + 1 + self.lane_instance_identity_dim + 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_identity_mlp = nn.Sequential(
                nn.Linear(c1 + 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.lane_instance_identity_dim),
            )
            self.lane_instance_pair_duplicate_mlp = nn.Sequential(
                nn.Linear(c1 * 3 + 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.lane_instance_topology_mlp = nn.Sequential(
                nn.Linear(c1 * 3 + 1 + self.lane_instance_identity_dim + 1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 2),
            )
            self.lane_instance_empty_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
            self.register_buffer(
                "lane_instance_start_prior",
                self._build_lane_instance_endpoint_prior(start=True),
                persistent=False,
            )
            self.register_buffer(
                "lane_instance_end_prior",
                self._build_lane_instance_endpoint_prior(start=False),
                persistent=False,
            )
        if self.gcs_mode == "ordered_slot":
            self.start_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, num_points),
            )
            self.end_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, num_points),
            )
            self.count_mlp = nn.Sequential(
                nn.Linear(c1, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, self.count_classes),
            )

        self.aux_mask = nn.Sequential(
            ConvBNAct(c1, c1, k=3),
            nn.Conv2d(c1, 2, kernel_size=1),
        )
        self.aux_edge = nn.Sequential(
            ConvBNAct(c1, c1 // 2, k=3),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
        )
        self.register_buffer("point_reference_logits", self._build_point_references(), persistent=False)
        self.register_buffer("fixed_y_anchors", self._build_fixed_y_anchors(), persistent=False)
        if not self.lane_instance_set_decoder_head:
            self._init_point_delta_head()
            self._init_point_valid_head()
            self._init_point_refine_head()
            self._init_point_valid_refine_head()
        if self.gcs_mode == "ordered_slot":
            self._init_interval_heads()
        if self.query_count_head:
            self._init_query_count_head()
        if self.dense_instance_head:
            self._init_dense_instance_head()
        if self.lane_instance_set_decoder_head:
            self._init_lane_instance_set_head()

    def _build_fixed_y_anchors(self):
        """Build shared bottom-to-top y anchors for fixed-y x-only prediction."""
        return torch.linspace(float(self.fixed_y_start), float(self.fixed_y_end), self.num_points)

    def _validate_fixed_y_anchors(self) -> None:
        """Fail fast unless fixed-y anchors match the TuSimple K56 contract."""
        anchors = self._build_fixed_y_anchors().detach().float().reshape(-1)
        if anchors.numel() != 56:
            raise ValueError(f"GCSLaneHead fixed_y_anchors: K mismatch, got {anchors.numel()}, expected 56.")
        anchors_px = anchors * 720.0
        expected_desc = torch.arange(710.0, 150.0, -10.0, dtype=anchors_px.dtype, device=anchors_px.device)
        if not torch.allclose(anchors_px, expected_desc, atol=1e-3):
            raise ValueError(
                "GCSLaneHead fixed_y_anchors mismatch; expected desc 710..160 step -10."
            )

    def _build_point_references(self):
        """Build query-specific bottom-to-top lane reference logits.

        A shared zero-bias point head makes every query initially predict the
        same centerline. These coarse perspective-shaped references give each
        query a distinct spatial role while still letting the MLP learn large
        offsets when the image geometry requires it.
        """
        y = self._build_fixed_y_anchors()
        bottom_x = torch.linspace(0.05, 0.95, self.num_queries)
        top_x = 0.5 + (bottom_x - 0.5) * 0.25
        t = torch.linspace(0.0, 1.0, self.num_points)
        x = bottom_x[:, None] * (1.0 - t[None]) + top_x[:, None] * t[None]
        if getattr(self, "point_mode", "free") == "fixed_y":
            return torch.logit(x.clamp(1e-4, 1.0 - 1e-4))
        points = torch.stack((x, y[None].expand(self.num_queries, -1)), dim=-1)
        return torch.logit(points.clamp(1e-4, 1.0 - 1e-4))

    def _build_lane_instance_endpoint_prior(self, start: bool) -> torch.Tensor:
        """Build a fixed endpoint prior for bottom-to-top interval logits."""
        index = torch.arange(self.num_points, dtype=torch.float32)
        target = 0.0 if bool(start) else float(self.num_points - 1)
        return -0.25 * (index - target).abs()

    def _init_point_delta_head(self):
        """Initialize point deltas near zero while keeping point gradients live."""
        final = self.point_mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def _init_point_valid_head(self):
        """Initialize per-point visibility logits near the BCE decision boundary."""
        final = self.point_valid_mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def _init_point_refine_head(self):
        """Initialize image-conditioned point refinement as a small residual update."""
        nn.init.zeros_(self.point_embed.weight)
        final = self.point_refine_mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(final.bias)

    def _init_point_valid_refine_head(self):
        """Initialize image-conditioned visibility refinement as a small residual update."""
        final = self.point_valid_refine_mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=5e-3)
        nn.init.zeros_(final.bias)

    def _init_interval_heads(self):
        """Initialize ordered-slot interval/count logits near neutral."""
        for mlp in (self.start_mlp, self.end_mlp, self.count_mlp):
            final = mlp[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)

    def _init_query_count_head(self):
        """Initialize query-mode explicit count logits near neutral."""
        final = self.query_count_mlp[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def _init_dense_instance_head(self):
        """Initialize dense evidence logits with conservative sparse-foreground priors."""
        nn.init.normal_(self.dense_instance_centerline.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.dense_instance_centerline.bias, -2.0)
        nn.init.normal_(self.dense_instance_endpoint.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.dense_instance_endpoint.bias, -3.0)
        nn.init.normal_(self.dense_instance_embed.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.dense_instance_embed.bias)
        if getattr(self, "dense_endpoint_offset_head", False):
            nn.init.normal_(self.dense_instance_endpoint_offset.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.dense_instance_endpoint_offset.bias)
        if getattr(self, "dense_candidate_head", False):
            nn.init.normal_(self.dense_instance_candidate_quality.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.dense_instance_candidate_quality.bias, -2.0)
            nn.init.normal_(self.dense_instance_candidate_replace.weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.dense_instance_candidate_replace.bias, -3.0)

    def _init_lane_instance_set_head(self):
        """Initialize the lane-instance-set branch with conservative finite logits."""
        nn.init.normal_(self.lane_instance_query_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lane_instance_row_embed.weight, mean=0.0, std=0.02)
        for mlp in (
            self.lane_instance_row_kernel_mlp,
            self.lane_instance_x_refine_mlp,
            self.lane_instance_start_mlp,
            self.lane_instance_end_mlp,
            self.lane_instance_survival_mlp,
            self.lane_instance_quality_mlp,
            self.lane_instance_novelty_mlp,
            self.lane_instance_identity_mlp,
            self.lane_instance_pair_duplicate_mlp,
            self.lane_instance_topology_mlp,
            self.lane_instance_empty_mlp,
        ):
            final = mlp[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)
        nn.init.constant_(self.lane_instance_survival_mlp[-1].bias, -2.0)
        nn.init.constant_(self.lane_instance_pair_duplicate_mlp[-1].bias, -3.0)

    def _sample_point_features(self, xs, points):
        """Sample high-resolution image features at normalized point coordinates.

        Args:
            xs: P2-P5 BiFPN features with equal channel count.
            points: B x Q x K x 2 normalized coordinates in x,y order.

        Returns:
            B x Q x K x C image-conditioned point tokens.
        """
        if points.ndim != 4 or points.shape[-1] != 2:
            raise ValueError(f"Expected points with shape B x Q x K x 2, got {tuple(points.shape)}.")
        b, q, k, _ = points.shape
        grid = points.mul(2.0).sub(1.0).view(b, q * k, 1, 2)
        sampled = []
        for level, feat in enumerate(xs[:3]):
            if feat.shape[1] != self.c1:
                raise ValueError(f"Expected {self.c1} channels at refinement level {level}, got {feat.shape[1]}.")
            token = F.grid_sample(
                feat,
                grid.to(device=feat.device, dtype=feat.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            token = token.squeeze(-1).transpose(1, 2).reshape(b, q, k, self.c1)
            sampled.append(token)
        return torch.stack(sampled, dim=0).mean(dim=0)

    def _point_refine_tokens(self, xs, hs, points):
        """Build point-level tokens from query state, point index, coordinates, and sampled image features."""
        b, q, k, _ = points.shape
        image_tokens = self._sample_point_features(xs, points)
        query_tokens = hs.unsqueeze(2).expand(-1, -1, k, -1)
        point_tokens = self.point_embed.weight.to(device=hs.device, dtype=hs.dtype).view(1, 1, k, self.c1)
        coord_tokens = self.point_coord_mlp(points.to(device=hs.device, dtype=hs.dtype))

        prior_tokens = self.point_refine_norm(query_tokens + point_tokens + coord_tokens)
        image_tokens = self.point_image_norm(image_tokens)
        return torch.cat((prior_tokens, image_tokens), dim=-1)

    def _refine_fixed_y_logits(self, xs, hs, coarse_logits, fixed_y):
        """Refine fixed-y x logits with point-level sampled image features."""
        b, q, k = coarse_logits.shape
        y = fixed_y.to(device=coarse_logits.device, dtype=coarse_logits.dtype).view(1, 1, k).expand(b, q, -1)
        coarse_x = torch.sigmoid(coarse_logits)
        coarse_points = torch.stack((coarse_x, y), dim=-1)

        refine_tokens = self._point_refine_tokens(xs, hs, coarse_points)
        refine_delta = self.point_refine_mlp(refine_tokens).squeeze(-1)
        return coarse_logits + refine_delta

    def _refine_fixed_y_valid_logits(self, xs, hs, pred_points):
        """Refine fixed-y point visibility logits with point-level sampled image features."""
        coarse_valid = self.point_valid_mlp(hs).view(hs.shape[0], self.num_queries, self.num_points)
        refine_tokens = self._point_refine_tokens(xs, hs, pred_points.detach())
        valid_delta = self.point_valid_refine_mlp(refine_tokens).squeeze(-1)
        return coarse_valid + valid_delta

    def _query_valid_interval_delta(self, base_valid_logits, point_tokens):
        """Return a contiguous valid-logit residual from zero-initialized start/end boundary offsets."""
        b, q, k = base_valid_logits.shape
        probability = base_valid_logits.detach().sigmoid()
        base_mask = probability >= self.query_valid_interval_base_thr
        anchor_index = torch.arange(k, device=base_valid_logits.device)
        first = torch.where(base_mask, anchor_index.view(1, 1, k), k).amin(dim=-1)
        last = torch.where(base_mask, anchor_index.view(1, 1, k), -1).amax(dim=-1)
        peak = probability.argmax(dim=-1)
        has_valid = base_mask.any(dim=-1)
        first = torch.where(has_valid, first, peak)
        last = torch.where(has_valid, last, peak)

        token_dim = point_tokens.shape[-1]
        first_token = point_tokens.gather(
            2, first[..., None, None].expand(b, q, 1, token_dim)
        ).squeeze(2)
        last_token = point_tokens.gather(
            2, last[..., None, None].expand(b, q, 1, token_dim)
        ).squeeze(2)
        offsets = self.query_valid_interval_max_shift * torch.tanh(
            self.query_valid_interval_offset_mlp(torch.cat((first_token, last_token), dim=-1))
        )

        base_start = first.to(dtype=base_valid_logits.dtype)
        base_end = last.to(dtype=base_valid_logits.dtype)
        endpoint_a = (base_start + offsets[..., 0]).clamp(0.0, float(k - 1))
        endpoint_b = (base_end + offsets[..., 1]).clamp(0.0, float(k - 1))
        corrected_start = torch.minimum(endpoint_a, endpoint_b)
        corrected_end = torch.maximum(endpoint_a, endpoint_b)

        position = anchor_index.to(dtype=base_valid_logits.dtype).view(1, 1, k)
        base_field = torch.minimum(position - base_start[..., None] + 0.5, base_end[..., None] - position + 0.5)
        corrected_field = torch.minimum(
            position - corrected_start[..., None] + 0.5,
            corrected_end[..., None] - position + 0.5,
        )
        delta = corrected_field - base_field
        base_bounds = torch.stack((base_start, base_end), dim=-1)
        corrected_bounds = torch.stack((corrected_start, corrected_end), dim=-1)
        return delta, offsets, base_bounds, corrected_bounds

    def _lane_instance_set_outputs(
        self,
        xs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build image-local Q x K lane-instance candidates from fused P2/P3 evidence."""
        p2, p3, _, _ = xs
        p2_features = self.lane_instance_p2_stem(p2)
        p3_features = self.lane_instance_p3_stem(p3)
        p3_features = F.interpolate(p3_features, size=p2_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.lane_instance_fuse(torch.cat((p2_features, p3_features), dim=1))
        b, _, _, width = features.shape

        fixed_y = self.fixed_y_anchors.to(device=features.device, dtype=features.dtype)
        grid_x = torch.linspace(-1.0, 1.0, width, device=features.device, dtype=features.dtype)
        row_grid = torch.stack(
            (
                grid_x.view(1, 1, width).expand(b, self.num_points, -1),
                fixed_y.mul(2.0).sub(1.0).view(1, self.num_points, 1).expand(b, -1, width),
            ),
            dim=-1,
        )
        row_features = F.grid_sample(
            features,
            row_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        row_features = row_features.permute(0, 2, 3, 1).contiguous()

        query_token = self.lane_instance_query_embed.weight.to(device=features.device, dtype=features.dtype)
        row_token = self.lane_instance_row_embed.weight.to(device=features.device, dtype=features.dtype)
        query_row_token = query_token[:, None, :] + row_token[None, :, :]
        row_kernel = F.normalize(self.lane_instance_row_kernel_mlp(query_row_token), dim=-1)
        normalized_row_features = F.normalize(row_features, dim=-1)
        row_logits = torch.einsum("qkc,bkwc->bqkw", row_kernel, normalized_row_features)
        row_logits = row_logits * math.sqrt(float(self.c1))

        x_axis = torch.linspace(0.0, 1.0, width, device=features.device, dtype=features.dtype)
        row_probability = torch.softmax(row_logits, dim=-1)
        seed_x = (row_probability * x_axis.view(1, 1, 1, width)).sum(dim=-1)
        row_context = (row_probability.unsqueeze(-1) * row_features[:, None, :, :, :]).sum(dim=3)
        query_row_context = query_row_token.view(1, self.num_queries, self.num_points, self.c1).expand(b, -1, -1, -1)
        point_token = self.lane_instance_point_norm(row_context + query_row_context)

        x_refine = 0.05 * torch.tanh(
            self.lane_instance_x_refine_mlp(torch.cat((point_token, row_context), dim=-1)).squeeze(-1)
        )
        pred_x = (seed_x + x_refine).clamp(0.0, 1.0)
        pred_y = fixed_y.view(1, 1, self.num_points).expand(b, self.num_queries, -1)
        pred_points = torch.stack((pred_x, pred_y), dim=-1)

        start_prior = self.lane_instance_start_prior.to(device=features.device, dtype=features.dtype).view(1, 1, -1)
        end_prior = self.lane_instance_end_prior.to(device=features.device, dtype=features.dtype).view(1, 1, -1)
        start_logits = self.lane_instance_start_mlp(point_token).squeeze(-1) + start_prior
        end_logits = self.lane_instance_end_mlp(point_token).squeeze(-1) + end_prior
        valid_logits, interval_start, interval_end = interval_valid_logits_from_start_end(
            start_logits,
            end_logits,
            sharpness=self.lane_instance_interval_sharpness,
        )
        valid_prob = valid_logits.sigmoid()
        interval_weight = valid_prob / valid_prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        candidate_token = self.lane_instance_candidate_norm((point_token * interval_weight.unsqueeze(-1)).sum(dim=2))

        candidate_valid_detached = valid_prob.detach()
        visible_len = valid_prob.mean(dim=-1)
        geometry_span = (interval_end - interval_start).clamp_min(0.0) / max(float(self.num_points - 1), 1.0)
        scalar_features = torch.stack((visible_len, geometry_span), dim=-1)
        identity = F.normalize(
            self.lane_instance_identity_mlp(torch.cat((candidate_token, scalar_features), dim=-1)),
            dim=-1,
        )
        identity_cos = (identity[:, :, None, :] * identity[:, None, :, :]).sum(dim=-1)
        identity_diagonal = torch.eye(self.num_queries, device=features.device, dtype=torch.bool).view(
            1, self.num_queries, self.num_queries
        )
        identity_neighbor = identity_cos.masked_fill(identity_diagonal, -1.0)
        identity_stats = torch.stack(
            (
                identity_neighbor.amax(dim=-1),
                identity_neighbor.sum(dim=-1) / max(float(self.num_queries - 1), 1.0),
            ),
            dim=-1,
        )
        pair_diff = (pred_x[:, :, None, :] - pred_x[:, None, :, :]).abs()
        pair_weight = candidate_valid_detached[:, :, None, :] * candidate_valid_detached[:, None, :, :]
        pair_dist = (pair_diff * pair_weight).sum(dim=-1) / pair_weight.sum(dim=-1).clamp_min(1e-6)
        nearest_candidate_dist = pair_dist.masked_fill(identity_diagonal, 1.0).amin(dim=-1)
        quality_logits = self.lane_instance_quality_mlp(
            torch.cat((candidate_token, scalar_features), dim=-1)
        ).squeeze(-1)
        novelty_logits = self.lane_instance_novelty_mlp(
            torch.cat((candidate_token, nearest_candidate_dist.unsqueeze(-1), identity, identity_stats), dim=-1)
        ).squeeze(-1)

        left_token = candidate_token[:, :, None, :].expand(-1, -1, self.num_queries, -1)
        right_token = candidate_token[:, None, :, :].expand(-1, self.num_queries, -1, -1)
        pair_features = torch.cat(
            (
                left_token,
                right_token,
                (left_token - right_token).abs(),
                pair_dist.unsqueeze(-1),
                identity_cos.unsqueeze(-1),
            ),
            dim=-1,
        )
        pair_duplicate_logits = self.lane_instance_pair_duplicate_mlp(pair_features).squeeze(-1)
        pair_duplicate_logits = 0.5 * (pair_duplicate_logits + pair_duplicate_logits.transpose(1, 2))
        diagonal = identity_diagonal
        pair_duplicate_logits = pair_duplicate_logits.masked_fill(diagonal, 20.0)

        mean_x = (pred_x * interval_weight).sum(dim=-1)
        signed_mean_diff = mean_x[:, :, None] - mean_x[:, None, :]
        topology_features = torch.cat(
            (
                left_token,
                right_token,
                left_token - right_token,
                signed_mean_diff.unsqueeze(-1),
                (identity[:, :, None, :] - identity[:, None, :, :]).abs(),
                identity_cos.unsqueeze(-1),
            ),
            dim=-1,
        )
        topology_logits = self.lane_instance_topology_mlp(topology_features)
        left_logits = topology_logits[..., 0] - 12.0 * signed_mean_diff
        right_logits = topology_logits[..., 1] + 12.0 * signed_mean_diff
        left_logits = left_logits.masked_fill(diagonal, -20.0)
        right_logits = right_logits.masked_fill(diagonal, -20.0)

        duplicate_risk = pair_duplicate_logits.sigmoid().masked_fill(diagonal, 0.0).amax(dim=-1)
        left_probability = left_logits.sigmoid()
        right_probability = right_logits.sigmoid()
        topology_pair_confidence = torch.maximum(left_probability, right_probability) * (
            left_probability - right_probability
        ).abs()
        topology_confidence = topology_pair_confidence.masked_fill(diagonal, 0.0).sum(dim=-1) / max(
            float(self.num_queries - 1), 1.0
        )
        visibility_quality = torch.sqrt((visible_len * geometry_span).clamp_min(1e-6))
        utility_components = torch.stack(
            (
                quality_logits.sigmoid(),
                novelty_logits.sigmoid(),
                visibility_quality,
                duplicate_risk,
                topology_confidence,
            ),
            dim=-1,
        )
        survival_logits = self.lane_instance_survival_mlp(
            torch.cat((candidate_token, scalar_features, identity, identity_stats, utility_components), dim=-1)
        ).squeeze(-1)

        image_token = F.adaptive_avg_pool2d(features, 1).flatten(1)
        empty_logit = self.lane_instance_empty_mlp(image_token).squeeze(-1)
        return {
            "pred_lane_instance_points": pred_points,
            "pred_lane_instance_start_logits": start_logits,
            "pred_lane_instance_end_logits": end_logits,
            "pred_lane_instance_valid_logits": valid_logits,
            "pred_lane_instance_interval_start": interval_start,
            "pred_lane_instance_interval_end": interval_end,
            "pred_lane_instance_identity": identity,
            "pred_lane_instance_pair_duplicate_logits": pair_duplicate_logits,
            "pred_lane_instance_novelty_logits": novelty_logits,
            "pred_lane_instance_left_logits": left_logits,
            "pred_lane_instance_right_logits": right_logits,
            "pred_lane_instance_geometry_quality_logits": quality_logits,
            "pred_lane_instance_survival_logits": survival_logits,
            "pred_lane_instance_empty_logit": empty_logit,
        }

    def _residual_proposal_outputs(
        self,
        p2: torch.Tensor,
        base_points: torch.Tensor,
        base_logits: torch.Tensor,
        base_valid_logits: torch.Tensor,
        base_query_token: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build differentiable full-K56 proposals from proposal-specific dense instance responses."""
        features = self.residual_proposal_stem(p2)
        mask_logits = self.residual_proposal_mask(features)
        b, proposal_count, _, width = mask_logits.shape
        fixed_y = self.fixed_y_anchors.to(device=mask_logits.device, dtype=mask_logits.dtype)
        grid_x = torch.linspace(-1.0, 1.0, width, device=mask_logits.device, dtype=mask_logits.dtype)
        grid = torch.stack(
            (
                grid_x.view(1, 1, width).expand(b, self.num_points, -1),
                fixed_y.mul(2.0).sub(1.0).view(1, self.num_points, 1).expand(b, -1, width),
            ),
            dim=-1,
        )
        row_logits = F.grid_sample(mask_logits, grid, mode="bilinear", padding_mode="border", align_corners=True)
        x_axis = torch.linspace(0.0, 1.0, width, device=mask_logits.device, dtype=mask_logits.dtype)
        row_probability = torch.softmax(row_logits / 0.1, dim=-1)
        pred_x = (row_probability * x_axis.view(1, 1, 1, width)).sum(dim=-1)
        pred_y = fixed_y.view(1, 1, self.num_points).expand(b, proposal_count, -1)
        pred_points = torch.stack((pred_x, pred_y), dim=-1)
        row_evidence = torch.logsumexp(row_logits, dim=-1) - math.log(float(width))
        valid_logits = row_evidence + self.residual_proposal_valid_mlp(row_evidence)
        residual_valid_prob = valid_logits.detach().sigmoid()
        base_x = base_points[:, :, :, 0].detach()
        base_valid_prob = base_valid_logits.detach().sigmoid()
        base_score = base_logits.detach().sigmoid()
        proposal_token = self.residual_relation_proposal_mlp(torch.cat((row_evidence, residual_valid_prob), dim=-1))
        base_token = self.residual_relation_base_mlp(
            torch.cat((base_x, base_valid_prob, base_score.unsqueeze(-1)), dim=-1)
        )

        residual_base_diff = (pred_x[:, :, None, :] - base_x[:, None, :, :]).abs()
        residual_valid_pair = residual_valid_prob[:, :, None, :].expand(-1, -1, base_x.shape[1], -1)
        base_valid_pair = base_valid_prob[:, None, :, :].expand(-1, proposal_count, -1, -1)
        base_score_pair = base_score[:, None, :, None].expand(-1, proposal_count, -1, -1)
        residual_base_pair = self.residual_relation_base_pair_mlp(
            torch.cat((residual_base_diff, residual_valid_pair, base_valid_pair, base_score_pair), dim=-1)
        )
        victim_pair_token = residual_base_pair
        base_attention = torch.softmax(base_score[:, None, :] - 10.0 * residual_base_diff.mean(dim=-1), dim=-1)
        base_context = (
            base_attention.unsqueeze(-1) * (base_token[:, None, :, :] + residual_base_pair)
        ).sum(dim=2)

        residual_pair_diff = (pred_x[:, :, None, :] - pred_x[:, None, :, :]).abs()
        residual_valid_left = residual_valid_prob[:, :, None, :].expand(-1, -1, proposal_count, -1)
        residual_valid_right = residual_valid_prob[:, None, :, :].expand(-1, proposal_count, -1, -1)
        residual_pair = self.residual_relation_proposal_pair_mlp(
            torch.cat((residual_pair_diff, residual_valid_left, residual_valid_right), dim=-1)
        )
        residual_attention_logits = -10.0 * residual_pair_diff.mean(dim=-1)
        diagonal = torch.eye(proposal_count, device=pred_x.device, dtype=torch.bool).unsqueeze(0)
        residual_attention = torch.softmax(residual_attention_logits.masked_fill(diagonal, -1e4), dim=-1)
        proposal_context = (
            residual_attention.unsqueeze(-1) * (proposal_token[:, None, :, :] + residual_pair)
        ).sum(dim=2)

        relation_token = self.residual_relation_norm(proposal_token + base_context + proposal_context)
        identity = F.normalize(self.residual_proposal_identity_mlp(relation_token), dim=-1)
        base_novelty = residual_base_diff.mean(dim=-1).min(dim=-1).values
        quality_features = torch.cat(
            (relation_token, base_context, proposal_context, base_novelty.unsqueeze(-1)),
            dim=-1,
        )
        quality_logits = self.residual_proposal_quality_mlp(quality_features).squeeze(-1)
        topology_token = relation_token
        if hasattr(self, "residual_topology_base_pair_mlp"):
            signed_residual_base_diff = pred_x[:, :, None, :] - base_x[:, None, :, :]
            topology_base_pair = self.residual_topology_base_pair_mlp(
                torch.cat((signed_residual_base_diff, residual_valid_pair, base_valid_pair, base_score_pair), dim=-1)
            )
            victim_pair_token = topology_base_pair
            topology_base_context = (
                base_attention.unsqueeze(-1) * (base_token[:, None, :, :] + topology_base_pair)
            ).sum(dim=2)
            signed_residual_pair_diff = pred_x[:, :, None, :] - pred_x[:, None, :, :]
            topology_proposal_pair = self.residual_topology_proposal_pair_mlp(
                torch.cat((signed_residual_pair_diff, residual_valid_left, residual_valid_right), dim=-1)
            )
            topology_proposal_context = (
                residual_attention.unsqueeze(-1) * (proposal_token[:, None, :, :] + topology_proposal_pair)
            ).sum(dim=2)
            topology_token = self.residual_topology_norm(
                proposal_token + topology_base_context + topology_proposal_context
            )
            topology_quality_features = torch.cat(
                (topology_token, topology_base_context, topology_proposal_context, base_novelty.unsqueeze(-1)),
                dim=-1,
            )
            quality_logits = quality_logits + self.residual_topology_quality_mlp(topology_quality_features).squeeze(-1)
        replace_pair_features = torch.cat(
            (
                topology_token[:, :, None, :].expand(-1, -1, base_x.shape[1], -1),
                victim_pair_token,
                base_query_token[:, None, :, :].expand(-1, proposal_count, -1, -1),
            ),
            dim=-1,
        )
        replace_logits = self.residual_replace_pair_mlp(replace_pair_features).squeeze(-1)
        proposal_grid = torch.stack((pred_x.mul(2.0).sub(1.0), pred_y.mul(2.0).sub(1.0)), dim=-1)
        sampled_visual = F.grid_sample(
            features,
            proposal_grid.reshape(b, proposal_count * self.num_points, 1, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled_visual = sampled_visual.squeeze(-1).transpose(1, 2).reshape(
            b, proposal_count, self.num_points, self.c1
        )
        visual_weight = residual_valid_prob.unsqueeze(-1)
        visual_pool = (sampled_visual * visual_weight).sum(dim=2) / visual_weight.sum(dim=2).clamp_min(1.0)
        visual_token = self.residual_visual_token_mlp(visual_pool)
        visual_pair_features = torch.cat(
            (
                topology_token[:, :, None, :].expand(-1, -1, base_x.shape[1], -1),
                visual_token[:, :, None, :].expand(-1, -1, base_x.shape[1], -1),
                victim_pair_token,
                base_query_token[:, None, :, :].expand(-1, proposal_count, -1, -1),
            ),
            dim=-1,
        )
        replace_logits = replace_logits + self.residual_replace_visual_pair_mlp(visual_pair_features).squeeze(-1)
        outputs = {
            "pred_residual_points": pred_points,
            "pred_residual_valid_logits": valid_logits,
            "pred_residual_exist_logits": self.residual_proposal_exist_mlp(row_evidence).squeeze(-1),
            "pred_residual_start_logits": self.residual_proposal_start_mlp(row_evidence),
            "pred_residual_end_logits": self.residual_proposal_end_mlp(row_evidence),
            "pred_residual_row_logits": row_logits,
            "pred_residual_mask_logits": mask_logits,
            "pred_residual_identity": identity,
            "pred_residual_quality_logits": quality_logits,
            "pred_residual_base_novelty": base_novelty,
            "pred_residual_relation_token": relation_token,
            "pred_residual_topology_token": topology_token,
            "pred_residual_base_token": base_token,
            "pred_residual_base_visual_token": base_query_token,
            "pred_residual_victim_pair_token": victim_pair_token,
            "pred_residual_replace_logits": replace_logits,
            "pred_residual_visual_token": visual_token,
        }
        if self.residual_replace_listwise_head:
            selected_count = min(5, int(base_score.shape[1]))
            selected_score, selected_index = base_score.topk(selected_count, dim=1)
            selected_query_token = base_query_token.gather(
                1,
                selected_index.unsqueeze(-1).expand(-1, -1, base_query_token.shape[-1]),
            )
            selected_valid = base_valid_prob.gather(
                1,
                selected_index.unsqueeze(-1).expand(-1, -1, base_valid_prob.shape[-1]),
            )
            sorted_score = base_score.sort(dim=1, descending=True).values
            boundary_gap = (
                sorted_score[:, selected_count - 1] - sorted_score[:, selected_count]
                if int(sorted_score.shape[1]) > selected_count
                else sorted_score[:, selected_count - 1]
            )
            noop_stats = torch.stack(
                (
                    selected_score.mean(dim=1),
                    selected_score.amin(dim=1),
                    selected_score.amax(dim=1),
                    boundary_gap,
                    selected_valid.mean(dim=(1, 2)),
                    residual_valid_prob.mean(dim=(1, 2)),
                ),
                dim=-1,
            )
            noop_features = torch.cat(
                (
                    selected_query_token.mean(dim=1),
                    topology_token.mean(dim=1),
                    visual_token.mean(dim=1),
                    noop_stats,
                ),
                dim=-1,
            )
            outputs["pred_residual_noop_logit"] = self.residual_noop_mlp(noop_features).squeeze(-1)
        return outputs

    def aux_output_size(self, orig_size=None):
        """Return auxiliary supervision size from the explicit original input image size."""
        if orig_size is not None:
            return tuple(int(v) for v in orig_size)
        raise ValueError("GCSLaneHead requires orig_size=(H, W) for auxiliary mask/edge outputs.")

    def _attach_aux_outputs(self, out, p2, orig_size=None):
        """Attach auxiliary mask and edge outputs when requested."""
        if not (self.aux and (self.training or self.return_aux)):
            return out
        aux_size = self.aux_output_size(orig_size=orig_size)
        out["aux_mask_logits"] = F.interpolate(
            self.aux_mask(p2),
            size=aux_size,
            mode="bilinear",
            align_corners=False,
        )
        out["aux_edge_logits"] = F.interpolate(
            self.aux_edge(p2),
            size=aux_size,
            mode="bilinear",
            align_corners=False,
        )
        return out

    def profile_flops(self, xs):
        """Estimate inference GFLOPs for the query decoder and prediction MLPs."""
        if len(xs) != 4:
            raise ValueError(f"GCSLaneHead expects [P2, P3, P4, P5], got {len(xs)} feature maps.")

        b = int(xs[0].shape[0])
        q = int(self.num_queries)
        d = int(self.c1)
        k = int(self.num_points)
        point_dims = int(getattr(self, "point_dims", 2))
        tokens = int(sum(x.shape[-2] * x.shape[-1] for x in xs))
        layers = len(self.decoder.layers)
        ff_dim = d * 4

        # MACs for PyTorch TransformerDecoderLayer in inference mode:
        # self-attention, cross-attention, feed-forward network, then point/existence MLP heads.
        self_attn_macs = q * (4 * d * d) + 2 * q * q * d
        cross_attn_macs = (2 * q + 2 * tokens) * d * d + 2 * q * tokens * d
        ffn_macs = 2 * q * d * ff_dim
        decoder_macs = b * layers * (self_attn_macs + cross_attn_macs + ffn_macs)
        point_mlp_macs = b * q * (2 * d * d + d * (k * point_dims))
        point_valid_mlp_macs = b * q * (d * d + d * k)
        sample_macs = b * q * k * min(3, len(xs)) * d * 4
        coord_mlp_macs = b * q * k * (2 * d + d * d)
        refine_mlp_macs = b * q * k * ((2 * d) * d + d)
        valid_refine_mlp_macs = b * q * k * ((2 * d) * d + d)
        exist_mlp_macs = b * q * (d * d + d)
        return (
            2.0
            * (
                decoder_macs
                + point_mlp_macs
                + point_valid_mlp_macs
                + 2 * sample_macs
                + 2 * coord_mlp_macs
                + refine_mlp_macs
                + valid_refine_mlp_macs
                + exist_mlp_macs
            )
            / 1e9
        )

    def flatten_features(self, xs):
        """Flatten P2-P5 feature maps to spatial tokens with position and level embeddings."""
        if len(xs) != 4:
            raise ValueError(f"GCSLaneHead expects [P2, P3, P4, P5], got {len(xs)} feature maps.")

        tokens = []
        feature_shapes = []
        tokens_per_level = []
        for level, x in enumerate(xs):
            b, c, h, w = x.shape
            if c != self.c1:
                raise ValueError(f"GCSLaneHead expected {self.c1} channels at level {level}, got {c}.")
            if h <= 1 or w <= 1:
                raise ValueError(
                    f"GCSLaneHead received collapsed level {level} feature map {tuple(x.shape)}. "
                    "Structured lane prediction requires spatial P2-P5 tokens, not 1x1 global pooled features."
                )

            token = x.flatten(2).transpose(1, 2)
            pos = build_2d_sincos_position_embedding(h, w, c, x.device).to(dtype=token.dtype)
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            level_pos = self.level_embed[level].view(1, 1, c).to(dtype=token.dtype)
            tokens.append(token + pos + level_pos)
            feature_shapes.append((int(b), int(c), int(h), int(w)))
            tokens_per_level.append(int(h * w))

        memory = torch.cat(tokens, dim=1)
        min_tokens = int(getattr(self, "min_spatial_tokens", 1024) or 0)
        if min_tokens > 0 and memory.shape[1] < min_tokens:
            raise ValueError(
                f"GCSLaneHead has too few spatial tokens: {tuple(memory.shape)} from {feature_shapes}. "
                "Check that P2/P3/P4/P5 were not globally pooled before the lane head."
            )
        self._last_spatial_debug = {
            "feature_shapes": feature_shapes,
            "tokens_per_level": tokens_per_level,
            "memory_shape": (int(memory.shape[0]), int(memory.shape[1]), int(memory.shape[2])),
        }
        return memory

    def forward(self, xs, orig_size=None):
        """Predict normalized lane point sequences and existence logits."""
        p2, _, _, _ = xs
        b = p2.shape[0]

        if self.lane_instance_set_decoder_head:
            lane_instance_outputs = self._lane_instance_set_outputs(xs)
            out = {
                "pred_points": lane_instance_outputs["pred_lane_instance_points"],
                "pred_logits": lane_instance_outputs["pred_lane_instance_survival_logits"],
                "pred_valid_logits": lane_instance_outputs["pred_lane_instance_valid_logits"],
            }
            out.update(lane_instance_outputs)
            return self._attach_aux_outputs(out, p2, orig_size=orig_size)

        memory = self.flatten_features(xs)
        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(tgt=query, memory=memory)

        point_dims = int(getattr(self, "point_dims", 2))
        point_delta = self.point_mlp(hs).view(b, self.num_queries, self.num_points, point_dims)
        point_ref = getattr(self, "point_reference_logits", None)
        point_mode = getattr(self, "point_mode", "free")
        if point_mode == "fixed_y":
            if point_ref is None:
                x_logits = point_delta.squeeze(-1)
            else:
                point_ref = point_ref.to(device=point_delta.device, dtype=point_delta.dtype).unsqueeze(0)
                x_logits = point_delta.squeeze(-1) + point_ref
            fixed_y = getattr(self, "fixed_y_anchors", None)
            if fixed_y is None:
                fixed_y = self._build_fixed_y_anchors()
            x_logits = self._refine_fixed_y_logits(xs, hs, x_logits, fixed_y)
            pred_x = torch.sigmoid(x_logits)
            y = fixed_y.to(device=pred_x.device, dtype=pred_x.dtype).view(1, 1, self.num_points)
            pred_y = y.expand(b, self.num_queries, -1)
            pred_points = torch.stack((pred_x, pred_y), dim=-1)
        elif point_ref is None:
            # Backward compatibility for checkpoints created before query-specific references existed.
            pred_points = torch.sigmoid(point_delta)
        else:
            point_ref = point_ref.to(device=point_delta.device, dtype=point_delta.dtype).unsqueeze(0)
            pred_points = torch.sigmoid(point_delta + point_ref)
        base_logits = self.exist_mlp(hs).squeeze(-1)
        if hasattr(self, "point_valid_mlp"):
            if point_mode == "fixed_y" and hasattr(self, "point_valid_refine_mlp"):
                pred_valid_logits = self._refine_fixed_y_valid_logits(xs, hs, pred_points)
            else:
                pred_valid_logits = self.point_valid_mlp(hs).view(b, self.num_queries, self.num_points)
        else:
            pred_valid_logits = base_logits.new_full((b, self.num_queries, self.num_points), 20.0)

        pred_logits = base_logits
        survival_delta_logits = None
        survival_point_delta = None
        survival_valid_delta = None
        interval_offsets = None
        interval_base_bounds = None
        interval_bounds = None
        base_points = pred_points
        base_valid_logits = pred_valid_logits
        if getattr(self, "query_survival_head", False):
            survival_features = torch.cat(
                (
                    hs,
                    pred_points[..., 0],
                    pred_valid_logits.sigmoid(),
                    base_logits.sigmoid().unsqueeze(-1),
                ),
                dim=-1,
            )
            survival_tokens = self.query_survival_state_mlp(survival_features)
            survival_context, _ = self.query_survival_attention(
                survival_tokens,
                survival_tokens,
                survival_tokens,
                need_weights=False,
            )
            survival_tokens = self.query_survival_norm(survival_tokens + survival_context)
            survival_delta_logits = self.query_survival_delta_mlp(survival_tokens).squeeze(-1)
            pred_logits = base_logits + survival_delta_logits
            if getattr(self, "query_survival_geometry_adapter", False):
                survival_point_delta = 0.1 * torch.tanh(self.query_survival_point_delta_mlp(survival_tokens))
                survival_valid_delta = self.query_survival_valid_delta_mlp(survival_tokens)
                pred_x = (base_points[..., 0] + survival_point_delta).clamp(0.0, 1.0)
                pred_points = torch.stack((pred_x, base_points[..., 1]), dim=-1)
                pred_valid_logits = base_valid_logits + survival_valid_delta
        if getattr(self, "query_valid_local_adapter", False):
            local_valid_tokens = self._point_refine_tokens(xs, hs, pred_points.detach())
            survival_valid_delta = self.query_valid_local_delta_mlp(local_valid_tokens).squeeze(-1)
            pred_valid_logits = base_valid_logits + survival_valid_delta
        if getattr(self, "query_valid_interval_adapter", False):
            interval_tokens = self._point_refine_tokens(xs, hs, pred_points.detach())
            survival_valid_delta, interval_offsets, interval_base_bounds, interval_bounds = (
                self._query_valid_interval_delta(base_valid_logits, interval_tokens)
            )
            pred_valid_logits = base_valid_logits + survival_valid_delta
        out = {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
            "pred_valid_logits": pred_valid_logits,
        }
        if survival_delta_logits is not None:
            out["pred_base_logits"] = base_logits
            out["pred_survival_delta_logits"] = survival_delta_logits
        if survival_point_delta is not None or survival_valid_delta is not None:
            out["pred_base_valid_logits"] = base_valid_logits
            out["pred_survival_valid_delta"] = survival_valid_delta
        if survival_point_delta is not None:
            out["pred_base_points"] = base_points
            out["pred_survival_point_delta"] = survival_point_delta
        if interval_offsets is not None:
            out["pred_valid_interval_offsets"] = interval_offsets
            out["pred_valid_interval_base_bounds"] = interval_base_bounds
            out["pred_valid_interval_bounds"] = interval_bounds
        if getattr(self, "query_count_head", False):
            out["pred_count_logits"] = self.query_count_mlp(hs.mean(dim=1))
        if getattr(self, "dense_instance_head", False):
            dense_features = self.dense_instance_stem(p2)
            out["pred_dense_centerline_logits"] = self.dense_instance_centerline(dense_features)
            out["pred_dense_endpoint_logits"] = self.dense_instance_endpoint(dense_features)
            out["pred_dense_instance_embed"] = self.dense_instance_embed(dense_features)
            if getattr(self, "dense_endpoint_offset_head", False):
                out["pred_dense_endpoint_offsets"] = self.dense_instance_endpoint_offset(dense_features)
            if getattr(self, "dense_candidate_head", False):
                out["pred_dense_candidate_quality_logits"] = self.dense_instance_candidate_quality(dense_features)
                out["pred_dense_candidate_replace_logits"] = self.dense_instance_candidate_replace(dense_features)
        if getattr(self, "residual_proposal_head", False):
            out.update(self._residual_proposal_outputs(p2, pred_points, pred_logits, pred_valid_logits, hs))
        if self.gcs_mode == "ordered_slot":
            out["pred_exist_logits"] = pred_logits
            out["pred_start_logits"] = self.start_mlp(hs).view(b, self.num_queries, self.num_points)
            out["pred_end_logits"] = self.end_mlp(hs).view(b, self.num_queries, self.num_points)
            out["pred_count_logits"] = self.count_mlp(hs.mean(dim=1))

        return self._attach_aux_outputs(out, p2, orig_size=orig_size)


class LaneBiFPN(nn.Module):
    """Lane-aware bidirectional feature pyramid for P2, P3, P4, and P5 features."""

    def __init__(self, channels, out_channels=128):
        """Initialize Lane-BiFPN with input channels [P2, P3, P4, P5]."""
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"LaneBiFPN expects 4 input channel values for P2-P5, got {channels}.")

        c2, c3, c4, c5 = channels

        self.p2_in = ConvBNAct(c2, out_channels, k=1, p=0)
        self.p3_in = ConvBNAct(c3, out_channels, k=1, p=0)
        self.p4_in = ConvBNAct(c4, out_channels, k=1, p=0)
        self.p5_in = ConvBNAct(c5, out_channels, k=1, p=0)

        self.fuse_p4_td = WeightedFusion(2)
        self.fuse_p3_td = WeightedFusion(2)
        self.fuse_p2_td = WeightedFusion(2)

        self.fuse_p3_out = WeightedFusion(3)
        self.fuse_p4_out = WeightedFusion(3)
        self.fuse_p5_out = WeightedFusion(2)

        self.p4_td_conv = ConvBNAct(out_channels, out_channels)
        self.p3_td_conv = ConvBNAct(out_channels, out_channels)
        self.p2_td_conv = ConvBNAct(out_channels, out_channels)

        self.p3_out_conv = ConvBNAct(out_channels, out_channels)
        self.p4_out_conv = ConvBNAct(out_channels, out_channels)
        self.p5_out_conv = ConvBNAct(out_channels, out_channels)

    @staticmethod
    def _downsample_to(x, size):
        """Downsample by BiFPN max-pooling, then align odd/non-standard sizes if needed."""
        if x.shape[-2] >= 2 and x.shape[-1] >= 2:
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="nearest")
        return x

    def forward(self, xs):
        """Fuse P2-P5 features bidirectionally and return four aligned feature maps."""
        if len(xs) != 4:
            raise ValueError(f"LaneBiFPN expects [P2, P3, P4, P5], got {len(xs)} feature maps.")

        p2, p3, p4, p5 = xs

        p2 = self.p2_in(p2)
        p3 = self.p3_in(p3)
        p4 = self.p4_in(p4)
        p5 = self.p5_in(p5)

        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode="nearest")
        p4_td = self.p4_td_conv(self.fuse_p4_td([p4, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode="nearest")
        p3_td = self.p3_td_conv(self.fuse_p3_td([p3, p4_up]))

        p3_up = F.interpolate(p3_td, size=p2.shape[-2:], mode="nearest")
        p2_td = self.p2_td_conv(self.fuse_p2_td([p2, p3_up]))

        p2_down = self._downsample_to(p2_td, p3.shape[-2:])
        p3_out = self.p3_out_conv(self.fuse_p3_out([p3, p3_td, p2_down]))

        p3_down = self._downsample_to(p3_out, p4.shape[-2:])
        p4_out = self.p4_out_conv(self.fuse_p4_out([p4, p4_td, p3_down]))

        p4_down = self._downsample_to(p4_out, p5.shape[-2:])
        p5_out = self.p5_out_conv(self.fuse_p5_out([p5, p4_down]))

        return [p2_td, p3_out, p4_out, p5_out]

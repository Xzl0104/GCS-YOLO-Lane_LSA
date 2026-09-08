# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""GCS-YOLO-Lane neural network modules."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        fixed_y_original_h=720,
        fixed_y_start_px=710,
        fixed_y_end_px=160,
        geometry_aware_exist: bool = False,
        two_stage_refine: bool = False,
        gated_multiscale: bool = False,
        proposal_state_refine: bool = False,
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
            self.num_slots = int(num_slots)
            if self.num_slots <= 0:
                raise ValueError(f"GCSLaneHead num_slots must be positive, got {num_slots}.")
            self.min_lanes = int(min_lanes)
            self.max_lanes = int(max_lanes)
            if self.min_lanes < 0 or self.max_lanes < self.min_lanes:
                raise ValueError(
                    f"GCSLaneHead query lane bounds must satisfy 0 <= min_lanes <= max_lanes, "
                    f"got {min_lanes}/{max_lanes}."
                )
            if self.max_lanes > int(num_queries):
                raise ValueError(
                    f"GCSLaneHead query max_lanes={self.max_lanes} exceeds num_queries={num_queries}."
                )
            expected_count_classes = self.max_lanes - self.min_lanes + 1
            self.count_classes = int(count_classes) if count_classes is not None else expected_count_classes
            if self.count_classes != expected_count_classes:
                raise ValueError(
                    f"GCSLaneHead query count_classes must be max_lanes-min_lanes+1={expected_count_classes}, "
                    f"got {self.count_classes}."
                )
        self.num_queries = int(num_queries)
        self.num_points = int(num_points)
        self.aux = aux
        self.point_mode = str(point_mode).lower()
        if self.point_mode in {"fixed-y", "fixedy"}:
            self.point_mode = "fixed_y"
        if self.point_mode not in {"free", "fixed_y"}:
            raise ValueError(f"GCSLaneHead point_mode must be 'free' or 'fixed_y', got {point_mode!r}.")
        if proposal_state_refine and not two_stage_refine:
            raise ValueError(
                "GCSLaneHead proposal_state_refine requires two_stage_refine=True "
                "so the updated proposal state can drive a second prediction stage."
            )
        if proposal_state_refine and self.point_mode != "fixed_y":
            raise ValueError(
                "GCSLaneHead proposal_state_refine currently requires point_mode='fixed_y'."
            )
        self.fixed_y_start = float(fixed_y_start)
        self.fixed_y_end = float(fixed_y_end)
        self.fixed_y_original_h = int(fixed_y_original_h)
        self.fixed_y_start_px = float(fixed_y_start_px)
        self.fixed_y_end_px = float(fixed_y_end_px)
        self.geometry_aware_exist = bool(geometry_aware_exist)
        self.two_stage_refine = bool(two_stage_refine)
        self.gated_multiscale = bool(gated_multiscale)
        self.proposal_state_refine = bool(proposal_state_refine)
        if not (0.0 <= self.fixed_y_end < self.fixed_y_start <= 1.0):
            raise ValueError(
                f"Expected 0 <= fixed_y_end < fixed_y_start <= 1, got "
                f"{self.fixed_y_end} < {self.fixed_y_start}."
            )
        if self.fixed_y_original_h <= 1:
            raise ValueError(f"GCSLaneHead fixed_y_original_h must be > 1, got {fixed_y_original_h}.")
        if not (0.0 <= self.fixed_y_end_px < self.fixed_y_start_px < self.fixed_y_original_h):
            raise ValueError(
                "GCSLaneHead fixed-y pixel anchors must satisfy "
                f"0 <= end < start < original_h, got {self.fixed_y_end_px}, "
                f"{self.fixed_y_start_px}, {self.fixed_y_original_h}."
            )
        normalized_start = self.fixed_y_start_px / float(self.fixed_y_original_h)
        normalized_end = self.fixed_y_end_px / float(self.fixed_y_original_h)
        if abs(self.fixed_y_start - normalized_start) > 1e-6 or abs(self.fixed_y_end - normalized_end) > 1e-6:
            raise ValueError(
                "GCSLaneHead fixed-y normalized and pixel contracts disagree: "
                f"normalized=({self.fixed_y_start}, {self.fixed_y_end}), "
                f"pixel=({self.fixed_y_start_px}, {self.fixed_y_end_px})/{self.fixed_y_original_h}."
            )
        if self.point_mode == "fixed_y":
            self._validate_fixed_y_anchors()
        self.point_dims = 1 if self.point_mode == "fixed_y" else 2
        self.return_aux = False
        self.min_spatial_tokens = 1024
        self._last_spatial_debug = None

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
        if self.two_stage_refine:
            self.point_refine_mlp_stage2 = nn.Sequential(
                nn.Linear(c1 * 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 1),
            )
        if self.proposal_state_refine:
            self.proposal_state_update_mlp = nn.Sequential(
                nn.Linear(c1 * 4 + 9, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, c1),
            )
            self.proposal_state_delta_norm = nn.LayerNorm(c1)
        if self.gated_multiscale:
            self.multiscale_gate = nn.Sequential(
                nn.Linear(c1 * 3 + 2, c1),
                nn.ReLU(inplace=True),
                nn.Linear(c1, 3),
            )
        self.exist_mlp = nn.Sequential(
            nn.Linear(c1, c1),
            nn.ReLU(inplace=True),
            nn.Linear(c1, 1),
        )
        if self.geometry_aware_exist:
            self.geometry_exist_delta_mlp = nn.Sequential(
                nn.Linear(c1 * 4 + 9, c1),
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
        self._init_point_delta_head()
        self._init_point_valid_head()
        self._init_point_refine_head()
        self._init_point_valid_refine_head()
        if self.two_stage_refine:
            self._init_point_refine_stage2_head()
        if self.proposal_state_refine:
            self._init_proposal_state_update()
        if self.gated_multiscale:
            self._init_multiscale_gate()
        if self.geometry_aware_exist:
            self._init_geometry_exist_delta_head()
        if self.gcs_mode == "ordered_slot":
            self._init_interval_heads()
        if self.query_count_head:
            self._init_query_count_head()

    def _build_fixed_y_anchors(self):
        """Build shared bottom-to-top y anchors for fixed-y x-only prediction."""
        return torch.linspace(float(self.fixed_y_start), float(self.fixed_y_end), self.num_points)

    def _validate_fixed_y_anchors(self) -> None:
        """Fail fast unless fixed-y anchors match this head's explicit contract."""
        anchors = self._build_fixed_y_anchors().detach().float().reshape(-1)
        if anchors.numel() != self.num_points:
            raise ValueError(
                f"GCSLaneHead fixed_y_anchors: K mismatch, got {anchors.numel()}, expected {self.num_points}."
            )
        anchors_px = anchors * float(self.fixed_y_original_h)
        expected_desc = torch.linspace(
            float(self.fixed_y_start_px),
            float(self.fixed_y_end_px),
            self.num_points,
            dtype=anchors_px.dtype,
            device=anchors_px.device,
        )
        if not torch.allclose(anchors_px, expected_desc, atol=1e-3, rtol=0.0):
            raise ValueError(
                "GCSLaneHead fixed_y_anchors mismatch with the configured explicit pixel contract: "
                f"expected {self.fixed_y_start_px:g}..{self.fixed_y_end_px:g} "
                f"for original_h={self.fixed_y_original_h}, K={self.num_points}."
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

    def _init_point_refine_stage2_head(self):
        """Initialize the second geometry refinement stage as a no-op residual."""
        final = self.point_refine_mlp_stage2[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _init_proposal_state_update(self):
        """Initialize proposal state refinement as an exact identity residual."""
        final = self.proposal_state_update_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _init_multiscale_gate(self):
        """Start gated P2/P3/P4 fusion from the existing equal-weight behavior."""
        final = self.multiscale_gate[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _init_geometry_exist_delta_head(self):
        """Start geometry-aware existence as a zero residual over the baseline score."""
        final = self.geometry_exist_delta_mlp[-1]
        nn.init.zeros_(final.weight)
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
        sampled = torch.stack(sampled, dim=3)
        if not self.gated_multiscale:
            return sampled.mean(dim=3)
        if sampled.shape[3] != 3:
            raise ValueError(
                "GCSLaneHead gated multi-scale refinement requires exactly P2/P3/P4 features."
            )
        gate_input = torch.cat(
            (
                sampled.reshape(b, q, k, self.c1 * 3),
                points.to(device=sampled.device, dtype=sampled.dtype).clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        gate = torch.softmax(self.multiscale_gate(gate_input), dim=-1)
        return (sampled * gate.unsqueeze(-1)).sum(dim=3)

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

    def _refine_fixed_y_logits_once(self, xs, hs, logits, fixed_y, refine_head):
        """Apply one fixed-y point refinement stage using the supplied proposal state."""
        b, q, k = logits.shape
        y = fixed_y.to(device=logits.device, dtype=logits.dtype).view(1, 1, k).expand(b, q, -1)
        current_x = torch.sigmoid(logits)
        current_points = torch.stack((current_x, y), dim=-1)
        refine_tokens = self._point_refine_tokens(xs, hs, current_points)
        refine_delta = refine_head(refine_tokens).squeeze(-1)
        return logits + refine_delta

    def _update_proposal_state(self, xs, hs, stage1_points):
        """Update each query state from its stage-1 geometry and sampled image context."""
        point_features = self._sample_point_features(xs, stage1_points)
        coarse_valid = self.point_valid_mlp(hs).view(hs.shape[0], self.num_queries, self.num_points)
        valid_prob = torch.sigmoid(coarse_valid)
        x = stage1_points[..., 0]
        dx = x[..., 1:] - x[..., :-1]
        curvature = dx[..., 1:] - dx[..., :-1]
        b, q, _, _ = stage1_points.shape
        bottom_count = min(3, self.num_points)
        geometry_summary = torch.stack(
            (
                x[:, :, 0],
                x[:, :, -1],
                x.mean(dim=2),
                x.std(dim=2, unbiased=False),
                dx.abs().mean(dim=2) if dx.shape[-1] > 0 else x.new_zeros((b, q)),
                curvature.abs().mean(dim=2) if curvature.shape[-1] > 0 else x.new_zeros((b, q)),
            ),
            dim=-1,
        )
        valid_summary = torch.stack(
            (
                valid_prob.mean(dim=2),
                valid_prob.amax(dim=2),
                valid_prob.std(dim=2, unbiased=False),
            ),
            dim=-1,
        )
        proposal_context = torch.cat(
            (
                hs,
                point_features.mean(dim=2),
                point_features.amax(dim=2),
                point_features[:, :, :bottom_count].mean(dim=2),
                valid_summary,
                geometry_summary,
            ),
            dim=-1,
        )
        delta = self.proposal_state_update_mlp(proposal_context)
        return hs + self.proposal_state_delta_norm(delta)

    def _refine_fixed_y_logits(self, xs, hs, coarse_logits, fixed_y):
        """Run stage-wise fixed-y refinement and return final logits plus final query state."""
        fixed_y = fixed_y.to(device=coarse_logits.device, dtype=coarse_logits.dtype)
        stage1_logits = self._refine_fixed_y_logits_once(
            xs, hs, coarse_logits, fixed_y, self.point_refine_mlp
        )
        proposal_hs = hs
        if self.proposal_state_refine:
            b, q, k = stage1_logits.shape
            y = fixed_y.view(1, 1, k).expand(b, q, -1)
            stage1_points = torch.stack((torch.sigmoid(stage1_logits), y), dim=-1)
            proposal_hs = self._update_proposal_state(xs, hs, stage1_points)
        if not self.two_stage_refine:
            return stage1_logits, proposal_hs
        stage2_logits = self._refine_fixed_y_logits_once(
            xs, proposal_hs, stage1_logits, fixed_y, self.point_refine_mlp_stage2
        )
        return stage2_logits, proposal_hs

    def _refine_fixed_y_valid_logits(self, xs, hs, pred_points):
        """Refine fixed-y point visibility logits with point-level sampled image features."""
        coarse_valid = self.point_valid_mlp(hs).view(hs.shape[0], self.num_queries, self.num_points)
        refine_tokens = self._point_refine_tokens(xs, hs, pred_points.detach())
        valid_delta = self.point_valid_refine_mlp(refine_tokens).squeeze(-1)
        return coarse_valid + valid_delta

    def aux_output_size(self, orig_size=None):
        """Return auxiliary supervision size from the explicit original input image size."""
        if orig_size is not None:
            return tuple(int(v) for v in orig_size)
        raise ValueError("GCSLaneHead requires orig_size=(H, W) for auxiliary mask/edge outputs.")

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
        refine_passes = 1 + int(self.two_stage_refine)
        sample_calls = refine_passes + 1 + int(self.geometry_aware_exist)
        coord_calls = refine_passes + 1
        geometry_exist_macs = 0
        gate_macs = 0
        proposal_state_macs = 0
        if self.geometry_aware_exist:
            geometry_exist_macs = b * q * ((4 * d + 9) * d + d)
        if self.gated_multiscale:
            gate_macs = sample_calls * b * q * k * ((3 * d + 2) * d + 3 * d)
        if self.proposal_state_refine:
            sample_calls += 1
            proposal_state_macs = b * q * ((4 * d + 9) * d + d)
        return (
            2.0
            * (
                decoder_macs
                + point_mlp_macs
                + point_valid_mlp_macs
                + sample_calls * sample_macs
                + coord_calls * coord_mlp_macs
                + refine_passes * refine_mlp_macs
                + valid_refine_mlp_macs
                + exist_mlp_macs
                + geometry_exist_macs
                + gate_macs
                + proposal_state_macs
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

        memory = self.flatten_features(xs)
        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(tgt=query, memory=memory)
        prediction_hs = hs

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
            x_logits, prediction_hs = self._refine_fixed_y_logits(xs, hs, x_logits, fixed_y)
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
        base_pred_logits = self.exist_mlp(prediction_hs).squeeze(-1)
        if hasattr(self, "point_valid_mlp"):
            if point_mode == "fixed_y" and hasattr(self, "point_valid_refine_mlp"):
                pred_valid_logits = self._refine_fixed_y_valid_logits(xs, prediction_hs, pred_points)
            else:
                pred_valid_logits = self.point_valid_mlp(prediction_hs).view(b, self.num_queries, self.num_points)
        else:
            pred_valid_logits = base_pred_logits.new_full((b, self.num_queries, self.num_points), 20.0)

        if self.geometry_aware_exist:
            point_features = self._sample_point_features(xs, pred_points)
            valid_prob = torch.sigmoid(pred_valid_logits)
            x = pred_points[..., 0]
            dx = x[..., 1:] - x[..., :-1]
            curvature = dx[..., 1:] - dx[..., :-1]
            point_feature_mean = point_features.mean(dim=2)
            point_feature_max = point_features.amax(dim=2)
            bottom_count = min(3, self.num_points)
            bottom_feature = point_features[:, :, :bottom_count].mean(dim=2)
            geometry_summary = torch.stack(
                (
                    x[:, :, 0],
                    x[:, :, -1],
                    x.mean(dim=2),
                    x.std(dim=2, unbiased=False),
                    dx.abs().mean(dim=2),
                    curvature.abs().mean(dim=2) if curvature.shape[-1] > 0 else x.new_zeros((b, self.num_queries)),
                ),
                dim=-1,
            )
            valid_summary = torch.stack(
                (
                    valid_prob.mean(dim=2),
                    valid_prob.amax(dim=2),
                    valid_prob.std(dim=2, unbiased=False),
                ),
                dim=-1,
            )
            exist_features = torch.cat(
                (
                    prediction_hs,
                    point_feature_mean,
                    point_feature_max,
                    bottom_feature,
                    valid_summary,
                    geometry_summary,
                ),
                dim=-1,
            )
            pred_logits = base_pred_logits + self.geometry_exist_delta_mlp(exist_features).squeeze(-1)
        else:
            pred_logits = base_pred_logits

        out = {
            "pred_points": pred_points,
            "pred_logits": pred_logits,
            "pred_valid_logits": pred_valid_logits,
        }
        if getattr(self, "query_count_head", False):
            out["pred_count_logits"] = self.query_count_mlp(prediction_hs.mean(dim=1))
        if self.gcs_mode == "ordered_slot":
            out["pred_exist_logits"] = pred_logits
            out["pred_start_logits"] = self.start_mlp(prediction_hs).view(b, self.num_queries, self.num_points)
            out["pred_end_logits"] = self.end_mlp(prediction_hs).view(b, self.num_queries, self.num_points)
            out["pred_count_logits"] = self.count_mlp(prediction_hs.mean(dim=1))

        if self.aux and (self.training or self.return_aux):
            aux_size = self.aux_output_size(orig_size=orig_size)
            # Keep the dense auxiliary branch in FP32 under AMP. Its BatchNorm
            # statistics are sensitive to large P2 activations and do not affect
            # the structured lane output contract.
            force_fp32_aux = self.training and p2.device.type in {"cuda", "cpu"}
            aux_context = torch.autocast(device_type=p2.device.type, enabled=False) if force_fp32_aux else nullcontext()
            with aux_context:
                aux_input = p2.float() if force_fp32_aux else p2
                aux_mask_logits = self.aux_mask(aux_input)
                aux_mask_logits = F.interpolate(
                    aux_mask_logits,
                    size=aux_size,
                    mode="bilinear",
                    align_corners=False,
                )

                aux_edge_logits = self.aux_edge(aux_input)
                aux_edge_logits = F.interpolate(
                    aux_edge_logits,
                    size=aux_size,
                    mode="bilinear",
                    align_corners=False,
                )

            out["aux_mask_logits"] = aux_mask_logits
            out["aux_edge_logits"] = aux_edge_logits

        return out


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

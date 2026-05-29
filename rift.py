"""
RIFT: crack segmentation with task-aligned structural-directional modeling.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, dims=(-2, -1)) -> None:
        super().__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.87, dice_weight: float = 0.13) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(pred_logits, target)
        dice = self.dice(pred_logits.sigmoid(), target)
        return self.bce_weight * bce + self.dice_weight * dice


def _make_gn(num_channels: int, requested_groups: int = 8) -> nn.GroupNorm:
    groups = min(requested_groups, num_channels)
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        act: bool = True,
        gn_groups: int = 8,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = _make_gn(out_ch, gn_groups)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DropPath(nn.Module):
    """Drop paths per sample. Local implementation to avoid timm dependency."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class StructuralDirectionalBlock(nn.Module):
    """Local representation plus gated directional continuity."""

    def __init__(
        self,
        dim: int,
        expand_ratio: float = 2.0,
        kernel_size: int = 11,
        drop_path: float = 0.0,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(dim, int(round(dim * expand_ratio)))
        k = int(kernel_size)
        if k % 2 == 0:
            k += 1

        self.norm = _make_gn(dim, gn_groups)
        self.expand = nn.Conv2d(dim, hidden, kernel_size=1, bias=False)
        self.local = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False)

        self.h_conv = nn.Conv2d(hidden, hidden, kernel_size=(1, k), padding=(0, k // 2), groups=hidden, bias=False)
        self.v_conv = nn.Conv2d(hidden, hidden, kernel_size=(k, 1), padding=(k // 2, 0), groups=hidden, bias=False)
        dilation = max(1, k // 4)
        self.diag_conv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=hidden,
            bias=False,
        )

        self.gate = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.mix_norm = _make_gn(hidden, gn_groups)
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(hidden, dim, kernel_size=1, bias=False)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.expand(self.norm(x))
        local = self.local(z)
        line = self.h_conv(z) + self.v_conv(z) + self.diag_conv(z)
        gate = self.gate(local + line)
        out = local + gate * line
        out = self.project(self.act(self.mix_norm(out)))
        return residual + self.drop_path(out)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 8) -> None:
        super().__init__()
        self.proj = ConvGNAct(in_ch, out_ch, kernel_size=3, stride=2, padding=1, gn_groups=gn_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class RIFTEncoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        dims: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (2, 2, 3, 2),
        kernel_size: int = 11,
        expand_ratio: float = 2.0,
        drop_path_rate: float = 0.05,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("RIFT expects four stages for dims and depths.")

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, dims[0] // 2, kernel_size=3, stride=2, padding=1, gn_groups=gn_groups),
            ConvGNAct(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1, gn_groups=gn_groups),
        )

        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        cursor = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_idx, (dim, depth) in enumerate(zip(dims, depths)):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    StructuralDirectionalBlock(
                        dim=dim,
                        expand_ratio=expand_ratio,
                        kernel_size=kernel_size,
                        drop_path=dp_rates[cursor],
                        gn_groups=gn_groups,
                    )
                )
                cursor += 1
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx < len(dims) - 1:
                self.downsamples.append(Downsample(dim, dims[stage_idx + 1], gn_groups=gn_groups))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        outs: List[torch.Tensor] = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            outs.append(x)
            if idx < len(self.downsamples):
                x = self.downsamples[idx](x)
        return outs


class GatedFusion(nn.Module):
    def __init__(self, channels: int, gn_groups: int = 8) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ConvGNAct(channels, channels, kernel_size=3, padding=1, gn_groups=gn_groups)

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        gate = self.gate(torch.cat([high, low], dim=1))
        return self.refine(high + gate * low)


class RIFTDecoder(nn.Module):
    def __init__(
        self,
        dims: Sequence[int] = (32, 64, 128, 192),
        decoder_dim: int = 64,
        out_size: Tuple[int, int] = (512, 512),
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.out_size = out_size
        self.laterals = nn.ModuleList(
            [ConvGNAct(dim, decoder_dim, kernel_size=1, padding=0, gn_groups=gn_groups) for dim in dims]
        )
        self.fuse43 = GatedFusion(decoder_dim, gn_groups=gn_groups)
        self.fuse32 = GatedFusion(decoder_dim, gn_groups=gn_groups)
        self.fuse21 = GatedFusion(decoder_dim, gn_groups=gn_groups)
        self.head = nn.Sequential(
            ConvGNAct(decoder_dim, decoder_dim, kernel_size=3, padding=1, gn_groups=gn_groups),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x1, x2, x3, x4 = [lat(feat) for lat, feat in zip(self.laterals, features)]
        x = self.fuse43(x4, x3)
        x = self.fuse32(x, x2)
        x = self.fuse21(x, x1)
        x = self.head(x)
        return F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)


class RIFT(nn.Module):
    def __init__(
        self,
        dims: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (2, 2, 3, 2),
        kernel_size: int = 11,
        expand_ratio: float = 2.0,
        drop_path_rate: float = 0.05,
        decoder_dim: int = 64,
        out_size: Tuple[int, int] = (512, 512),
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = RIFTEncoder(
            dims=dims,
            depths=depths,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
            drop_path_rate=drop_path_rate,
            gn_groups=gn_groups,
        )
        self.decoder = RIFTDecoder(dims=dims, decoder_dim=decoder_dim, out_size=out_size, gn_groups=gn_groups)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def parse_int_list(value: str | Iterable[int], default: Sequence[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.strip() == "":
            return list(default)
        return [int(v.strip()) for v in value.split(",")]
    return [int(v) for v in value]


def build_model(args):
    dims = parse_int_list(args.dims, default=(32, 64, 128, 192))
    depths = parse_int_list(args.depths, default=(2, 2, 3, 2))
    model = RIFT(
        dims=dims,
        depths=depths,
        kernel_size=args.kernel_size,
        expand_ratio=args.expand_ratio,
        drop_path_rate=args.drop_path,
        decoder_dim=args.decoder_dim,
        out_size=(args.load_height, args.load_width),
        gn_groups=args.gn_groups,
    )
    criterion = BCEDiceLoss(bce_weight=args.BCELoss_ratio, dice_weight=args.DiceLoss_ratio)
    return model, criterion


__all__ = [
    "BCEDiceLoss",
    "DiceLoss",
    "RIFT",
    "RIFTEncoder",
    "RIFTDecoder",
    "StructuralDirectionalBlock",
    "build_model",
    "parse_int_list",
]

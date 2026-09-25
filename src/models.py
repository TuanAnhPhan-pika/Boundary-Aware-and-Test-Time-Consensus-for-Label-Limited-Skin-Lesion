"""Lightweight U-Nets, boundary objectives, and cached four-view TTA."""

import time

import torch
from torch import nn
from torch.nn import functional as F

class NumericalDivergenceError(RuntimeError):
    """Raised when optimization produces an invalid loss or gradient."""

    def __init__(self, message, value=None):
        self.value = value
        detail = message if value is None else f"{message}: {value}"
        super().__init__(detail)

def _group_count(channels):
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1

class ConvBlock(nn.Module):
    """Two depthwise-separable convolutions with group normalization."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = _group_count(out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)

class CompactUNetBCESoftDice(nn.Module):
    """Four-level compact U-Net with BCE plus soft-Dice training."""

    def __init__(self, widths=(16, 32, 64, 128), dice_epsilon=1e-6):
        super().__init__()
        self.widths = tuple(widths)
        self.dice_epsilon = dice_epsilon

        self.encoder1 = ConvBlock(3, widths[0])
        self.encoder2 = ConvBlock(widths[0], widths[1])
        self.encoder3 = ConvBlock(widths[1], widths[2])
        self.bottleneck = ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)

        self.decoder3 = ConvBlock(widths[3] + widths[2], widths[2])
        self.decoder2 = ConvBlock(widths[2] + widths[1], widths[1])
        self.decoder1 = ConvBlock(widths[1] + widths[0], widths[0])
        self.output_layer = nn.Conv2d(widths[0], 1, 1)

    def forward(self, images):
        encoder1 = self.encoder1(images)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = F.interpolate(
            bottleneck,
            size=encoder3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder3 = self.decoder3(
            torch.cat((decoder3, encoder3), dim=1)
        )

        decoder2 = F.interpolate(
            decoder3,
            size=encoder2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder2 = self.decoder2(
            torch.cat((decoder2, encoder2), dim=1)
        )

        decoder1 = F.interpolate(
            decoder2,
            size=encoder1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoder1 = self.decoder1(
            torch.cat((decoder1, encoder1), dim=1)
        )
        return self.output_layer(decoder1)

    def segmentation_loss(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum((1, 2, 3))
        denominator = (
            probabilities.sum((1, 2, 3))
            + targets.sum((1, 2, 3))
        )
        soft_dice = (
            2.0 * intersection + self.dice_epsilon
        ) / (denominator + self.dice_epsilon)
        return bce + (1.0 - soft_dice.mean())

    def train_step(
        self,
        batch,
        optimizer,
        scaler,
        device,
        gradient_clip_norm,
        epoch,
    ):
        del epoch
        images, masks, _, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = self.forward(images)
            loss = self.segmentation_loss(logits, masks)

        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss).item() or loss_value > 100.0:
            raise NumericalDivergenceError(
                "NaN/divergence detected", loss_value
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            self.parameters(), gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm).item():
            raise NumericalDivergenceError(
                "non-finite gradient", float(gradient_norm)
            )

        scaler.step(optimizer)
        scaler.update()
        return loss_value

    @torch.no_grad()
    def predict_proba(self, images, temperature=1.0):
        logits = torch.clamp(self.forward(images), -15.0, 15.0)
        return torch.sigmoid(logits / temperature)

    @torch.no_grad()
    def predict(self, images, threshold=0.5, temperature=1.0):
        return self.predict_proba(images, temperature) >= threshold

class UniformDistanceBoundaryUNet(CompactUNetBCESoftDice):
    """Compact U-Net with uniformly weighted signed-distance loss."""

    def __init__(
        self,
        widths=(16, 32, 64, 128),
        dice_epsilon=1e-6,
        lambda_boundary=0.1,
        boundary_warmup_epochs=10,
    ):
        super().__init__(widths, dice_epsilon)
        self.lambda_boundary = lambda_boundary
        self.boundary_warmup_epochs = boundary_warmup_epochs

    def train_step(
        self,
        batch,
        optimizer,
        scaler,
        device,
        gradient_clip_norm,
        epoch,
    ):
        images, masks, distances, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        distances = distances.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = self.forward(images)
            probabilities = torch.sigmoid(logits)
            base_loss = self.segmentation_loss(logits, masks)
            boundary_loss = torch.mean(probabilities * distances)
            warmup = min(
                1.0,
                float(epoch + 1) / self.boundary_warmup_epochs,
            )
            loss = (
                base_loss
                + warmup * self.lambda_boundary * boundary_loss
            )

        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss).item() or loss_value > 100.0:
            raise NumericalDivergenceError(
                "NaN/divergence detected", loss_value
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            self.parameters(), gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm).item():
            raise NumericalDivergenceError(
                "non-finite gradient", float(gradient_norm)
            )

        scaler.step(optimizer)
        scaler.update()
        return loss_value

class FourViewGeometry:
    """Identity, horizontal flip, vertical flip, and 180-degree rotation."""

    @staticmethod
    def transform_views(images):
        return (
            images,
            torch.flip(images, dims=(-1,)),
            torch.flip(images, dims=(-2,)),
            torch.rot90(images, 2, dims=(-2, -1)),
        )

    @staticmethod
    def inverse_align(values, view_index):
        if view_index == 0:
            return values
        if view_index == 1:
            return torch.flip(values, dims=(-1,))
        if view_index == 2:
            return torch.flip(values, dims=(-2,))
        if view_index == 3:
            return torch.rot90(values, 2, dims=(-2, -1))
        raise ValueError(f"unknown view index: {view_index}")

class MeanProbabilityFourViewTTA(FourViewGeometry):
    """Arithmetic probability mean over cached aligned logits."""

    def __init__(self, threshold=0.5, num_views=4):
        if num_views != 4:
            raise ValueError("this implementation requires four views")
        self.threshold = threshold
        self.num_views = num_views

    @torch.no_grad()
    def cache_aligned_logits(self, model, images):
        aligned = []
        for view_index, view in enumerate(
            self.transform_views(images)
        ):
            logits = torch.clamp(model(view), -15.0, 15.0)
            aligned.append(self.inverse_align(logits, view_index))
        return torch.stack(aligned, dim=1)

    @staticmethod
    def aggregate(aligned_logits, temperature=1.0):
        probabilities = torch.sigmoid(
            aligned_logits / temperature
        )
        return probabilities.mean(dim=1)

    def predict_proba(self, model, images, temperature=1.0):
        cached = self.cache_aligned_logits(model, images)
        return self.aggregate(cached, temperature)

    def benchmark_latency(
        self,
        model,
        example,
        temperature,
        warmup_iterations,
        timed_iterations,
    ):
        device = example.device
        timings = []
        for _ in range(warmup_iterations):
            self.predict_proba(model, example, temperature)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(timed_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.predict_proba(model, example, temperature)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - start))
        return timings

class UniformBoundaryMeanProbabilityFourViewArm(
    MeanProbabilityFourViewTTA
):
    """Mean TTA restricted to a frozen uniform-boundary checkpoint."""

    def __init__(self, uniform_boundary_model, threshold=0.5):
        super().__init__(threshold=threshold, num_views=4)
        self.model = uniform_boundary_model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def predict_proba(self, images, temperature=1.0):
        cached = self.cache_aligned_logits(self.model, images)
        return self.aggregate(cached, temperature)

class RobustLogitConsensusFourViewTTA(FourViewGeometry):
    """Trim the locally most discordant aligned logit per pixel."""

    def __init__(
        self,
        local_discordance_window=5,
        retained_views_per_pixel=3,
        logit_clip=15.0,
        threshold=0.5,
    ):
        if local_discordance_window % 2 != 1:
            raise ValueError("discordance window must be odd")
        if retained_views_per_pixel != 3:
            raise ValueError("four-view trimming must retain 3 views")
        self.window = local_discordance_window
        self.retained_views = retained_views_per_pixel
        self.logit_clip = logit_clip
        self.threshold = threshold

    def compute_local_discordance(self, aligned_logits):
        logits = aligned_logits.squeeze(2)
        median = logits.median(dim=1, keepdim=True).values
        deviations = torch.abs(logits - median)
        return F.avg_pool2d(
            deviations,
            kernel_size=self.window,
            stride=1,
            padding=self.window // 2,
        )

    def predict_proba(self, aligned_logits, temperature=1.0):
        logits = torch.clamp(
            aligned_logits,
            -self.logit_clip,
            self.logit_clip,
        )
        scores = self.compute_local_discordance(logits)
        rejected = scores.argmax(dim=1, keepdim=True)
        keep = torch.ones_like(scores, dtype=torch.bool)
        keep.scatter_(1, rejected, False)
        keep = keep.unsqueeze(2)

        retained_count = keep.sum(dim=1)
        if not torch.all(retained_count == self.retained_views):
            raise RuntimeError("robust-consensus invariant failed")

        consensus = (
            logits * keep.to(logits.dtype)
        ).sum(dim=1) / retained_count.to(logits.dtype)
        return torch.sigmoid(consensus / temperature)

    @staticmethod
    def predict_disagreement(aligned_logits, temperature=1.0):
        probabilities = torch.sigmoid(
            aligned_logits / temperature
        )
        return probabilities.var(dim=1, unbiased=False)

class UncertaintyGatedDistanceBoundaryUNet(
    CompactUNetBCESoftDice
):
    """Boundary supervision gated by frozen-reference TTA variance."""

    def __init__(
        self,
        frozen_reference,
        widths=(16, 32, 64, 128),
        dice_epsilon=1e-6,
        lambda_boundary=0.1,
        boundary_warmup_epochs=10,
        tau=0.01,
        variance_floor=1e-8,
    ):
        super().__init__(widths, dice_epsilon)
        if tau <= 0:
            raise ValueError("tau must be positive")

        self.frozen_reference = frozen_reference
        self.lambda_boundary = lambda_boundary
        self.boundary_warmup_epochs = boundary_warmup_epochs
        self.tau = tau
        self.variance_floor = variance_floor

        self.frozen_reference.eval()
        for parameter in self.frozen_reference.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.frozen_reference.eval()
        return self

    @torch.no_grad()
    def generate_frozen_tta_variance(self, images):
        tta = MeanProbabilityFourViewTTA()
        logits = tta.cache_aligned_logits(
            self.frozen_reference, images
        )
        probabilities = torch.sigmoid(logits)
        return probabilities.var(dim=1, unbiased=False)

    def compute_reliability_weights(self, variance):
        scaled = torch.clamp(
            (variance + self.variance_floor) / self.tau,
            min=0.0,
            max=50.0,
        )
        return torch.exp(-scaled)

    def train_step(
        self,
        batch,
        optimizer,
        scaler,
        device,
        gradient_clip_norm,
        epoch,
    ):
        images, masks, distances, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        distances = distances.to(device, non_blocking=True)

        with torch.no_grad():
            variance = self.generate_frozen_tta_variance(
                images
            ).detach()
            reliability = self.compute_reliability_weights(
                variance
            ).detach()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = self.forward(images)
            probabilities = torch.sigmoid(logits)
            base_loss = self.segmentation_loss(logits, masks)
            boundary_values = probabilities * distances
            gated_boundary_loss = (
                reliability * boundary_values
            ).sum() / reliability.sum().clamp_min(
                self.dice_epsilon
            )
            warmup = min(
                1.0,
                float(epoch + 1) / self.boundary_warmup_epochs,
            )
            loss = (
                base_loss
                + warmup
                * self.lambda_boundary
                * gated_boundary_loss
            )

        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss).item() or loss_value > 100.0:
            raise NumericalDivergenceError(
                "NaN/divergence detected", loss_value
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        trainable = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        gradient_norm = nn.utils.clip_grad_norm_(
            trainable, gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm).item():
            raise NumericalDivergenceError(
                "non-finite gradient", float(gradient_norm)
            )

        scaler.step(optimizer)
        scaler.update()
        return loss_value

class LatencyMatchedWiderUNet(CompactUNetBCESoftDice):
    """One-pass capacity control selected by validation-free timing."""

    def __init__(self, widths, dice_epsilon=1e-6):
        super().__init__(widths, dice_epsilon)
        self.selected_widths = tuple(widths)

    @torch.no_grad()
    def benchmark_latency(
        self,
        example,
        warmup_iterations,
        timed_iterations,
    ):
        self.eval()
        device = example.device
        timings = []

        for _ in range(warmup_iterations):
            self.predict_proba(example)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(timed_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.predict_proba(example)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(1000.0 * (time.perf_counter() - start))

        return timings
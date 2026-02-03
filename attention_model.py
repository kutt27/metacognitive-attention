"""
Itti-Koch Saliency Model: Bio-plausible Visual Attention
Implements center-surround attention, color opponency, and multi-scale saliency mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class CenterSurroundLayer(nn.Module):
    """Implements center-surround filtering using Difference of Gaussians (DoG)"""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        sigma_center: float = 0.5,
        sigma_surround: float = 3.0,
    ):
        super(CenterSurroundLayer, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Create coordinate grid
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        y = torch.arange(kernel_size) - (kernel_size - 1) / 2
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        dist_sq = grid_x**2 + grid_y**2

        # Calculate Gaussian kernels
        center = torch.exp(-dist_sq / (2 * sigma_center**2))
        surround = torch.exp(-dist_sq / (2 * sigma_surround**2))

        # Difference of Gaussians (On-Center / Off-Surround)
        dog_kernel = (center / center.sum()) - (surround / surround.sum())

        # Reshape for depthwise convolution
        dog_kernel = dog_kernel.view(1, 1, kernel_size, kernel_size)
        self.register_buffer("weight", dog_kernel.repeat(channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise center-surround convolution"""
        return F.conv2d(
            x, self.weight, padding=self.kernel_size // 2, groups=self.channels
        )


class AdaptiveCenterSurround(nn.Module):
    """Learnable center-surround layer with adaptive sigma parameters"""

    def __init__(self, channels: int, kernel_size: int = 15):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size

        # Learnable sigma parameters
        self.sigma_center = nn.Parameter(torch.tensor([1.0]))
        self.sigma_surround = nn.Parameter(torch.tensor([3.0]))

        # Pre-calculate coordinate grid
        x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        y = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        self.register_buffer("dist_sq", grid_x**2 + grid_y**2)

    def get_kernel(self) -> torch.Tensor:
        """Generate dynamic DoG kernel based on learnable parameters"""
        # Ensure valid sigma values
        s_c = torch.clamp(self.sigma_center, min=0.5)
        s_s = torch.clamp(self.sigma_surround, min=s_c.item() + 0.5)

        # Generate Gaussians
        center = torch.exp(-self.dist_sq / (2 * s_c**2))
        surround = torch.exp(-self.dist_sq / (2 * s_s**2))

        # Normalize and subtract
        dog_kernel = (center / center.sum()) - (surround / surround.sum())
        return dog_kernel.view(1, 1, self.kernel_size, self.kernel_size).repeat(
            self.channels, 1, 1, 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive center-surround convolution"""
        kernel = self.get_kernel()
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=self.channels)


class ColorOpponency(nn.Module):
    """Color opponency module implementing primate visual system pathways"""

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract intensity and color opponent channels (R/G, B/Y)"""
        # x shape: [B, 3, H, W]
        r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]

        # Intensity channel
        intensity = (r + g + b) / 3

        # Color opponency channels
        R = r - (g + b) / 2
        G = g - (r + b) / 2
        B = b - (r + g) / 2
        Y = (r + g) / 2 - torch.abs(r - g) / 2 - b

        # Opponent responses
        RG = torch.abs(R - G).unsqueeze(1)
        BY = torch.abs(B - Y).unsqueeze(1)

        return intensity.unsqueeze(1), RG, BY


class FullIttiKochModel(nn.Module):
    """Complete Itti-Koch visual saliency model with multi-scale processing"""

    def __init__(self, kernel_size: int = 15, num_scales: int = 4):
        super().__init__()
        self.color_engine = ColorOpponency()
        self.cs_layer = AdaptiveCenterSurround(channels=1, kernel_size=kernel_size)
        self.num_scales = num_scales

    def process_feature(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale center-surround processing to a feature map"""
        # Build Gaussian pyramid
        pyramid = [feature_map]
        for _ in range(self.num_scales - 1):
            pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2))

        # Apply center-surround and upscale back
        results = []
        for p in pyramid:
            out = self.cs_layer(p)
            results.append(
                F.interpolate(
                    out,
                    size=(feature_map.shape[2], feature_map.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        return torch.mean(torch.stack(results), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate saliency map from input image"""
        # Extract color opponent channels
        intensity, RG, BY = self.color_engine(x)

        # Process each pathway through multi-scale center-surround
        saliency_I = self.process_feature(intensity)
        saliency_RG = self.process_feature(RG)
        saliency_BY = self.process_feature(BY)

        # Combine saliency maps
        combined_saliency = (saliency_I + saliency_RG + saliency_BY) / 3
        return torch.abs(combined_saliency)


class FovealAttentionNet(nn.Module):
    """Attention-gated classifier combining saliency with CNN"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.saliency_extractor = FullIttiKochModel()

        # Simple CNN classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with attention gating"""
        # Generate attention mask
        mask = self.saliency_extractor(x)

        # Normalize mask
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Apply foveation (multiply by attention mask)
        gated_x = x * mask

        # Classify gated input
        logits = self.classifier(gated_x)
        return logits, gated_x, mask


def visualize_attention(
    image_path: str, model: FullIttiKochModel, device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate saliency visualization for an image

    Returns:
        overlay: RGB overlay of heatmap on original image
        heatmap: Colored heatmap
        gray_saliency: Grayscale saliency map
    """
    from PIL import Image
    import torchvision.transforms as T

    # Load and preprocess image
    raw_img = Image.open(image_path).convert("RGB")
    original_size = raw_img.size

    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    # Generate saliency
    model.eval()
    with torch.no_grad():
        saliency = model(input_tensor)

    # Process saliency map
    s_map = saliency[0, 0].cpu().numpy()
    s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)

    # Resize to original dimensions
    s_map_resized = cv2.resize(s_map, original_size)
    gray_saliency = (s_map_resized * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(gray_saliency, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Create overlay
    raw_img_cv = cv2.cvtColor(np.array(raw_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(raw_img_cv, 0.6, heatmap, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay, heatmap, gray_saliency


def get_model(device: str = "cpu") -> FullIttiKochModel:
    """Get initialized model on specified device"""
    model = FullIttiKochModel()
    model.to(device)
    model.eval()
    return model

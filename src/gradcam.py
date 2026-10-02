"""
gradcam.py - Grad-CAM (Selvaraju et al., 2017).

Recipe:
 1. forward pass, remember the feature maps A (C x h x w) of the last conv layer
 2. backward pass from the score of the chosen class, remember dScore/dA
 3. channel weight = average of the gradient over space
 4. heatmap = ReLU( sum_c weight_c * A_c ), scaled to 0..1, resized to the image

IMPORTANT (say this in the viva): it shows regions that INFLUENCED the score of
that class - it is a visual hint, NOT proof of how the model "reasons". Models
can look at the right region for wrong reasons (e.g. background).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing import get_eval_transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def remove(self):
        self._handle.remove()

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) normalised tensor. Returns (cam[h,w] in 0..1 at input size, class_idx, probs)."""
        x = x.clone().requires_grad_(True)       # guarantees a graph even if weights are frozen
        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach()
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam[None, None], size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        return cam.cpu().numpy(), class_idx, probs


def overlay_heatmap(pil_image, cam, alpha=0.45):
    """Blend the JET-coloured heatmap onto the ORIGINAL photo. Returns RGB uint8 array."""
    img = np.array(pil_image.convert("RGB"))
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return np.uint8((1 - alpha) * img + alpha * heat)


def explain(model, target_layer, pil_image, img_size, device="cpu", class_idx=None):
    """Convenience: PIL image -> (overlay RGB array, class_idx, probs)."""
    x = get_eval_transform(img_size)(pil_image).unsqueeze(0).to(device)
    cam_obj = GradCAM(model, target_layer)
    try:
        cam, idx, probs = cam_obj(x, class_idx)
    finally:
        cam_obj.remove()
    return overlay_heatmap(pil_image, cam), idx, probs

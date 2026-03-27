"""
gradcam.py — Grad-CAM : visualisation des régions d'activation
==============================================================
Grad-CAM (Gradient-weighted Class Activation Mapping) permet de comprendre
quelles zones de la radiographie influencent la décision du CNN.

Références :
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks",
    ICCV 2017.  https://arxiv.org/abs/1610.02391

Fonctionnement :
    1. Passer une image en avant (forward pass)
    2. Calculer le gradient de la sortie par rapport à la dernière couche conv
    3. Moyenner les gradients → poids d'importance par canal
    4. Pondérer les feature maps et appliquer ReLU
    5. Superposer la heatmap sur l'image originale

Usage :
    python src/gradcam.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --image      data/chest_xray/test/PNEUMONIA/person1_virus_006.jpeg \
        --output     outputs/figures/gradcam.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import BaseCNN, TransferModel, get_model


# ─────────────────────────────────────────────────────────────
# Classe GradCAM
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Implémente Grad-CAM pour tout modèle PyTorch.

    Paramètres
    ----------
    model      : modèle entraîné (en mode eval)
    target_layer : couche convolutive cible (ex: model.features[-1].block[-1])
    device     : "cuda" ou "cpu"
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None

        # Hooks : capturent les activations (forward) et gradients (backward)
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,   # shape (1, 3, H, W)
    ) -> np.ndarray:
        """
        Génère la heatmap Grad-CAM normalisée ∈ [0, 1].

        Paramètres
        ----------
        image_tensor : (1, 3, 224, 224) normalisé

        Retour
        ------
        np.ndarray (H, W) : heatmap entre 0 et 1
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad_()

        # Forward
        output = self.model(image_tensor)          # (1, 1) logit brut
        self.model.zero_grad()

        # Backward sur le score de la classe positive (pneumonie)
        output.backward()

        # Poids d'importance = moyenne globale des gradients par canal
        # shape gradients : (1, C, H', W')
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Carte d'activation pondérée
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam)                      # ne conserver que l'influence positive
        cam = cam.squeeze().cpu().numpy()          # (H', W')

        # Normalisation dans [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


# ─────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────

def load_image(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    Charge une image, retourne le tenseur normalisé ET l'image RGB originale.
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)    # (1, 3, H, W)
    img_np = np.array(img_pil.resize((IMAGE_SIZE, IMAGE_SIZE)))  # HWC, uint8
    return img_tensor, img_np


def overlay_heatmap(
    cam: np.ndarray,
    original_img: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Superpose la heatmap Grad-CAM sur l'image originale.

    Paramètres
    ----------
    cam          : (H', W') float ∈ [0, 1]
    original_img : (H, W, 3) uint8 RGB
    alpha        : transparence de la heatmap (0 = invisible, 1 = opaque)

    Retour
    ------
    np.ndarray (H, W, 3) uint8
    """
    # Redimensionnement de la CAM à la taille de l'image
    cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_rgb, alpha, 0)
    return superimposed


def get_target_layer(model: nn.Module) -> nn.Module:
    """Retourne automatiquement la dernière couche conv du BaseCNN."""
    if isinstance(model, BaseCNN):
        # model.features[-1] est le 4e ConvBlock ; son block[-1] est le MaxPool
        # On cible la ReLU, juste avant le MaxPool (index -2)
        return model.features[-1].block[-2]
    raise ValueError("get_target_layer ne supporte que BaseCNN pour l'instant.")


# ─────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────

def run_gradcam(
    checkpoint: str,
    image_path: str,
    output_path: str,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Chargement modèle
    ckpt = torch.load(checkpoint, map_location=device)
    arch = ckpt.get("architecture", "baseline")
    threshold = float(ckpt.get("threshold", 0.5))
    if arch != "baseline":
        raise ValueError("Grad-CAM ne supporte que les checkpoints baseline/BaseCNN pour l'instant.")
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    if arch != "baseline":
        print("⚠ Grad-CAM est configuré uniquement pour le BaseCNN dans ce script.")
    model = get_model(arch, **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    # Chargement image
    img_tensor, img_np = load_image(image_path)
    img_tensor = img_tensor.to(device)
    prob = torch.sigmoid(model(img_tensor)).item()
    pred_class = "PNEUMONIA" if prob >= threshold else "NORMAL"

    # Grad-CAM
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer, device=device)
    cam = gradcam.generate(img_tensor)
    superimposed = overlay_heatmap(cam, img_np)

    # Visualisation
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np);       axes[0].set_title("Image originale");   axes[0].axis("off")
    axes[1].imshow(cam, cmap="jet"); axes[1].set_title("Heatmap CAM");     axes[1].axis("off")
    axes[2].imshow(superimposed); axes[2].set_title(
        f"Superposition\nPréd: {pred_class} ({prob:.2f})"
    ); axes[2].axis("off")

    plt.suptitle(f"Grad-CAM — {Path(image_path).name}", fontsize=11)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Grad-CAM sauvegardée → {out}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère une heatmap Grad-CAM")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--image",      required=True, type=str)
    parser.add_argument("--output",  default="outputs/figures/gradcam.png", type=str)
    parser.add_argument("--device",  default=None, type=str)
    args = parser.parse_args()
    run_gradcam(args.checkpoint, args.image, args.output, args.device)

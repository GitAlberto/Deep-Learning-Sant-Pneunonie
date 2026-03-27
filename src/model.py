"""
model.py — Architecture CNN baseline + extensions Transfer Learning
====================================================================
Ce module expose :
    1. BaseCNN       : CNN personnalisé entraîné from scratch
    2. TransferModel : wrapper autour de ResNet18 / DenseNet121 / EfficientNet-B0
    3. get_model()   : factory function utilisée par train.py et eval.py

Architecture CNN baseline (BaseCNN)
─────────────────────────────────────────────────────────────────────
Input  : (B, 3, 224, 224)   ← batch de radiographies normalisées RGB
   │
   ├─► Block Conv 1 → Conv(3→32, k=3)   → BN → ReLU → MaxPool(2×2)
   │   Sortie : (B, 32, 112, 112)
   │
   ├─► Block Conv 2 → Conv(32→64, k=3)  → BN → ReLU → MaxPool(2×2)
   │   Sortie : (B, 64, 56, 56)
   │
   ├─► Block Conv 3 → Conv(64→128, k=3) → BN → ReLU → MaxPool(2×2)
   │   Sortie : (B, 128, 28, 28)
   │
   ├─► Block Conv 4 → Conv(128→256, k=3) → BN → ReLU → MaxPool(2×2)
   │   Sortie : (B, 256, 14, 14)
   │
   ├─► Global Average Pooling (AdaptiveAvgPool → 1×1)
   │   Sortie : (B, 256)
   │
   ├─► Dropout(p=0.5)           ← régularisation principale
   │
   ├─► FC 256 → 128  → ReLU
   ├─► Dropout(p=0.3)
   └─► FC 128 → 1    → Sigmoid  ← probabilité de pneumonie
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────────────────────
# 1. CNN Baseline (from scratch)
# ─────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Bloc élémentaire : Conv2d → BatchNorm2d → ReLU → MaxPool2d.

    Paramètres
    ----------
    in_ch  : nombre de canaux en entrée
    out_ch : nombre de filtres (canaux en sortie)
    pool   : si True, applique MaxPool(2×2) pour diviser la résolution par 2
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            # Convolution 3×3, padding=1 → preserve la taille spatiale
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # BatchNorm stabilise et accélère la convergence
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaseCNN(nn.Module):
    """
    CNN baseline pour la classification binaire NORMAL / PNEUMONIA.

    Paramètres
    ----------
    num_classes : 1 pour classification binaire avec BCEWithLogitsLoss
    dropout1    : taux de dropout après le GAP  (défaut 0.5)
    dropout2    : taux de dropout dans le MLP  (défaut 0.3)
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout1: float = 0.5,
        dropout2: float = 0.3,
    ) -> None:
        super().__init__()

        # ── Extracteur de caractéristiques (feature extractor) ──────────────
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # 224 → 112
            ConvBlock(32,  64),   # 112 → 56
            ConvBlock(64,  128),  # 56  → 28
            ConvBlock(128, 256),  # 28  → 14
        )

        # ── Pooling global : réduit (B, 256, 14, 14) → (B, 256, 1, 1) ───────
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # ── Classificateur (MLP) ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(128, num_classes),
            # Pas de Sigmoid ici → on utilise BCEWithLogitsLoss (plus stable)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # extraction de features
        x = self.gap(x)            # pooling global
        x = torch.flatten(x, 1)   # (B, 256)
        x = self.classifier(x)    # logit brut
        return x


# ─────────────────────────────────────────────────────────────
# 2. Transfer Learning (extension J6+)
# ─────────────────────────────────────────────────────────────

class TransferModel(nn.Module):
    """
    Wrapper pour les modèles pré-entraînés : ResNet18, DenseNet121, EfficientNet-B0.

    Deux modes :
    - fine-tune  : seule la tête de classification est remplacée
    - full       : tous les paramètres sont entraînables

    Paramètres
    ----------
    backbone    : "resnet18" | "densenet121" | "efficientnet_b0"
    pretrained  : charger les poids ImageNet (défaut True)
    freeze_backbone : geler les couches convolutives (True = fine-tune rapide)
    """

    BACKBONES = ("resnet18", "densenet121", "efficientnet_b0")

    def __init__(
        self,
        backbone: Literal["resnet18", "densenet121", "efficientnet_b0"] = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in self.BACKBONES:
            raise ValueError(f"backbone doit être l'un de {self.BACKBONES}")

        weights_flag = True if pretrained else False

        # ── Chargement du backbone ───────────────────────────────────────────
        if backbone == "resnet18":
            base = models.resnet18(pretrained=weights_flag)
            in_features = base.fc.in_features   # 512
            base.fc = nn.Linear(in_features, 1) # remplace la tête ImageNet

        elif backbone == "densenet121":
            base = models.densenet121(pretrained=weights_flag)
            in_features = base.classifier.in_features  # 1024
            base.classifier = nn.Linear(in_features, 1)

        elif backbone == "efficientnet_b0":
            base = models.efficientnet_b0(pretrained=weights_flag)
            in_features = base.classifier[1].in_features  # 1280
            base.classifier[1] = nn.Linear(in_features, 1)

        self.model = base

        # ── Gel optionnel des couches convolutives ───────────────────────────
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                # On ne gèle pas la nouvelle tête de classification
                if "fc" not in name and "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze(self) -> None:
        """Dégèle tous les paramètres (fine-tuning complet à lr faible)."""
        for param in self.model.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────────────────────
# 3. Factory
# ─────────────────────────────────────────────────────────────

def get_model(
    architecture: str = "baseline",
    **kwargs,
) -> nn.Module:
    """
    Factory function : retourne le modèle correspondant à `architecture`.

    Paramètres
    ----------
    architecture : "baseline" | "resnet18" | "densenet121" | "efficientnet_b0"
    **kwargs     : transmis au constructeur correspondant

    Exemple
    -------
    >>> model = get_model("resnet18", pretrained=True, freeze_backbone=True)
    """
    architecture = architecture.lower()
    if architecture == "baseline":
        return BaseCNN(**kwargs)
    elif architecture in TransferModel.BACKBONES:
        return TransferModel(backbone=architecture, **kwargs)
    else:
        raise ValueError(
            f"Architecture inconnue : '{architecture}'. "
            f"Choix valides : baseline, {', '.join(TransferModel.BACKBONES)}"
        )


# ─────────────────────────────────────────────────────────────
# Vérification rapide
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)  # batch de 4 images fictives

    print("── BaseCNN ──────────────────────────────")
    m = get_model("baseline")
    out = m(x)
    print(f"  Entrée : {x.shape}  →  Sortie : {out.shape}")  # (4, 1)

    print("\n── ResNet18 (transfer) ──────────────────")
    m2 = get_model("resnet18", pretrained=False)
    out2 = m2(x)
    print(f"  Entrée : {x.shape}  →  Sortie : {out2.shape}")
    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m2.parameters())
    print(f"  Paramètres entraînables : {trainable:,} / {total:,}")
    print("\nModèle OK ✓")

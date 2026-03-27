"""
utils.py — Fonctions utilitaires transverses
============================================
Ce module regroupe des outils réutilisables :
    - Reproductibilité (seed globale)
    - Comptage des classes du dataset
    - Résumé du modèle
    - Sauvegarde / chargement de métriques au format JSON
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
# Reproductibilité
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fixe toutes les sources d'aléa pour la reproductibilité.

    À appeler en début de script, avant toute opération aléatoire.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force cuDNN à être déterministe (légèrement plus lent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  Seed fixée à {seed} ✓")


# ─────────────────────────────────────────────────────────────
# Informations dataset
# ─────────────────────────────────────────────────────────────

def count_classes(data_dir: str = "data/chest_xray") -> dict[str, dict[str, int]]:
    """
    Compte le nombre d'images par classe et par split.

    Paramètres
    ----------
    data_dir : chemin vers le répertoire chest_xray/

    Retour
    ------
    dict : {"train": {"NORMAL": 1341, "PNEUMONIA": 3875}, "val": {...}, "test": {...}}
    """
    base = Path(data_dir)
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_path = base / split
        if not split_path.exists():
            continue
        result[split] = {}
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                n = len(list(class_dir.glob("*.jpeg")) +
                        list(class_dir.glob("*.jpg")) +
                        list(class_dir.glob("*.png")))
                result[split][class_dir.name] = n

    print("\n  Distribution des classes :")
    for split, counts in result.items():
        total = sum(counts.values())
        parts = [f"{cls}: {n}" for cls, n in counts.items()]
        print(f"  [{split:5s}] {', '.join(parts)}  |  Total: {total}")
    return result


# ─────────────────────────────────────────────────────────────
# Résumé modèle
# ─────────────────────────────────────────────────────────────

def model_summary(model: torch.nn.Module) -> None:
    """Affiche le nombre total et entraînable de paramètres."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n  Paramètres totaux     : {total:>12,}")
    print(f"  Paramètres entraîn.  : {trainable:>12,}")
    print(f"  Paramètres gelés     : {frozen:>12,}")


# ─────────────────────────────────────────────────────────────
# Sauvegarde métriques
# ─────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, output_path: str) -> None:
    """Sauvegarde un dictionnaire de métriques au format JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Métriques sauvegardées → {path}")


def load_metrics(output_path: str) -> dict:
    """Charge des métriques depuis un fichier JSON."""
    with open(output_path, "r", encoding="utf-8") as f:
        return json.load(f)

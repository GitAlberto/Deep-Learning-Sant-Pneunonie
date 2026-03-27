"""
dataset.py — Chargement et préparation des données Chest X-Ray
================================================================
Responsabilités :
    - Définir les transformations (resize, normalisation, augmentation légère)
    - Encapsuler les splits train / val / test via torchvision.datasets.ImageFolder
    - Fournir des DataLoaders prêts à l'emploi

Arborescence attendue sous `data/chest_xray/` :
    train/
        NORMAL/      ← exemples négatifs (poumons sains)
        PNEUMONIA/   ← exemples positifs (pneumonie)
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────
# Constantes globales
# ─────────────────────────────────────────────────────────────

# Taille d'entrée choisie pour le CNN baseline (224×224 = standard)
IMAGE_SIZE: int = 224

# Statistiques ImageNet – utilisées aussi pour les modèles pré-entraînés
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Classes : l'ordre est imposé par ImageFolder (ordre alphabétique)
# NORMAL → 0  (négatif),  PNEUMONIA → 1  (positif)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# ─────────────────────────────────────────────────────────────
# Transformations
# ─────────────────────────────────────────────────────────────

def get_transforms(mode: str = "train") -> transforms.Compose:
    """
    Retourne le pipeline de transformations adapté au split demandé.

    Paramètres
    ----------
    mode : str
        "train"  → augmentation légère + normalisation
        "val"    → resize + normalisation uniquement
        "test"   → idem val

    Retour
    ------
    transforms.Compose : pipeline torchvision prêt à l'emploi
    """
    if mode == "train":
        return transforms.Compose([
            # 1. Redimensionnement légèrement plus grand pour le crop aléatoire
            transforms.Resize((256, 256)),
            # 2. Crop aléatoire à la taille cible → légère variation de cadrage
            transforms.RandomCrop(IMAGE_SIZE),
            # 3. Flip horizontal aléatoire (50 %) → invariance gauche/droite
            transforms.RandomHorizontalFlip(p=0.5),
            # 4. Légère rotation (±10°) → robustesse aux orientations
            transforms.RandomRotation(degrees=10),
            # 5. Légère variation de luminosité/contraste
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # 6. Conversion en tenseur [0, 1]
            transforms.ToTensor(),
            # 7. Normalisation autour de la moyenne ImageNet
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # "val" ou "test" : pas d'augmentation
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────

def get_datasets(data_dir: str = "data/chest_xray") -> Dict[str, datasets.ImageFolder]:
    """
    Construit les trois datasets (train, val, test) depuis l'arborescence disque.

    Paramètres
    ----------
    data_dir : str
        Chemin vers le répertoire contenant les sous-dossiers train/val/test.

    Retour
    ------
    dict : {"train": Dataset, "val": Dataset, "test": Dataset}
    """
    base = Path(data_dir)
    splits = {
        "train": base / "train",
        "val":   base / "val",
        "test":  base / "test",
    }

    datasets_dict: Dict[str, datasets.ImageFolder] = {}
    for split, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Le répertoire '{path}' est introuvable. "
                "Vérifiez que le dataset Kaggle a été téléchargé et décompressé."
            )
        dataset = datasets.ImageFolder(
            root=str(path),
            transform=get_transforms(mode=split),
        )
        datasets_dict[split] = dataset
        print(
            f"  [{split:5s}] {len(dataset):5d} images  |  "
            f"classes : {dataset.class_to_idx}"
        )

    return datasets_dict


# ─────────────────────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = "data/chest_xray",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Retourne les DataLoaders train, val et test.

    Paramètres
    ----------
    data_dir   : chemin vers chest_xray/
    batch_size : taille des mini-batchs (défaut 32)
    num_workers: parallélisme de chargement (mettre 0 sous Windows si erreur)

    Retour
    ------
    Tuple (train_loader, val_loader, test_loader)
    """
    dsets = get_datasets(data_dir)

    train_loader = DataLoader(
        dsets["train"],
        batch_size=batch_size,
        shuffle=True,           # mélange à chaque époque
        num_workers=num_workers,
        pin_memory=True,        # accélère le transfert CPU→GPU
        drop_last=True,         # évite les batchs incomplets en fin d'époque
    )
    val_loader = DataLoader(
        dsets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dsets["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Point d'entrée rapide (vérification)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/chest_xray"
    print("=== Vérification du dataset ===")
    train_dl, val_dl, test_dl = get_dataloaders(data_dir=data_path)
    batch_imgs, batch_labels = next(iter(train_dl))
    print(f"\nForme d'un batch : {batch_imgs.shape}   labels : {batch_labels.shape}")
    print("  min pixel (normalisé) :", batch_imgs.min().item())
    print("  max pixel (normalisé) :", batch_imgs.max().item())
    print("\nDataset pret - OK")

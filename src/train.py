"""
train.py — Boucle d'entraînement et de validation
===================================================
Fonctionnalités :
    - Instanciation du modèle (baseline ou transfer)
    - Optimisation avec Adam + LR scheduler (ReduceLROnPlateau)
    - Gestion du déséquilibre de classes via pos_weight
    - Early stopping pour éviter l'overfitting
    - Sauvegarde automatique du meilleur checkpoint (.pt)
    - Affichage et export des courbes loss / accuracy

Usage :
    python src/train.py \
        --data_dir  data/chest_xray \
        --arch      baseline \
        --epochs    30 \
        --batch     32 \
        --lr        1e-3 \
        --dropout   0.5 \
        --output    outputs/checkpoints
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # mode sans affichage (serveur / HPC)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import get_model


# ─────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────

def predictions_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calcule l'accuracy binaire à partir des logits (avant sigmoid).

    On seuille à 0.5 sur les probabilités sigmoid.
    """
    return (torch.sigmoid(logits) >= threshold).long()


def save_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    output_dir: Path,
) -> None:
    """Sauvegarde les courbes d'entraînement dans `output_dir/training_curves.png`."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe de perte
    axes[0].plot(epochs, train_losses, label="Train Loss", color="royalblue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="tomato", linestyle="--")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("BCEWithLogitsLoss")
    axes[0].set_title("Courbe de perte")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Courbe d'accuracy
    axes[1].plot(epochs, train_accs, label="Train Acc", color="royalblue")
    axes[1].plot(epochs, val_accs,   label="Val Acc",   color="tomato", linestyle="--")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Courbe d'accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Courbes sauvegardées → {save_path}")


# ─────────────────────────────────────────────────────────────
# Boucle principale
# ─────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    arch: str,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    output_dir: str,
    num_workers: int = 4,
    patience: int = 5,          # early stopping : nombre d'époques sans amélioration
    device: str | None = None,
) -> None:
    """
    Entraîne le modèle et sauvegarde le meilleur checkpoint.

    Paramètres
    ----------
    data_dir   : chemin vers chest_xray/
    arch       : architecture ("baseline", "resnet18", etc.)
    epochs     : nombre max d'époques
    batch_size : taille du batch
    lr         : learning rate initial
    dropout    : taux de dropout (uniquement pour BaseCNN)
    output_dir : répertoire de sauvegarde des checkpoints et courbes
    num_workers: workers DataLoader
    patience   : early stopping patience
    device     : "cuda", "cpu" ou None (autodetect)
    """

    # ── Appareil ──────────────────────────────────────────────────────────────
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Entraînement sur : {device.upper()} ===")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Données ───────────────────────────────────────────────────────────────
    print("\n[1/4] Chargement des données…")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    print("\n[2/4] Instanciation du modèle…")
    extra = {"dropout1": dropout} if arch == "baseline" else {}
    model = get_model(arch, **extra).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : {arch}  |  Paramètres entraînables : {n_params:,}")

    # ── Gestion du déséquilibre de classes ────────────────────────────────────
    # Chest X-Ray : ~3× plus de cas PNEUMONIA que NORMAL dans train
    # pos_weight = N_neg / N_pos  ≈  1/3 ≈ 0.33 → pénalise moins les FP
    # Adapter selon le comptage réel du dataset
    pos_weight = torch.tensor([0.33]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiseur et scheduler ───────────────────────────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,   # régularisation L2 légère
    )
    # Réduit le lr par 0.5 si la loss de validation ne s'améliore pas sur 3 époques
    try:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    print("\n[3/4] Entraînement…")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)    # (B,)
            loss   = criterion(logits, labels)
            loss.backward()
            # Gradient clipping pour stabiliser l'entraînement
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = predictions_from_logits(logits)
            batch_size_curr = labels.size(0)
            running_loss += loss.item() * batch_size_curr
            train_correct += (preds == labels.long()).sum().item()
            train_total += batch_size_curr

        train_loss = running_loss / max(train_total, 1)
        train_acc  = train_correct / max(train_total, 1)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_targets: list[int] = []
        val_preds: list[int] = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.float().to(device)
                logits = model(imgs).squeeze(1)
                preds = predictions_from_logits(logits)
                batch_size_curr = labels.size(0)
                val_loss_sum += criterion(logits, labels).item() * batch_size_curr
                val_correct += (preds == labels.long()).sum().item()
                val_total += batch_size_curr
                val_targets.extend(labels.long().cpu().tolist())
                val_preds.extend(preds.cpu().tolist())

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc  = val_correct / max(val_total, 1)
        val_prec = precision_score(val_targets, val_preds, zero_division=0)
        val_rec  = recall_score(val_targets, val_preds, zero_division=0)
        val_f1   = f1_score(val_targets, val_preds, zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Époque {epoch:02d}/{epochs}"
            f"  | Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}"
            f"  | Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
            f"  P: {val_prec:.4f}  R: {val_rec:.4f}  F1: {val_f1:.4f}"
            f"  | LR: {current_lr:.2e}"
            f"  | {elapsed:.1f}s"
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Sauvegarde du meilleur modèle ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt_path = out_path / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "architecture": arch,
                    "model_kwargs": extra,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                    "val_f1": val_f1,
                    "threshold": 0.5,
                    "config": {
                        "data_dir": data_dir,
                        "arch": arch,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "dropout": dropout,
                        "output_dir": str(out_path),
                        "num_workers": num_workers,
                        "patience": patience,
                        "device": device,
                    },
                },
                ckpt_path,
            )
            print(f"    ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\n  Early stopping déclenché après {epoch} époques.")
            break

    # ── Export des courbes ────────────────────────────────────────────────────
    print("\n[4/4] Export des courbes…")
    figures_dir = Path(output_dir).parent.parent / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_curves(train_losses, val_losses, train_accs, val_accs, figures_dir)

    print(f"\n=== Entraînement terminé. Meilleure val_loss : {best_val_loss:.4f} ===")


# ─────────────────────────────────────────────────────────────
# Interface ligne de commande
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraîne le CNN de classification radiographies"
    )
    parser.add_argument("--data_dir", default="data/chest_xray",   type=str)
    parser.add_argument("--arch",     default="baseline",           type=str,
                        help="baseline | resnet18 | densenet121 | efficientnet_b0")
    parser.add_argument("--epochs",   default=30,                   type=int)
    parser.add_argument("--batch",    default=32,                   type=int)
    parser.add_argument("--lr",       default=1e-3,                 type=float)
    parser.add_argument("--dropout",  default=0.5,                  type=float)
    parser.add_argument("--output",   default="outputs/checkpoints",type=str)
    parser.add_argument("--workers",  default=4,                    type=int)
    parser.add_argument("--patience", default=5,                    type=int)
    parser.add_argument("--device",   default=None,                 type=str,
                        help="cuda | cpu (autodetect si non précisé)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        dropout=args.dropout,
        output_dir=args.output,
        num_workers=args.workers,
        patience=args.patience,
        device=args.device,
    )

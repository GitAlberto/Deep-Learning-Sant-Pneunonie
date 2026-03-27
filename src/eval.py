"""
eval.py — Évaluation finale sur le jeu de test
================================================
Ce script charge le meilleur checkpoint et calcule :
    - Accuracy globale
    - Precision, Recall, F1-score (binaire, seuil 0.5)
    - AUC-ROC
    - Matrice de confusion complète
    - Courbe ROC
    - Visualisation des erreurs (FN et FP)

Usage :
    python src/eval.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --data_dir   data/chest_xray \
        --output     outputs/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset import CLASS_NAMES, get_dataloaders
from model import TransferModel, get_model


# ─────────────────────────────────────────────────────────────
# Inférence sur le set de test
# ─────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    test_loader,
    device: str,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fait passer tout le test set dans le modèle.

    Retour
    ------
    all_labels : (N,) entiers  0=NORMAL / 1=PNEUMONIA
    all_probs  : (N,) probabilités sigmoid ∈ [0, 1]
    all_preds  : (N,) prédictions binaires (seuil 0.5)
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_labels = np.concatenate(all_labels)
    all_probs  = np.concatenate(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)

    return all_labels, all_probs, all_preds


# ─────────────────────────────────────────────────────────────
# Affichage des métriques
# ─────────────────────────────────────────────────────────────

def print_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> dict:
    """Calcule et affiche toutes les métriques. Retourne un dict récapitulatif."""
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc_score = roc_auc_score(labels, probs)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "═" * 50)
    print("  RÉSULTATS D'ÉVALUATION FINALE")
    print("═" * 50)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  ← critique (détection pneumonie)")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc_score:.4f}")
    print("\n  Matrice de confusion :")
    print(f"  {'':15s} Préd. NORMAL  Préd. PNEUMONIA")
    print(f"  {'Réel NORMAL':15s} TN={tn:5d}        FP={fp:5d}")
    print(f"  {'Réel PNEUMONIA':15s} FN={fn:5d}        TP={tp:5d}")
    print("\n  Rapport complet sklearn :")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    # ── Analyse des faux négatifs ─────────────────────────────────────────────
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    print(f"  ⚠ Taux de faux négatifs (FNR) : {fnr:.4f}  ({fn} cas)")
    print(f"    → {fn} pneumonies non détectées (risque clinique élevé !)")
    if fn > 0:
        print("    Analyser ces cas prioritairement (cf. visualisation des erreurs).")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc_score,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


# ─────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    """Heatmap de la matrice de confusion."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_marks = range(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(CLASS_NAMES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_ylabel("Réel")
    ax.set_xlabel("Prédit")
    ax.set_title("Matrice de confusion — Test set")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Matrice de confusion → {path}")


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, output_dir: Path) -> None:
    """Courbe ROC avec AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"Courbe ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Taux de faux positifs (FPR)")
    ax.set_ylabel("Taux de vrais positifs (TPR / Recall)")
    ax.set_title("Courbe ROC — Détection de pneumonie")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = output_dir / "roc_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Courbe ROC → {path}")


def visualize_errors(
    model: torch.nn.Module,
    test_loader,
    device: str,
    output_dir: Path,
    n_show: int = 8,
) -> None:
    """
    Visualise les premières erreurs significatives (FP et FN).

    Affiche côté à côté : image, vraie classe, classe prédite, confiance.
    """
    model.eval()
    errors: list[dict] = []

    # Dénormalisation pour l'affichage
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs_gpu = imgs.to(device)
            logits   = model(imgs_gpu).squeeze(1)
            probs    = torch.sigmoid(logits).cpu().numpy()
            preds    = (probs >= 0.5).astype(int)
            labels_np = labels.numpy()

            for i, (img, lbl, pred, prob) in enumerate(
                zip(imgs.numpy(), labels_np, preds, probs)
            ):
                if lbl != pred:
                    # Dénormalisation : CHW → HWC
                    img_disp = img.transpose(1, 2, 0) * std + mean
                    img_disp = np.clip(img_disp, 0, 1)
                    error_type = "FN" if lbl == 1 else "FP"
                    errors.append({
                        "img": img_disp,
                        "true": CLASS_NAMES[lbl],
                        "pred": CLASS_NAMES[pred],
                        "prob": float(prob),
                        "type": error_type,
                    })
            if len(errors) >= n_show:
                break

    if not errors:
        print("  Aucune erreur trouvée sur le test set !")
        return

    n = min(n_show, len(errors))
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 6))
    axes = axes.flatten()

    for idx, (ax, err) in enumerate(zip(axes, errors[:n])):
        ax.imshow(err["img"])
        color = "red" if err["type"] == "FN" else "orange"
        ax.set_title(
            f"{err['type']} | Vrai: {err['true']}\nPrédit: {err['pred']}"
            f" ({err['prob']:.2f})",
            color=color, fontsize=8,
        )
        ax.axis("off")

    plt.suptitle("Erreurs significatives — FN (rouge) et FP (orange)", fontsize=11)
    plt.tight_layout()
    path = output_dir / "error_analysis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Visualisation des erreurs → {path}")


# ─────────────────────────────────────────────────────────────
# Programme principal
# ─────────────────────────────────────────────────────────────

def evaluate(
    checkpoint: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | None = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Évaluation sur : {device.upper()} ===")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Chargement checkpoint ─────────────────────────────────────────────────
    print(f"\n[1/3] Chargement du checkpoint : {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)
    arch = ckpt.get("architecture", "baseline")
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    if arch in TransferModel.BACKBONES:
        model_kwargs["pretrained"] = False
    model = get_model(arch, **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = float(ckpt.get("threshold", 0.5))
    print(f"  Époque sauvegardée : {ckpt.get('epoch', '?')}")
    print(f"  Val loss (training) : {ckpt.get('val_loss', '?'):.4f}")
    print(f"  Seuil de dÃ©cision   : {threshold:.2f}")

    # ── Données ───────────────────────────────────────────────────────────────
    print("\n[2/3] Chargement du test set…")
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=num_workers
    )

    # ── Inférence + métriques ─────────────────────────────────────────────────
    print("\n[3/3] Inférence…")
    labels, probs, preds = run_inference(model, test_loader, device, threshold=threshold)
    metrics = print_metrics(labels, probs, preds)

    # ── Graphiques ────────────────────────────────────────────────────────────
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, out_path)
    plot_roc_curve(labels, probs, out_path)
    visualize_errors(model, test_loader, device, out_path)

    print(f"\n=== Évaluation terminée. Résultats dans : {out_path} ===")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue le meilleur modèle sur le test set")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt", type=str)
    parser.add_argument("--data_dir",   default="data/chest_xray",                   type=str)
    parser.add_argument("--output",     default="outputs/figures",                   type=str)
    parser.add_argument("--batch",      default=32,                                  type=int)
    parser.add_argument("--workers",    default=4,                                   type=int)
    parser.add_argument("--device",     default=None,                                type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output,
        batch_size=args.batch,
        num_workers=args.workers,
        device=args.device,
    )

"""
app.py — Interface Streamlit pour le diagnostic IA de Radiographies Thoraciques
=================================================================================
Interface médicale premium pour tester le CNN de détection de pneumonie.

Usage :
    streamlit run app.py
    # ou depuis le dossier medical-cnn-project :
    streamlit run app.py --server.port 8501
"""

import io
import os
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# ── Chemin vers les modules src ──────────────────────────────────────────────
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from gradcam import GradCAM, get_target_layer, overlay_heatmap
from model import BaseCNN, TransferModel, get_model

warnings.filterwarnings("ignore")

# ── Chemins par défaut ───────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = Path(__file__).parent / "outputs" / "checkpoints" / "best_model.pt"
OUTPUTS_DIR = Path(__file__).parent / "outputs" / "figures"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# CONFIG PAGE STREAMLIT
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="PneumoScan AI — Diagnostic Radiologique",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Premium ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Import Google Fonts ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Global Reset ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── Background ──────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2e 40%, #091420 100%);
    color: #e2e8f0;
}

/* ─── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #091420 100%);
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}
[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

/* ─── Header Hero ─────────────────────────────────────────── */
.hero-container {
    background: linear-gradient(135deg, rgba(6,182,212,0.12) 0%, rgba(59,130,246,0.08) 50%, rgba(99,102,241,0.12) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

/* ─── Metric Cards ────────────────────────────────────────── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s, transform 0.2s;
}
.metric-card:hover {
    border-color: rgba(56,189,248,0.4);
    transform: translateY(-2px);
}
.metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}

/* ─── Result Banner ───────────────────────────────────────── */
.result-normal {
    background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.08) 100%);
    border: 1.5px solid rgba(16,185,129,0.4);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    text-align: center;
}
.result-pneumonia {
    background: linear-gradient(135deg, rgba(239,68,68,0.15) 0%, rgba(185,28,28,0.08) 100%);
    border: 1.5px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    text-align: center;
    animation: pulse-border 2s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { border-color: rgba(239,68,68,0.4); }
    50%       { border-color: rgba(239,68,68,0.85); }
}
.result-title {
    font-size: 1.7rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.result-confidence {
    font-size: 0.9rem;
    opacity: 0.75;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Probability Bar ─────────────────────────────────────── */
.prob-bar-container {
    margin: 1rem 0;
}
.prob-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 6px;
    font-weight: 500;
}
.prob-bar-bg {
    background: rgba(30,41,59,0.8);
    border-radius: 100px;
    height: 10px;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.1);
}
.prob-bar-fill-normal {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #10b981, #34d399);
    transition: width 1s ease;
}
.prob-bar-fill-pneumonia {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #ef4444, #f87171);
    transition: width 1s ease;
}

/* ─── Section Titles ──────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.3px;
    margin: 1.5rem 0 0.8rem 0;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ─── Info Box ────────────────────────────────────────────── */
.info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.6;
}
.warn-box {
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #fbbf24;
    line-height: 1.6;
}

/* ─── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 0.6rem 1.4rem;
    transition: opacity 0.25s, transform 0.2s;
    letter-spacing: 0.3px;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
}

/* ─── File Uploader ───────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(56,189,248,0.25);
    border-radius: 14px;
    background: rgba(15,23,42,0.5);
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.5);
}

/* ─── Slider ──────────────────────────────────────────────── */
.stSlider [data-testid="stTickBar"] { background: transparent; }

/* ─── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,23,42,0.6);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.15) !important;
    color: #38bdf8 !important;
}

/* ─── Spinner ─────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #38bdf8 !important;
}

/* ─── Divider ─────────────────────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid rgba(56,189,248,0.12);
    margin: 1.5rem 0;
}

/* ─── Images ──────────────────────────────────────────────── */
img { border-radius: 10px; }

/* ─── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 100px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str) -> tuple:
    """Charge le modèle depuis le checkpoint et le met en cache."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture", "baseline")
    model_kwargs = dict(ckpt.get("model_kwargs", {}))
    if arch in TransferModel.BACKBONES:
        model_kwargs["pretrained"] = False
    model = get_model(arch, **model_kwargs).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    threshold = float(ckpt.get("threshold", 0.5))
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)
    return model, device, threshold, arch, epoch, val_loss


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    """Transforme un PIL.Image en tenseur normalisé (1, 3, 224, 224)."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def predict(model, tensor: torch.Tensor, device: str, threshold: float):
    """Retourne (prob_pneumonia, pred_class, pred_label)."""
    model.eval()
    with torch.no_grad():
        logit = model(tensor.to(device)).squeeze()
        prob = torch.sigmoid(logit).item()
    pred = 1 if prob >= threshold else 0
    label = "PNEUMONIA" if pred == 1 else "NORMAL"
    return prob, pred, label


def compute_gradcam(model, tensor: torch.Tensor, device: str) -> tuple:
    """Génère la heatmap Grad-CAM + image superposée."""
    try:
        target_layer = get_target_layer(model)
        gc = GradCAM(model, target_layer, device=device)
        cam = gc.generate(tensor.clone().to(device))
        return cam, True
    except Exception as e:
        return None, False


def fig_to_pil(fig) -> Image.Image:
    """Convertit une figure matplotlib en PIL.Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=130)
    buf.seek(0)
    return Image.open(buf).copy()


def gradcam_figure(cam, img_np):
    """Crée la figure matplotlib Grad-CAM (3 colonnes)."""
    superimposed = overlay_heatmap(cam, img_np, alpha=0.45)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0a0e1a")
    titles = ["Radiographie originale", "Carte d'activation (CAM)", "Superposition Grad-CAM"]
    images_to_show = [img_np, cam, superimposed]
    cmaps = [None, "jet", None]
    for ax, title, img, cmap in zip(axes, titles, images_to_show, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=8, fontweight="600")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1.5)
    return fig


def prob_bar_html(prob_pneumonia: float) -> str:
    prob_normal = 1 - prob_pneumonia
    pct_p = f"{prob_pneumonia*100:.1f}%"
    pct_n = f"{prob_normal*100:.1f}%"
    return f"""
    <div class="prob-bar-container">
      <div class="prob-bar-label"><span>🫁 Normal</span><span>{pct_n}</span></div>
      <div class="prob-bar-bg"><div class="prob-bar-fill-normal" style="width:{prob_normal*100:.1f}%"></div></div>
    </div>
    <div class="prob-bar-container">
      <div class="prob-bar-label"><span>🦠 Pneumonie</span><span>{pct_p}</span></div>
      <div class="prob-bar-bg"><div class="prob-bar-fill-pneumonia" style="width:{prob_pneumonia*100:.1f}%"></div></div>
    </div>
    """


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;margin-bottom:1.5rem;">
      <div style="font-size:3rem;line-height:1;">🫁</div>
      <div style="font-size:1.15rem;font-weight:800;color:#38bdf8;letter-spacing:-0.3px;">PneumoScan AI</div>
      <div style="font-size:0.75rem;color:#475569;margin-top:2px;">Diagnostic Radiologique</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    # Checkpoint
    checkpoint_path = st.text_input(
        "📂 Chemin du checkpoint",
        value=str(DEFAULT_CHECKPOINT),
        help="Chemin vers le fichier .pt généré par train.py"
    )

    # Seuil de décision
    st.markdown("**🎚️ Seuil de décision**")
    threshold_override = st.slider(
        "Seuil (probabilité de pneumonie)",
        min_value=0.1, max_value=0.9,
        value=0.5, step=0.05,
        label_visibility="collapsed"
    )
    st.caption(f"Classification PNEUMONIA si P ≥ **{threshold_override:.2f}**")

    # Grad-CAM toggle
    st.markdown("---")
    st.markdown("### 🔬 Interprétabilité")
    enable_gradcam = st.toggle("Activer Grad-CAM", value=True,
                               help="Disponible uniquement pour le modèle BaseCNN")
    gradcam_alpha = st.slider("Intensité heatmap", 0.1, 0.9, 0.45, 0.05,
                              disabled=not enable_gradcam)

    # Info
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
      <strong>📊 Dataset</strong><br>
      Chest X-Ray (Kaggle)<br>
      NORMAL / PNEUMONIA<br><br>
      <strong>🧠 Architectures</strong><br>
      BaseCNN · ResNet18 · DenseNet121 · EfficientNet-B0
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="warn-box">
      ⚠️ <strong>Usage éducatif uniquement.</strong><br>
      Ne remplace pas un avis médical professionnel.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HEADER HERO
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
  <div class="hero-badge">✦ Intelligence Artificielle Médicale</div>
  <h1 class="hero-title">🫁 PneumoScan AI</h1>
  <p class="hero-subtitle">Système de détection automatique de pneumonie par analyse de radiographies thoraciques.<br>
  Basé sur un CNN entraîné sur le dataset Chest X-Ray (Kaggle) — <strong style="color:#38bdf8">Classification binaire : NORMAL / PNEUMONIA</strong></p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════

model_loaded = False
model_info = {}

checkpoint_file = Path(checkpoint_path)
if not checkpoint_file.exists():
    st.markdown("""
    <div class="warn-box">
      ⚠️ <strong>Aucun checkpoint trouvé.</strong><br>
      Entraînez d'abord le modèle : <code>python src/train.py</code><br>
      Le fichier attendu : <code>outputs/checkpoints/best_model.pt</code>
    </div>
    """, unsafe_allow_html=True)
else:
    with st.spinner("⚡ Chargement du modèle..."):
        try:
            model, device, ckpt_threshold, arch, epoch, val_loss = load_model(str(checkpoint_file))
            # On respecte le seuil du checkpoint sauf si l'utilisateur l'a modifié
            effective_threshold = threshold_override
            model_loaded = True
            model_info = {
                "arch": arch,
                "device": device,
                "epoch": epoch,
                "val_loss": val_loss,
                "threshold": ckpt_threshold,
            }
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle : {e}")

# ── Indicateurs modèle ───────────────────────────────────────────────────────
if model_loaded:
    device_icon = "⚡ GPU" if model_info["device"] == "cuda" else "💻 CPU"
    val_loss_str = f"{model_info['val_loss']:.4f}" if model_info["val_loss"] is not None else "—"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-value" style="color:#38bdf8">{model_info['arch'].upper()}</div>
        <div class="metric-label">Architecture</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#818cf8">{model_info['epoch']}</div>
        <div class="metric-label">Époque sauvegardée</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#06b6d4">{val_loss_str}</div>
        <div class="metric-label">Val Loss</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#f59e0b; font-size:1.3rem">{device_icon}</div>
        <div class="metric-label">Matériel</div>
      </div>
      <div class="metric-card">
        <div class="metric-value" style="color:#10b981">{effective_threshold:.2f}</div>
        <div class="metric-label">Seuil actif</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# ONGLETS PRINCIPAUX
# ════════════════════════════════════════════════════════════════════════════

tab_analyse, tab_batch, tab_pipeline, tab_about = st.tabs([
    "🔬 Analyse d'Image",
    "📂 Analyse par Lot",
    "🔁 Pipeline Complet",
    "ℹ️ À propos du Modèle"
])


# ────────────────────────────────────────────────────────────────────────────
# Tab 1 — Analyse d'image unique
# ────────────────────────────────────────────────────────────────────────────
with tab_analyse:

    col_upload, col_result = st.columns([1, 1.35], gap="large")

    with col_upload:
        st.markdown('<div class="section-title">📤 Charger une Radiographie</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Glissez-déposez ou cliquez pour sélectionner",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            label_visibility="collapsed",
            key="single_upload"
        )

        if uploaded_file:
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption=f"📷 {uploaded_file.name}", use_container_width=True)

            # Infos image
            w, h = pil_img.size
            st.markdown(f"""
            <div class="info-box">
              📐 Résolution originale : <strong>{w} × {h} px</strong><br>
              📄 Nom du fichier : <strong>{uploaded_file.name}</strong><br>
              💾 Taille : <strong>{uploaded_file.size / 1024:.1f} Ko</strong>
            </div>
            """, unsafe_allow_html=True)

            analyse_btn = st.button("🚀 Lancer l'analyse IA", use_container_width=True,
                                    disabled=not model_loaded)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:2rem;">
              🫁<br><br>
              <strong>Importez une radiographie thoracique</strong><br>
              Formats acceptés : JPG, PNG, BMP, TIFF
            </div>
            """, unsafe_allow_html=True)
            analyse_btn = False

    with col_result:
        st.markdown('<div class="section-title">📊 Résultat du Diagnostic</div>', unsafe_allow_html=True)

        if uploaded_file and model_loaded and analyse_btn:
            with st.spinner("🧠 Analyse en cours..."):
                tensor = preprocess_image(pil_img)
                prob, pred, label = predict(model, tensor, device, effective_threshold)
                prob_normal = 1 - prob

            # Bannière résultat
            if label == "NORMAL":
                st.markdown(f"""
                <div class="result-normal">
                  <div class="result-title" style="color:#10b981">✅ NORMAL</div>
                  <div class="result-confidence">Poumons apparemment sains</div>
                  <div style="font-size:0.85rem;color:#6ee7b7;margin-top:0.5rem;">
                    Confiance : {prob_normal*100:.1f}% NORMAL
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-pneumonia">
                  <div class="result-title" style="color:#ef4444">🦠 PNEUMONIE DÉTECTÉE</div>
                  <div class="result-confidence">Signes radiologiques suspects</div>
                  <div style="font-size:0.85rem;color:#fca5a5;margin-top:0.5rem;">
                    Confiance : {prob*100:.1f}% PNEUMONIE
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Barres de probabilité
            st.markdown(prob_bar_html(prob), unsafe_allow_html=True)

            # Jauge circulaire matplotlib
            fig_gauge, ax_g = plt.subplots(figsize=(4, 2.5),
                                           subplot_kw=dict(aspect="equal"))
            fig_gauge.patch.set_facecolor("#0a0e1a")
            ax_g.set_facecolor("#0a0e1a")
            color_p = "#ef4444" if prob >= effective_threshold else "#10b981"
            ax_g.barh(0, 1, height=0.4, color="#1e293b", left=0)
            ax_g.barh(0, prob, height=0.4, color=color_p, left=0)
            ax_g.set_xlim(0, 1); ax_g.set_ylim(-0.5, 0.5)
            ax_g.axis("off")
            ax_g.text(0.5, -0.42, f"P(Pneumonie) = {prob:.4f}",
                      ha="center", va="top", color="#94a3b8",
                      fontsize=10, transform=ax_g.transAxes,
                      fontfamily="monospace")
            plt.tight_layout(pad=0.3)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

            # ── Grad-CAM ─────────────────────────────────────────────────────
            if enable_gradcam:
                st.markdown('<div class="section-title">🔥 Visualisation Grad-CAM</div>',
                            unsafe_allow_html=True)
                with st.spinner("🔬 Génération Grad-CAM..."):
                    cam, ok = compute_gradcam(model, tensor, device)
                    if ok and cam is not None:
                        img_np = np.array(pil_img.convert("RGB").resize(
                            (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
                        ))
                        fig_gc = gradcam_figure(cam, img_np)
                        # màj alpha
                        overlay = overlay_heatmap(cam, img_np, alpha=gradcam_alpha)
                        st.pyplot(fig_gc, use_container_width=True)
                        plt.close(fig_gc)
                        st.markdown("""
                        <div class="info-box">
                          🔥 <strong>Grad-CAM</strong> : les zones <span style="color:#ef4444">rouges</span>
                          influencent le plus la décision du modèle.<br>
                          Permet de vérifier que le CNN « regarde » bien les poumons.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ Grad-CAM disponible uniquement pour le modèle **BaseCNN**.")

        elif not model_loaded:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:2rem;">
              🔌 <strong>Modèle non chargé</strong><br>
              Vérifiez le chemin du checkpoint dans la barre latérale.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="text-align:center;padding:2.5rem;">
              👈 <strong>Chargez une image et cliquez sur Analyser</strong><br><br>
              Le résultat apparaîtra ici avec les probabilités<br>et la carte d'activation Grad-CAM.
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Tab 2 — Analyse par lot
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-title">📂 Analyse Multiple</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.warning("⚠️ Chargez un checkpoint dans la barre latérale pour activer cette fonctionnalité.")
    else:
        st.markdown("""
        <div class="info-box">
          Importez plusieurs radiographies en simultané.<br>
          Le tableau récapitulatif vous donnera la prédiction et la probabilité pour chaque image.
        </div>
        """, unsafe_allow_html=True)

        batch_files = st.file_uploader(
            "Sélectionner plusieurs images",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            accept_multiple_files=True,
            label_visibility="visible",
            key="batch_upload"
        )

        if batch_files:
            run_batch = st.button(f"🚀 Analyser {len(batch_files)} image(s)", use_container_width=True)

            if run_batch:
                results = []
                progress_bar = st.progress(0, text="Analyse en cours…")
                cols_per_row = 3
                cols = st.columns(cols_per_row)
                col_idx = 0

                for i, f in enumerate(batch_files):
                    pil = Image.open(f).convert("RGB")
                    tensor = preprocess_image(pil)
                    prob, pred, label = predict(model, tensor, device, effective_threshold)
                    results.append({
                        "Fichier": f.name,
                        "Prédiction": label,
                        "P(Pneumonie)": f"{prob*100:.1f}%",
                        "P(Normal)": f"{(1-prob)*100:.1f}%",
                        "Statut": "🔴 Pneumonie" if label == "PNEUMONIA" else "🟢 Normal",
                    })

                    with cols[col_idx % cols_per_row]:
                        border_color = "#ef4444" if label == "PNEUMONIA" else "#10b981"
                        st.image(pil.resize((200, 200)), use_container_width=True)
                        st.markdown(
                            f"<div style='text-align:center;font-size:0.8rem;"
                            f"color:{border_color};font-weight:700;"
                            f"margin-top:4px;'>{'🦠 '+label if label=='PNEUMONIA' else '✅ '+label}</div>"
                            f"<div style='text-align:center;font-size:0.75rem;"
                            f"color:#64748b;'>{f.name[:25]}</div>",
                            unsafe_allow_html=True
                        )
                    col_idx += 1

                    progress_bar.progress((i + 1) / len(batch_files),
                                          text=f"Analyse : {f.name}")

                progress_bar.empty()

                # Tableau récap
                st.markdown('<div class="section-title">📋 Récapitulatif</div>',
                            unsafe_allow_html=True)
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True, hide_index=True)

                n_pneumo = sum(1 for r in results if r["Prédiction"] == "PNEUMONIA")
                n_normal = len(results) - n_pneumo
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total analysé", len(results))
                with c2:
                    st.metric("🦠 Pneumonie", n_pneumo)
                with c3:
                    st.metric("✅ Normal", n_normal)


# ────────────────────────────────────────────────────────────────────────────
# Tab 3 — Pipeline Complet
# ────────────────────────────────────────────────────────────────────────────
with tab_pipeline:

    st.markdown('<div class="section-title">🔁 Pipeline Complet — Du Téléchargement à l\'Application</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      Ce pipeline reproduit l'intégralité du cycle de vie d'un modèle de Deep Learning médical :
      de la collecte des données jusqu'au déploiement de l'interface de diagnostic.<br>
      Chaque étape explique <strong>ce qui est fait</strong> et <strong>pourquoi ces valeurs ont été choisies</strong>.
    </div>
    """, unsafe_allow_html=True)

    # ── Diagramme ASCII du pipeline ─────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(15,23,42,0.8);border:1px solid rgba(56,189,248,0.2);
                border-radius:14px;padding:1.2rem 1.5rem;font-family:'JetBrains Mono',monospace;
                font-size:0.8rem;color:#94a3b8;line-height:2;overflow-x:auto;margin:1rem 0;">
      <span style="color:#38bdf8;font-weight:700;">① Téléchargement</span> ─►
      <span style="color:#818cf8;font-weight:700;">② EDA</span> ─►
      <span style="color:#06b6d4;font-weight:700;">③ Split &amp; Prétraitement</span> ─►
      <span style="color:#10b981;font-weight:700;">④ Modèle CNN</span> ─►
      <span style="color:#f59e0b;font-weight:700;">⑤ Entraînement</span> ─►
      <span style="color:#ef4444;font-weight:700;">⑥ Évaluation</span> ─►
      <span style="color:#a78bfa;font-weight:700;">⑦ Application</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════
    # ÉTAPE 1 — TÉLÉCHARGEMENT
    # ════════════════════════════════════════════════════════
    with st.expander("① 📥  Téléchargement du Dataset", expanded=True):
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown("""
            <div class="section-title" style="margin-top:0">📦 Dataset : Chest X-Ray (Kaggle)</div>

            Le dataset utilisé est le **Chest X-Ray Images (Pneumonia)** publié par Paul Mooney sur Kaggle.
            Il contient des radiographies thoraciques pédiatriques collectées au **Guangzhou Women and Children's Medical Center**.

            <div class="info-box" style="margin-top:0.8rem">
              🔗 <strong>Source :</strong> <code>kaggle datasets download -d paultimothymooney/chest-xray-pneumonia</code><br><br>
              📂 <strong>Structure attendue :</strong><br>
              <code>data/chest_xray/</code><br>
              <code>&nbsp;&nbsp;├── train/NORMAL/ &nbsp;&nbsp;&nbsp;(1 341 images)</code><br>
              <code>&nbsp;&nbsp;├── train/PNEUMONIA/ (3 875 images)</code><br>
              <code>&nbsp;&nbsp;├── val/NORMAL/ &nbsp;&nbsp;&nbsp;&nbsp;(8 images)</code><br>
              <code>&nbsp;&nbsp;├── val/PNEUMONIA/ &nbsp;&nbsp;(8 images)</code><br>
              <code>&nbsp;&nbsp;├── test/NORMAL/ &nbsp;&nbsp;&nbsp;(234 images)</code><br>
              <code>&nbsp;&nbsp;└── test/PNEUMONIA/ &nbsp;(390 images)</code>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card" style="margin-bottom:0.6rem">
              <div class="metric-value" style="color:#38bdf8">5 856</div>
              <div class="metric-label">Images Totales</div>
            </div>
            <div class="metric-card" style="margin-bottom:0.6rem">
              <div class="metric-value" style="color:#ef4444">74%</div>
              <div class="metric-label">PNEUMONIA (déséquilibre)</div>
            </div>
            <div class="metric-card">
              <div class="metric-value" style="color:#10b981">26%</div>
              <div class="metric-label">NORMAL</div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 2 — EDA
    # ════════════════════════════════════════════════════════
    with st.expander("② 🔍  Analyse Exploratoire (EDA)"):
        st.markdown("""
        <div class="section-title" style="margin-top:0">📊 Que révèle l'EDA ?</div>
        """, unsafe_allow_html=True)

        col_eda1, col_eda2 = st.columns(2, gap="large")
        with col_eda1:
            st.markdown("""
            <div class="info-box">
              <strong>🔢 Déséquilibre de classes</strong><br><br>
              Train : 3 875 PNEUMONIA vs 1 341 NORMAL → ratio ≈ <strong>3:1</strong><br><br>
              <strong>⚠️ Conséquence :</strong> Sans correction, le modèle apprend à prédire
              systématiquement PNEUMONIA (74% accuracy triviale).<br><br>
              <strong>✅ Solution choisie :</strong> <code>pos_weight = 0.33</code> dans la BCEWithLogitsLoss
              pour compenser ce déséquilibre au niveau de la fonction de perte.
            </div>
            """, unsafe_allow_html=True)

        with col_eda2:
            st.markdown("""
            <div class="info-box">
              <strong>🖼️ Caractéristiques des images</strong><br><br>
              • Format : JPEG, niveaux de gris convertis en RGB<br>
              • Résolution variable : 400–2000 px (nécessite redimensionnement)<br>
              • Contraste faible sur NORMAL, opacités sur PNEUMONIA<br><br>
              <strong>📈 Observations clés :</strong><br>
              – Pneumonie virale : infiltrats interstitiels diffus<br>
              – Pneumonie bactérienne : opacification lobaire dense<br>
              – NORMAL : poumons clairs, coupoles nettes
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:0.8rem">
          📓 <strong>Notebook :</strong> <code>notebooks/01_eda.ipynb</code> — Visualisation de la distribution par classe,
          analyse de la résolution, histogrammes de luminosité, et exemples annotés.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 3 — SPLIT & PRÉTRAITEMENT
    # ════════════════════════════════════════════════════════
    with st.expander("③ ✂️  Split du Dataset & Prétraitement"):
        st.markdown('<div class="section-title" style="margin-top:0">📐 Transformations & Augmentation</div>', unsafe_allow_html=True)

        col_s1, col_s2 = st.columns(2, gap="large")
        with col_s1:
            st.markdown("""
            <div class="info-box">
              <strong>🏋️ Transformations Train (augmentation)</strong><br><br>
              <table style="width:100%;font-size:0.8rem;color:#94a3b8;border-collapse:collapse">
                <tr><td style="color:#38bdf8;padding:3px 0;font-weight:600">Opération</td><td style="color:#38bdf8;font-weight:600">Valeur</td><td style="color:#38bdf8;font-weight:600">Pourquoi ?</td></tr>
                <tr><td>Resize</td><td>256×256</td><td>Marge pour crop aléatoire</td></tr>
                <tr><td>RandomCrop</td><td>224×224</td><td>Variation de cadrage</td></tr>
                <tr><td>RandomHorizontalFlip</td><td>p=0.5</td><td>Invariance gauche/droite</td></tr>
                <tr><td>RandomRotation</td><td>±10°</td><td>Robustesse orientations</td></tr>
                <tr><td>ColorJitter</td><td>brightness=0.2<br>contrast=0.2</td><td>Variabilité d\'appareils RX</td></tr>
                <tr><td>Normalize</td><td>μ=[0.485,0.456,0.406]<br>σ=[0.229,0.224,0.225]</td><td>Statistiques ImageNet</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with col_s2:
            st.markdown("""
            <div class="info-box">
              <strong>🧪 Transformations Val/Test (pas d\'augmentation)</strong><br><br>
              <table style="width:100%;font-size:0.8rem;color:#94a3b8;border-collapse:collapse">
                <tr><td style="color:#38bdf8;padding:3px 0;font-weight:600">Opération</td><td style="color:#38bdf8;font-weight:600">Valeur</td><td style="color:#38bdf8;font-weight:600">Pourquoi ?</td></tr>
                <tr><td>Resize</td><td>224×224</td><td>Taille standard du modèle</td></tr>
                <tr><td>ToTensor</td><td>[0, 1]</td><td>Conversion float32</td></tr>
                <tr><td>Normalize (ImageNet)</td><td>μ, σ</td><td>Cohérence avec Train</td></tr>
              </table>
              <br>
              <strong>ℹ️ Pourquoi 224×224 ?</strong><br>
              Taille standard pour ResNet/DenseNet/EfficientNet pré-entraînés sur ImageNet.
              Le BaseCNN adopte la même taille pour cohérence.
              <br><br>
              <strong>ℹ️ Pourquoi les stats ImageNet ?</strong><br>
              Même les modèles from-scratch bénéficient d\'une normalisation stable.
              Les radiographies en niveaux de gris sont converties en RGB (3 canaux identiques)
              avant normalisation.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box" style="margin-top:0.8rem">
          ⚠️ <strong>Validation set Kaggle (16 images)</strong> : Le split val officiel est très petit.
          Pendant l\'entraînement, la val_loss peut être bruitée. L\'évaluation finale se fait
          obligatoirement sur le <strong>test set (624 images)</strong> via <code>src/eval.py</code>.
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 4 — MODÈLE CNN
    # ════════════════════════════════════════════════════════
    with st.expander("④ 🧠  Architecture CNN"):
        st.markdown('<div class="section-title" style="margin-top:0">🏗️ Choix d\'Architecture</div>', unsafe_allow_html=True)

        col_m1, col_m2 = st.columns([1.1, 1], gap="large")
        with col_m1:
            st.markdown("""
            <div class="info-box">
              <strong>BaseCNN — Entraîné from scratch</strong><br><br>
              4 blocs Conv (Conv2d + BatchNorm + ReLU + MaxPool) :<br>
              3→32→64→128→256 filtres, kernel 3×3, padding=1<br><br>
              <strong>Pourquoi BatchNorm ?</strong><br>
              Stabilise les activations inter-couches, accélère la convergence,
              réduit la sensibilité au learning rate.<br><br>
              <strong>Pourquoi Global Average Pooling (GAP) ?</strong><br>
              Remplace Flatten + FC sur la carte spatiale entière.
              Réduit drastiquement le nombre de paramètres, régularise implicitement
              et supporte des images de tailles différentes.<br><br>
              <strong>Pourquoi Dropout(0.5) + Dropout(0.3) ?</strong><br>
              Double régularisation dans le MLP classifier.
              Drop 0.5 après GAP : fort taux pour forcer la redondance.
              Drop 0.3 dans FC intermédiaire : taux modéré pour conserver l\'expressivité.
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown("""
            <div class="info-box">
              <strong>Transfer Learning — Alternatives</strong><br><br>
              <span style="color:#38bdf8">ResNet-18</span> : skip connections (residual),
              idéal pour éviter le vanishing gradient. 11M params.<br><br>
              <span style="color:#818cf8">DenseNet-121</span> : connexions denses entre toutes les couches,
              réutilise les features. Fort pour les images médicales.<br><br>
              <span style="color:#06b6d4">EfficientNet-B0</span> : compromis optimal taille/performance.
              Architecture issue d\'une NAS (Neural Architecture Search). 5M params.<br><br>
              <strong>Stratégie en 2 phases :</strong><br>
              Phase 1 : backbone gelé (<code>freeze_backbone=True</code>)<br>
              → Entraîne uniquement la tête de classification<br>
              Phase 2 : dégel total à lr × 0.1<br>
              → Fine-tuning complet<br><br>
              <strong>Pourquoi geler d\'abord ?</strong><br>
              Évite de corrompre les poids ImageNet avec un gradient fort
              avant que la tête soit stabilisée.
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 5 — ENTRAÎNEMENT
    # ════════════════════════════════════════════════════════
    with st.expander("⑤ 🏋️  Entraînement — Hyperparamètres & Justifications"):
        st.markdown('<div class="section-title" style="margin-top:0">⚙️ Tous les Hyperparamètres Expliqués</div>', unsafe_allow_html=True)

        col_h1, col_h2 = st.columns(2, gap="large")
        with col_h1:
            st.markdown("""
            <div class="info-box">
              <strong>📉 Fonction de Perte : BCEWithLogitsLoss</strong><br><br>
              Binary Cross-Entropy combinée avec Sigmoid (plus stable numériquement
              que Sigmoid + BCE séparés).<br><br>
              <code>pos_weight = 0.33</code> → pondère les positifs (PNEUMONIA)
              par 0.33 pour compenser le déséquilibre 3:1.<br>
              Formule : <em>loss = -[w × y·log(σ(x)) + (1−y)·log(1−σ(x))]</em><br><br>
              <strong>❓ Pourquoi 0.33 ?</strong><br>
              <em>pos_weight ≈ N_négatifs / N_positifs = 1 341 / 3 875 ≈ 0.346</em><br>
              On arrondit à 0.33 : le modèle est ainsi plus prudent sur les FP
              (ne surestime pas la pneumonie).
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:0.8rem">
              <strong>⚡ Optimiseur : Adam</strong><br><br>
              Adam (Adaptive Moment Estimation) combine les avantages de
              RMSprop et Momentum.<br><br>
              <code>lr = 1e-3 (0.001)</code> : valeur par défaut recommandée par
              les auteurs d\'Adam (Kingma & Ba, 2014). Bonne convergence initiale.<br><br>
              <code>weight_decay = 1e-4 (0.0001)</code> : régularisation L2 légère.
              Pénalise les grands poids sans bloquer l\'apprentissage.
            </div>
            """, unsafe_allow_html=True)

        with col_h2:
            st.markdown("""
            <div class="info-box">
              <strong>📅 Scheduler : ReduceLROnPlateau</strong><br><br>
              Réduit automatiquement le lr quand la validation loss stagne.<br><br>
              <code>mode = "min"</code> → surveille la val_loss (minimiser)<br>
              <code>factor = 0.5</code> → lr ← lr × 0.5 à chaque déclenchement<br>
              <code>patience = 3</code> → attend 3 époques sans amélioration<br><br>
              <strong>❓ Pourquoi factor=0.5 ?</strong><br>
              Réduction douce. Factor=0.1 (classique) serait trop agressif
              pour un petit dataset médical — risque de stagnation prématurée.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:0.8rem">
              <strong>🛑 Early Stopping</strong><br><br>
              <code>patience = 5</code> : arrête l\'entraînement si la val_loss
              ne s\'améliore pas pendant 5 époques consécutives.<br><br>
              Le <strong>meilleur modèle</strong> est sauvegardé à chaque amélioration
              dans <code>outputs/checkpoints/best_model.pt</code>.<br><br>
              <strong>❓ Pourquoi patience=5 ?</strong><br>
              Le scheduler patience=3 s\'active en premier et réduit le lr.
              Les 2 époques supplémentaires (3+2=5) laissent le temps au
              modèle de bénéficier du lr réduit avant d\'arrêter.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:0.8rem">
          <table style="width:100%;font-size:0.82rem;color:#94a3b8;border-collapse:collapse">
            <tr style="border-bottom:1px solid rgba(56,189,248,0.15)">
              <td style="color:#38bdf8;font-weight:700;padding:6px 8px">Hyperparamètre</td>
              <td style="color:#38bdf8;font-weight:700;padding:6px 8px">Valeur</td>
              <td style="color:#38bdf8;font-weight:700;padding:6px 8px">Justification</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">epochs</td><td style="padding:5px 8px"><code>30</code></td>
              <td style="padding:5px 8px">Plafond raisonnable — early stopping arrête avant si convergé</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">batch_size</td><td style="padding:5px 8px"><code>32</code></td>
              <td style="padding:5px 8px">Compromis mémoire GPU / bruit de gradient (32 = standard)</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">learning_rate</td><td style="padding:5px 8px"><code>0.001</code></td>
              <td style="padding:5px 8px">Défaut Adam, convergence stable sur images médicales</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">dropout1</td><td style="padding:5px 8px"><code>0.5</code></td>
              <td style="padding:5px 8px">Fort taux après GAP pour prévenir l\'overfitting</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">dropout2</td><td style="padding:5px 8px"><code>0.3</code></td>
              <td style="padding:5px 8px">Taux modéré dans FC → conserve l\'expressivité</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">weight_decay</td><td style="padding:5px 8px"><code>0.0001</code></td>
              <td style="padding:5px 8px">Régularisation L2 légère, évite les grands poids</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">scheduler factor</td><td style="padding:5px 8px"><code>0.5</code></td>
              <td style="padding:5px 8px">Réduction douce du LR (÷2), pas trop agressive</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">scheduler patience</td><td style="padding:5px 8px"><code>3</code></td>
              <td style="padding:5px 8px">3 époques de grâce avant de réduire le LR</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">early stop patience</td><td style="padding:5px 8px"><code>5</code></td>
              <td style="padding:5px 8px">Scheduler(3) + buffer(2) = arrêt propre après 5 sans progrès</td>
            </tr>
            <tr style="border-bottom:1px solid rgba(56,189,248,0.08)">
              <td style="padding:5px 8px">gradient clip</td><td style="padding:5px 8px"><code>1.0</code></td>
              <td style="padding:5px 8px">Prévient l\'explosion du gradient (exploding gradient)</td>
            </tr>
            <tr>
              <td style="padding:5px 8px">pos_weight</td><td style="padding:5px 8px"><code>0.33</code></td>
              <td style="padding:5px 8px">N_normal/N_pneumo ≈ 1341/3875 → correction déséquilibre</td>
            </tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:0.8rem">
          <strong>🔄 Commande d\'entraînement :</strong><br>
          <code>python src/train.py --data_dir data/chest_xray --arch baseline --epochs 30 --batch 32 --lr 1e-3</code>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 6 — ÉVALUATION
    # ════════════════════════════════════════════════════════
    with st.expander("⑥ 📊  Évaluation & Métriques"):
        st.markdown('<div class="section-title" style="margin-top:0">📈 Métriques Choisies & Justifications</div>', unsafe_allow_html=True)

        col_e1, col_e2 = st.columns(2, gap="large")
        with col_e1:
            st.markdown("""
            <div class="info-box">
              <strong>🎯 Recall (Sensibilité) — Métrique prioritaire</strong><br><br>
              <em>Recall = TP / (TP + FN)</em><br><br>
              <strong>❓ Pourquoi le Recall avant l\'Accuracy ?</strong><br>
              En médecine, un <strong>Faux Négatif</strong> (pneumonie non détectée)
              est bien plus grave qu\'un Faux Positif (renvoi inutile chez un radiologue).<br><br>
              Un modèle avec Recall=0.95 mais Precision=0.70 est préférable à<br>
              Recall=0.75 / Precision=0.95 dans ce contexte clinique.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:0.8rem">
              <strong>📉 AUC-ROC</strong><br><br>
              Mesure la capacité discriminante <em>indépendamment du seuil</em>.<br>
              AUC = 1.0 → classifieur parfait | AUC = 0.5 → aléatoire.<br><br>
              Particulièrement utile pour comparer les architectures<br>
              (BaseCNN vs ResNet vs DenseNet) à seuil identique.
            </div>
            """, unsafe_allow_html=True)

        with col_e2:
            st.markdown("""
            <div class="info-box">
              <strong>📊 Matrice de Confusion — Analyse des Erreurs</strong><br><br>
              <table style="width:100%;font-size:0.8rem;color:#94a3b8;border-collapse:collapse;text-align:center">
                <tr><td></td><td style="color:#38bdf8;font-weight:600">Prédit NORMAL</td><td style="color:#38bdf8;font-weight:600">Prédit PNEUMONIA</td></tr>
                <tr><td style="color:#38bdf8;font-weight:600">Réel NORMAL</td>
                    <td style="background:rgba(16,185,129,0.2);padding:8px">✅ TN</td>
                    <td style="background:rgba(251,191,36,0.15);padding:8px">⚠️ FP</td></tr>
                <tr><td style="color:#38bdf8;font-weight:600">Réel PNEUMONIA</td>
                    <td style="background:rgba(239,68,68,0.2);padding:8px">🚨 FN (critique)</td>
                    <td style="background:rgba(16,185,129,0.2);padding:8px">✅ TP</td></tr>
              </table><br>
              Les <strong>FN en rouge</strong> sont les cas les plus critiques :
              des pneumonies non détectées par le modèle.
              On les surveille avec le <em>FNR (False Negative Rate)</em>.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-box" style="margin-top:0.8rem">
              <strong>🎚️ Seuil de décision (0.5)</strong><br><br>
              Seuil par défaut sauvegardé dans le checkpoint.<br>
              Peut être ajusté dans la sidebar de cette app pour<br>
              maximiser le Recall au détriment de la Precision<br>
              (seuil ↓ → plus de détections → Recall ↑, FP ↑).
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:0.8rem">
          <strong>▶️ Lancer l\'évaluation :</strong><br>
          <code>python src/eval.py --checkpoint outputs/checkpoints/best_model.pt --data_dir data/chest_xray</code>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # ÉTAPE 7 — APPLICATION
    # ════════════════════════════════════════════════════════
    with st.expander("⑦ 🚀  Application Streamlit (cette interface)"):
        st.markdown('<div class="section-title" style="margin-top:0">🎛️ Fonctionnalités de l\'Interface</div>', unsafe_allow_html=True)

        col_app1, col_app2 = st.columns(2, gap="large")
        with col_app1:
            st.markdown("""
            <div class="info-box">
              <strong>🔬 Inférence en temps réel</strong><br><br>
              1. Chargement du checkpoint (<code>@st.cache_resource</code> → une seule fois)<br>
              2. Prétraitement identique à l\'entraînement (Resize 224, Normalize ImageNet)<br>
              3. Forward pass → logit → Sigmoid → probabilité<br>
              4. Seuillage à la valeur configurée dans la sidebar<br><br>
              <strong>Grad-CAM en direct :</strong><br>
              Génère la heatmap d\'activation pour la dernière couche conv4 du BaseCNN,
              superposée sur la radiographie originale.
            </div>
            """, unsafe_allow_html=True)

        with col_app2:
            st.markdown("""
            <div class="info-box">
              <strong>📂 Analyse par lot</strong><br><br>
              Traitement séquentiel avec barre de progression.<br>
              Export du tableau de résultats (Pandas DataFrame).<br><br>
              <strong>▶️ Lancer l\'application :</strong><br>
              <code>streamlit run app.py</code><br><br>
              <strong>🔧 Fichiers du projet :</strong><br>
              <code>src/model.py</code> — Architectures CNN<br>
              <code>src/dataset.py</code> — Chargement données<br>
              <code>src/train.py</code> — Boucle entraînement<br>
              <code>src/eval.py</code> — Évaluation finale<br>
              <code>src/gradcam.py</code> — Visualisation Grad-CAM<br>
              <code>config.yaml</code> — Configuration centrale<br>
              <code>app.py</code> — Cette interface Streamlit
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Tab 4 — À propos du modèle
# ────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<div class="section-title">🧠 Architecture du Modèle</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="info-box">
          <strong>🏗️ BaseCNN (Baseline)</strong><br><br>
          <table style="width:100%;font-size:0.82rem;color:#94a3b8;border-collapse:collapse;">
            <tr><td style="padding:4px 0;color:#38bdf8;font-weight:600;">Couche</td><td style="color:#38bdf8;font-weight:600;">Sortie</td></tr>
            <tr><td>Input (radiographie RGB)</td><td>3 × 224 × 224</td></tr>
            <tr><td>ConvBlock 1 (Conv+BN+ReLU+Pool)</td><td>32 × 112 × 112</td></tr>
            <tr><td>ConvBlock 2</td><td>64 × 56 × 56</td></tr>
            <tr><td>ConvBlock 3</td><td>128 × 28 × 28</td></tr>
            <tr><td>ConvBlock 4</td><td>256 × 14 × 14</td></tr>
            <tr><td>Global Average Pooling</td><td>256</td></tr>
            <tr><td>Dropout(0.5) + FC(256→128)</td><td>128</td></tr>
            <tr><td>Dropout(0.3) + FC(128→1)</td><td>1 (logit)</td></tr>
            <tr><td><strong>Sigmoid → P(Pneumonie)</strong></td><td>[0, 1]</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="info-box">
          <strong>🔄 Transfer Learning</strong><br><br>
          Architectures pré-entraînées sur ImageNet :<br><br>
          • <strong style="color:#38bdf8">ResNet-18</strong> — 11M paramètres<br>
          • <strong style="color:#818cf8">DenseNet-121</strong> — 8M paramètres<br>
          • <strong style="color:#06b6d4">EfficientNet-B0</strong> — 5M paramètres<br><br>
          Stratégie <em>fine-tuning</em> en 2 phases :<br>
          1️⃣ Phase 1 : backbone gelé, tête entraînée<br>
          2️⃣ Phase 2 : backbone dégelé à lr réduit
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Grad-CAM — Interprétabilité</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      <strong>Comment fonctionne Grad-CAM ?</strong><br><br>
      1. Passe avant (forward) de l'image dans le réseau<br>
      2. Calcul du gradient de la sortie par rapport à la dernière couche convolutive<br>
      3. Moyenne des gradients → poids d'importance par canal de feature map<br>
      4. Combinaison pondérée + ReLU → heatmap d'activation<br>
      5. Superposition sur la radiographie originale<br><br>
      📖 Référence : <em>Selvaraju et al., ICCV 2017 — Grad-CAM: Visual Explanations from Deep Networks</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗂️ Dataset</div>', unsafe_allow_html=True)
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value" style="color:#38bdf8">5,216</div>
          <div class="metric-label">Images Train</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value" style="color:#818cf8">16</div>
          <div class="metric-label">Images Val</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d3:
        st.markdown("""
        <div class="metric-card">
          <div class="metric-value" style="color:#06b6d4">624</div>
          <div class="metric-label">Images Test</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box" style="margin-top:1rem;">
      ⚠️ <strong>Avertissement médical</strong> : Cette application est développée à des fins éducatives et de recherche uniquement.
      Les prédictions du modèle ne constituent pas un diagnostic médical et ne remplacent en aucun cas l'expertise d'un radiologue qualifié.
      Toute décision médicale doit être prise par un professionnel de santé agréé.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.8rem;padding:0.5rem 0;">
  PneumoScan AI — Deep Learning pour la Radiologie Thoracique &nbsp;·&nbsp;
  <span style="color:#38bdf8;">CNN Chest X-Ray</span> &nbsp;·&nbsp;
  Entraîné sur Kaggle Dataset<br>
  <span style="opacity:0.5;">Interface conçue pour l'évaluation et la démonstration de modèles IA médicaux</span>
</div>
""", unsafe_allow_html=True)

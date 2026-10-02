"""
app.py - Streamlit interface for EcoSort AI.   Run:  streamlit run app.py

Pages: Home | Prediction | Webcam | Model Information | About
The app loads the model chosen by evaluate.py (results/selected_model.json) or,
if that does not exist, the first checkpoint it finds. All numbers shown on the
"Model Information" page are READ FROM the files evaluate.py wrote - nothing is
hard-coded.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from src.config import CLASS_NAMES, DISPLAY_NAMES, RESULTS_DIR, LOW_CONFIDENCE
from src.gradcam import explain
from src.models import gradcam_target_layer
from src.predict import InvalidImageError, load_checkpoint, load_image_safe, predict_image

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide")

ICONS = {"dry": "🧻", "wet": "🍌", "recyclable": "♻️", "e_waste": "🔌"}
DESCRIPTIONS = {
    "dry": "Non-recyclable dry items such as clothes, shoes, general trash.",
    "wet": "Food scraps and other organic / biodegradable waste.",
    "recyclable": "Paper, cardboard, plastic, metal and glass.",
    "e_waste": "Batteries and discarded electronics.",
}


@st.cache_resource
def get_model():
    return load_checkpoint(None, "cpu")


def show_result(pil_img):
    model, meta = get_model()
    res = predict_image(model, pil_img, meta["img_size"])
    cam_img, _, _ = explain(model, gradcam_target_layer(model, meta["model_name"]), pil_img, meta["img_size"])

    col1, col2 = st.columns(2)
    col1.image(pil_img, caption="Original image", use_container_width=True)
    col2.image(cam_img, caption="Grad-CAM overlay (red = more influential)", use_container_width=True)

    st.markdown("### Prediction")
    name = DISPLAY_NAMES[res["predicted_class"]]
    st.success(f"{ICONS[res['predicted_class']]}  **{name}**  —  confidence **{res['confidence'] * 100:.1f}%**")
    if res["warning"]:
        st.warning("⚠️ " + res["warning"])
    st.markdown("**Top alternatives**")
    for i, (c, p) in enumerate(res["top3"], 1):
        st.write(f"{i}. {DISPLAY_NAMES[c]} — {p * 100:.1f}%")
        st.progress(min(max(p, 0.0), 1.0))
    st.caption("Grad-CAM is a visual indication of image regions that influenced this prediction. "
               "It does not prove how the model 'reasons'. The model can only choose among the four "
               "classes, so unrelated objects will still get a label - check the confidence.")


def page_home():
    st.title("♻️ EcoSort AI")
    st.write("EcoSort AI is a deep-learning image classifier that looks at a photo of waste and "
             "predicts which of four categories it belongs to, with a confidence score and a "
             "Grad-CAM visual explanation.")
    cols = st.columns(4)
    for col, c in zip(cols, CLASS_NAMES):
        col.markdown(f"### {ICONS[c]} {DISPLAY_NAMES[c]}")
        col.write(DESCRIPTIONS[c])
    st.info("Use the sidebar to upload a photo (Prediction) or take one (Webcam).")


def page_prediction():
    st.header("Prediction")
    up = st.file_uploader("Upload a waste image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if up is None:
        return
    try:
        img = load_image_safe(up.getvalue(), filename=up.name)
    except InvalidImageError as e:
        st.error(f"❌ {e}")
        return
    st.image(img, caption="Uploaded image", width=350)
    if st.button("Predict", type="primary"):
        with st.spinner("Classifying..."):
            show_result(img)


def page_webcam():
    st.header("Webcam")
    st.write("Take a photo with your camera (needs browser permission; not available on every device).")
    shot = st.camera_input("Capture waste item")
    if shot is not None:
        try:
            img = load_image_safe(shot.getvalue(), filename="capture.png")
        except InvalidImageError as e:
            st.error(f"❌ {e}")
            return
        show_result(img)


def page_model_info():
    st.header("Model Information")
    model, meta = get_model()
    st.write(f"**Architecture:** `{meta['model_name']}`  |  **Input size:** {meta['img_size']}×{meta['img_size']}  |  "
             f"**Classes ({len(CLASS_NAMES)}):** {', '.join(CLASS_NAMES)}")
    st.write(f"**Trained with augmentation:** {meta.get('augment')}  |  "
             f"**Best validation macro-F1:** {meta.get('best_val_macro_f1', float('nan')):.3f}")
    rep = RESULTS_DIR / "dataset_report.json"
    if rep.exists():
        st.subheader("Training dataset")
        st.json(json.loads(rep.read_text()))
    run = Path(meta["checkpoint"]).parent.name
    m = RESULTS_DIR / run / "test_metrics.json"
    if m.exists():
        d = json.loads(m.read_text())
        st.subheader("Test-set metrics (frozen test set)")
        c = st.columns(4)
        c[0].metric("Accuracy", f"{d['accuracy']:.3f}"); c[1].metric("Macro F1", f"{d['macro_f1']:.3f}")
        c[2].metric("Weighted F1", f"{d['weighted_f1']:.3f}"); c[3].metric("CPU ms/img", d["cpu_latency_ms"])
        st.dataframe(pd.DataFrame(d["per_class"]).T)
        cm = RESULTS_DIR / run / "confusion_matrix.png"
        if cm.exists():
            st.image(str(cm), width=450)
    else:
        st.info("Run `python src/evaluate.py` to generate test metrics.")
    comp = RESULTS_DIR / "model_comparison.csv"
    if comp.exists():
        st.subheader("Model comparison (same test set)")
        st.dataframe(pd.read_csv(comp))
    curves = RESULTS_DIR / run / "training_curves.png"
    if curves.exists():
        st.image(str(curves))


def page_about():
    st.header("About")
    st.markdown("""
**Objective:** classify waste photos into Dry, Wet, Recyclable and E-Waste to support segregation.

**Dataset:** public waste-image datasets remapped to four classes (see README → Dataset), de-duplicated and split
70/15/15 without leakage.

**Technologies:** Python, PyTorch, torchvision, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn, Streamlit, Grad-CAM.

**Limitations**
- Only four classes: unrelated images are still forced into one of them.
- Dataset photos are cleaner than real life; real-world accuracy is usually lower.
- One item per photo is assumed; mixed waste piles are unreliable.
- Class boundaries (e.g. dry vs recyclable) are partly a human convention and vary by city.
- Grad-CAM shows influential regions, not a proof of reasoning.
""")


PAGES = {"Home": page_home, "Prediction": page_prediction, "Webcam": page_webcam,
         "Model Information": page_model_info, "About": page_about}
choice = st.sidebar.radio("Navigate", list(PAGES))
try:
    PAGES[choice]()
except FileNotFoundError as e:
    st.error(f"Model not found: {e}\n\nTrain one first:  `python src/train.py --model resnet18`")

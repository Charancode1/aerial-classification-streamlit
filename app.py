# app.py — Streamlit app for Bird vs Drone classification + Grad-CAM
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import requests
import io

# --------- Config ----------
MODEL_PATH = "best_model_for_streamlit.pth"  # file in repo root (or use MODEL_URL in secrets)
IMG_SIZE = 224
CLASS_NAMES = ["bird", "drone"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Aerial Classifier", layout="centered")

st.title("Aerial Object Classifier — Bird vs Drone")
st.write("Upload an image and the model predicts Bird or Drone with probability. Grad-CAM visualization attempted when available.")

# --------- Build model helpers ----------
def build_efficientnet(num_classes=2):
    from torchvision import models
    m = models.efficientnet_b0(pretrained=False)
    in_f = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_f, num_classes)
    return m

def build_simplecnn(num_classes=2):
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=2, dropout_prob=0.4):
            super().__init__()
            def conv_block(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            self.features = nn.Sequential(
                conv_block(3, 32),
                conv_block(32, 64),
                conv_block(64, 128),
                conv_block(128, 256),
            )
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout_prob/2),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = self.classifier(x)
            return x
    return SimpleCNN(num_classes=num_classes)

# --------- Robust torch.load utilities ----------
def safe_torch_load(path_or_bytes, map_location="cpu"):
    """
    Try multiple safe ways to load a checkpoint:
      - path_or_bytes: either a filesystem path or raw bytes (downloaded)
    Returns the loaded object or raises RuntimeError.
    """
    import torch as _torch
    import numpy as _np
    last_err = None

    def try_load(obj):
        try:
            return _torch.load(obj, map_location=map_location)
        except Exception as e:
            raise e

    # If path_or_bytes is bytes-like, convert to BytesIO for torch.load
    obj = path_or_bytes
    is_bytes = isinstance(path_or_bytes, (bytes, bytearray, io.BytesIO))
    if is_bytes:
        obj = io.BytesIO(path_or_bytes)

    # 1) Try plain load
    try:
        return try_load(obj)
    except Exception as e1:
        last_err = e1

    # 2) Try weights_only=False (PyTorch >=2.6)
    try:
        try:
            return _torch.load(obj, map_location=map_location, weights_only=False)
        except TypeError:
            # this torch version doesn't accept weights_only
            raise
    except Exception as e2:
        last_err = e2

    # 3) Try to allowlist numpy scalar global then load again
    try:
        # add safe globals for numpy scalar if the API exists
        if hasattr(_torch.serialization, "add_safe_globals"):
            try:
                _torch.serialization.add_safe_globals([_np._core.multiarray.scalar])
            except Exception:
                pass
        elif hasattr(_torch.serialization, "safe_globals"):
            try:
                _torch.serialization.safe_globals([_np._core.multiarray.scalar])
            except Exception:
                pass

        # Try loading again; attempt weights_only=False first
        try:
            return _torch.load(obj, map_location=map_location, weights_only=False)
        except TypeError:
            return _torch.load(obj, map_location=map_location)
    except Exception as e3:
        last_err = e3

    raise RuntimeError(f"Failed to load checkpoint. Last error: {last_err}")

def extract_state_dict(ck):
    """
    Given a loaded checkpoint object, return a plain state_dict mapping.
    Handles common formats: {'model_state': ...}, {'state_dict': ...}, direct state_dict.
    """
    import torch as _torch
    if isinstance(ck, dict):
        if "model_state" in ck:
            return ck["model_state"]
        if "state_dict" in ck:
            return ck["state_dict"]
        # heuristic: if values are tensors -> treat as state_dict
        vals = list(ck.values())
        if len(vals) and all(isinstance(v, _torch.Tensor) for v in vals[:min(10, len(vals))]):
            return ck
    raise ValueError("Could not extract state_dict from checkpoint. Keys: " + (str(list(ck.keys())) if isinstance(ck, dict) else str(type(ck))))

# --------- Transforms ----------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --------- Load model once (supports MODEL_URL secret) ----------
@st.cache_resource
def get_model():
    # Determine path or download from MODEL_URL if provided in secrets
    path = None
    model_bytes = None
    # If user set MODEL_URL in Streamlit secrets, download model at runtime
    secret_url = None
    try:
        secret_url = st.secrets["MODEL_URL"]
    except Exception:
        secret_url = None

    if secret_url:
        try:
            resp = requests.get(secret_url, timeout=60)
            resp.raise_for_status()
            model_bytes = resp.content
        except Exception as e:
            st.error(f"Failed to download model from MODEL_URL: {e}")
            return None, None
    else:
        # use local model path in repo
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found at {MODEL_PATH}. Upload it to the repo root or set MODEL_URL in secrets.")
            return None, None
        path = MODEL_PATH

    # Load checkpoint safely
    try:
        ck = None
        if model_bytes is not None:
            ck = safe_torch_load(model_bytes, map_location=DEVICE)
        else:
            ck = safe_torch_load(path, map_location=DEVICE)
    except Exception as e:
        st.error(f"Failed to load checkpoint safely: {e}")
        return None, None

    # Extract state_dict
    try:
        state = extract_state_dict(ck)
    except Exception as e:
        # Try common fallback keys
        if isinstance(ck, dict) and "model_state" in ck:
            state = ck["model_state"]
        else:
            st.error(f"Failed to extract state_dict from checkpoint: {e}")
            return None, None

    # Try EfficientNet then SimpleCNN
    try:
        model = build_efficientnet(num_classes=2)
        model.load_state_dict(state)
        model.to(DEVICE)
        return model.eval(), "efficientnet_b0"
    except Exception:
        try:
            model = build_simplecnn(num_classes=2)
            model.load_state_dict(state)
            model.to(DEVICE)
            return model.eval(), "SimpleCNN"
        except Exception as e:
            st.error(f"Failed to load model into known architectures: {e}")
            return None, None

model, model_name = get_model()
if model is None:
    st.stop()

# --------- UI: file uploader ----------
uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
col1, col2 = st.columns([1,1])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    with col1:
        st.image(img, caption="Input image", use_column_width=True)
    # preprocess
    x = transform(img).unsqueeze(0).to(DEVICE)
    with st.spinner("Predicting..."):
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
            pred_idx = int(probs.argmax())
            pred_label = CLASS_NAMES[pred_idx]
            pred_prob = float(probs[pred_idx])

    with col2:
        st.markdown(f"### Prediction: **{pred_label}**")
        st.markdown(f"**Probability:** {pred_prob:.3f}")
        st.bar_chart({CLASS_NAMES[0]: float(probs[0]), CLASS_NAMES[1]: float(probs[1])})

    # Grad-CAM (best-effort)
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # pick last conv layer heuristically
        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                break
        if target_layer is not None:
            # grad-cam expects CPU/cuda depending on use, use CPU to be safe here
            cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
            grayscale_cam = cam(input_tensor=x.cpu(), targets=[ClassifierOutputTarget(pred_idx)])[0]
            rgb_img = np.array(img.resize((IMG_SIZE,IMG_SIZE))).astype(np.float32) / 255.0
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            st.image(cam_image, caption="Grad-CAM", use_column_width=True)
        else:
            st.info("Grad-CAM not available for this model.")
    except Exception as e:
        # do not fail the app if grad-cam not installed or errors out
        st.info("Grad-CAM unavailable: " + str(e))

st.write("---")
st.write("Loaded model:", model_name)
st.write("Classes:", CLASS_NAMES)

import os
from functools import lru_cache

import timm
import torch
from huggingface_hub import hf_hub_download

from modules.labels import LABELS, NUM_CLASSES  # noqa: F401  (re-exported)

DEFAULT_HF_REPO = "Haseeb88/neurovision-model"
DEFAULT_HF_FILE = "best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def _setting(name, default=None):
    """Read a setting from environment variables, then Streamlit secrets."""
    value = os.environ.get(name)

    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name)
    except Exception:
        value = None

    return value if value else default


def build_model():
    return timm.create_model(
        "swinv2_tiny_window8_256",
        pretrained=False,
        num_classes=NUM_CLASSES,
        in_chans=3
    )


def download_checkpoint():

    return hf_hub_download(
        repo_id=_setting("NEUROVISION_HF_REPO", DEFAULT_HF_REPO),
        filename=_setting("NEUROVISION_HF_FILE", DEFAULT_HF_FILE),
        revision=_setting("NEUROVISION_HF_REVISION", None),
        token=_setting("HF_TOKEN", None)
    )


def load_state_dict(path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # Accept a plain state_dict or {"model_state_dict": ...}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    return checkpoint


@lru_cache(maxsize=1)
def get_model():
    """Build the model and load the weights once per process."""

    model = build_model()

    model.load_state_dict(
        load_state_dict(download_checkpoint()),
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model


def __getattr__(name):
    # Keeps `from modules.model_handler import model` working (gradcam.py).
    if name == "model":
        return get_model()

    raise AttributeError(name)


def predict(tensor):

    model = get_model()

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probs = torch.sigmoid(
            logits
        ).cpu().numpy()[0]

    return probs

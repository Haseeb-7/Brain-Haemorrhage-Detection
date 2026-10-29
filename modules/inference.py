import numpy as np
import torch

from modules.model_handler import predict
from modules.preprocessor import (
    read_dicom,
    slice_from_dataset,
    generate_triple_window_stack,
    resize_for_model
)

IMG_SIZE = 256


def preprocess_dicom(data, frame=None, auto_fix=True):
    """
    data: raw bytes of a DICOM file.
    Returns (tensor, processed_img, info, warnings).
    processed_img has shape (3, 256, 256); channel 0 is the brain window.
    """
    ds = read_dicom(data)

    hu, info, warnings = slice_from_dataset(
        ds,
        frame=frame,
        auto_fix=auto_fix
    )

    img = generate_triple_window_stack(hu)
    img = resize_for_model(img, IMG_SIZE)

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    return tensor, img, info, warnings


def run_inference(data, frame=None, auto_fix=True):

    tensor, processed_img, info, warnings = preprocess_dicom(
        data,
        frame=frame,
        auto_fix=auto_fix
    )

    probs = predict(tensor)

    return np.asarray(probs), processed_img, tensor, info, warnings

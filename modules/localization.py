import torch
import cv2
import numpy as np

from modules.model_handler import get_model, DEVICE


def get_localization_map(input_tensor):

    model = get_model()

    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():

        x = model.patch_embed(input_tensor)

        x = model.layers[0](x)
        x = model.layers[1](x)
        x = model.layers[2](x)
        x = model.layers[3](x)

        feature_map = x

    feature_map = feature_map[0]

    feature_map = feature_map.mean(dim=-1)

    feature_map = feature_map.cpu().numpy()

    feature_map -= feature_map.min()

    if feature_map.max() > 0:
        feature_map /= feature_map.max()

    heatmap = cv2.resize(
        feature_map,
        (256, 256),
        interpolation=cv2.INTER_CUBIC
    )

    return np.clip(heatmap, 0.0, 1.0)


def get_bounding_box(heatmap):

    mask = (heatmap > 0.60).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest
    )

    return (x, y, w, h)

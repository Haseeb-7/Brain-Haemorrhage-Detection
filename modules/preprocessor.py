import io

import cv2
import numpy as np
import pydicom


def read_dicom(data):
    """Parse DICOM bytes. Raises ValueError if the file is not a readable DICOM."""
    try:
        return pydicom.dcmread(io.BytesIO(data))
    except Exception as exc:
        raise ValueError(
            "This file is not a readable DICOM (.dcm) file: " + str(exc)
        )


def slice_from_dataset(ds, frame=None, auto_fix=True):
    """
    Convert a DICOM dataset into a 2D Hounsfield-unit image.

    Returns (hu_image, info, warnings).
    Handles: multi-frame files, colour (non-CT) images, MONOCHROME1,
    missing rescale tags and compressed pixel data.
    """
    warnings = []

    try:
        arr = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise ValueError(
            "Could not decode the pixel data. The file may use a compression "
            "format that is not supported: " + str(exc)
        )

    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    n_frames = 1
    idx = 0
    color = samples > 1

    if color:
        if arr.ndim == 4:
            n_frames = arr.shape[0]
            idx = n_frames // 2 if frame is None or frame < 0 else min(frame, n_frames - 1)
            arr = arr[idx]
        if arr.ndim != 3:
            raise ValueError("Unsupported pixel data layout: " + str(arr.shape))
        raw = arr.astype(np.float32).mean(axis=-1)
        warnings.append(
            "This is a colour image, not a CT scan with Hounsfield values. "
            "Results are not meaningful."
        )
    else:
        if arr.ndim == 3:
            n_frames = arr.shape[0]
            idx = n_frames // 2 if frame is None or frame < 0 else min(frame, n_frames - 1)
            arr = arr[idx]
        if arr.ndim != 2:
            raise ValueError("Unsupported pixel data layout: " + str(arr.shape))
        raw = arr.astype(np.float32)

    if n_frames > 1:
        warnings.append(
            "This file has " + str(n_frames) + " frames; analysing frame "
            + str(idx + 1) + " only."
        )

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    if not color and photometric == "MONOCHROME1":
        raw = raw.max() + raw.min() - raw
        warnings.append("MONOCHROME1 image detected; intensities were inverted.")

    modality = getattr(ds, "Modality", None)
    if modality is not None and str(modality) != "CT":
        warnings.append("Modality is " + str(modality) + ", not CT.")

    has_slope = "RescaleSlope" in ds
    has_intercept = "RescaleIntercept" in ds
    slope = float(ds.RescaleSlope) if has_slope else 1.0
    intercept = float(ds.RescaleIntercept) if has_intercept else 0.0

    raw_min = float(raw.min())
    raw_max = float(raw.max())

    if not color and not has_intercept:
        if raw_max <= 255:
            warnings.append(
                "The image looks 8-bit and has no rescale tags, so it is probably "
                "an exported picture and not true CT data."
            )
        elif auto_fix and raw_min >= 0 and raw_max > 1500:
            intercept = -1024.0
            warnings.append(
                "RescaleIntercept is missing and the values look offset; "
                "assumed an intercept of -1024. Verify this is correct."
            )
        else:
            warnings.append(
                "RescaleIntercept is missing; assuming the values are already "
                "in Hounsfield units."
            )

    hu = raw * slope + intercept
    hu = hu.astype(np.float32)

    brain_fraction = float(np.mean((hu >= 0) & (hu <= 80)))
    if brain_fraction < 0.03:
        warnings.append(
            "Very few pixels fall in the brain-tissue range (0 to 80 HU). "
            "This may not be a head CT, or the values may not be in Hounsfield units."
        )

    if float(hu.min()) > -300:
        warnings.append(
            "No air-like pixels (below -300 HU) were found. CT scans normally "
            "contain air; the intensity values may not be in Hounsfield units."
        )

    info = {
        "modality": None if modality is None else str(modality),
        "photometric_interpretation": photometric,
        "rows": int(arr.shape[0]),
        "columns": int(arr.shape[1]),
        "frames": int(n_frames),
        "frame_used": int(idx + 1),
        "rescale_slope_used": slope,
        "rescale_intercept_used": intercept,
        "has_rescale_tags": bool(has_slope and has_intercept),
        "hu_min": float(hu.min()),
        "hu_max": float(hu.max()),
        "hu_mean": float(hu.mean()),
        "fraction_pixels_0_to_80_HU": round(brain_fraction, 4),
    }

    return hu, info, warnings


def get_hounsfield_units(dicom_file):
    """Backward-compatible helper: file object -> (hu_image, dataset)."""
    dicom_file.seek(0)
    ds = read_dicom(dicom_file.read())
    hu, _, _ = slice_from_dataset(ds)
    return hu, ds


def window_image(img, center, width):

    lower = center - width // 2
    upper = center + width // 2

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower + 1e-6)

    return img


def generate_triple_window_stack(hu_image):

    brain = window_image(hu_image, 40, 80)
    subdural = window_image(hu_image, 80, 200)
    bone = window_image(hu_image, 600, 2800)

    img = np.stack(
        [brain, subdural, bone],
        axis=-1
    )

    return img


def resize_for_model(img, size=256):

    img = np.stack([
        cv2.resize(np.ascontiguousarray(img[:, :, 0]), (size, size)),
        cv2.resize(np.ascontiguousarray(img[:, :, 1]), (size, size)),
        cv2.resize(np.ascontiguousarray(img[:, :, 2]), (size, size))
    ])

    return img

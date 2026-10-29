import os
import sqlite3
import hashlib

import numpy as np
import cv2
import streamlit as st

st.set_page_config(
    page_title="NeuroVision",
    layout="wide"
)

from modules.labels import LABELS, summarise_predictions
from modules.inference import run_inference
from modules.localization import (
    get_localization_map,
    get_bounding_box
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "users.db")

os.makedirs(DB_DIR, exist_ok=True)


@st.cache_resource
def get_connection():
    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


@st.cache_resource(show_spinner="Loading the model (first start can take a minute)...")
def load_model_once():
    from modules.model_handler import get_model
    return get_model()


conn = get_connection()
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

conn.commit()


def hash_password(password):
    return hashlib.sha256(
        password.encode()
    ).hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def analyse(data, frame, auto_fix):
    """Run the full pipeline once per (file, frame, option) combination."""

    probs, processed_img, tensor, info, warnings = run_inference(
        data,
        frame=frame,
        auto_fix=auto_fix
    )

    warnings = list(warnings)
    heatmap = None
    bbox = None

    try:
        heatmap = get_localization_map(tensor)
        bbox = get_bounding_box(heatmap)
    except Exception as exc:
        warnings.append(
            "Localization heatmap could not be computed: " + str(exc)
        )

    return {
        "probs": probs,
        "img": processed_img,
        "heatmap": heatmap,
        "bbox": bbox,
        "info": info,
        "warnings": warnings
    }


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""


if not st.session_state.logged_in:

    st.title("🧠 NeuroVision")

    tab1, tab2 = st.tabs(
        ["Login", "Register"]
    )

    with tab1:

        username = st.text_input(
            "Username"
        )

        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            c.execute(
                """
                SELECT * FROM users
                WHERE username=?
                AND password=?
                """,
                (
                    username,
                    hash_password(password)
                )
            )

            user = c.fetchone()

            if user:

                st.session_state.logged_in = True
                st.session_state.username = username

                st.rerun()

            else:

                st.error(
                    "Invalid username or password"
                )

    with tab2:

        new_username = st.text_input(
            "New Username"
        )

        new_password = st.text_input(
            "New Password",
            type="password"
        )

        if st.button("Register"):

            if not new_username or not new_password:

                st.error(
                    "Enter a username and a password"
                )

            else:

                try:

                    c.execute(
                        """
                        INSERT INTO users
                        (username,password)
                        VALUES (?,?)
                        """,
                        (
                            new_username,
                            hash_password(new_password)
                        )
                    )

                    conn.commit()

                    st.success(
                        "Registration successful"
                    )

                except Exception:

                    st.error(
                        "Username already exists"
                    )

else:

    st.sidebar.success(
        f"Logged in as {st.session_state.username}"
    )

    if st.sidebar.button("Logout"):

        st.session_state.logged_in = False
        st.session_state.username = ""

        st.rerun()

    st.sidebar.header("Settings")

    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
        help=(
            "A label is flagged when its probability is at or above this value. "
            "Lower values flag more scans (fewer misses, more false alarms). "
            "This threshold has not been tuned on a held-out test set."
        )
    )

    auto_fix = st.sidebar.checkbox(
        "Assume -1024 offset if rescale tags are missing",
        value=True,
        help=(
            "Some DICOM files lack the rescale tags needed to convert pixel "
            "values to Hounsfield units. When values look offset, this applies "
            "an intercept of -1024."
        )
    )

    st.title(
        "🧠 NeuroVision Brain Hemorrhage Detection"
    )

    st.warning(
        "Experimental research prototype. It is not clinically validated and "
        "must not be used for diagnosis or as a substitute for interpretation "
        "by a qualified clinician. In an emergency, seek medical care immediately."
    )

    st.write(
        "Upload a single head CT slice in DICOM format (.dcm). The model "
        "estimates an independent probability for each of six labels: "
        + ", ".join(LABELS) + "."
    )

    try:
        load_model_once()
    except Exception as exc:
        st.error(
            "The model could not be loaded, so predictions are unavailable: "
            + str(exc)
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a single head CT slice (DICOM)"
    )

    if uploaded_file:

        data = uploaded_file.getvalue()

        result = None
        error = None

        with st.spinner(
            "Running SwinV2 inference..."
        ):

            try:

                result = analyse(
                    data,
                    -1,
                    auto_fix
                )

                n_frames = result["info"]["frames"]

                if n_frames > 1:

                    chosen = st.sidebar.number_input(
                        "Frame to analyse",
                        min_value=1,
                        max_value=n_frames,
                        value=result["info"]["frame_used"],
                        step=1
                    )

                    result = analyse(
                        data,
                        int(chosen) - 1,
                        auto_fix
                    )

            except ValueError as exc:

                error = str(exc)

            except Exception as exc:

                error = "Unexpected error while analysing the file: " + str(exc)

        if error:

            st.error(error)

        else:

            probs = result["probs"]
            processed_img = result["img"]
            localization_map = result["heatmap"]
            bbox = result["bbox"]

            for message in result["warnings"]:
                st.warning(message)

            summary = summarise_predictions(probs, threshold)

            display_probs = summary["adjusted"]
            flagged = summary["flagged"]

            if summary["any_flagged"]:

                detail = ""

                if summary["flagged_subtypes"]:
                    detail = " Flagged subtypes: " + ", ".join(
                        summary["flagged_subtypes"]
                    ) + "."

                st.error(
                    "⚠ Possible hemorrhage flagged at threshold "
                    f"{threshold:.2f}." + detail
                )

            else:

                st.success(
                    "✅ No hemorrhage flagged at threshold "
                    f"{threshold:.2f}"
                )

            st.caption(
                "These are model scores, not calibrated clinical probabilities. "
                "A single slice can look normal even in a scan that contains a "
                "hemorrhage elsewhere."
            )

            st.subheader(
                "Label Scores"
            )

            st.table({
                "Label": LABELS,
                "Score": [f"{p:.4f}" for p in display_probs],
                "Flagged": ["YES" if f else "no" for f in flagged]
            })

            for label, prob in zip(LABELS, display_probs):

                st.progress(
                    min(max(prob, 0.0), 1.0),
                    text=f"{label}: {prob:.3f}"
                )

            brain_img = np.ascontiguousarray(
                (processed_img[0] * 255).astype(np.uint8)
            )

            brain_bgr = cv2.cvtColor(
                brain_img,
                cv2.COLOR_GRAY2BGR
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.subheader(
                    "Brain Window"
                )

                st.image(
                    brain_img,
                    use_container_width=True
                )

            if localization_map is not None:

                heatmap_color = cv2.applyColorMap(
                    np.uint8(localization_map * 255),
                    cv2.COLORMAP_JET
                )

                overlay = cv2.addWeighted(
                    brain_bgr,
                    0.7,
                    heatmap_color,
                    0.3,
                    0
                )

                bbox_img = brain_bgr.copy()

                if bbox is not None:

                    x, y, w, h = bbox

                    cv2.rectangle(
                        bbox_img,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                with col2:

                    st.subheader(
                        "Localization Heatmap"
                    )

                    st.image(
                        overlay,
                        channels="BGR",
                        use_container_width=True
                    )

                with col3:

                    st.subheader(
                        "Highlighted Region"
                    )

                    st.image(
                        bbox_img,
                        channels="BGR",
                        use_container_width=True
                    )

                st.caption(
                    "The heatmap shows where the model's internal features are "
                    "strongest. It is indicative only and has not been validated "
                    "as a hemorrhage locator."
                )

                if bbox is not None:

                    x, y, w, h = bbox

                    st.write(
                        f"Region (x, y, width, height): {x}, {y}, {w}, {h}"
                    )

            with st.expander("Scan diagnostics"):

                st.json(result["info"])

                st.write(
                    "Raw model outputs by index (before the Any adjustment): "
                    + ", ".join(
                        f"[{i}] {label}={float(p):.4f}"
                        for i, (label, p) in enumerate(zip(LABELS, probs))
                    )
                )

"""Single source of truth for the model's output labels (pure Python)."""
import json
import math
import os

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "class_mapping.json"
)


def load_class_mapping(path=_CONFIG_PATH):

    with open(path, encoding="utf-8") as f:
        config = json.load(f)

    order = config["output_order"]
    names = config["display_names"]

    if len(order) != 6 or len(set(order)) != 6:
        raise ValueError("class_mapping.json must list six unique labels")

    if "any" not in order:
        raise ValueError("class_mapping.json must contain the 'any' label")

    missing = [k for k in order if k not in names]

    if missing:
        raise ValueError("Missing display names for: " + ", ".join(missing))

    return config


CONFIG = load_class_mapping()
OUTPUT_ORDER = list(CONFIG["output_order"])
LABELS = [CONFIG["display_names"][k] for k in OUTPUT_ORDER]
ANY_INDEX = OUTPUT_ORDER.index("any")
NUM_CLASSES = len(OUTPUT_ORDER)


def summarise_predictions(probs, threshold):
    """
    probs: six sigmoid outputs in OUTPUT_ORDER.
    The 'any' score is raised to at least the strongest subtype score.
    """
    values = [float(p) for p in probs]

    if len(values) != NUM_CLASSES:
        raise ValueError(
            "Expected " + str(NUM_CLASSES) + " outputs, got " + str(len(values))
        )

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Model returned a non-finite value")

    strongest_subtype = max(
        v for i, v in enumerate(values) if i != ANY_INDEX
    )

    adjusted = list(values)
    adjusted[ANY_INDEX] = max(values[ANY_INDEX], strongest_subtype)

    flagged = [v >= threshold for v in adjusted]

    return {
        "labels": list(LABELS),
        "raw": values,
        "adjusted": adjusted,
        "flagged": flagged,
        "any_flagged": flagged[ANY_INDEX],
        "flagged_subtypes": [
            LABELS[i]
            for i in range(NUM_CLASSES)
            if i != ANY_INDEX and flagged[i]
        ]
    }

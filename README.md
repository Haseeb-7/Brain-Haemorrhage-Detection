# NeuroVision: Brain Haemorrhage Detection

Streamlit web app that estimates hemorrhage scores on a single head-CT slice (DICOM) using a SwinV2 transformer.
It reports six independent scores: **any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural**.

> **Research prototype. Not clinically validated. Not for diagnosis.** Trained only on the RSNA Intracranial
> Hemorrhage dataset. Never use it as a substitute for a qualified clinician. In an emergency seek care immediately.

## Important fix in this version

The label names in the original training notebook were shifted by one (pandas sorts pivoted columns
alphabetically; the notebook renamed them by position). The old app therefore read the *subdural* output as
"any hemorrhage". This version uses the correct order from `config/class_mapping.json`.
The existing trained weights remain valid and **do not need re-uploading**. Details: `AUDIT.md`.

## Project layout

```
app.py                      Streamlit app (login, upload, scores, heatmap, diagnostics)
requirements.txt
config/class_mapping.json   output index -> label (single source of truth)
modules/
  labels.py                 loads the mapping, summarises scores
  preprocessor.py           DICOM -> Hounsfield units -> 3 CT windows -> 256x256
  inference.py              shared inference pipeline used by the app
  model_handler.py          downloads and loads weights from Hugging Face (cached, lazy)
  localization.py           indicative heatmap / bounding box
  gradcam.py                Grad-CAM class (not used by the app)
training/
  train.py                  corrected training (patient-level split, augmentation, early stopping)
  evaluate.py               one-time evaluation on the held-out test split
  verify_label_order.py     checks which true label each model output responds to
  common.py data_utils.py augment.py
tests/                      unittest suite
db/users.db                 existing user accounts (see "Security notes")
model/README.md             where the weights live and how to configure them
AUDIT.md                    findings, evidence and limitations
```

## Run locally (Mac or Windows)

```
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The first start downloads the model from Hugging Face (about 110 MB).

## Deploy

1. **GitHub.** Upload the *contents* of this folder (not the zip) to your repository
   (`app.py`, `requirements.txt`, `README.md`, `.gitignore`, `config/`, `modules/`, `training/`, `tests/`, `db/`, `model/`, `AUDIT.md`).
   Delete the old top-level and old `modules/` files first so no outdated copies remain.
   Do **not** upload any `.pth` file.
2. **Hugging Face.** The model repository `Haseeb88/neurovision-model` already contains `best_model.pth`.
   Nothing to change. To use another repository, set `NEUROVISION_HF_REPO` (see below).
3. **Streamlit Community Cloud** (share.streamlit.io): *Create app -> Deploy a public app from GitHub*.
   Repository: yours; branch `main`; main file path `app.py`; *Advanced settings -> Python 3.11*.
4. **Secrets / variables (optional).** In the app's *Settings -> Secrets*:
   ```
   NEUROVISION_HF_REPO = "Haseeb88/neurovision-model"
   # NEUROVISION_HF_REVISION = "main"
   # HF_TOKEN = "hf_xxx"      # only needed if the model repository is private
   ```
5. If an update does not appear, delete the app and deploy it again so the packages are reinstalled.

Free-tier limits: PyTorch plus this model is close to Streamlit Community Cloud's memory limit, and
registrations stored in `db/users.db` are lost when the app restarts. Both are platform limitations.

## Quick sanity check after deploying

Use the DICOM samples your Kaggle notebook exports (`rsna_sample_images.zip`, folders named by true label).
Upload several slices from the `epidural`, `subdural` and `intraventricular` folders and open
*Scan diagnostics*. The score for the matching label should usually be the highest.
This is a smoke test on training-distribution data, not an evaluation.

## Verify the label mapping (recommended, once, on Kaggle)

```
python training/verify_label_order.py \
  --csv /kaggle/input/.../stage_2_train.csv \
  --dicom-dir /kaggle/input/.../stage_2_train \
  --checkpoint best_model.pth
```
It prints a 6x6 ROC-AUC matrix (model output vs. true label). The largest value in each row should sit under
the label that `config/class_mapping.json` assigns to that output index. The script prints whether it matches.

## Retrain (optional, to improve generalisation; GPU recommended)

```
python training/train.py --csv .../stage_2_train.csv --dicom-dir .../stage_2_train --out-dir run1
python training/evaluate.py --checkpoint run1/best_model.pth --splits run1/splits.csv \
       --dicom-dir .../stage_2_train --out-dir run1/eval
```
`train.py` selects labels by name, splits by patient (stratified), augments only the training split,
weights positives per label, stops early on validation macro AUC and writes `best_model.pth`,
`class_mapping.json`, `preprocessing.json`, `splits.csv`, `history.json`.
Run `evaluate.py` once, after training is final: thresholds are tuned on validation, then applied to the test split.
Evaluation reports per-label ROC-AUC, average precision, precision, recall, F1, specificity, support and one 2x2
confusion matrix per label (a six-class confusion matrix does not exist for a multi-label task).
To use a retrained model, upload its `best_model.pth` to your Hugging Face model repository
(Files and versions -> Add file -> Upload files) and keep `config/class_mapping.json` unchanged.

## Tests

```
python -m unittest discover -s tests -t . -v
```
Tests that need PyTorch, timm or the real `pydicom` are skipped automatically when those are not installed.

## Status of this release (honest summary)

| Item | Status |
|---|---|
| Label-order bug identified and reproduced | Done (executed) |
| App uses the corrected mapping | Done |
| Unit tests that need no PyTorch (23 tests: labels, DICOM edge cases, split, augmentation) | Executed and passed in the build environment |
| Tests needing PyTorch / timm / pydicom (7 tests) | **Not run** (not installable in the build environment) |
| Real model run, Hugging Face download, Streamlit app run | **Not run** by the author of this release; verify after deployment |
| Retraining and new evaluation metrics | **Not performed**; training code is provided but has never been executed |
| Clinical validation | None |

## Security notes

* `db/users.db` contains hashed accounts from the original app. Passwords are hashed with unsalted SHA-256
  (unchanged, so existing logins keep working). Use a salted hash (for example `bcrypt`) before any real use,
  and remove `db/users.db` from a public repository if those accounts matter.
* Do not upload real patient scans to a public deployment.

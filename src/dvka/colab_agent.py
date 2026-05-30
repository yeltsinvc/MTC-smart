from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .experiments import load_queue
from .kaggle_agent import build_workspace, slugify
from .paths import WORKSPACES


def render_colab_notebook(script: str, title: str, dataset_slug: str) -> dict[str, Any]:
    owner, _, name = dataset_slug.partition("/")
    setup = f'''# Colab bootstrap for {title}
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["KAGGLE_INPUT_DIR"] = "/content/kaggle/input"
os.environ["KAGGLE_WORKING_DIR"] = "/content/working"
os.environ["KAGGLE_TEMP_DIR"] = "/content/temp"

input_root = Path(os.environ["KAGGLE_INPUT_DIR"])
dataset_root = input_root / "datasets" / "{owner}" / "{name}"
dataset_root.mkdir(parents=True, exist_ok=True)
Path(os.environ["KAGGLE_WORKING_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["KAGGLE_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

kaggle_json = Path("/root/.kaggle/kaggle.json")
if not kaggle_json.exists():
    from google.colab import files
    print("Upload kaggle.json from your Kaggle account.")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("kaggle.json was not uploaded.")
    kaggle_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.move("kaggle.json", kaggle_json)
    kaggle_json.chmod(0o600)

if not (dataset_root / "train.csv").exists():
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", "{dataset_slug}",
        "-p", str(dataset_root),
        "--unzip",
    ])

print("Dataset root:", dataset_root)
print("Working dir:", os.environ["KAGGLE_WORKING_DIR"])
print("Temp dir:", os.environ["KAGGLE_TEMP_DIR"])
'''
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\nColab-ready DVKA training notebook. Select a GPU runtime before running.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in setup.splitlines()],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in script.splitlines()],
            },
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_colab(exp_id: str, output_dir: Path | None = None, single_gpu: bool = True) -> Path:
    queue = load_queue()
    matches = [item for item in queue if str(item.get("id")) == exp_id]
    if not matches:
        raise ValueError(f"Experiment not found in queue: {exp_id}")
    exp_for_colab = dict(matches[0])
    if single_gpu:
        exp_for_colab["id"] = f"{exp_for_colab['id']}_colab1g"
        exp_for_colab["batch"] = max(1, int(exp_for_colab.get("batch") or 1) // 2)
        exp_for_colab["notes"] = f"{exp_for_colab.get('notes', '')} Colab single-GPU batch adjustment.".strip()
    ws = build_workspace(exp_for_colab)
    exp = json.loads((ws / "experiment.json").read_text(encoding="utf-8"))
    script = (ws / "train_kernel.py").read_text(encoding="utf-8")
    title = f"DVKA Colab {slugify(exp_id)}"
    dataset_slug = str(exp.get("dataset_slug") or "")
    if not dataset_slug:
        raise ValueError(f"Experiment {exp_id} has no dataset_slug.")
    out_dir = output_dir or (WORKSPACES / "colab")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(exp_id)}-colab.ipynb"
    out_path.write_text(json.dumps(render_colab_notebook(script, title, dataset_slug), indent=2), encoding="utf-8")
    return out_path

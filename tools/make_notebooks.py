"""Generate the Colab notebooks in ``notebooks/`` from a single template.

Every notebook shares the same header (mount Drive, clone/update the repo branch,
install extras) and then runs one ``python -m cocogoat.tasks.<task>`` command whose
outputs (checkpoints, metrics, samples, logs, errors) go to Google Drive, so runs
resume automatically after a Colab disconnect.

Run:  python tools/make_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/Martyn0324/Cocogoat.git"
BRANCH = "claude/cocogoat-revival-glfnmz"
ROOT = Path(__file__).resolve().parents[1]

HEADER_MD = """# Cocogoat 2026 - {title}

{description}

**How to run:** `Runtime -> Change runtime type -> G4 (RTX PRO 6000)`, then `Runtime -> Run all`.
Keep this tab open (Colab Pro stops idle/closed sessions). Everything is written to
`MyDrive/Cocogoat/runs/...`; if Colab disconnects, just *Run all* again - training resumes from the last checkpoint.
You do not need to inspect results: metrics, sample grids and logs are saved to Drive for analysis."""

SETUP_CELLS = [
    """# 1) Mount Google Drive (outputs + datasets live there)
from google.colab import drive
drive.mount('/content/drive')""",
    f"""# 2) Get / update the code (public repo, branch {BRANCH})
import os, subprocess
REPO, BRANCH = "{REPO}", "{BRANCH}"
if not os.path.exists('/content/Cocogoat/.git'):
    subprocess.run(['git', 'clone', '-b', BRANCH, '--depth', '1', REPO, '/content/Cocogoat'], check=True)
else:
    subprocess.run(['git', '-C', '/content/Cocogoat', 'fetch', '--depth', '1', 'origin', BRANCH], check=True)
    subprocess.run(['git', '-C', '/content/Cocogoat', 'reset', '--hard', 'FETCH_HEAD'], check=True)
print(subprocess.run(['git', '-C', '/content/Cocogoat', 'log', '-1', '--oneline'], capture_output=True, text=True).stdout)
os.chdir('/content/Cocogoat')""",
    """# 3) Extra packages (torch/torchvision/transformers come with Colab)
!pip -q install -r /content/Cocogoat/requirements-colab.txt
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# Colab Secrets are only readable from the notebook kernel: export HF_TOKEN for the training subprocesses
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN exported')
except Exception as exc:
    print('No HF_TOKEN secret (only needed for gated Hugging Face datasets):', type(exc).__name__)
os.environ['PYTHONUNBUFFERED'] = '1'""",
]


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def notebook(title: str, description: str, run_cells: list[str]) -> dict:
    cells = [md_cell(HEADER_MD.format(title=title, description=description))]
    cells += [code_cell(c) for c in SETUP_CELLS]
    cells += [code_cell(c) for c in run_cells]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "G4", "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


NOTEBOOKS = {
    "00_probe.ipynb": notebook(
        "Probe (environment, data, datasets, pretrained weights)",
        "Short run (~15-25 min). Checks the GPU, benchmarks a few kernels, inspects your datasets in `MyDrive/Python/Projects/GANs/` (and `MyDrive/Cocogoat/data/` if present), "
        "lists candidate Hugging Face datasets and tests every pretrained-weight download the project needs. "
        "Writes `MyDrive/Cocogoat/runs/probe/report.md` (+ preview images).",
        [
            """# 4) Optional: a Hugging Face token (Colab 'Secrets' panel, name HF_TOKEN) is only needed for gated datasets.
!python -m cocogoat.tasks.probe --data_dir /content/drive/MyDrive/Python/Projects/GANs /content/drive/MyDrive/Cocogoat/data --out /content/drive/MyDrive/Cocogoat/runs/probe""",
            """# 5) Show the report
from IPython.display import Markdown, display
display(Markdown(open('/content/drive/MyDrive/Cocogoat/runs/probe/report.md').read()))""",
        ],
    ),

    "01_task1_gan64.ipynb": notebook(
        "Task 1: modern GAN at 64x64",
        "R3GAN objective (RpGAN + R1 + R2) with normalization-free depthwise-separable blocks and self-attention, "
        "trained on your Genshin fan-art set (`Complete_dataset.npy`, 45,935 images) with CLIP k-means pseudo-classes "
        "(cluster-conditioned GAN) against mode collapse. Budget: ~3.5 h of training + ~10 min data prep + final eval. "
        "Outputs: `MyDrive/Cocogoat/runs/task1/`.",
        [
            """# 4) GPU smoke test (tiny model, ~10 steps). The first run also converts the 11 GB float64 file into a
#    1.4 GB uint8 cache (MyDrive/Cocogoat/cache) - a few minutes, reused by every later run.
!python -m cocogoat.tasks.task1_gan64 --smoke --dataset genshin --run_dir /content/drive/MyDrive/Cocogoat/runs/task1_smoke --force""",
            """# 5) Full training run (resumes automatically from MyDrive/Cocogoat/runs/task1/ckpt/latest.pt)
!python -m cocogoat.tasks.task1_gan64 --dataset genshin --hours 3.5""",
            """# 6) Results
import json
from IPython.display import Image, Markdown, display
run = '/content/drive/MyDrive/Cocogoat/runs/task1'
display(Markdown(open(f'{run}/report.md').read()))
display(Image(f'{run}/samples/final.png'))
display(Image(f'{run}/samples/final_per_class.png'))""",
        ],
    ),

    "02_task2_vae_world_model.ipynb": notebook(
        "Task 2: VAE + latent world model (64x64 video)",
        "Per-frame beta-VAE (ConvNeXt depthwise blocks, PixelUnshuffle with DC-AE shortcuts, bottleneck attention, "
        "EQ-VAE regulariser, adversarial fine-tuning) + a space-time transformer world model with EMA memory that "
        "predicts the next frame's latent. Two datasets: Colored Moving MNIST (controlled dynamics, the first rung of "
        "your ViVAE curriculum) and your `Shelterdataset.npz`. Budget: ~1.5 h per dataset. "
        "Outputs: `MyDrive/Cocogoat/runs/task2/<dataset>/{vae,wm}/`.",
        [
            """# 4) GPU smoke test (tiny models, a few steps)
!python -m cocogoat.tasks.task2_world --smoke --dataset mmnist --run_root /content/drive/MyDrive/Cocogoat/runs/task2_smoke --force""",
            """# 5) Colored Moving MNIST: VAE (45 min) then world model (45 min); resumes automatically
!python -m cocogoat.tasks.task2_world --dataset mmnist""",
            """# 6) Shelter (your anime music-video frames): VAE (45 min) then world model (45 min)
!python -m cocogoat.tasks.task2_world --dataset shelter""",
            """# 7) Results
from IPython.display import Image, Markdown, display
for ds in ['mmnist', 'shelter']:
    run = f'/content/drive/MyDrive/Cocogoat/runs/task2/{ds}'
    try:
        display(Markdown(f'## {ds}\\n' + open(f'{run}/wm/report.md').read()))
        display(Image(f'{run}/vae/samples/recon_final.png'))
        display(Image(f'{run}/wm/samples/rollout_final.png'))
    except FileNotFoundError as exc:
        print(ds, 'not finished:', exc)""",
        ],
    ),

    "03_task3_latent_gan.ipynb": notebook(
        "Task 3: class-conditional latent GAN (z -> G -> z' -> Decoder -> D)",
        "Stage A trains a beta-VAE on the Genshin set (1 h) and freezes it. Stage B trains a generator that works "
        "entirely in latent space (noise with the latent's shape -> displacement z' = z + G(z, class)), decoded by "
        "the frozen decoder and judged by two critics: an image critic on decoded fakes vs *decoded reals* and a "
        "latent critic on z' vs posterior samples (keeps z' on the decoder's manifold). Same CLIP pseudo-classes and "
        "metrics as Task 1, so the two are directly comparable. Budget: ~4 h. Outputs: `MyDrive/Cocogoat/runs/task3/`.",
        [
            """# 4) GPU smoke test (tiny models, a few steps)
!python -m cocogoat.tasks.task3_latent_gan --smoke --dataset genshin --run_dir /content/drive/MyDrive/Cocogoat/runs/task3_smoke --force""",
            """# 5) VAE (1 h) then latent GAN (3 h); resumes automatically
!python -m cocogoat.tasks.task3_latent_gan --dataset genshin""",
            """# 6) Results
from IPython.display import Image, Markdown, display
run = '/content/drive/MyDrive/Cocogoat/runs/task3'
display(Markdown(open(f'{run}/gan/report.md').read()))
display(Image(f'{run}/vae/samples/recon_final.png'))
display(Image(f'{run}/gan/samples/final.png'))
display(Image(f'{run}/gan/samples/final_per_class.png'))""",
        ],
    ),

    "04_task4_hires_1024.ipynb": notebook(
        "Task 4: prompt-conditional 1024x1024 GAN (~10.5 h)",
        "FFHQ-1024 (70k faces) downloaded from the Hugging Face mirror on Colab. Base GAN at 256x256 conditioned on "
        "CLIP ViT-L/14 embeddings (language-free LAFITE training: image embeddings + noise while training, text "
        "prompts at inference), with a pixel critic and a projected critic on frozen DINOv2-S features "
        "(~7 h). Then a x4 super-resolution GAN trained from scratch on real 1024 crops (~2.5 h). Final evaluation at "
        "1024: KID/FID, patch-KID on native-resolution crops, precision/recall, CLIP score for text prompts. "
        "Outputs: `MyDrive/Cocogoat/runs/task4/`. Needs ~130 GB of the Colab disk (dataset + decoded arrays).",
        [
            """# 4) GPU smoke test (tiny models on synthetic shards, a few steps)
!python -m cocogoat.tasks.task4_hires --smoke --run_dir /content/drive/MyDrive/Cocogoat/runs/task4_smoke --force""",
            """# 5) Data (~20-30 min: 64 GB download + decode + CLIP embeddings), base GAN (7 h), SR (2.5 h), 1024 evaluation.
#    Everything resumes: if Colab disconnects, Run all again (the data step re-downloads on a fresh VM).
!python -m cocogoat.tasks.task4_hires""",
            """# 6) Results
from IPython.display import Image, Markdown, display
run = '/content/drive/MyDrive/Cocogoat/runs/task4'
display(Markdown(open(f'{run}/eval/report.md').read()))
display(Image(f'{run}/eval/samples/prompts_grid.png'))
display(Image(f'{run}/sr/samples/sr_final.png'))""",
        ],
    ),
}


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        (out / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        print("wrote", out / name)


if __name__ == "__main__":
    main()

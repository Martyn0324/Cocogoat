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

**How to run:** `Runtime -> Change runtime type -> G4 (RTX PRO 6000)`, set the budgets in the settings cell (5),
then `Runtime -> Run all`.
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
            """# 4) GPU smoke test (tiny model, ~10 steps, always from scratch). The first run also converts the 11 GB
#    float64 file into a 1.4 GB uint8 cache (MyDrive/Cocogoat/cache) - a few minutes, reused by every later run.
!python -m cocogoat.tasks.task1_gan64 --smoke --dataset genshin --run_dir /content/drive/MyDrive/Cocogoat/runs/task1_smoke --force""",
            """# 5) Settings - edit freely. HOURS = training time budget (evaluation/data time not included).
#    Raising HOURS later on the same RUN_DIR continues training from the last checkpoint.
HOURS = 3.5
RUN_DIR = '/content/drive/MyDrive/Cocogoat/runs/task1'""",
            """# 6) Training (resumes automatically after a disconnect)
!python -m cocogoat.tasks.task1_gan64 --dataset genshin --hours {HOURS} --run_dir {RUN_DIR}""",
            """# 7) Results
import json
from IPython.display import Image, Markdown, display
run = RUN_DIR
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
            """# 4) GPU smoke test (tiny models, a few steps, always from scratch)
!python -m cocogoat.tasks.task2_world --smoke --dataset mmnist --run_root /content/drive/MyDrive/Cocogoat/runs/task2_smoke --force""",
            """# 5) Settings - edit freely. Training time budgets in hours, per dataset (evaluation/data time not included).
#    Quick test: small budgets AND a separate folder (e.g. '.../runs/task2_test'): a finished Task 2 stage is not
#    extended later, so a test inside runs/task2 would block the full run there.
VAE_HOURS, WM_HOURS = 0.75, 0.75
RUN_ROOT = '/content/drive/MyDrive/Cocogoat/runs/task2'
# World-model evaluation: EVAL_CONTEXT real frames are shown, then EVAL_HORIZON frames are generated (any length;
# clips are rolled out in batches). A Shelter clip must have CONTEXT + HORIZON frames without a scene cut.
EVAL_CONTEXT, EVAL_HORIZON = 64, 64
# Shelter: drop the black intro (the anime starts around frame 500); 0 keeps every frame.
SHELTER_START_FRAME = 500
EVAL = f'--eval_context {EVAL_CONTEXT} --eval_horizon {EVAL_HORIZON}'""",
            """# 6) Colored Moving MNIST: VAE, then world model (resumes automatically after a disconnect)
!python -m cocogoat.tasks.task2_world --dataset mmnist --vae_hours {VAE_HOURS} --wm_hours {WM_HOURS} --run_root {RUN_ROOT} {EVAL}""",
            """# 7) Shelter (your anime music-video frames): VAE, then world model
!python -m cocogoat.tasks.task2_world --dataset shelter --vae_hours {VAE_HOURS} --wm_hours {WM_HOURS} --run_root {RUN_ROOT} {EVAL} --start_frame {SHELTER_START_FRAME}""",
            """# 8) Results
from IPython.display import Image, Markdown, display
for ds in ['mmnist', 'shelter']:
    run = f'{RUN_ROOT}/{ds}'
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
            """# 4) GPU smoke test (tiny models, a few steps, always from scratch)
!python -m cocogoat.tasks.task3_latent_gan --smoke --dataset genshin --run_dir /content/drive/MyDrive/Cocogoat/runs/task3_smoke --force""",
            """# 5) Settings - edit freely. Training time budgets in hours: stage A = VAE, stage B = latent GAN.
#    Quick test: small budgets AND a separate folder (e.g. '.../runs/task3_test'): the finished VAE stage is not
#    extended later. (Raising GAN_HOURS on the same RUN_DIR does continue the GAN's training.)
VAE_HOURS, GAN_HOURS = 1.0, 3.0
RUN_DIR = '/content/drive/MyDrive/Cocogoat/runs/task3'""",
            """# 6) VAE, then latent GAN (resumes automatically after a disconnect)
!python -m cocogoat.tasks.task3_latent_gan --dataset genshin --vae_hours {VAE_HOURS} --hours {GAN_HOURS} --run_dir {RUN_DIR}""",
            """# 7) Results
from IPython.display import Image, Markdown, display
run = RUN_DIR
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
            """# 4) GPU smoke test (tiny models on synthetic shards, a few steps, always from scratch)
!python -m cocogoat.tasks.task4_hires --smoke --run_dir /content/drive/MyDrive/Cocogoat/runs/task4_smoke --force""",
            """# 5) Settings - edit freely. Training time budgets in hours (data/evaluation time not included).
#    MAX_SHARDS: FFHQ shards of ~1,000 faces each; 0 = all 70 (~64 GB download, ~20-30 min with decoding).
#    Quick test: e.g. MAX_SHARDS = 5 with small budgets, in a separate folder (e.g. '.../runs/task4_test').
#    Raising BASE_HOURS / SR_HOURS later on the same RUN_DIR continues training from the checkpoints.
MAX_SHARDS, BASE_HOURS, SR_HOURS = 0, 7.0, 2.5
RUN_DIR = '/content/drive/MyDrive/Cocogoat/runs/task4'""",
            """# 6) Data (download + decode + CLIP embeddings), base GAN, SR, 1024 evaluation. Everything resumes: if Colab
#    disconnects, Run all again (the data step re-downloads on a fresh VM).
!python -m cocogoat.tasks.task4_hires --max_shards {MAX_SHARDS} --base_hours {BASE_HOURS} --sr_hours {SR_HOURS} --run_dir {RUN_DIR}""",
            """# 7) Results
from IPython.display import Image, Markdown, display
run = RUN_DIR
display(Markdown(open(f'{run}/eval/report.md').read()))
display(Image(f'{run}/eval/samples/prompts_grid.png'))
display(Image(f'{run}/sr/samples/sr_final.png'))""",
        ],
    ),

    "05_task5_world_gan_1024.ipynb": notebook(
        "Task 5: latent world-GAN - images and videos at 1024x1024 (~10 h)",
        "Inter4K (1,000 real 4K clips, 15 fps) downloaded on Colab. Per-frame beta-VAE at 256 px (1.5 h), then a "
        "latent image generator + an adversarial world model trained jointly (5.5 h) with three critics (latent "
        "frame, decoded frame, temporal window), then a x4 SR GAN on native 1024 crops (2 h). One system, two "
        "modes: world model OFF -> 1024 px images, ON -> 1024 px videos (mp4 + GIF). Text prompts work through CLIP. "
        "Outputs: `MyDrive/Cocogoat/runs/task5/`.",
        [
            """# 4) GPU smoke test (tiny models on synthetic clips, always from scratch)
!python -m cocogoat.tasks.task5_world_gan --smoke --run_dir /content/drive/MyDrive/Cocogoat/runs/task5_smoke --force""",
            """# 5) Settings - edit freely. Training time budgets in hours (data/evaluation time not included).
#    MAX_CLIPS: Inter4K clips to use; 0 = all 1,000 (~15-25 min to download and decode).
#    Quick test: e.g. MAX_CLIPS = 100 with small budgets, in a separate folder (e.g. '.../runs/task5_test'):
#    the finished VAE stage is not extended later. (Raising GAN_HOURS / SR_HOURS on the same RUN_DIR continues them.)
MAX_CLIPS, VAE_HOURS, GAN_HOURS, SR_HOURS = 0, 1.5, 5.5, 2.0
RUN_DIR = '/content/drive/MyDrive/Cocogoat/runs/task5'
# Frames generated by the world model: EVAL_HORIZON in periodic evaluations (after 4 real context frames),
# VIDEO_FRAMES in the final videos.
EVAL_HORIZON, VIDEO_FRAMES = 8, 24""",
            """# 6) Data, VAE, world-GAN, SR, evaluation + videos (resumes automatically after a disconnect)
!python -m cocogoat.tasks.task5_world_gan --max_clips {MAX_CLIPS} --vae_hours {VAE_HOURS} --gan_hours {GAN_HOURS} --sr_hours {SR_HOURS} --run_dir {RUN_DIR} --eval_horizon {EVAL_HORIZON} --video_frames {VIDEO_FRAMES}""",
            """# 7) Results
from IPython.display import Image, Markdown, Video, display
run = RUN_DIR
display(Markdown(open(f'{run}/eval/report.md').read()))
display(Image(f'{run}/eval/samples/prompts_images.png'))
display(Image(f'{run}/eval/videos/prompt0.gif'))""",
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

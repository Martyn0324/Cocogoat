"""Colab environment + data probe.

Runs a set of independent checks and writes a machine-readable ``report.json``
plus a readable ``report.md`` (and a few preview images) to ``--out``.  Every
section is isolated: a failure is recorded in the report instead of aborting.

Usage (Colab):
    python -m cocogoat.tasks.probe --data_dir /content/drive/MyDrive/Cocogoat/data \
        --out /content/drive/MyDrive/Cocogoat/runs/probe
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

REPORT: dict = {"sections": {}}

HF_CANDIDATES = [
    # (repo_id, purpose)
    ("zh-plus/tiny-imagenet", "task1/3 fallback: 64x64, 200 classes"),
    ("gaunernst/ffhq-1024-wds", "task4: FFHQ 1024 (webdataset)"),
    ("yangtao9009/FFHQ1024", "task4: FFHQ 1024 (zip)"),
    ("AnimeFans/Iwara_MMD_all", "task5: anime MMD videos (may be NSFW)"),
    ("ACCA225/iwara_metadata", "task5: Iwara metadata (ratings/tags)"),
    ("SynthForensics/SynthForensics", "task2/5: people-centric synthetic videos (gated)"),
    ("ermu2001/Inter4kPllavaRecaption", "task5: Inter4K mirror (4K clips)"),
]


def section(name):
    """Decorator: run a probe section, record its result or traceback."""

    def deco(fn):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            print(f"\n=== [{name}] ===", flush=True)
            try:
                result = fn(*args, **kwargs)
                REPORT["sections"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": result}
            except Exception as exc:  # noqa: BLE001 - we want every failure in the report
                tb = traceback.format_exc()
                print(tb, flush=True)
                REPORT["sections"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                            "error": f"{type(exc).__name__}: {exc}", "traceback": tb[-4000:]}
            return REPORT["sections"][name]

        return wrapper

    return deco


def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _save_grid(images: np.ndarray, path: Path, max_side: int = 1024) -> None:
    """Save up to 16 HWC uint8 images as a 4x4 grid PNG."""
    from PIL import Image

    imgs = images[:16]
    n = len(imgs)
    if n == 0:
        return
    h, w = imgs[0].shape[:2]
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=2)
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im[..., :3]
    pil = Image.fromarray(grid)
    scale = max_side / max(pil.size)
    if scale < 1:
        pil = pil.resize((int(pil.size[0] * scale), int(pil.size[1] * scale)), Image.BILINEAR)
    pil.save(path)


def _to_uint8_hwc(x: np.ndarray) -> np.ndarray:
    """Best-effort conversion of an image batch to uint8 NHWC for previews."""
    x = np.asarray(x)
    if x.ndim == 3:  # N,H,W (grayscale) or H,W,C single image
        x = x[None] if x.shape[-1] in (1, 3, 4) else x[..., None]
    if x.ndim == 4 and x.shape[1] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (0, 2, 3, 1))
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi <= 1.0 + 1e-6 and lo >= -1.0 - 1e-6:
        x = (x + 1.0) * 127.5 if lo < 0 else x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------

@section("system")
def probe_system():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            mem = {line.split(":")[0]: line.split(":")[1].strip() for line in f}
        info["ram_total"] = mem.get("MemTotal")
        info["ram_available"] = mem.get("MemAvailable")
    except OSError:
        pass
    for p in ["/content", "/tmp", "/"]:
        if os.path.exists(p):
            du = shutil.disk_usage(p)
            info[f"disk_{p}"] = {"total": _human(du.total), "free": _human(du.free)}
    try:
        import importlib.metadata as md

        pkgs = {}
        for name in ["torch", "torchvision", "numpy", "transformers", "huggingface_hub", "datasets",
                     "opencv-python", "opencv-python-headless", "pillow", "lpips", "pytorch-fid",
                     "scikit-learn", "pandas", "pyarrow", "timm", "accelerate", "av", "imageio"]:
            try:
                pkgs[name] = md.version(name)
            except md.PackageNotFoundError:
                pkgs[name] = None
        info["packages"] = pkgs
    except Exception as exc:  # noqa: BLE001
        info["packages_error"] = str(exc)
    print(json.dumps(info, indent=1))
    return info


@section("gpu")
def probe_gpu():
    import torch

    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return info
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    info.update(name=props.name, capability=f"{props.major}.{props.minor}", total_memory=_human(props.total_memory),
                multiprocessors=props.multi_processor_count, bf16=torch.cuda.is_bf16_supported())
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters

    a = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    t = bench(lambda: a @ b)
    info["matmul_bf16_tflops"] = round(2 * 8192 ** 3 / t / 1e12, 1)
    a32 = a.float()
    b32 = b.float()
    t = bench(lambda: a32 @ b32, iters=5)
    info["matmul_tf32_tflops"] = round(2 * 8192 ** 3 / t / 1e12, 1)
    del a, b, a32, b32

    # dense vs depthwise conv (forward+backward), channels_last bf16
    x = torch.randn(128, 256, 32, 32, device=dev, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    x.requires_grad_(True)
    dense = torch.nn.Conv2d(256, 256, 3, padding=1).to(dev, torch.bfloat16).to(memory_format=torch.channels_last)
    dw = torch.nn.Conv2d(256, 256, 3, padding=1, groups=256).to(dev, torch.bfloat16).to(memory_format=torch.channels_last)
    pw = torch.nn.Conv2d(256, 256, 1).to(dev, torch.bfloat16).to(memory_format=torch.channels_last)
    info["conv_dense3x3_fwdbwd_ms"] = round(1000 * bench(lambda: dense(x).sum().backward()), 2)
    info["conv_depthwise3x3_fwdbwd_ms"] = round(1000 * bench(lambda: dw(x).sum().backward()), 2)
    info["conv_pointwise1x1_fwdbwd_ms"] = round(1000 * bench(lambda: pw(x).sum().backward()), 2)
    del x, dense, dw, pw

    # attention: SDPA kernels + double backward through math attention (needed for R1/R2)
    import torch.nn.functional as F

    q = torch.randn(64, 8, 256, 64, device=dev, dtype=torch.bfloat16, requires_grad=True)
    info["sdpa_fwdbwd_ms"] = round(1000 * bench(lambda: F.scaled_dot_product_attention(q, q, q).sum().backward()), 2)
    qf = torch.randn(4, 4, 64, 32, device=dev, requires_grad=True)
    attn = torch.softmax(qf @ qf.transpose(-1, -2) / 32 ** 0.5, dim=-1) @ qf
    (g,) = torch.autograd.grad(attn.square().sum(), qf, create_graph=True)
    g.square().sum().backward()
    info["manual_attention_double_backward"] = True
    try:
        qf2 = torch.randn(4, 4, 64, 32, device=dev, requires_grad=True)
        out = F.scaled_dot_product_attention(qf2, qf2, qf2)
        (g2,) = torch.autograd.grad(out.square().sum(), qf2, create_graph=True)
        g2.square().sum().backward()
        info["sdpa_double_backward"] = True
    except Exception as exc:  # noqa: BLE001
        info["sdpa_double_backward"] = f"unsupported: {type(exc).__name__}"
    info["max_memory_allocated"] = _human(torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
    print(json.dumps(info, indent=1))
    return info


@section("drive_data")
def probe_drive_data(data_dir: Path, out_dir: Path):
    res = {"data_dir": str(data_dir), "exists": data_dir.exists(), "files": []}
    if not data_dir.exists():
        print(f"{data_dir} does not exist")
        return res
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        entry = {"path": str(p.relative_to(data_dir)), "size": _human(p.stat().st_size)}
        suffix = p.suffix.lower()
        try:
            if suffix == ".npy":
                arr = np.load(p, mmap_mode="r")
                entry.update(shape=list(arr.shape), dtype=str(arr.dtype))
                sample = np.asarray(arr[: min(16, len(arr))])
                entry.update(sample_min=float(sample.min()), sample_max=float(sample.max()))
                _save_grid(_to_uint8_hwc(sample), out_dir / f"preview_{p.stem}.png")
            elif suffix == ".npz":
                with np.load(p, allow_pickle=False) as z:
                    keys = {}
                    for k in z.files:
                        arr = z[k]
                        keys[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                        if arr.dtype != object and arr.size > 0 and arr.ndim >= 3:
                            flat = arr.reshape(-1, *arr.shape[-3:]) if arr.ndim > 4 else arr
                            sample = np.asarray(flat[: min(16, len(flat))])
                            keys[k].update(sample_min=float(sample.min()), sample_max=float(sample.max()))
                            _save_grid(_to_uint8_hwc(sample), out_dir / f"preview_{p.stem}_{k}.png")
                    entry["arrays"] = keys
            elif suffix in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
                import cv2

                cap = cv2.VideoCapture(str(p))
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                entry.update(frames=n, fps=fps, width=w, height=h, seconds=round(n / max(fps, 1e-6), 1))
                frames = []
                for idx in np.linspace(0, max(n - 1, 0), 16).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ok, fr = cap.read()
                    if ok:
                        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                        fr = cv2.resize(fr, (256, int(256 * h / max(w, 1))))
                        frames.append(fr)
                cap.release()
                if frames:
                    _save_grid(np.stack(frames), out_dir / f"preview_{p.stem[:40]}.png")
            elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
                from PIL import Image

                with Image.open(p) as im:
                    entry.update(image_size=list(im.size), mode=im.mode)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        print(entry, flush=True)
        res["files"].append(entry)
    return res


@section("huggingface")
def probe_huggingface(out_dir: Path, max_entries: int = 3000):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore

            token = userdata.get("HF_TOKEN")
        except Exception:  # noqa: BLE001
            token = None
    api = HfApi(token=token)
    res = {"has_token": bool(token), "repos": {}}
    for repo_id, purpose in HF_CANDIDATES:
        entry = {"purpose": purpose}
        try:
            info = api.dataset_info(repo_id)
            entry.update(gated=getattr(info, "gated", None), private=getattr(info, "private", None),
                         last_modified=str(getattr(info, "last_modified", "")),
                         card_size_category=(info.card_data or {}).get("size_categories") if info.card_data else None)
            files, total, ext = [], 0, Counter()
            for i, item in enumerate(api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)):
                if i >= max_entries:
                    entry["truncated_listing_at"] = max_entries
                    break
                size = getattr(item, "size", None)
                if size is None:  # folder
                    continue
                total += size
                ext[Path(item.path).suffix.lower() or "<none>"] += 1
                if len(files) < 40:
                    files.append(f"{item.path} ({_human(size)})")
            entry.update(n_files_seen=sum(ext.values()), bytes_seen=_human(total), extensions=dict(ext.most_common(10)),
                         first_files=files)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        print(repo_id, json.dumps(entry, indent=1)[:3000], flush=True)
        res["repos"][repo_id] = entry

    # Peek inside the first FFHQ webdataset shard without downloading it fully.
    try:
        import tarfile

        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=token)
        shards = sorted(p for p in fs.glob("datasets/gaunernst/ffhq-1024-wds/**/*.tar"))
        peek = {"n_shards": len(shards), "first": shards[:3]}
        if shards:
            with fs.open(shards[0], "rb", block_size=8 * 1024 * 1024) as f:
                with tarfile.open(fileobj=f, mode="r|") as tf:
                    members = []
                    for j, m in enumerate(tf):
                        members.append(f"{m.name} ({_human(m.size)})")
                        if m.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and "image" not in peek:
                            from io import BytesIO

                            from PIL import Image

                            data = tf.extractfile(m).read()
                            with Image.open(BytesIO(data)) as im:
                                peek["image"] = {"size": list(im.size), "mode": im.mode, "format": im.format}
                        if j >= 7:
                            break
            peek["members"] = members
        res["ffhq_wds_peek"] = peek
        print("ffhq peek", peek, flush=True)
    except Exception as exc:  # noqa: BLE001
        res["ffhq_wds_peek"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    # Tiny-ImageNet: read the small validation parquet and decode one image.
    try:
        import io

        import pandas as pd
        from huggingface_hub import hf_hub_download
        from PIL import Image

        files = [f for f in api.list_repo_files("zh-plus/tiny-imagenet", repo_type="dataset") if f.endswith(".parquet")]
        valid = [f for f in files if "valid" in f] or files
        path = hf_hub_download("zh-plus/tiny-imagenet", valid[0], repo_type="dataset")
        df = pd.read_parquet(path)
        col_img = [c for c in df.columns if "image" in c.lower()][0]
        cell = df.iloc[0][col_img]
        raw = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(raw)) as im:
            img_info = {"size": list(im.size), "mode": im.mode}
        res["tinyimagenet_peek"] = {"parquet_files": files, "columns": list(df.columns), "rows": len(df),
                                    "dtypes": {c: str(t) for c, t in df.dtypes.items()}, "image": img_info}
        print(res["tinyimagenet_peek"], flush=True)
    except Exception as exc:  # noqa: BLE001
        res["tinyimagenet_peek"] = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
    return res


@section("pretrained")
def probe_pretrained():
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}

    def timed(name, fn):
        t0 = time.time()
        try:
            out = fn()
            res[name] = {"ok": True, "seconds": round(time.time() - t0, 1), **(out or {})}
        except Exception as exc:  # noqa: BLE001
            res[name] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
        print(name, res[name], flush=True)

    def inception():
        from pytorch_fid.inception import InceptionV3

        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        with torch.no_grad():
            f = net(torch.rand(4, 3, 64, 64, device=dev))[0]
        return {"feature_shape": list(f.shape)}

    def lpips_vgg():
        import lpips

        net = lpips.LPIPS(net="vgg", verbose=False).to(dev)
        with torch.no_grad():
            d = net(torch.rand(2, 3, 64, 64, device=dev) * 2 - 1, torch.rand(2, 3, 64, 64, device=dev) * 2 - 1)
        return {"out_shape": list(d.shape)}

    def clip_model(name):
        def fn():
            from transformers import CLIPModel, CLIPTokenizer

            model = CLIPModel.from_pretrained(name).to(dev).eval()
            tok = CLIPTokenizer.from_pretrained(name)
            with torch.no_grad():
                size = model.config.vision_config.image_size
                img = model.get_image_features(pixel_values=torch.rand(2, 3, size, size, device=dev))
                txt = model.get_text_features(**tok(["a photo of a cat"], return_tensors="pt").to(dev))
            img = img if torch.is_tensor(img) else img.pooler_output
            txt = txt if torch.is_tensor(txt) else txt.pooler_output
            return {"image_emb": list(img.shape), "text_emb": list(txt.shape), "image_size": size}

        return fn

    def dinov2():
        from transformers import AutoModel

        model = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
        with torch.no_grad():
            out = model(pixel_values=torch.rand(2, 3, 224, 224, device=dev), output_hidden_states=True)
        return {"n_hidden_states": len(out.hidden_states), "hidden_shape": list(out.hidden_states[-1].shape)}

    def mnist():
        import torchvision

        ds = torchvision.datasets.MNIST("/content/cache/mnist", train=True, download=True)
        return {"n": len(ds)}

    def vgg19():
        import torchvision

        torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
        return {}

    timed("inception_fid_weights", inception)
    timed("lpips_vgg", lpips_vgg)
    timed("vgg19_imagenet", vgg19)
    timed("clip_vit_b32", clip_model("openai/clip-vit-base-patch32"))
    timed("clip_vit_l14", clip_model("openai/clip-vit-large-patch14"))
    timed("dinov2_small", dinov2)
    timed("mnist", mnist)
    return res


def write_report(out_dir: Path) -> None:
    REPORT["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "report.json").write_text(json.dumps(REPORT, indent=1, default=str))
    lines = ["# Cocogoat probe report", "", f"finished: {REPORT['finished']}", ""]
    for name, sec in REPORT["sections"].items():
        status = "OK" if sec.get("ok") else "FAILED"
        lines.append(f"## {name} - {status} ({sec.get('seconds')} s)")
        if sec.get("ok"):
            lines.append("```json")
            lines.append(json.dumps(sec.get("result"), indent=1, default=str)[:20000])
            lines.append("```")
        else:
            lines.append(f"**{sec.get('error')}**")
            lines.append("```")
            lines.append(sec.get("traceback", "")[-3000:])
            lines.append("```")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"\nReport written to {out_dir / 'report.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", default="/content/drive/MyDrive/Cocogoat/data")
    ap.add_argument("--out", default="/content/drive/MyDrive/Cocogoat/runs/probe")
    ap.add_argument("--skip", nargs="*", default=[], help="sections to skip: system gpu drive_data huggingface pretrained")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT["args"] = vars(args)
    if "system" not in args.skip:
        probe_system()
    if "gpu" not in args.skip:
        probe_gpu()
    if "drive_data" not in args.skip:
        probe_drive_data(Path(args.data_dir), out_dir)
    write_report(out_dir)  # partial report early, in case later sections hang
    if "huggingface" not in args.skip:
        probe_huggingface(out_dir)
    write_report(out_dir)
    if "pretrained" not in args.skip:
        probe_pretrained()
    write_report(out_dir)


if __name__ == "__main__":
    main()

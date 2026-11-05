"""Array-to-uint8 conversion of the user's datasets and the evaluation GIF writer."""

import numpy as np
import pytest
import torch

from cocogoat.data.images import load_array_images


def _video(n=600, black=500, seed=0):
    """uint8 frames whose first ``black`` frames are black (like Shelter's intro)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(0, 256, size=(n, 16, 16, 3), dtype=np.uint8)
    v[:black] = 0
    return v


def _save(tmp_path, arr, kind):
    if kind == "npz":
        path = tmp_path / "data.npz"
        np.savez_compressed(path, frames=arr)
    else:
        path = tmp_path / "data.npy"
        np.save(path, arr)
    return path


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@pytest.mark.parametrize("kind", ["npz", "npy"])
@pytest.mark.parametrize("encoding", ["unit", "signed", "255", "imagenet", "unit_nchw"])
def test_float_arrays_decode_to_the_original_pixels(tmp_path, kind, encoding):
    u8 = _video()
    x = u8.astype(np.float32)
    enc = {"unit": x / 255, "signed": x / 127.5 - 1, "255": x, "imagenet": (x / 255 - MEAN) / STD,
           "unit_nchw": (x / 255).transpose(0, 3, 1, 2)}[encoding]
    out = np.asarray(load_array_images(_save(tmp_path, enc, kind), cache_dir=tmp_path / "cache", chunk=128))
    assert out.shape == u8.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - u8.astype(int)).max() <= 1


def test_faulty_cache_is_rebuilt(tmp_path):
    u8 = _video()
    path = _save(tmp_path, u8.astype(np.float32) / 255, "npz")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "data_u8.npy", (u8 > 127).astype(np.uint8))  # what the old conversion produced: values 0/1
    out = np.asarray(load_array_images(path, cache_dir=cache, chunk=128))
    assert np.array_equal(out, u8)


def test_black_dataset_raises(tmp_path):
    path = _save(tmp_path, np.zeros((40, 8, 8, 3), np.float32), "npz")
    with pytest.raises(ValueError, match="black"):
        load_array_images(path, cache_dir=tmp_path / "cache")


def test_video_gif(tmp_path):
    from PIL import Image

    from cocogoat.train.utils import save_video_gif

    vids = torch.rand(4, 7, 3, 8, 8) * 2 - 1
    colours = torch.zeros(4, 7, 3, dtype=torch.uint8)
    colours[:, 3:] = torch.tensor([220, 0, 0], dtype=torch.uint8)
    save_video_gif(vids, tmp_path / "v.gif", nrow=2, fps=5, borders=colours)
    im = Image.open(tmp_path / "v.gif")
    assert im.n_frames == 7 and im.size == (2 * 12, 2 * 12)

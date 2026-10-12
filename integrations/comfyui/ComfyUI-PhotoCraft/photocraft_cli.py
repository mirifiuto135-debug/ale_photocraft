"""Bridge between ComfyUI tensors and the headless ``photocraft-cli`` binary.

Every node in this package writes its input image to a PNG in a temporary folder, runs
``photocraft-cli run <in.png> --cmd <id> --params <json> --out <out.png>`` and reads the PNG back.
The binary is found, in this order, through the ``PHOTOCRAFT_CLI`` environment variable, a
``bin/`` folder next to this file, or the ``PATH``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
BINARY = "photocraft-cli.exe" if sys.platform.startswith("win") else "photocraft-cli"
#: Seconds a single command may take (Remove Background on a large image is the slow one).
TIMEOUT = float(os.environ.get("PHOTOCRAFT_CLI_TIMEOUT", "600"))


class PhotoCraftError(RuntimeError):
    """A command failed or the CLI could not be found; the message is meant for the ComfyUI user."""


def find_cli() -> str:
    """Path of the ``photocraft-cli`` binary, or a `PhotoCraftError` explaining where to put it."""
    candidates = []
    env = os.environ.get("PHOTOCRAFT_CLI")
    if env:
        candidates.append(Path(env))
    candidates.append(HERE / "bin" / BINARY)
    on_path = shutil.which("photocraft-cli")
    if on_path:
        candidates.append(Path(on_path))
    for c in candidates:
        if c.is_file():
            return str(c)
    raise PhotoCraftError(
        "photocraft-cli not found. Build it with `cargo build --release -p photocraft-cli` and either "
        "set the PHOTOCRAFT_CLI environment variable to the binary, copy it to "
        f"{HERE / 'bin' / BINARY}, or put it on the PATH."
    )


# ---------------------------------------------------------------- tensor <-> numpy <-> PNG


def to_numpy(x) -> np.ndarray:
    """A ComfyUI tensor (or a numpy array) as a float32 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def from_numpy(a: np.ndarray):
    """A numpy array as a torch tensor when torch is importable (ComfyUI), else the array itself."""
    try:
        import torch  # noqa: WPS433 (optional at test time)
    except ImportError:  # pragma: no cover - only outside ComfyUI
        return a
    return torch.from_numpy(np.ascontiguousarray(a))


def image_to_rgba8(image: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """One ``[H, W, 3]`` float image (0..1) plus an optional ``[H, W]`` alpha as ``[H, W, 4]`` uint8."""
    if image.ndim != 3 or image.shape[2] < 3:
        raise PhotoCraftError(f"expected an image of shape [H, W, 3], got {list(image.shape)}")
    h, w = image.shape[:2]
    rgb = np.clip(image[:, :, :3], 0.0, 1.0)
    if alpha is None:
        a = np.ones((h, w), dtype=np.float32)
    else:
        a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
        if a.shape != (h, w):
            raise PhotoCraftError(f"alpha of shape {list(a.shape)} does not match the image {h}x{w}")
    rgba = np.concatenate([rgb, a[:, :, None]], axis=2)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


def write_png(path: Path, image: np.ndarray, alpha: np.ndarray | None) -> None:
    """Write one image as an 8-bit PNG: RGB when fully opaque (so PhotoCraft opens it as a
    Background layer, like Photoshop), RGBA when it carries transparency."""
    rgba = image_to_rgba8(image, alpha)
    if alpha is None or np.all(rgba[:, :, 3] == 255):
        Image.fromarray(rgba[:, :, :3], "RGB").save(path, format="PNG")
    else:
        Image.fromarray(rgba, "RGBA").save(path, format="PNG")


def read_png(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A PNG as ``([H, W, 3] float32 RGB, [H, W] float32 alpha)``."""
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    return np.ascontiguousarray(rgba[:, :, :3]), np.ascontiguousarray(rgba[:, :, 3])


# ---------------------------------------------------------------- running commands


def run_commands(in_png: Path, out_png: Path, steps: list[tuple[str, dict]]) -> list[dict]:
    """Open ``in_png``, run ``steps`` (command id, params) in order, save ``out_png``.

    Returns each command's JSON result. Raises `PhotoCraftError` with the CLI's own message when a
    command is refused (bad params, nothing to trim to, no subject found...)."""
    args = [find_cli(), "run", str(in_png)]
    for cmd, params in steps:
        args += ["--cmd", cmd, "--params", json.dumps(params or {})]
    args += ["--out", str(out_png), "--format", "png"]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT, check=False)
    except subprocess.TimeoutExpired as e:
        raise PhotoCraftError(f"photocraft-cli timed out after {TIMEOUT:.0f}s running {steps}") from e
    except OSError as e:
        raise PhotoCraftError(f"could not start photocraft-cli ({args[0]}): {e}") from e
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "").strip() or f"exit code {proc.returncode}"
        raise PhotoCraftError(f"PhotoCraft: {msg}")
    results = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not out_png.is_file():
        raise PhotoCraftError(f"PhotoCraft wrote no output file ({proc.stderr.strip()})")
    return results


def apply(images, alphas, steps: list[tuple[str, dict]]):
    """Run ``steps`` on every image of a ComfyUI batch.

    ``images`` is ``[B, H, W, 3]``, ``alphas`` ``[B, H, W]`` (1 = opaque) or None. Returns
    ``(images, alphas, results)`` as tensors plus the list of per-image command results. Every
    output must come out the same size, as a ComfyUI batch is one tensor."""
    imgs = to_numpy(images)
    if imgs.ndim == 3:
        imgs = imgs[None]
    als = None if alphas is None else to_numpy(alphas)
    if als is not None and als.ndim == 2:
        als = als[None]
    if als is not None and als.shape[0] not in (1, imgs.shape[0]):
        raise PhotoCraftError(f"{als.shape[0]} alphas for {imgs.shape[0]} images")
    out_imgs, out_als, results = [], [], []
    with tempfile.TemporaryDirectory(prefix="photocraft-comfy-") as tmp:
        tmp = Path(tmp)
        for i in range(imgs.shape[0]):
            a = None if als is None else als[min(i, als.shape[0] - 1)]
            src, dst = tmp / f"in-{i}.png", tmp / f"out-{i}.png"
            write_png(src, imgs[i], a)
            results.append(run_commands(src, dst, steps))
            rgb, alpha = read_png(dst)
            out_imgs.append(rgb)
            out_als.append(alpha)
    sizes = {o.shape[:2] for o in out_imgs}
    if len(sizes) > 1:
        raise PhotoCraftError(
            f"the images of the batch came out at different sizes {sorted(sizes)}; "
            "process them one at a time or give the command a fixed size"
        )
    return from_numpy(np.stack(out_imgs)), from_numpy(np.stack(out_als)), results


def size_of(images) -> tuple[int, int]:
    """``(width, height)`` of a ComfyUI image batch."""
    shape = tuple(images.shape)
    if len(shape) == 4:
        return int(shape[2]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[0])
    raise PhotoCraftError(f"expected an image batch [B, H, W, C], got {list(shape)}")

"""End-to-end tests of the nodes against a built photocraft-cli (numpy only, no torch).

Run from the repository root after `cargo build --release -p photocraft-cli`:
    python3 -m unittest discover -s integrations/comfyui/ComfyUI-PhotoCraft/tests
Set PHOTOCRAFT_CLI to the binary when it is not at target/release/photocraft-cli.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

PKG_DIR = Path(__file__).resolve().parents[1]
REPO = PKG_DIR.parents[2]
sys.path.insert(0, str(PKG_DIR.parent))
os.environ.setdefault("PHOTOCRAFT_CLI", str(REPO / "target" / "release" / ("photocraft-cli.exe" if sys.platform.startswith("win") else "photocraft-cli")))

nodes = cli = None


def _load():
    # The package folder is "ComfyUI-PhotoCraft": import it under a Python-friendly alias.
    import importlib.util

    spec = importlib.util.spec_from_file_location("photocraft_nodes", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["photocraft_nodes"] = mod
    spec.loader.exec_module(mod)
    return sys.modules["photocraft_nodes.nodes"], sys.modules["photocraft_nodes.photocraft_cli"]


try:
    nodes, cli = _load()
    HAVE_CLI = Path(cli.find_cli()).is_file()
except Exception:  # noqa: BLE001
    HAVE_CLI = False


def disc(w=160, h=120, r=40, bg=(0.2, 0.3, 0.8), fg=(0.9, 0.5, 0.1)):
    """A flat disc on a flat background: easy for Select Subject, with a known bbox."""
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 <= r * r
    img = np.empty((h, w, 3), np.float32)
    img[:] = bg
    img[inside] = fg
    rng = np.random.default_rng(1)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)[None], inside


@unittest.skipUnless(HAVE_CLI, "photocraft-cli not built (set PHOTOCRAFT_CLI)")
class NodeTests(unittest.TestCase):
    def test_image_size_proportional(self):
        img, _ = disc()
        out, alpha, w, h = nodes.PhotoCraftImageSize().run(img, 80, 0, "bicubic")
        self.assertEqual((w, h), (80, 60))
        self.assertEqual(out.shape, (1, 60, 80, 3))
        self.assertEqual(alpha.shape, (1, 60, 80))
        self.assertTrue(np.all(alpha == 1.0))

    def test_image_size_every_resample(self):
        img, _ = disc(40, 20)
        for r in nodes.RESAMPLE:
            out, _, w, h = nodes.PhotoCraftImageSize().run(img, 20, 10, r)
            self.assertEqual((w, h), (20, 10), r)

    def test_scale_by(self):
        img, _ = disc()
        _, _, w, h = nodes.PhotoCraftScaleBy().run(img, 0.25, "lanczos")
        self.assertEqual((w, h), (40, 30))

    def test_canvas_size_transparent_extension(self):
        img, _ = disc(40, 30)
        out, alpha, w, h = nodes.PhotoCraftCanvasSize().run(img, 60, 50, False, "topLeft", "transparent", "#ffffff")
        self.assertEqual((w, h), (60, 50))
        self.assertTrue(np.all(alpha[0, :30, :40] == 1.0))
        self.assertTrue(np.all(alpha[0, 30:, :] == 0.0))
        self.assertTrue(np.all(alpha[0, :, 40:] == 0.0))

    def test_canvas_size_relative_custom_color(self):
        img, _ = disc(40, 30)
        out, alpha, w, h = nodes.PhotoCraftCanvasSize().run(img, 10, 10, True, "center", "custom", "ff0000")
        self.assertEqual((w, h), (50, 40))
        self.assertTrue(np.all(alpha == 1.0))
        np.testing.assert_allclose(out[0, 0, 0], [1.0, 0.0, 0.0], atol=1 / 255)

    def test_crop_and_crop_to_target(self):
        img, _ = disc()
        out, _, w, h = nodes.PhotoCraftCrop().run(img, 10, 20, 50, 40, 0, 0, "bicubic")
        self.assertEqual((w, h), (50, 40))
        np.testing.assert_allclose(out[0, 0, 0], img[0, 20, 10], atol=2 / 255)
        _, _, w, h = nodes.PhotoCraftCrop().run(img, 10, 20, 50, 40, 100, 80, "nearest")
        self.assertEqual((w, h), (100, 80))

    def test_trim_transparent_uses_alpha_input(self):
        img, _ = disc(50, 40)
        alpha = np.zeros((1, 40, 50), np.float32)
        alpha[0, 5:25, 10:40] = 1.0
        out, al, w, h = nodes.PhotoCraftTrim().run(img, "transparent", True, True, True, True, alpha=alpha)
        self.assertEqual((w, h), (30, 20))
        self.assertTrue(np.all(al == 1.0))

    def test_trim_nothing_is_an_error_not_a_crash(self):
        img, _ = disc(20, 20)
        with self.assertRaises(cli.PhotoCraftError):
            nodes.PhotoCraftTrim().run(np.ones((1, 20, 20, 3), np.float32), "topLeft", True, True, True, True)

    def test_rotate_and_flip(self):
        img, _ = disc(30, 20)
        out, _ = nodes.PhotoCraftRotateFlip().run(img, "90° clockwise")
        self.assertEqual(out.shape, (1, 30, 20, 3))
        out, _ = nodes.PhotoCraftRotateFlip().run(img, "flip horizontal")
        np.testing.assert_allclose(out[0, :, ::-1], img[0], atol=1 / 255)

    def test_remove_background(self):
        img, inside = disc()
        out, alpha = nodes.PhotoCraftRemoveBackground().run(img, True, False, "transparent", "#000000")
        self.assertEqual(alpha.shape, (1, 120, 160))
        a = alpha[0] > 0.5
        agree = (a == inside).mean()
        self.assertGreater(agree, 0.97, f"subject mask agrees on {agree:.3f} of pixels")
        out, alpha = nodes.PhotoCraftRemoveBackground().run(img, True, True, "white", "#000000")
        self.assertLess(out.shape[1], 120)
        self.assertLess(out.shape[2], 160)
        corner = out[0, 0, 0]
        self.assertTrue(np.all(corner > 0.9), corner)

    def test_generic_command_and_steps(self):
        img, _ = disc(40, 30)
        out, _, result = nodes.PhotoCraftCommand().run(img, "image.imageSize", '{"width": 20}', '["image.imageRotation.90cw"]')
        self.assertEqual(out.shape, (1, 20, 15, 3))
        self.assertIn("image.imageSize", result)
        with self.assertRaises(cli.PhotoCraftError):
            nodes.PhotoCraftCommand().run(img, "no.such.command", "{}", "")
        with self.assertRaises(cli.PhotoCraftError):
            nodes.PhotoCraftCommand().run(img, "image.imageSize", "not json", "")

    def test_batch(self):
        img, _ = disc(40, 30)
        batch = np.concatenate([img, img[:, :, ::-1]], axis=0)
        out, alpha, w, h = nodes.PhotoCraftImageSize().run(batch, 20, 15, "bilinear")
        self.assertEqual(out.shape, (2, 15, 20, 3))
        self.assertEqual(alpha.shape, (2, 15, 20))


class HelperTests(unittest.TestCase):
    def test_hex(self):
        if nodes is None:
            self.skipTest("package failed to import")
        self.assertEqual(nodes._hex("FFaa00"), "#ffaa00")
        with self.assertRaises(cli.PhotoCraftError):
            nodes._hex("#12345")

    def test_missing_cli_message(self):
        if nodes is None:
            self.skipTest("package failed to import")
        saved_cli, saved_path = os.environ.pop("PHOTOCRAFT_CLI", None), os.environ.get("PATH", "")
        try:
            os.environ["PATH"] = ""
            with self.assertRaises(cli.PhotoCraftError) as e:
                cli.find_cli()
            self.assertIn("PHOTOCRAFT_CLI", str(e.exception))
        finally:
            os.environ["PATH"] = saved_path
            if saved_cli:
                os.environ["PHOTOCRAFT_CLI"] = saved_cli


if __name__ == "__main__":
    unittest.main()

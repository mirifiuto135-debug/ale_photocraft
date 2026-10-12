"""ComfyUI nodes that run PhotoCraft engine commands on an image.

Conventions shared by every node:

* ``image`` is a ComfyUI ``IMAGE`` batch ``[B, H, W, 3]`` in 0..1.
* ``alpha`` is an optional ``MASK`` ``[B, H, W]`` with **1 = visible pixel** (the image's
  transparency, so a background removed by one node survives into the next). Every node also
  returns the resulting ``alpha``. ComfyUI's *Load Image* mask is the opposite (1 = transparent):
  run it through *Invert Mask* when feeding it in.
* Each node is one engine command; chain nodes to chain commands, or use *PhotoCraft Command* for
  any command of the registry (``photocraft-cli commands`` lists them with their params).
"""

from __future__ import annotations

import json

from .photocraft_cli import PhotoCraftError, apply, size_of

CATEGORY = "PhotoCraft"

RESAMPLE = ["bicubic", "automatic", "preserveDetails", "bicubicSmoother", "bicubicSharper", "bicubicAutomatic", "bilinear", "nearest", "lanczos"]
ANCHORS = ["center", "topLeft", "top", "topRight", "left", "right", "bottomLeft", "bottom", "bottomRight"]
EXTENSION = ["transparent", "white", "black", "background", "foreground", "custom"]
ROTATIONS = {
    "90° clockwise": "image.imageRotation.90cw",
    "90° counter clockwise": "image.imageRotation.90ccw",
    "180°": "image.imageRotation.180",
    "flip horizontal": "image.imageRotation.flipCanvasHorizontal",
    "flip vertical": "image.imageRotation.flipCanvasVertical",
}
MAX_SIDE = 30000


def _alpha_input():
    return {"alpha": ("MASK", {"tooltip": "Transparency of the image, 1 = visible. Optional."})}


def _hex(color: str) -> str:
    c = color.strip()
    if not c.startswith("#"):
        c = "#" + c
    if len(c) != 7 or any(ch not in "0123456789abcdefABCDEF" for ch in c[1:]):
        raise PhotoCraftError(f"custom color must be #rrggbb, got {color!r}")
    return c.lower()


class PhotoCraftImageSize:
    """Image › Image Size: resample to a new pixel size (0 keeps the aspect ratio for that side)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": 0, "max": MAX_SIDE, "tooltip": "0 = proportional to the height"}),
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_SIDE, "tooltip": "0 = proportional to the width"}),
                "resample": (RESAMPLE, {"default": "bicubic"}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "alpha", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Resize the image with PhotoCraft's Image Size (Photoshop's resampling methods, including Preserve Details)."

    def run(self, image, width, height, resample, alpha=None):
        if width == 0 and height == 0:
            raise PhotoCraftError("Image Size needs a width or a height (0 = proportional)")
        params = {"resample": resample}
        if width:
            params["width"] = width
        if height:
            params["height"] = height
        img, al, _ = apply(image, alpha, [("image.imageSize", params)])
        w, h = size_of(img)
        return (img, al, w, h)


class PhotoCraftScaleBy:
    """Image Size by a factor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 16.0, "step": 0.01}),
                "resample": (RESAMPLE, {"default": "bicubic"}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "alpha", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Scale the image by a factor with PhotoCraft's Image Size."

    def run(self, image, scale, resample, alpha=None):
        w, h = size_of(image)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if max(nw, nh) > MAX_SIDE:
            raise PhotoCraftError(f"{nw}x{nh} exceeds {MAX_SIDE} px per side")
        img, al, _ = apply(image, alpha, [("image.imageSize", {"width": nw, "height": nh, "resample": resample})])
        w, h = size_of(img)
        return (img, al, w, h)


class PhotoCraftCanvasSize:
    """Image › Canvas Size: grow or shrink the canvas around an anchor without resampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 1024, "min": -MAX_SIDE, "max": MAX_SIDE}),
                "height": ("INT", {"default": 1024, "min": -MAX_SIDE, "max": MAX_SIDE}),
                "relative": ("BOOLEAN", {"default": False, "tooltip": "Width/height are added to the current size (negative shrinks)"}),
                "anchor": (ANCHORS, {"default": "center"}),
                "extension": (EXTENSION, {"default": "transparent", "tooltip": "What fills the new area"}),
                "custom_color": ("STRING", {"default": "#ffffff", "tooltip": "Used when extension = custom"}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "alpha", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Change the canvas size (padding or cropping) around an anchor, as Photoshop's Canvas Size."

    def run(self, image, width, height, relative, anchor, extension, custom_color, alpha=None):
        if not relative and (width <= 0 or height <= 0):
            raise PhotoCraftError("Canvas Size needs a positive width and height (or relative = true)")
        color = _hex(custom_color) if extension == "custom" else extension
        params = {"width": width, "height": height, "relative": relative, "anchor": anchor, "extensionColor": color}
        img, al, _ = apply(image, alpha, [("image.canvasSize", params)])
        w, h = size_of(img)
        return (img, al, w, h)


class PhotoCraftCrop:
    """Image › Crop to a rectangle, optionally resampled to a target size in the same step."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": -MAX_SIDE, "max": MAX_SIDE}),
                "y": ("INT", {"default": 0, "min": -MAX_SIDE, "max": MAX_SIDE}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_SIDE}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_SIDE}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": MAX_SIDE, "tooltip": "0 = keep the cropped size; else resample the crop to this size"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": MAX_SIDE}),
                "resample": (RESAMPLE, {"default": "bicubic"}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "alpha", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Crop the image to a rectangle; a crop outside the image is padded with transparency."

    def run(self, image, x, y, width, height, target_width, target_height, resample, alpha=None):
        params = {"x": x, "y": y, "width": width, "height": height}
        if (target_width > 0) != (target_height > 0):
            raise PhotoCraftError("set both target_width and target_height, or neither")
        if target_width:
            params.update({"targetWidth": target_width, "targetHeight": target_height, "resample": resample})
        img, al, _ = apply(image, alpha, [("image.crop", params)])
        w, h = size_of(img)
        return (img, al, w, h)


class PhotoCraftTrim:
    """Image › Trim: crop away transparent (or uniform) borders."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "based_on": (["transparent", "topLeft", "bottomRight"], {"default": "transparent", "tooltip": "Trim transparent pixels, or pixels of the top-left / bottom-right corner colour"}),
                "top": ("BOOLEAN", {"default": True}),
                "bottom": ("BOOLEAN", {"default": True}),
                "left": ("BOOLEAN", {"default": True}),
                "right": ("BOOLEAN", {"default": True}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "alpha", "width", "height")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Trim transparent or single-colour borders, as Photoshop's Image › Trim."

    def run(self, image, based_on, top, bottom, left, right, alpha=None):
        params = {"basedOn": based_on, "top": top, "bottom": bottom, "left": left, "right": right}
        img, al, _ = apply(image, alpha, [("image.trim", params)])
        w, h = size_of(img)
        return (img, al, w, h)


class PhotoCraftRotateFlip:
    """Image › Image Rotation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "operation": (list(ROTATIONS), {"default": "90° clockwise"})},
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Rotate the whole image by 90°/180° or flip it."

    def run(self, image, operation, alpha=None):
        img, al, _ = apply(image, alpha, [(ROTATIONS[operation], {})])
        return (img, al)


class PhotoCraftRemoveBackground:
    """Remove Background (Photoshop's Quick Action): Select Subject + edge refinement."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "refine_edge": ("BOOLEAN", {"default": True, "tooltip": "Smart-radius edge refinement for hair and soft edges"}),
                "trim": ("BOOLEAN", {"default": False, "tooltip": "Also crop to the subject"}),
                "fill": (["transparent", "white", "black", "custom"], {"default": "transparent", "tooltip": "Colour behind the subject in the image output (the alpha output is unchanged)"}),
                "custom_color": ("STRING", {"default": "#00ff00"}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Cut the subject out with PhotoCraft's Remove Background; alpha = 1 on the subject."

    def run(self, image, refine_edge, trim, fill, custom_color, alpha=None):
        steps = [("layer.removeBackground", {"refine": refine_edge})]
        if trim:
            steps.append(("image.trim", {"basedOn": "transparent"}))
        img, al, _ = apply(image, alpha, steps)
        if fill != "transparent":
            import numpy as np

            from .photocraft_cli import from_numpy, to_numpy

            rgb = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0)}.get(fill)
            if rgb is None:
                h = _hex(custom_color)
                rgb = tuple(int(h[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
            a = to_numpy(al)[..., None]
            img = from_numpy(to_numpy(img) * a + np.asarray(rgb, dtype=np.float32) * (1.0 - a))
        return (img, al)


class PhotoCraftCommand:
    """Any engine command by id, with JSON params: the whole registry from one node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "command": ("STRING", {"default": "image.imageSize", "tooltip": "Command id (photocraft-cli commands)"}),
                "params": ("STRING", {"default": '{"width": 512}', "multiline": True, "tooltip": "JSON object of params"}),
                "more_steps": ("STRING", {"default": "", "multiline": True, "tooltip": 'Optional JSON list of further steps: [["image.trim", {}], ...]'}),
            },
            "optional": _alpha_input(),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "alpha", "result")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Run any PhotoCraft engine command (filters, adjustments, selections...) on the image."

    def run(self, image, command, params, more_steps, alpha=None):
        command = command.strip()
        if not command:
            raise PhotoCraftError("command id is empty")
        try:
            p = json.loads(params) if params.strip() else {}
        except json.JSONDecodeError as e:
            raise PhotoCraftError(f"params is not valid JSON: {e}") from e
        if not isinstance(p, dict):
            raise PhotoCraftError("params must be a JSON object")
        steps = [(command, p)]
        if more_steps.strip():
            try:
                extra = json.loads(more_steps)
            except json.JSONDecodeError as e:
                raise PhotoCraftError(f"more_steps is not valid JSON: {e}") from e
            for step in extra if isinstance(extra, list) else [extra]:
                if isinstance(step, str):
                    steps.append((step, {}))
                elif isinstance(step, list) and len(step) == 2 and isinstance(step[0], str) and isinstance(step[1], dict):
                    steps.append((step[0], step[1]))
                elif isinstance(step, dict) and isinstance(step.get("command"), str):
                    steps.append((step["command"], step.get("params") or {}))
                else:
                    raise PhotoCraftError(f"bad step {step!r}: use \"id\", [\"id\", {{...}}] or {{\"command\": id, \"params\": {{...}}}}")
        img, al, results = apply(image, alpha, steps)
        return (img, al, json.dumps(results[0] if len(results) == 1 else results, indent=2))


NODE_CLASS_MAPPINGS = {
    "PhotoCraftImageSize": PhotoCraftImageSize,
    "PhotoCraftScaleBy": PhotoCraftScaleBy,
    "PhotoCraftCanvasSize": PhotoCraftCanvasSize,
    "PhotoCraftCrop": PhotoCraftCrop,
    "PhotoCraftTrim": PhotoCraftTrim,
    "PhotoCraftRotateFlip": PhotoCraftRotateFlip,
    "PhotoCraftRemoveBackground": PhotoCraftRemoveBackground,
    "PhotoCraftCommand": PhotoCraftCommand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoCraftImageSize": "PhotoCraft Image Size",
    "PhotoCraftScaleBy": "PhotoCraft Scale By",
    "PhotoCraftCanvasSize": "PhotoCraft Canvas Size",
    "PhotoCraftCrop": "PhotoCraft Crop",
    "PhotoCraftTrim": "PhotoCraft Trim",
    "PhotoCraftRotateFlip": "PhotoCraft Rotate / Flip",
    "PhotoCraftRemoveBackground": "PhotoCraft Remove Background",
    "PhotoCraftCommand": "PhotoCraft Command",
}

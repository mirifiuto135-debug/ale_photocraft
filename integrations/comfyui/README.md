# PhotoCraft for ComfyUI

`ComfyUI-PhotoCraft/` is a ComfyUI custom-node package that runs PhotoCraft's engine commands on
ComfyUI images through the headless command-line tool `photocraft-cli` (`apps/photocraft-cli`).
ComfyUI is Python, so the nodes are Python; PhotoCraft itself stays Rust. Each node writes its
input to a PNG in a temporary folder, runs `photocraft-cli run … --cmd <id> --params <json> --out …`
and reads the PNG back, so every command in the registry (`photocraft-cli commands`) is reachable.

## Nodes (category **PhotoCraft**)

| Node | Engine command | Notes |
|---|---|---|
| PhotoCraft Image Size | `image.imageSize` | width/height (0 = proportional), every Photoshop resampling method incl. Preserve Details |
| PhotoCraft Scale By | `image.imageSize` | by a factor |
| PhotoCraft Canvas Size | `image.canvasSize` | absolute or relative, 9 anchors, transparent/white/black/custom extension |
| PhotoCraft Crop | `image.crop` | rectangle, optional resample to a target size in the same step |
| PhotoCraft Trim | `image.trim` | transparent or corner-colour borders, per side |
| PhotoCraft Rotate / Flip | `image.imageRotation.*` | 90° cw/ccw, 180°, flip H/V |
| PhotoCraft Remove Background | `layer.removeBackground` | Select Subject + edge refinement; optional trim and fill colour |
| PhotoCraft Command | any | command id + JSON params (+ optional further steps); returns the JSON result too |

Conventions: `image` is ComfyUI's `IMAGE` (`[B, H, W, 3]`, 0..1). `alpha` is an optional `MASK`
with **1 = visible**, and every node returns the resulting `alpha`, so a removed background
survives into the next node. ComfyUI's *Load Image* mask is the opposite (1 = transparent): put an
*Invert Mask* node in between. Remove Background's `alpha` is 1 on the subject. Batches are
processed image by image; the outputs must come out the same size.

## Install

1. Build the CLI (Rust 1.95+): `cargo build --release -p photocraft-cli` → `target/release/photocraft-cli[.exe]`.
2. Copy `ComfyUI-PhotoCraft/` into `ComfyUI/custom_nodes/`.
3. Tell the nodes where the binary is: the `PHOTOCRAFT_CLI` environment variable, or copy it to
   `ComfyUI/custom_nodes/ComfyUI-PhotoCraft/bin/`, or put it on the `PATH`.
4. Restart ComfyUI. The nodes need only `numpy` and `Pillow`, which ComfyUI already has.

`PHOTOCRAFT_CLI_TIMEOUT` (seconds, default 600) bounds one command.

### Windows (PowerShell)

```powershell
# 1. build the CLI (from the PhotoCraft checkout)
cd C:\src\ale_photocraft
cargo build --release -p photocraft-cli

# 2. install the nodes into ComfyUI (adjust the ComfyUI path)
$comfy = "C:\ComfyUI"
Copy-Item -Recurse -Force integrations\comfyui\ComfyUI-PhotoCraft "$comfy\custom_nodes\ComfyUI-PhotoCraft"
New-Item -ItemType Directory -Force "$comfy\custom_nodes\ComfyUI-PhotoCraft\bin" | Out-Null
Copy-Item target\release\photocraft-cli.exe "$comfy\custom_nodes\ComfyUI-PhotoCraft\bin\"

# 3. (alternative to the copy) point the nodes at the binary for every future session
[Environment]::SetEnvironmentVariable("PHOTOCRAFT_CLI", "C:\src\ale_photocraft\target\release\photocraft-cli.exe", "User")
```

## Tests

```sh
cargo build --release -p photocraft-cli
python3 -m unittest discover -s integrations/comfyui/ComfyUI-PhotoCraft/tests
```

The tests use numpy only (no torch, no ComfyUI) and run every node against the built binary:
sizes, anchors, alpha round trips, batch handling, the subject mask of a synthetic disc, and that
bad params come back as errors rather than crashes.

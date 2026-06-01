# Replicant Character Lab — Wan2GP Plugin

> # ⚠️ THIS IS NOT READY YET. CLONE AT YOUR OWN RISK.
>
> Active, early development — incomplete, unstable, and changing constantly. Things will break.

A [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) plugin that ports the **character-creator wizard** from [SupremeDiffusion](https://github.com/saintorphan/SupremeDiffusion) into Wan2GP's Gradio UI.

![Replicant Character Lab](replicant.png)

## What it does

Guides you through creating a reusable character — from name + description to a trained LoRA — as a single branded tab inside Wan2GP. Six steps:

1. **Setup** — name, description, style, reference image, and positive/negative prompt generation (abliterated Qwen3.5 enhancer)
2. **Baseline** — generate candidate base images and pick one (or reimagine/skip a supplied reference)
3. **Face / Body Swap** — optional identity locking
4. **Touch Up** — optional inpaint + cohesion (img2img normalize) passes
5. **Poses** — generate + approve pose variants
6. **Datasets & Training** — build the LoRA datasets and train the character LoRA

Save/Load Character and Clear are session actions in the header (top-right), usable at any step.

## Install

In Wan2GP, open the **Plugins** manager and "install from GitHub URL":

```
https://github.com/saintorphan/Replicant-CharLab-Wan2GP
```

Then enable **Replicant Character Lab** in the plugin list and restart WanGP. The
installer clones the repo into `plugins/`, installs `requirements.txt`
(InsightFace + onnxruntime — required for face swap / dataset crops), and the
plugin creates its data directories on first run.

> Not yet an official Wan2GP plugin — install via the GitHub URL above.

### Data layout

Created under `<wan2gp_root>/character_lab/` (all three roots are configurable
and persisted from the wizard's **Prerequisites → Directories** panel):

```
character_lab/
├── characters/<Name>/   character.json, base.png, poses/
├── datasets/<Name>/     video512/ highres/ full/ face/  (NNN.png + NNN.txt)
└── models/face/         downloaded swap/enhancer models
```

### Models (downloaded from the Prerequisites → Models panel)

| Model | Required | Source |
|-------|----------|--------|
| `inswapper_128.onnx` | yes | facefusion-assets |
| `GFPGANv1.4.onnx` | optional | Gourieff/ReActor |
| `codeformer.onnx` | optional | facefusion/models-3.0.0 |
| `face_yolov8s.pt` | optional | Bingsu/adetailer |

InsightFace's `buffalo_l` is fetched automatically on first use.

## Status

🚧 Early development. Data layer + wizard UI complete and tested; GPU-bound
generation (base gen, swaps, pose gen, prompt enhance, training) wires to
Wan2GP's native pipelines next.

## Design notes

- Single top-level Wan2GP tab; logo banner header.
- 7-step wizard via visibility-toggled groups with a clickable step rail + Back/Next (Gradio 5.29 has no native stepper).
- Reuses portable logic from SupremeDiffusion (character schema, dataset builder/compositor, pose specs); standard generation routed through Wan2GP's native pipelines.
- Prompt enhancement via Wan2GP's abliterated Qwen3.5 enhancer.

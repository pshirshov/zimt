# zimt — todo / design notes

## Reference-image conditioning (`/ref <image>`)

Goal: a command that conditions generation on a user-supplied
reference image, akin to SDXL IP-Adapter. The wiring is straightforward
on SDXL but materially different on Z-Image — the two surfaces need
distinct backends behind a shared command shape.

### SDXL family — IP-Adapter (well-trodden)

Diffusers ships `pipe.load_ip_adapter(repo_id, subfolder, weight_name)`
+ `pipe(..., ip_adapter_image=PIL)`. Variants:

- **IP-Adapter base / plus** — general style+content transfer (~700 MB
  for plus).
- **IP-Adapter face / plus-face** — face-biased.
- **InstantID / PuLID** — stronger face-identity preservation, each
  adds a ControlNet + adapter pair (heavier wiring, defer).

All current zimt SDXL fine-tunes (Pony, Animagine, Juggernaut) are
SDXL-arch so one wiring covers them. Tradeoffs:

- First command in the grammar to carry a filesystem path → needs a
  resolver. Sandbox can only read paths under the project dir or
  `/tmp/exchange`; user has to drop images there, or web UI adds an
  upload affordance.
- ~700 MB extra weights per adapter variant. Stacks with LoRAs in
  VRAM — likely forces `cpuoffload` for low-VRAM users.

### Z-Image — different story (as of 2026-05)

Z-Image-Turbo is a 6B single-stream DiT ("S3-DiT") with **Qwen3-4B** as
sole text encoder and Flux VAE for latents. Text + visual + image-
latent tokens are concatenated into one sequence, not fed via separate
cross-attention streams. **SDXL IP-Adapter weights are not
architecturally portable** (different token topology, different encoder,
different attention shape).

Sources: paper [arXiv:2511.22699](https://arxiv.org/abs/2511.22699);
[Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image); local
`src/zimt/models/zimage.py` already encodes the Qwen3-4B + bf16 facts.

State of prior art (verified 2026-05):

- **No IP-Adapter for Z-Image.** Zero hits on HF / GitHub / ModelScope.
- **No PuLID / InstantID / FaceID port.** Only generic comparisons of
  these techniques on SDXL/Flux.
- **No architecturally-portable neighbour.** Qwen-Image shares the
  single-stream MM-DiT paradigm but is 20B+ with different dims — not
  weight-portable. Z-Image is not a Flux/SD3/PixArt-Sigma descendant.
- **The 685 listed Z-Image adapters on HF are all LoRAs.** No image-
  prompt entries.
- **ControlNet exists and is mature.** Alibaba-PAI ships
  [`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)
  (~71k monthly downloads, updated 2026-02-26). Supports
  Canny/HED/Depth/Pose/MLSD/Scribble/Gray + inpainting. **Structural
  conditioning, not image-prompt.**
- **The architecturally-native reference path is wired in diffusers
  but the weights have not shipped.** `ZImageOmniPipeline` consumes
  `siglip: Siglip2VisionModel` + `siglip_processor` — the SigLIP-2
  reference-image route. Requires `Z-Image-Omni-Base` / `Z-Image-Edit`
  weights, which remain "coming soon" per the
  [z-image.me 2026-02-06 blog](https://z-image.me/en/blog/Not_Z-Image-Base_but_Z-Image-Omni-Base_en)
  and [Tongyi-MAI/Z-Image#40](https://github.com/Tongyi-MAI/Z-Image/issues/40).
  The Tongyi-MAI HF org currently lists only `Z-Image` and
  `Z-Image-Turbo`.
- **Diffusers upstream support is complete.** `diffusers 0.39.0.dev0`
  ships `ZImagePipeline`, `ZImageImg2ImgPipeline`,
  `ZImageInpaintPipeline`, `ZImageControlNetPipeline`,
  `ZImageControlNetInpaintPipeline`, `ZImageControlNetModel`,
  `ZImageOmniPipeline`, `ZImageTransformer2DModel`. No custom pipeline
  required.

### Buildable today on Z-Image (in lieu of IP-Adapter)

1. **img2img via `ZImageImg2ImgPipeline`** — weak SDXL-style fallback.
   Preserves structure and palette, not concept. Useful for "redraw in
   a different style"; useless for "put this specific dragon in a new
   scene".
2. **ControlNet-Union 2.1 via `ZImageControlNetPipeline`** — real
   structural conditioning. Requires preprocessing the reference into
   a control map (Canny/Depth/Pose/etc.). Best results today for
   structure-preserving conditioning.
3. **VLM caption → text condition.** Run the reference image through a
   captioner (Qwen2.5-VL family pairs naturally — same Qwen lineage as
   Z-Image's text encoder; `Qwen2.5-VL-3B-Instruct` ~7–8 GiB bf16,
   `Qwen2.5-VL-7B-Instruct` ~15 GiB bf16). Describe-then-redraw — gist
   and style words, not identity.

### Proposed `/ref` shape

A single command with a mode flag rather than per-backend commands:

- `/ref <path>` — default mode per active model. SDXL: IP-Adapter.
  Z-Image: img2img.
- `/ref-mode <mode>` — explicit override. Modes:
  - `ipadapter` (SDXL only) — uses IP-Adapter base/plus.
  - `ipadapter-face` (SDXL only) — uses IP-Adapter face variant.
  - `img2img` (both) — denoise-strength-controlled redraw.
  - `controlnet-canny` / `controlnet-depth` / `controlnet-pose` /
    `controlnet-union` (both, weights differ) — structural.
  - `caption` (both) — VLM-captioning fallback.
  - `omni` (Z-Image, future) — SigLIP-2 reference tokens via
    `ZImageOmniPipeline`. Reserved; errors with "weights not released"
    until `Z-Image-Omni-Base` ships.
- `/ref-weight <float>` — adapter / control / denoise strength,
  semantics per mode.
- `/ref-clear` — drop the reference.

The state surface (`AppState`) must record which mode was selected and
which weights were actually loaded, so the UI can show "Z-Image: using
img2img — IP-Adapter equivalent unavailable" rather than silently
falling back. Caveat in the report: report the active mode in the
generation metadata (PNG `Parameters` block) so post-hoc the user
knows whether identity transfer was even attempted.

### Order of build

1. **SDXL IP-Adapter wiring + sandbox-aware path resolver.** Highest
   user value, well-trodden code path.
2. **Z-Image img2img mode.** Smallest delta — same pipeline class
   family, already in diffusers.
3. **Z-Image ControlNet-Union 2.1 mode** with an automatic Canny/Depth
   preprocessor. Bigger lift (extra weights, preprocessor pipeline)
   but the strongest Z-Image result available today.
4. **VLM-caption fallback.** Cross-model, one captioner serves both
   surfaces.
5. **Z-Image Omni mode.** Stub the code path now; flip the switch
   when Omni-Base weights ship.

## Probing what the U-Net can actually depict

The `/tokenize` command shows tokenization; it does not show what the
**U-Net** was trained on. CLIP "knows" Higgs boson; SDXL cannot draw
one. The text encoder is at best a proxy. Everything below probes the
diffusion path itself.

Constraint: there is no introspection-only shortcut. U-Net knowledge
only manifests through denoising. So every option below either piggy-
backs on a generation we were already going to run, or pays for extra
generations, or pays once offline and caches.

### Option A — Cross-attention saliency hook on `/g` (free)

**What.** During a normal `/g` call, install attention-processor hooks
on `pipe.unet` (or `pipe.transformer` for DiT models). Accumulate, per
prompt-token, the attention mass that token received across all cross-
attention layers and all denoising timesteps. Emit a per-token
"attention mass" sparkline alongside the generated image.

**Signal.** A token with near-zero accumulated mass = the U-Net
ignored it. A token with mass concentrated in one shallow layer = used
weakly / mostly as a style nudge. Strongly attended tokens = the U-Net
is using them to shape content.

**Cost.** One generation we were doing anyway, plus the per-layer hook
overhead (negligible — same order as logging).

**Wiring.**
- Replace `pipe.unet.set_attn_processor(...)` (SDXL) or the DiT
  equivalent with a tracing processor that records attention weights
  per timestep per layer.
- Sum over (layer, timestep, head, query-spatial-position) → per-key-
  token scalar.
- Surface as part of generation metadata; render as a horizontal bar
  per prompt token in the web UI under the image.
- Gate behind `/g --explain` so the default path stays untouched.

**Cross-architecture.** Works on SDXL UNet *and* z-image DiT — both
expose cross-attention layers that diffusers' attention-processor API
can hook. Same code, different layer enumeration.

**Caveats.** Attention mass is a proxy for "use", not a guarantee.
A high-mass token can still be depicted wrongly. But low-mass = high
confidence that the token did not affect the image.

**Order of build: first.** Highest signal-to-noise per line of code.

### Option B — Counterfactual `/probe` (1 extra generation per probe)

**What.** `/probe <word> [base prompt]` generates the prompt twice at
the same seed: once with `<word>` included, once with it removed. Score
the two outputs against each other with CLIP-image similarity.

**Signal.**
- High similarity (≈1.0) → the word had no measurable effect on the
  image. U-Net effectively does not depict it (in this context).
- Low similarity → the word changed the image. Combined with a CLIP-
  image-vs-text-of-word score on the with-word output, you get
  "changed *and* in the expected direction" vs "changed but not toward
  the concept".

**Cost.** One extra generation per probe call. Acceptable as an on-
demand command, not as background telemetry.

**Wiring.** Reuses `/g` and the existing CLIP image-text scoring path
(would need a small CLIP-image encoder if not already loaded — same
model as the text encoder usually suffices for SDXL CLIP-L probes;
z-image would need a separate CLIP for image scoring).

**Caveats.** Context-sensitive — a word can be inert in one prompt and
load-bearing in another. So `/probe wyvern` on a bare prompt is
different from `/probe wyvern <full scene>`. Make this explicit in the
output.

**Order of build: second.** Composes cleanly with what we have.

### Option C — Cached offline probe-set per model (one-time batch)

**What.** Curate a probe vocabulary of ~200 words covering common
categories: creatures, materials, art styles, named entities,
technical objects, anatomy, attire, etc. Run once per registered
model, at a fixed seed, against a canonical prompt template. Score
each generated image against its corresponding word using CLIP-image-
vs-text similarity. Cache the resulting `(model, word) → score` table.

`/probe X` on a known-vocabulary word becomes a lookup → "Pony scores
0.31 on wyvern (baseline 0.27, top decile 0.41); weakly depictable."

**Signal.** A stable per-model fingerprint of concept coverage.
Particularly useful for distinguishing fine-tunes — Pony's animal
concept coverage vs. Animagine's vs. Juggernaut's becomes a real
table.

**Cost.** ~200 generations × N models, one time. At SDXL 8-step turbo
speeds (~2s/image warm), ~7 minutes per model. Storage trivial.

**Wiring.**
- A `tools/probe_baseline.py` script that loads each model, iterates
  the vocabulary, generates, scores, writes a JSON next to the model
  spec.
- The `/probe` command reads the JSON for the loaded model and
  reports the cached score plus the model's baseline and top decile
  for context.
- Re-run when a model is added or its weights change.

**Caveats.** The fixed prompt template anchors what "depictable" means
— a single template can't cover all framings of a concept. Document
the template explicitly so the score is interpretable. The CLIP
scorer's biases bake into the table; this is a known issue with any
CLIP-based evaluation.

**Order of build: third (highest commitment, best UX once built).**

### Option D — Generation-variance probe (4+ generations per probe)

**What.** For prompt X, generate 4 images at 4 seeds, compute pairwise
CLIP-image similarity. High variance + low text-alignment = the model
is fishing; low variance + high alignment = solid concept; high
variance + high alignment = creative but on-target.

**Cost.** 4× one probe. Too expensive for routine use.

**Order of build: skip unless A/B/C prove insufficient.**

### Honest framing for the user

Whatever surfaces ship, the UI should be explicit that:

1. "Known to the text encoder" ≠ "depictable by the U-Net". The
   tokenize report answers the first; only A–D answer the second.
2. Every U-Net probe is context-sensitive. The same word can be inert
   in one prompt and load-bearing in another.
3. CLIP-image scoring inherits CLIP's biases. Treat scores as ordinal,
   not absolute.

### Recommended build order

1. **A** — cross-attention saliency on `/g --explain`. Free, immediate,
   architecture-portable.
2. **B** — counterfactual `/probe`. Composes with existing `/g`.
3. **C** — cached probe-set table per model. Biggest UX win, biggest
   upfront commitment.
4. **D** — defer.

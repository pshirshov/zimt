"""Built-in LoRA registry.

A small, conservative starter set of well-known HuggingFace SDXL LoRAs.
Each entry advertises ``compatible_with`` tags that must overlap with
the active base model's :attr:`ModelSpec.compatibility_tags` for the
adapter to be applicable. SDXL fine-tunes (Pony, Illustrious, …) all
carry the ``"sdxl"`` tag, so anything tagged ``["sdxl"]`` here loads on
any SDXL pipeline — quality varies of course, since the LoRA was
trained against one specific base.

Add your own by dropping JSON descriptors under ``CUSTOM_DIR/loras/``
(see :mod:`zimt.models.custom`) or via the "+ add custom" form in the
models tab.
"""

from __future__ import annotations

from .spec import LoraSpec

LORAS: dict[str, LoraSpec] = {
    "pixel-art-xl": LoraSpec(
        name="pixel-art-xl",
        description="Pixel-art style for SDXL (by Nerijs). Use trigger 'pixel art'.",
        repo_id="nerijs/pixel-art-xl",
        family="sdxl",
        compatible_with=["sdxl"],
        default_weight=1.0,
        trigger_tags="pixel art",
    ),
    "ascii-art": LoraSpec(
        name="ascii-art",
        description="ASCII-art style for SDXL (CiroN2022). Use trigger 'ascii_art'.",
        repo_id="CiroN2022/ascii-art",
        family="sdxl",
        compatible_with=["sdxl"],
        default_weight=0.9,
        trigger_tags="ascii_art",
    ),
    "studio-ghibli-style": LoraSpec(
        name="studio-ghibli-style",
        description="Studio Ghibli style for SDXL (KappaNeuro). Trigger 'Studio Ghibli Style'.",
        repo_id="KappaNeuro/studio-ghibli-style",
        family="sdxl",
        compatible_with=["sdxl"],
        default_weight=0.9,
        trigger_tags="Studio Ghibli Style",
    ),

    # ---- Z-Image-Turbo LoRAs ----
    # The repo ships two training checkpoints; pin the later one. Adult/NSFW
    # content — apache-2.0 licensed. Trigger word is 'pronmstr'.
    "zimage-pornmaster": LoraSpec(
        name="zimage-pornmaster",
        description="Pornmaster v1 — uncensored/NSFW realism for Z-Image-Turbo. Trigger 'pronmstr'.",
        repo_id="RomixERR/Pornmaster_v1-Z-Images-Turbo",
        family="zimage",
        compatible_with=["zimage"],
        weight_name="Pornmaster_v1_000044700.safetensors",
        default_weight=0.8,
        trigger_tags="pronmstr",
    ),
}

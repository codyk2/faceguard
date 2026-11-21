"""ViT deepfake classifier (Hugging Face transformers).

P(fake) is read via model.config.id2label: we look up the label whose name contains
"fake" (covers "Fake", "Deepfake", ...). The class index is never hard-coded because the
candidate models order their labels differently.
"""
from __future__ import annotations

import re

import torch
from PIL import Image

from app.config import settings as default_settings

_FAKE_RE = re.compile(r"fake", re.IGNORECASE)
_NEGATED_RE = re.compile(r"\b(not|non|no)[\s_-]?", re.IGNORECASE)


class LabelMappingError(RuntimeError):
    """Raised when a model's id2label has no unambiguous 'fake' class."""


def find_fake_index(id2label: dict) -> int:
    """Return the class index whose label name means 'fake' (case-insensitive)."""
    hits = [
        int(i) for i, name in id2label.items()
        if _FAKE_RE.search(str(name)) and not _NEGATED_RE.match(str(name))
    ]
    if len(hits) != 1:
        raise LabelMappingError(
            f"cannot identify the fake class: id2label={dict(id2label)!r} "
            f"(expected exactly one label containing 'fake'/'deepfake', found {len(hits)})"
        )
    return hits[0]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ViTFakeDetector:
    def __init__(self, model_id: str | None = None, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.model_id = model_id or default_settings.detector_model_id
        self.device = device or pick_device()
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model = model.to(self.device).eval()
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.fake_index = find_fake_index(self.id2label)
        self.name = f"{self.model_id} (ViT-B/16)"

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[float, str]:
        """Return (P(fake), predicted label name) for an RGB face crop."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu()
        p_fake = float(probs[self.fake_index])
        label = self.id2label[int(torch.argmax(probs))]
        return p_fake, label

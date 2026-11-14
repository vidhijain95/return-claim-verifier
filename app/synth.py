"""Synthetic product / customer photos.

Used by the seed script (demo data), the offline mock vision tests and the eval harness.
Cracks are drawn as thin dark polylines. This is NOT a realistic damage simulator -
it exists so the whole pipeline runs and is testable without an API key or real photos.
For real numbers, evaluate with real photos and the Gemini provider (see README).
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = (320, 320)


def product_image(seed: int) -> Image.Image:
    """A 'listing photo': a blobby coloured product on a light background."""
    rng = random.Random(seed)
    img = Image.new("RGB", SIZE, (236, 236, 236))
    d = ImageDraw.Draw(img)
    x0, y0 = rng.randint(40, 90), rng.randint(40, 90)
    x1, y1 = SIZE[0] - rng.randint(40, 90), SIZE[1] - rng.randint(40, 90)
    body = tuple(rng.randint(40, 210) for _ in range(3))
    d.rounded_rectangle((x0, y0, x1, y1), radius=rng.randint(8, 40), fill=body)
    for _ in range(rng.randint(3, 6)):
        ax, ay = rng.randint(x0, x1 - 20), rng.randint(y0, y1 - 20)
        bx, by = min(ax + rng.randint(15, 70), x1), min(ay + rng.randint(15, 70), y1)
        shade = tuple(rng.randint(30, 240) for _ in range(3))
        if rng.random() < 0.5:
            d.rectangle((ax, ay, bx, by), fill=shade)
        else:
            d.ellipse((ax, ay, bx, by), fill=shade)
    return img.filter(ImageFilter.GaussianBlur(1.0))


def customer_photo(seed: int, cracks: int = 0, other_product: bool = False, noise: float = 2.0) -> Image.Image:
    """A 'customer photo' of the product (or of a different product) with optional cracks."""
    rng = random.Random(seed * 7919 + 13)
    img = product_image(seed + 7777 if other_product else seed)
    arr = np.asarray(img).astype(np.float32)
    arr = arr * rng.uniform(0.95, 1.05) + rng.uniform(-6, 6)
    arr += np.random.default_rng(seed).normal(0, noise, arr.shape)
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    for _ in range(cracks):
        x, y = rng.randint(70, 250), rng.randint(70, 250)
        pts = [(x, y)]
        for _ in range(rng.randint(4, 8)):
            x = min(max(x + rng.randint(-35, 35), 40), 280)
            y = min(max(y + rng.randint(-35, 35), 40), 280)
            pts.append((x, y))
        d.line(pts, fill=(15, 15, 15), width=rng.choice([2, 3]))
    return img

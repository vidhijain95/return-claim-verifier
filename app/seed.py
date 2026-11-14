"""Seed the fake commerce backend: orders, listing photos and a few ready-to-upload demo claim photos."""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

from . import config, db, synth

PRODUCTS = [  # (name, category, price range in rupees)
    ("Wireless Earbuds Pro", "electronics", (2500, 9000)),
    ("Smartphone 5G 128GB", "electronics", (12000, 35000)),
    ("Bluetooth Speaker", "electronics", (1500, 6000)),
    ("Cotton Kurta", "clothing", (500, 2000)),
    ("Denim Jacket", "clothing", (1200, 3500)),
    ("Running Shoes", "footwear", (1500, 6000)),
    ("Leather Sandals", "footwear", (800, 2500)),
    ("Crystal Wine Glass Set", "glassware", (700, 3000)),
    ("Glass Water Jug", "glassware", (400, 1500)),
]


def seed_orders(n: int = 24, seed: int = 42, demo_orders: int = 4) -> int:
    rng = random.Random(seed)
    listing_dir, demo_dir = Path(config.LISTING_DIR), Path(config.DEMO_DIR)
    listing_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        name, category, (lo, hi) = rng.choice(PRODUCTS)
        order_id, img_seed = f"ORD{1001 + i}", 1000 + i
        listing = listing_dir / f"{order_id}.png"
        synth.product_image(img_seed).save(listing)
        db.insert_order(
            order_id=order_id,
            customer_id=f"C{rng.randint(100, 999)}",
            product_name=name,
            category=category,
            price=round(rng.uniform(lo, hi) / 10) * 10,
            delivered_on=(date.today() - timedelta(days=rng.randint(1, 4))).isoformat(),
            listing_image=listing,
        )
        if i < demo_orders:  # photos you can upload in the UI for this order
            synth.customer_photo(img_seed, cracks=3).save(demo_dir / f"{order_id}_damaged.png")
            synth.customer_photo(img_seed, cracks=0).save(demo_dir / f"{order_id}_intact.png")
            synth.customer_photo(img_seed, cracks=3, other_product=True).save(demo_dir / f"{order_id}_other_product.png")
    return n


if __name__ == "__main__":
    db.init_db()
    print(f"Seeded {seed_orders()} orders; demo photos in {config.DEMO_DIR}")

"""Tiny SYNTHETIC support-ticket generator for tests and demos.

WARNING: labels are assigned by construction from templates, so a model trained on this data
proves nothing about real-world performance. Never report metrics from it as results.
Every row carries is_synthetic=True.

Run:  python -m supportx.data.sample
"""

import argparse
import random
from pathlib import Path

import pandas as pd

INTENTS: dict[str, tuple[str, ...]] = {
    "track_order": (
        "Where is my order {oid}?",
        "Can you tell me the delivery status of order {oid}?",
        "My package {oid} has not arrived yet.",
    ),
    "cancel_order": (
        "I want to cancel order {oid}.",
        "Please cancel my order {oid} before it ships.",
    ),
    "get_refund": (
        "I would like a refund for order {oid}.",
        "Please refund order {oid}, the item was damaged.",
    ),
    "check_refund_policy": (
        "What is your refund policy?",
        "How many days do I have to return an item?",
    ),
    "complaint": (
        "I am unhappy with the service on order {oid}.",
        "The quality of the product in order {oid} is poor.",
    ),
}

SENTIMENT_SUFFIX: dict[str, tuple[str, ...]] = {
    "negative": ("This is unacceptable.", "I am very disappointed."),
    "neutral": ("Please let me know.", "Thanks."),
    "positive": ("Thank you so much for your help!", "I appreciate your quick support."),
}

COLUMNS = ["text", "intent", "sentiment", "priority", "is_synthetic"]


def _priority(intent: str, sentiment: str) -> str:
    """Rule used only to label synthetic rows."""
    if sentiment == "negative" and intent in {"get_refund", "complaint", "cancel_order"}:
        return "high"
    if sentiment == "negative" or intent in {"get_refund", "cancel_order"}:
        return "medium"
    return "low"


def generate_sample_tickets(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Return n synthetic tickets. Same (n, seed) always gives the same rows."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        intent = rng.choice(list(INTENTS))
        sentiment = rng.choice(list(SENTIMENT_SUFFIX))
        oid = f"ORD-{rng.randint(10000, 99999)}"
        body = rng.choice(INTENTS[intent]).format(oid=oid)
        text = f"{body} {rng.choice(SENTIMENT_SUFFIX[sentiment])}"
        rows.append((text, intent, sentiment, _priority(intent, sentiment), True))
    return pd.DataFrame(rows, columns=COLUMNS)


def write_sample_csv(path: Path, n: int = 200, seed: int = 42) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_tickets(n, seed).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write synthetic sample tickets (demo only).")
    parser.add_argument("--out", type=Path, default=Path("data/sample/synthetic_tickets.csv"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Wrote {write_sample_csv(args.out, args.n, args.seed)} (SYNTHETIC, demo only)")


if __name__ == "__main__":
    main()

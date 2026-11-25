"""Safe workflow for REAL customer messages: import, privacy checks, labeling, leak-free splits.

NO REAL CUSTOMER DATA EXISTS IN THIS PROJECT. Nothing here fabricates messages or labels; this is
the pipeline for the day real tickets are available. Real data lives under `data/real/`
(git-ignored) and must never be committed.

IMPORT FORMAT (CSV, UTF-8), one row per customer message:

    message_id   required, unique, your own id (never an email or name)
    text         required, the message as the customer wrote it
    source       optional ("chat", "email", ...)
    received_at  optional ISO timestamp
    label        intent label; blank until a person labels it
    labeler      required whenever `label` is set: the person (or pseudonym) who chose it
    label2, labeler2   optional SECOND opinion; rows where the two labels differ are held out of
                       training and evaluation until someone adjudicates them

Steps:
    python -m supportx.data.real prepare raw.csv --out-dir data/real/processed
        validates, redacts, deduplicates and splits; writes all.csv, unlabeled.csv (the sheet to
        hand to labelers: fill `label` and `labeler`) and report.json
    python -m supportx.data.real merge-labels data/real/processed/all.csv labeled.csv

PRIVACY: emails, phone numbers, card-like numbers and long digit runs are replaced by placeholders
BEFORE anything is written next to the code; the report counts what was found. Automated redaction
is a safety net, not a guarantee (names and addresses are NOT detected): review samples by hand.

SPLITS: assigned by a hash of the normalized text, so the same (or trivially re-typed) message gets
the same split every time you import more data. Train/validation/test cannot leak through re-imports
or duplicates. The test split should be scored once per model (see `supportx.ml.real_eval`).
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from supportx.core import DataError, get_logger
from supportx.nlp import tokenize

logger = get_logger(__name__)

REQUIRED = ("message_id", "text")
SPLITS = ("train", "val", "test")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)")
_LONG_DIGITS = re.compile(r"\b\d{9,}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace emails, card numbers, phone numbers and long digit runs; count what was found."""
    counts = {"email": 0, "card": 0, "phone": 0, "digits": 0}

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            counts["card"] += 1
            return "<card>"
        return match.group()

    def swap(pattern: re.Pattern[str], token: str, key: str, value: str) -> str:
        def repl(_: re.Match[str]) -> str:
            counts[key] += 1
            return token

        return pattern.sub(repl, value)

    text = swap(_EMAIL, "<email>", "email", text)
    text = _CARD.sub(card, text)
    text = swap(_PHONE, "<phone>", "phone", text)
    text = swap(_LONG_DIGITS, "<number>", "digits", text)
    return text, counts


def normalized(text: str) -> str:
    """Words only, lowercase: case, spacing and punctuation never make a message "new"."""
    return " ".join(tokenize(text))


def split_of(text: str, salt: str = "supportx-real-v1") -> str:
    """Stable split from the normalized text: 70% train, 15% val, 15% test."""
    digest = hashlib.sha256(f"{salt}|{normalized(text)}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate, redact, deduplicate and split. Returns (frame with `split`, report)."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataError(f"Missing required columns: {missing}.")
    frame = raw.copy()
    frame["message_id"] = frame["message_id"].astype(str).str.strip()
    if frame["message_id"].duplicated().any() or (frame["message_id"] == "").any():
        raise DataError("message_id must be unique and non-empty.")
    if frame["message_id"].str.contains("@").any():
        raise DataError("message_id looks like an email address; use an opaque id.")
    frame = frame.dropna(subset=["text"])
    frame["text"] = frame["text"].astype(str)

    totals = {"email": 0, "card": 0, "phone": 0, "digits": 0}
    cleaned: list[str] = []
    for value in frame["text"]:
        redacted, counts = redact(value)
        cleaned.append(redacted.strip())
        for key, n in counts.items():
            totals[key] += n
    frame["text"] = cleaned
    frame = frame[frame["text"].str.len() > 0]

    before = len(frame)
    frame["_norm"] = frame["text"].map(normalized)
    frame = frame.drop_duplicates(subset="_norm").drop(columns="_norm")
    frame["split"] = frame["text"].map(split_of)

    for col in ("label", "labeler", "label2", "labeler2"):
        if col not in frame:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    if ((frame["label"] != "") & (frame["labeler"] == "")).any():
        raise DataError("Every labeled row needs a `labeler`.")
    if ((frame["label2"] != "") & (frame["labeler2"] == "")).any():
        raise DataError("Every row with `label2` needs a `labeler2`.")
    frame["needs_adjudication"] = (
        (frame["label"] != "") & (frame["label2"] != "") & (frame["label"] != frame["label2"])
    )
    report = {
        "rows_in": len(raw),
        "rows_out": len(frame),
        "duplicates_removed": before - len(frame),
        "redactions": totals,
        "labeled": int((frame["label"] != "").sum()),
        "needs_adjudication": int(frame["needs_adjudication"].sum()),
        "by_split": frame["split"].value_counts().to_dict(),
        "data_origin": "real",
    }
    return frame.reset_index(drop=True), report


def usable_labels(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rows of one split with a final label: labeled, and not in dispute between two labelers."""
    part = frame[(frame["split"] == split) & (frame["label"] != "") & ~frame["needs_adjudication"]]
    return part.reset_index(drop=True)


def merge_labels(frame: pd.DataFrame, labeled: pd.DataFrame) -> pd.DataFrame:
    """Add labels from a returned sheet: message_id, label, labeler [, label2, labeler2]."""
    needed = {"message_id", "label", "labeler"}
    if not needed <= set(labeled.columns):
        raise DataError(f"The labeling sheet needs columns {sorted(needed)}.")
    sheet = labeled.copy()
    sheet["message_id"] = sheet["message_id"].astype(str).str.strip()
    unknown = set(sheet["message_id"]) - set(frame["message_id"])
    if unknown:
        raise DataError(f"{len(unknown)} message_id(s) in the sheet are not in the data.")
    keep = [c for c in ("label", "labeler", "label2", "labeler2") if c in sheet.columns]
    merged = frame.drop(columns=[c for c in keep if c in frame.columns]).merge(
        sheet[["message_id", *keep]], on="message_id", how="left"
    )
    result, _ = prepare(merged.drop(columns=["split", "needs_adjudication"], errors="ignore"))
    return result


def cohens_kappa(first: list[str], second: list[str]) -> float:
    """Agreement between two labelers beyond chance (1 = perfect, 0 = chance)."""
    if not first or len(first) != len(second):
        raise DataError("Need two equally long, non-empty label lists.")
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    labels = set(first) | set(second)
    expected = sum((first.count(x) / n) * (second.count(x) / n) for x in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real customer messages.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("csv", type=Path)
    prep.add_argument("--out-dir", type=Path, default=Path("data/real/processed"))
    merge = sub.add_parser("merge-labels")
    merge.add_argument("all_csv", type=Path)
    merge.add_argument("labeled_csv", type=Path)
    args = parser.parse_args()

    if args.cmd == "prepare":
        frame, report = prepare(pd.read_csv(args.csv, encoding="utf-8", dtype=str))
        out: Path = args.out_dir
        out.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out / "all.csv", index=False, encoding="utf-8")
        frame[frame["label"] == ""][["message_id", "text"]].assign(label="", labeler="").to_csv(
            out / "unlabeled.csv", index=False, encoding="utf-8"
        )
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    else:
        frame = pd.read_csv(args.all_csv, encoding="utf-8", dtype=str, keep_default_na=False)
        merged = merge_labels(frame, pd.read_csv(args.labeled_csv, encoding="utf-8", dtype=str))
        merged.to_csv(args.all_csv, index=False, encoding="utf-8")
        print(json.dumps({"labeled": int((merged["label"] != "").sum())}))


if __name__ == "__main__":
    main()

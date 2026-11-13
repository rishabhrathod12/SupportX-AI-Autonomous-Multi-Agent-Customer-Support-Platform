"""Shared text cleaning and tokenization.

The SAME functions must be used at training time and at inference time, otherwise a model sees
differently-shaped text in production than it was trained on (train/serve skew).
"""

import re
import unicodedata

import pandas as pd

URL_TOKEN = "<url>"
EMAIL_TOKEN = "<email>"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", re.IGNORECASE)
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)  # runs before lowercasing
_SPACES = re.compile(r"\s+")
# a placeholder like <url>, or a word/number with an optional contraction suffix (don't, it's)
_TOKEN = re.compile(r"<[a-z_]+>|[a-z0-9]+(?:'[a-z]+)?")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def clean_text(text: str, lowercase: bool = True) -> str:
    """Normalize one message: unicode fix, URL/email placeholders, single spaces, lowercase."""
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES)
    text = _EMAIL.sub(f" {EMAIL_TOKEN} ", text)  # emails first, so their domain isn't read as a URL
    text = _URL.sub(f" {URL_TOKEN} ", text)
    text = _SPACES.sub(" ", text).strip()
    return text.lower() if lowercase else text


def tokenize(text: str) -> list[str]:
    """Clean the text, then split it into word tokens. Punctuation is dropped."""
    return _TOKEN.findall(clean_text(text))


def clean_series(texts: pd.Series, lowercase: bool = True) -> pd.Series:
    """Clean a column of messages. Missing values (NaN/None) become empty strings."""
    return texts.fillna("").astype(str).map(lambda t: clean_text(t, lowercase))

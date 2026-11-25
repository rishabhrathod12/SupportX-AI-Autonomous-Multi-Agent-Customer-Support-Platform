"""Query normalization for the LEXICAL retriever: map everyday phrasings onto the vocabulary the
policy documents use ("send back" -> "return", "smashed" -> "damaged", "stop an order" -> "cancel").

This is a small hand-written thesaurus, not semantic understanding: it only helps for the phrases
listed here, and a real embedding model (see `supportx.llm.OpenAIEmbedder`) is the proper fix. The
original question is kept and the canonical terms are APPENDED, so exact wording still matches.
Extend `THESAURUS` when real customer questions show new phrasings; do not tune it on the frozen
evaluation questions.
"""

import re
from collections.abc import Mapping

# canonical terms (what the documents say) -> everyday phrases that should add them
THESAURUS: Mapping[str, tuple[str, ...]] = {
    "return": (
        "send back",
        "send it back",
        "send them back",
        "send something back",
        "sent back", "give back", "giving back", "returning", "bring back"),
    "damaged": ("smashed", "broken", "cracked", "defective", "faulty", "shattered", "ruined"),
    "not arrived delayed": (
        "never showed up",
        "never arrived",
        "did not arrive",
        "didn't arrive",
        "hasn't arrived",
        "has not arrived",
        "late",
        "overdue",
        "missing",
        "lost",
        "delayed",
    ),
    "order parcel": ("package", "parcel", "shipment"),
    "refund": ("money back", "reimburse", "reimbursement", "get my money", "payment back"),
    "payment method": ("card", "bank account"),
    "shipped": (
        "dispatched",
        "left the warehouse",
        "on its way",
        "sent out",
        "in transit",
        "goes out",
        "go out",
    ),
    "cancel": ("stop", "call off", "withdraw", "abort", "undo"),
    "tracking": ("follow", "trace", "monitor"),
    "change delivery address": (
        "different address",
        "new address",
        "moved house",
        "somewhere else",
        "redirect",
        "swap the address",
    ),
    "final sale": ("clearance", "sale item", "discounted", "on sale"),
    "photo picture": ("picture", "photo"),
    "product page described": ("listing", "description", "different from"),
    "delivery times": (
        "how fast",
        "get here",
        "arrive by",
        "standard shipping",
        "express shipping",
    ),
    "window days": ("how many days", "period", "timeframe", "deadline", "time limit"),
    "approved reviewed support team": ("who decides", "decides", "agree", "agrees"),
}

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (
        canonical,
        re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE),
    )
    for canonical, phrases in THESAURUS.items()
)


def expand_query(question: str) -> str:
    """The question plus the canonical terms for any everyday phrase it contains."""
    extra = [canonical for canonical, pattern in _PATTERNS if pattern.search(question)]
    return f"{question} {' '.join(extra)}" if extra else question

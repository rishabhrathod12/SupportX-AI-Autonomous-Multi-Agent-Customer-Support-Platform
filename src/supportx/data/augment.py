"""Hand-written, realistic training phrasings for the intents the agents route on.

WHY: the Bitext data is templated ("where can I obtain refunds?", slot-style order numbers) and
lacks everyday phrasing ("my kettle arrived broken, refund me ORD-5001"). Evaluation on hand-written
probes (E7 in docs/EXPERIMENTS.md) showed most ordinary messages being escalated.

HOW: a few dozen hand-written core messages per intent are expanded with openers, closers, order
references and mild typos, deterministically (seeded). These rows are added to the TRAIN split only.

LIMITS (read before trusting scores): the same person wrote these phrasings and the probe messages,
so probe scores are optimistic about real customers. Texts that exactly match a probe message are
removed (`drop_texts`). Real customer tickets are still the proper fix.
"""

import random
from collections.abc import Collection

import pandas as pd

from supportx.nlp import clean_text

# Intents whose messages usually carry an order number.
_ORDER_INTENTS = {
    "track_order",
    "get_refund",
    "cancel_order",
    "track_refund",
    "change_shipping_address",
}

CORES: dict[str, list[str]] = {
    "track_order": [
        "where is my parcel",
        "i haven't got my delivery yet, where is it",
        "can you let me know where my order is at the moment",
        "any update on my shipment",
        "when is my order going to turn up",
        "i'd like to follow my order",
        "has my order shipped yet",
        "what's happening with my delivery",
        "tracking number for my order please",
        "my order hasn't arrived, can you look into it",
        "I want an update on the status of my order",
        "where's my order",
        "show me the tracking for my purchase",
        "is my order on its way",
        "still no sign of my package",
        "dispatch status of my order",
    ],
    "get_refund": [
        "I want a refund",
        "my item is broken, I'd like my money back",
        "the product I received is damaged and I want a refund",
        "please refund me, the item is not what I ordered",
        "I want to return this and get reimbursed",
        "can I get a refund for this purchase",
        "this arrived faulty, refund it please",
        "i'd like to send it back for a full refund",
        "the item doesn't work. I need my money back",
        "I'm requesting a refund",
        "wrong item delivered, I want a refund",
        "it never arrived so I want my money back",
        "I changed my mind and would like a refund",
        "refund my order please",
        "the size is wrong and I want to return it for a refund",
        "returning my purchase, please refund me",
    ],
    "cancel_order": [
        "I want to cancel my order",
        "please cancel the order I just made",
        "can you cancel this for me",
        "I ordered the wrong thing, cancel it",
        "I need to cancel my purchase",
        "cancel my order before it is dispatched",
        "i don't need it any more, please cancel",
        "stop my order",
        "can I still cancel my order?",
        "I would like to cancel an order I placed today",
        "I accidentally placed an order. Please cancel it",
        "need to cancel asap",
        "cancel it please",
        "I want to call off my order",
        "please withdraw my order",
        "undo my order, I changed my mind",
    ],
    "check_refund_policy": [
        "what is your refund policy",
        "how long do I have to return something",
        "what's the returns window",
        "can I return items I bought on sale",
        "are refunds possible after delivery",
        "what are the conditions for a refund",
        "do you accept returns",
        "how do returns work",
        "explain your return rules",
        "i want to know about your money back policy",
        "up to how many days can I send an item back",
        "can I return a product I've used",
        "what items can't be returned",
        "who pays for return shipping",
        "is it possible to get my money back if I don't like the item",
        "tell me the rules on refunds",
    ],
    "delivery_period": [
        "how long does shipping take",
        "how many days will delivery take",
        "when would my order get here if I order now",
        "what's the usual delivery time",
        "how quickly can you deliver",
        "what are the delivery lead times",
        "how many business days for shipping",
        "how soon will my parcel arrive after I order",
        "what is the average shipping duration",
        "how long until I receive my items",
        "estimated delivery time please",
        "how long does it take to get my order",
        "is delivery quick",
        "how many days to ship abroad",
        "delivery time for standard shipping",
    ],
    "track_refund": [
        "where is my refund",
        "has my refund been issued yet",
        "when will my refund arrive",
        "I returned my order, when do I get the money",
        "i'm still waiting on my refund",
        "check the status of my refund",
        "how long does a refund take to show up",
        "refund hasn't arrived in my account",
        "did you process my return",
        "update on my refund please",
        "it's been days since I returned the item. Where is my money",
        "I haven't received my reimbursement yet",
        "what's going on with my refund",
        "when will the refunded money reach my card",
        "could you check if my refund was approved",
        "refund not received yet",
    ],
    "complaint": [
        "i want to make a complaint",
        "I'm very unhappy with your service",
        "your service has been awful",
        "this is not acceptable, I want to complain",
        "I need to complain about how I was treated",
        "I am disgusted with this experience",
        "I'm writing to complain",
        "I have had a horrible experience with your company",
        "I want to report poor service",
        "I'm upset and want to file a complaint",
        "terrible experience, never again",
        "you have let me down badly and I want to complain",
        "I am so angry about the service I got",
        "I want to lodge a complaint about my order experience",
        "what a useless service",
        "I'd like to complain about a member of your staff",
    ],
    "payment_issue": [
        "I was charged twice",
        "my payment didn't work",
        "the payment keeps failing",
        "money was taken from my account but the order wasn't placed",
        "there's an unexpected charge on my card",
        "my card was declined",
        "I'm having trouble paying for my order",
        "i got billed the wrong amount",
        "double charge on my bank statement",
        "the website won't accept my card",
        "payment error at checkout",
        "you took the money twice",
        "my payment is stuck",
        "I was overcharged",
        "card payment problem",
    ],
    "change_shipping_address": [
        "i need to change my delivery address",
        "can I change where the order is sent",
        "I typed the wrong address",
        "please update the address for my order",
        "send my parcel somewhere else",
        "i've moved and need to change my shipping address",
        "change the delivery location",
        "the address on my order is wrong",
        "I'd like to correct my address",
        "can you deliver to a different address instead",
        "edit shipping details",
        "wrong street on my order",
        "new delivery address please",
        "I need the parcel sent to my work address",
        "fix my shipping address",
        "modify the delivery address",
    ],
}

_OPENERS = ["", "", "", "hi, ", "hello, ", "hey ", "please, ", "excuse me, ", "hi there. "]
_CLOSERS = ["", "", "", " please", " thanks", " thank you", " asap", "?", "."]
_ORDER_FORMS = ["order ORD-{n}", "ORD-{n}", "my order ORD-{n}", "(ORD-{n})", "purchase ORD-{n}"]


def _typo(text: str, rng: random.Random) -> str:
    """Swap two adjacent letters inside one word (a very common typing slip)."""
    positions = [i for i in range(len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha()]
    if not positions:
        return text
    i = rng.choice(positions)
    return text[:i] + text[i + 1] + text[i] + text[i + 2 :]


def build_augmentation(
    per_intent: int = 120,
    seed: int = 42,
    drop_texts: Collection[str] = (),
) -> pd.DataFrame:
    """Return a DataFrame with columns text, intent (about `per_intent` unique rows per intent).

    Rows whose cleaned text equals the cleaned text of anything in `drop_texts` are removed, so
    the evaluation probes are never in the training data verbatim.
    """
    rng = random.Random(seed)
    banned = {clean_text(t) for t in drop_texts}
    rows: list[tuple[str, str]] = []
    for intent, cores in CORES.items():
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_intent and attempts < per_intent * 40:
            attempts += 1
            text = rng.choice(cores)
            if intent in _ORDER_INTENTS and rng.random() < 0.5:
                ref = rng.choice(_ORDER_FORMS).format(n=rng.randint(1000, 99999))
                text = f"{ref} {text}" if rng.random() < 0.3 else f"{text} {ref}"
            text = rng.choice(_OPENERS) + text + rng.choice(_CLOSERS)
            if rng.random() < 0.25:
                text = text.lower()
            if rng.random() < 0.12:
                text = _typo(text, rng)
            key = clean_text(text)
            if key in seen or key in banned:
                continue
            seen.add(key)
            rows.append((text, intent))
    return pd.DataFrame(rows, columns=["text", "intent"])

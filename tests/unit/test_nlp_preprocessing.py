import pandas as pd
import pytest

from supportx.nlp import clean_series, clean_text, tokenize


def test_lowercase_and_whitespace() -> None:
    assert clean_text("  WHERE is\tmy\n ORDER?  ") == "where is my order?"


def test_lowercase_can_be_disabled() -> None:
    assert clean_text("Hello World", lowercase=False) == "Hello World"


def test_urls_become_placeholder() -> None:
    assert clean_text("see https://shop.com/o?id=1 or www.shop.com now") == "see <url> or <url> now"


def test_emails_become_placeholder() -> None:
    assert clean_text("mail me at Jane.Doe+x@mail.example.com please") == (
        "mail me at <email> please"
    )


def test_unicode_is_normalized() -> None:
    assert clean_text("ＯＲＤＥＲ") == "order"  # full-width letters
    assert clean_text("I don’t know") == "i don't know"  # curly apostrophe


def test_clean_text_is_idempotent() -> None:
    once = clean_text("Visit HTTP://X.com  ASAP!")
    assert once == "visit <url> asap!"
    assert clean_text(once) == once


def test_template_slots_become_single_tokens() -> None:
    text = "Where is order {{Order Number}} for {{ Person Name }}?"
    assert clean_text(text) == "where is order <order_number> for <person_name> ?"
    assert tokenize(text) == ["where", "is", "order", "<order_number>", "for", "<person_name>"]


def test_template_slot_cleaning_is_idempotent() -> None:
    once = clean_text("refund {{Refund Amount}} please")
    assert clean_text(once) == once


def test_clean_text_rejects_non_strings() -> None:
    with pytest.raises(TypeError):
        clean_text(123)  # type: ignore[arg-type]


def test_tokenize_basic_and_contractions() -> None:
    assert tokenize("Where's my order ORD-12345? I can't wait!") == [
        "where's", "my", "order", "<order_number>", "i", "can't", "wait",
    ]  # fmt: skip


def test_tokenize_keeps_placeholders() -> None:
    assert tokenize("Refund to a@b.com, see www.x.com") == [
        "refund", "to", "<email>", "see", "<url>",
    ]  # fmt: skip


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("?!...") == []


def test_clean_series_handles_missing_values() -> None:
    out = clean_series(pd.Series(["Hi  There", None, float("nan")]))
    assert out.tolist() == ["hi there", "", ""]


def test_real_order_numbers_match_the_dataset_slot_token() -> None:
    assert clean_text("Where is ORD-5001?") == clean_text("Where is {{Order Number}}?")
    assert clean_text("ord-77 and ORD-8") == "<order_number> and <order_number>"
    assert clean_text("keyword-1234 or WORD-5") == "keyword-1234 or word-5"

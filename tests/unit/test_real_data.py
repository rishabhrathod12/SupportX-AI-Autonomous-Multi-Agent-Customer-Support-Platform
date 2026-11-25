"""Real-data workflow tests. Every row below is an INVENTED fixture that exercises code paths;
none of it is customer data and no score here says anything about real customers."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from supportx.core import DataError
from supportx.data.real import cohens_kappa, merge_labels, prepare, redact, split_of, usable_labels
from supportx.ml import real_eval
from supportx.ml.real_eval import evaluate_real


def raw(n: int = 300, labeled: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message_id": [f"m{i}" for i in range(n)],
            "text": [f"fixture message number {i} about thing {i % 9}" for i in range(n)],
            "label": ["intent_a" if i % 2 else "intent_b" for i in range(n)] if labeled else "",
            "labeler": "tester" if labeled else "",
        }
    )


def test_redaction_removes_contact_details_and_counts_them() -> None:
    text = "mail jo.doe@example.com or +44 7700 900123, card 4111 1111 1111 1111, ref 1234567890"
    out, counts = redact(text)
    for leaked in ("jo.doe", "7700", "4111", "1234567890"):
        assert leaked not in out
    assert counts["email"] == 1 and counts["card"] == 1
    assert counts["phone"] + counts["digits"] == 2  # a bare 10-digit run counts as either
    assert redact("order ORD-1001 was 25 days ago")[0] == "order ORD-1001 was 25 days ago"
    assert redact("number 1234 5678 9012 3456")[1]["card"] == 0  # fails the Luhn check


def test_prepare_validates_dedupes_redacts_and_splits_stably() -> None:
    data = raw(300)
    data.loc[1, "text"] = data.loc[0, "text"].upper() + "!!"  # a re-typed duplicate
    data.loc[2, "text"] = "write to me at who@example.com about my parcel"
    frame, report = prepare(data)
    assert report["data_origin"] == "real" and report["duplicates_removed"] >= 1
    assert report["redactions"]["email"] == 1 and not frame["text"].str.contains("@").any()
    assert set(frame["split"]) == {"train", "val", "test"}
    again, _ = prepare(data.sample(frac=1.0, random_state=1))  # re-import in another order
    assert dict(zip(frame["message_id"], frame["split"], strict=True)) == dict(
        zip(again["message_id"], again["split"], strict=True)
    )
    assert split_of("Hello there") == split_of("hello   THERE!")  # same normalized text, same split


def test_the_same_text_can_never_land_in_two_splits() -> None:
    frame, _ = prepare(raw(400))
    by_text = frame.groupby(frame["text"].str.lower())["split"].nunique()
    assert (by_text == 1).all()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.drop(columns=["text"]),
        lambda d: d.assign(message_id="same"),
        lambda d: d.assign(message_id="jo@example.com"),
        lambda d: d.assign(labeler=""),  # labels without a labeler
    ],
)
def test_bad_input_is_rejected(mutate: Any) -> None:
    with pytest.raises(DataError):
        prepare(mutate(raw(50)))


def test_two_labelers_who_disagree_are_held_out_until_adjudicated() -> None:
    data = raw(300)
    data["label2"] = data["label"]
    data["labeler2"] = "second"
    data.loc[0:4, "label2"] = "something_else"
    frame, report = prepare(data)
    assert report["needs_adjudication"] == 5
    for split in ("train", "val", "test"):
        assert not usable_labels(frame, split)["needs_adjudication"].any()
    assert cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_labels_can_be_merged_from_a_returned_sheet() -> None:
    frame, _ = prepare(raw(300, labeled=False))
    sheet = pd.DataFrame(
        {"message_id": ["m0", "m1"], "label": ["intent_a", "intent_b"], "labeler": ["amy", "amy"]}
    )
    merged = merge_labels(frame, sheet)
    assert int((merged["label"] != "").sum()) == 2
    with pytest.raises(DataError):
        merge_labels(frame, pd.DataFrame({"message_id": ["zzz"], "label": ["x"], "labeler": ["a"]}))


class FakeModel:
    classes_ = ["intent_a", "intent_b"]

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        return ("intent_a", 0.9) if int(text.split()[3]) % 2 else ("intent_b", 0.4)


def test_real_evaluation_is_labelled_real_and_refuses_tiny_samples() -> None:
    frame, _ = prepare(raw(1200))
    report = evaluate_real(FakeModel(), frame, "val", min_messages=10)
    assert report["data_origin"] == "real" and report["labelers"] == ["tester"]
    assert report["n"] >= 10 and 0.0 <= report["confidently_wrong"] <= 1.0
    with pytest.raises(DataError, match="at least"):
        evaluate_real(FakeModel(), frame.head(30), "val")
    with pytest.raises(DataError):
        evaluate_real(FakeModel(), frame, "train")


def test_the_real_test_split_is_scored_once_per_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supportx.ml.intent import IntentClassifier

    frame, _ = prepare(raw(1500))
    frame["needs_adjudication"] = frame["needs_adjudication"].astype(str)
    data_path = tmp_path / "all.csv"
    frame.to_csv(data_path, index=False)
    texts = [f"fixture message number {i} about thing {i % 9}" for i in range(60)]
    labels = ["intent_a" if i % 2 else "intent_b" for i in range(60)]
    model_path = IntentClassifier().fit(texts, labels).save(tmp_path / "m.joblib")
    monkeypatch.setattr(real_eval, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setitem(real_eval.MIN_MESSAGES, "test", 10)
    first = real_eval.run(model_path, data_path, "test", 0.6)
    assert first["data_origin"] == "real"
    with pytest.raises(DataError, match="already scored"):
        real_eval.run(model_path, data_path, "test", 0.6)

import pytest

from src.train.metrics_schema import canonical_dataset_name, detect_task


def test_canonical_dataset_name():
    assert canonical_dataset_name("Salesforce/wikitext") == "wikitext"
    assert canonical_dataset_name("wmt16-en-fr") == "wmt16_en_fr"
    assert canonical_dataset_name("cnn_dailymail:3.0.0") == "cnn_dailymail"


def test_detect_task_explicit_override():
    cfg = {"task": "textdiff", "data": {"dataset": "wikitext103"}}
    assert detect_task(cfg) == "textdiff"


@pytest.mark.parametrize(
    ("field", "task"),
    [
        ("lm_dataset", "lm"),
        ("seq_dataset", "seq2seq"),
        ("textdiff_dataset", "textdiff"),
        ("vit_dataset", "vit"),
    ],
)
def test_detect_task_dataset_fields(field, task):
    cfg = {"data": {field: "wikitext2"}}
    assert detect_task(cfg) == task


def test_detect_task_dataset_mapping():
    cfg = {"data": {"dataset": "wmt16_en_fr"}}
    assert detect_task(cfg) == "seq2seq"

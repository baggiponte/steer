import pytest

from steer.pairs import ContrastPair, parse_generated_pairs


def test_parse_generated_pairs_accepts_surrounding_text():
    text = '```json\n[{"concept":"Go hard","neutral":"Try carefully"}]\n```'

    assert parse_generated_pairs(text, 1) == [ContrastPair("Go hard", "Try carefully")]


def test_parse_generated_pairs_requires_expected_count():
    with pytest.raises(ValueError, match="returned 1 pairs; expected 2"):
        parse_generated_pairs('[{"concept":"a","neutral":"b"}]', 2)

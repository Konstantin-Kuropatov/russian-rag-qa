import pytest

from rag_qa_eda.text import audit_span, normalize, question_type, token_window_contains


def test_correct_offset_is_preserved_even_with_multiple_occurrences():
    result = audit_span("кот и кот", "кот", 6)
    assert result["span_status"] == "exact"
    assert result["answer_start_fixed"] == 6


def test_unique_exact_relocation_only_changes_offset():
    result = audit_span("Кот поймал мышь.", "мышь", -1)
    assert result["span_status"] == "relocated_exact"
    assert result["answer_start_fixed"] == 11
    assert result["word_boundary_ok"]


def test_ambiguous_occurrence_is_not_arbitrarily_repaired():
    result = audit_span("кот и кот", "кот", -1)
    assert result["span_status"] == "ambiguous_exact"
    assert result["answer_start_fixed"] is None


def test_substring_is_not_automatically_usable_as_word_answer():
    result = audit_span("Дарданеллы", "да", 3)
    assert result["span_status"] == "exact"
    assert not result["word_boundary_ok"]


def test_empty_label_is_unknown_not_a_negative_class():
    result = audit_span("Здесь есть ответ", "", -1)
    assert result["span_status"] == "no_label"
    assert result["answer_start_fixed"] is None


@pytest.mark.parametrize("answer", ["мышь.", "Мышь", "МЫШЬ."])
def test_nonexact_normalized_matches_are_review_only(answer):
    result = audit_span("кот поймал мышь", answer, -1)
    assert "mismatch" in result["span_status"]
    assert result["answer_start_fixed"] is None


def test_normalization_keeps_semantic_punctuation_and_yo():
    assert normalize("  Ёлка\nСТОИТ ") == "ёлка стоит"
    assert normalize("елка") != normalize("ёлка")
    assert normalize("нет!") != normalize("нет?")


@pytest.mark.parametrize(
    ("start", "end", "size", "overlap", "expected"),
    [
        (120, 135, 128, 0, False),
        (120, 135, 128, 32, True),
        (0, 129, 128, 32, False),
        (299, 300, 128, 0, True),
    ],
)
def test_overlap_changes_boundary_coverage(start, end, size, overlap, expected):
    assert token_window_contains(start, end, 300, size, overlap) == expected


def test_invalid_overlap_fails_instead_of_looping():
    with pytest.raises(ValueError):
        token_window_contains(1, 3, 20, 5, 5)


def test_question_type_handles_preposition_and_yes_no():
    assert question_type("В каком году это произошло?") == "какой"
    assert question_type("Правда ли это?") == "ли / да-нет"

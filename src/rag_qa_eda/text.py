"""Transparent text features; heuristics are flags, not semantic ground truth."""

import hashlib
import re
import unicodedata

WORD = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", flags=re.UNICODE)
STOPWORDS = frozenset(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по "
    "только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если "
    "уже или ни быть был него до вас опять уж вам ведь там потом себя ничего ей может "
    "они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже "
    "себе под будет тогда кто этот того потому этого какой совсем ним здесь этом один "
    "почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два "
    "об другой хоть после над больше тот через эти нас про всего них какая много разве "
    "три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им "
    "более всегда конечно всю между также это который которая которые каких каком "
    "которого которой сколько почему какое каким какие чего чья чье чьи whose".split()
)


def normalize(text: str) -> str:
    """Conservative duplicate key; no spelling correction or lemmatization."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def words(text: str) -> list[str]:
    return WORD.findall(text.casefold())


def content_words(text: str) -> set[str]:
    return {word for word in words(text) if word not in STOPWORDS}


def question_type(text: str) -> str:
    tokens = words(text)
    if "ли" in tokens:
        return "ли / да-нет"
    for token in tokens[:3]:
        if token in {"что", "чего", "чем"}:
            return "что"
        if token in {"кто", "кого", "кому", "кем"}:
            return "кто"
        if token.startswith(("како", "каки", "кака", "каку")):
            return "какой"
        if token in {"где", "куда", "откуда", "когда", "сколько", "почему", "зачем", "как"}:
            return token
    return "другое"


def has_context_reference(text: str) -> bool:
    """High-recall review cue; a pronoun does not prove unanswerability."""
    return bool(
        re.search(
            r"\b(?:он|она|оно|они|его|ее|её|их|этому|этого|этой|этот|эта|это|эти|"
            r"данный|данная|данного|такой|такая|такого|выше|упомянут\w*)\b",
            text.casefold(),
        )
    )


def word_boundaries_valid(context: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(context)):
        return False
    left = start == 0 or not (context[start - 1].isalnum() and context[start].isalnum())
    right = end == len(context) or not (context[end - 1].isalnum() and context[end].isalnum())
    return left and right


def occurrences(context: str, answer: str) -> list[int]:
    if not answer:
        return []
    return [match.start() for match in re.finditer(f"(?={re.escape(answer)})", context)]


def audit_span(context: str, answer: str, start: int | None) -> dict:
    """Only unique, character-exact relocation is automatically accepted.

    Punctuation/whitespace tolerant matches are marked for review, never used as
    replacement gold labels or as evidence that a question is answerable.
    """
    if not answer.strip():
        return {
            "span_status": "no_label",
            "answer_start_fixed": None,
            "exact_occurrences": 0,
            "word_boundary_ok": False,
        }
    starts = occurrences(context, answer)
    if start is not None and 0 <= start and context[start : start + len(answer)] == answer:
        status, fixed = "exact", start
    elif len(starts) == 1:
        status, fixed = "relocated_exact", starts[0]
    elif len(starts) > 1:
        status, fixed = "ambiguous_exact", None
    else:
        # Diagnostic comparison; indices here intentionally never become offsets.
        compact = lambda s: "".join(ch for ch in s if ch.isalnum())  # noqa: E731
        compact_answer = compact(answer)
        if compact_answer and compact_answer in compact(context):
            status = "punctuation_or_space_mismatch"
        elif normalize(answer) in normalize(context):
            status = "case_or_unicode_mismatch"
        elif compact(normalize(answer)) and compact(normalize(answer)) in compact(
            normalize(context)
        ):
            status = "case_and_punctuation_mismatch"
        else:
            status = "not_found"
        fixed = None
    return {
        "span_status": status,
        "answer_start_fixed": fixed,
        "exact_occurrences": len(starts),
        "word_boundary_ok": (
            word_boundaries_valid(context, fixed, fixed + len(answer))
            if fixed is not None
            else False
        ),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def token_window_contains(start: int, end: int, n_tokens: int, size: int, overlap: int) -> bool:
    """Whether a nonempty half-open token span is fully covered by one window."""
    if not (0 <= overlap < size):
        raise ValueError("Require 0 <= overlap < size")
    if not (0 <= start < end <= n_tokens):
        return False
    stride = size - overlap
    latest_start = (start // stride) * stride
    return end <= min(latest_start + size, n_tokens)

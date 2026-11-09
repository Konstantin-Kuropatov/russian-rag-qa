"""Length, lexical, sentence and tokenizer diagnostics with explicit definitions."""

import logging
import re

import numpy as np
import pandas as pd
from razdel import sentenize
from tokenizers import Tokenizer

from .paths import RAW
from .text import (
    content_words,
    has_context_reference,
    jaccard,
    normalize,
    question_type,
    token_window_contains,
    words,
)

LOGGER = logging.getLogger(__name__)


def text_shape(text: str) -> dict:
    tokens = words(text)
    n = max(len(text), 1)
    return {
        "chars": len(text),
        "words": len(tokens),
        "unique_word_ratio": len(set(tokens)) / max(len(tokens), 1),
        "digit_ratio": sum(ch.isdigit() for ch in text) / n,
        "cyrillic_ratio": sum("а" <= ch.casefold() <= "я" or ch.casefold() == "ё" for ch in text)
        / n,
    }


def add_features(qa: pd.DataFrame, corpus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = Tokenizer.from_file(str(RAW / "tokenizer/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    canonical = corpus.drop_duplicates("context_hash").sort_values("corpus_id")
    context_cache = {}
    prefix = "passage: "
    LOGGER.info("Computing context and tokenizer features for %s unique passages", len(canonical))
    rows = list(canonical[["context_hash", "context"]].itertuples(index=False, name=None))
    for batch_start in range(0, len(rows), 256):
        batch = rows[batch_start : batch_start + 256]
        encodings = tokenizer.encode_batch([prefix + text for _, text in batch])
        for (cid, text), encoding in zip(batch, encodings, strict=True):
            sentences = list(sentenize(text))
            offsets = [
                (max(s - len(prefix), 0), e - len(prefix))
                for s, e in encoding.offsets
                if e > len(prefix)
            ]
            context_cache[cid] = {
                "shape": text_shape(text),
                "sentences": sentences,
                "words": content_words(text),
                "offsets_start": np.array([s for s, _ in offsets]),
                "offsets_end": np.array([e for _, e in offsets]),
                "e5_tokens": len(encoding.ids),
                "content_tokens": len(offsets),
                "retained_chars": {
                    size: max(0, max(e for _, e in encoding.offsets[: size - 1]) - len(prefix))
                    for size in (128, 256, 512)
                },
            }
    frame = qa.copy()
    question_encodings = tokenizer.encode_batch(["query: " + q for q in frame.question])
    feature_rows = []
    for row, encoding in zip(frame.itertuples(index=False), question_encodings, strict=True):
        ctx = context_cache[row.context_hash]
        qwords = content_words(row.question)
        features = {
            **{f"c_{k}": v for k, v in ctx["shape"].items()},
            **{f"q_{k}": v for k, v in text_shape(row.question).items()},
            **{
                f"a_{k}": v if row.label_available else np.nan
                for k, v in text_shape(row.answer_text).items()
            },
            "c_sentences": len(ctx["sentences"]),
            "c_e5_tokens": ctx["e5_tokens"],
            "q_e5_tokens": len(encoding.ids),
            "question_type": question_type(row.question),
            "context_reference_flag": has_context_reference(row.question),
            "ends_question_mark": row.question.rstrip().endswith("?"),
            "answer_form": (
                "не опубликован"
                if not row.label_available
                else "да / нет"
                if normalize(row.answer_text) in {"да", "нет"}
                else "только числа и знаки"
                if re.fullmatch(r"[\d\s.,+−\-–—/%№:()]+", row.answer_text.strip())
                else "текст / смешанный"
            ),
            "c_wiki_citation_flag": bool(re.search(r"\[\d+\]", row.context)),
            "q_context_jaccard": jaccard(qwords, ctx["words"]),
            "q_context_coverage": len(qwords & ctx["words"]) / max(len(qwords), 1),
            "answer_equals_context": bool(row.answer_text.strip())
            and row.answer_text.strip() == row.context.strip(),
            "answer_in_question": bool(row.answer_text.strip())
            and normalize(row.answer_text) in normalize(row.question),
            "a_context_char_ratio": len(row.answer_text) / max(len(row.context), 1)
            if row.label_available
            else np.nan,
            "a_position_relative": np.nan,
            "q_answer_sentence_jaccard": np.nan,
            "a_sentence_count": np.nan,
            **{f"answer_retained_at_{b}": np.nan for b in (128, 256, 512)},
            **{f"answer_window_{b}_overlap_{o}": np.nan for b in (128, 256) for o in (0, 32)},
        }
        if row.reader_eligible:
            start = int(row.answer_start_fixed)
            end = start + len(row.answer_text)
            features["a_position_relative"] = start / len(row.context)
            relevant_sentences = [s for s in ctx["sentences"] if s.start < end and s.stop > start]
            features["a_sentence_count"] = len(relevant_sentences)
            sentence_words = content_words(" ".join(s.text for s in relevant_sentences))
            features["q_answer_sentence_jaccard"] = jaccard(qwords, sentence_words)
            token_start = int(np.searchsorted(ctx["offsets_end"], start, side="right"))
            token_end = int(np.searchsorted(ctx["offsets_start"], end, side="left"))
            for size in (128, 256, 512):
                features[f"answer_retained_at_{size}"] = int(end <= ctx["retained_chars"][size])
            for size in (128, 256):
                for overlap in (0, 32):
                    # Content budget only; model prefixes/specials require additional capacity.
                    features[f"answer_window_{size}_overlap_{overlap}"] = int(
                        token_window_contains(
                            token_start, token_end, ctx["content_tokens"], size, overlap
                        )
                    )
        feature_rows.append(features)
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
    for field in ("q_words", "c_words", "a_words", "c_e5_tokens", "a_context_char_ratio"):
        reference = frame.loc[frame.split.eq("train") & frame.label_available, field]
        q1, q3 = reference.quantile([0.25, 0.75])
        low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        frame[f"iqr_flag_{field}"] = (frame[field].lt(low) | frame[field].gt(high)).astype(
            "boolean"
        )
        if field.startswith("a_"):
            frame.loc[~frame.label_available, f"iqr_flag_{field}"] = pd.NA
    context_features = pd.DataFrame(
        [
            {
                "context_hash": cid,
                **{f"c_{k}": v for k, v in cache["shape"].items()},
                "c_sentences": len(cache["sentences"]),
                "c_e5_tokens": cache["e5_tokens"],
            }
            for cid, cache in context_cache.items()
        ]
    )
    corpus = corpus.merge(context_features, on="context_hash", validate="many_to_one")
    return frame, corpus

"""Read source formats and join their identities by content, not by unrelated IDs."""

import logging

import numpy as np
import pandas as pd

from .paths import RAW
from .text import audit_span, normalize, text_hash

LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    qa_parts = []
    for split in SPLITS:
        frame = pd.read_parquet(RAW / "qa/sberquad" / f"{split}-00000-of-00001.parquet")
        frame["split"] = split
        frame["source_row"] = np.arange(len(frame))
        frame["row_key"] = split + ":" + frame["source_row"].astype(str)
        qa_parts.append(frame)
    qa = pd.concat(qa_parts, ignore_index=True)
    corpus = pd.read_json(RAW / "retrieval/corpus.jsonl", lines=True).rename(
        columns={"_id": "corpus_id", "text": "context"}
    )
    queries = pd.read_json(RAW / "retrieval/queries.jsonl", lines=True).rename(
        columns={"_id": "query_id", "text": "question"}
    )
    qrels = pd.concat(
        [
            pd.read_csv(RAW / "qrels" / f"{split}.tsv", sep="\t").assign(split=split)
            for split in SPLITS
        ],
        ignore_index=True,
    ).rename(columns={"query-id": "query_id", "corpus-id": "corpus_id"})
    return qa, corpus, queries, qrels


def flatten_labels(qa: pd.DataFrame) -> pd.DataFrame:
    frame = qa.copy()
    frame["answer_count"] = frame.answers.map(lambda a: len(a["text"]))
    frame["offset_count"] = frame.answers.map(lambda a: len(a["answer_start"]))
    if not (frame.answer_count.eq(1) & frame.offset_count.eq(1)).all():
        raise ValueError(
            "Unexpected multi-answer/malformed schema: inspect source before flattening"
        )
    frame["answer_text"] = frame.answers.map(lambda a: a["text"][0])
    frame["answer_start_raw"] = frame.answers.map(lambda a: int(a["answer_start"][0]))
    frame["label_available"] = frame.answer_text.str.strip().ne("")
    frame["context_hash"] = frame.context.map(text_hash)
    frame["context_normalized_hash"] = frame.context.map(lambda s: text_hash(normalize(s)))
    frame["question_normalized"] = frame.question.map(normalize)
    frame["pair_hash"] = [
        text_hash(q + "\x00" + c) for q, c in zip(frame.question, frame.context, strict=True)
    ]
    audits = pd.DataFrame(
        [
            audit_span(context, answer, start)
            for context, answer, start in zip(
                frame.context, frame.answer_text, frame.answer_start_raw, strict=True
            )
        ],
        index=frame.index,
    )
    frame = pd.concat([frame, audits], axis=1)
    frame["answer_start_fixed"] = frame.answer_start_fixed.astype("Int64")
    frame["reader_eligible"] = (
        frame.span_status.isin(["exact", "relocated_exact"]) & frame.word_boundary_ok
    )
    return frame.drop(columns="answers")


def join_retrieval(
    qa: pd.DataFrame, corpus: pd.DataFrame, queries: pd.DataFrame, qrels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if corpus.corpus_id.duplicated().any() or queries.query_id.duplicated().any():
        raise ValueError("Retrieval identifiers must be unique")
    if qrels.duplicated(["query_id", "corpus_id"]).any():
        raise ValueError("Duplicate qrels edges require explicit handling")
    expanded = qrels.merge(
        queries[["query_id", "question"]], on="query_id", how="left", validate="many_to_one"
    ).merge(corpus[["corpus_id", "context"]], on="corpus_id", how="left", validate="many_to_one")
    if expanded[["question", "context"]].isna().any().any():
        raise ValueError("Dangling qrels: unknown query or corpus identifier")
    expanded["pair_hash"] = [
        text_hash(q + "\x00" + c) for q, c in zip(expanded.question, expanded.context, strict=True)
    ]
    # Identical pair duplicates are matched by within-pair occurrence, with ambiguity retained.
    left = qa.copy()
    left["pair_occurrence"] = left.groupby(["split", "pair_hash"], sort=False).cumcount()
    expanded["pair_occurrence"] = expanded.groupby(["split", "pair_hash"], sort=False).cumcount()
    left["mapping_unique"] = ~left.duplicated(["split", "pair_hash"], keep=False)
    mapping = expanded[["split", "pair_hash", "pair_occurrence", "query_id", "corpus_id"]]
    merged = left.merge(
        mapping, on=["split", "pair_hash", "pair_occurrence"], how="left", validate="one_to_one"
    )
    if merged.query_id.isna().any() or len(merged) != len(expanded):
        raise ValueError("QA and retrieval contents are not in one-to-one correspondence")
    corpus = corpus.copy()
    corpus["context_normalized_hash"] = corpus.context.map(lambda s: text_hash(normalize(s)))
    corpus["context_hash"] = corpus.context.map(text_hash)
    corpus["canonical_id"] = corpus.groupby("context_normalized_hash").corpus_id.transform("min")
    merged = merged.merge(
        corpus[["corpus_id", "canonical_id"]], on="corpus_id", validate="many_to_one"
    )
    audit = {
        "queries": len(queries),
        "corpus_rows": len(corpus),
        "unique_exact_contexts": corpus.context.nunique(),
        "unique_normalized_contexts": corpus.canonical_id.nunique(),
        "qrels": len(qrels),
        "qrels_score_values": sorted(qrels.score.unique().tolist()),
        "unmapped_qa_rows": int(merged.query_id.isna().sum()),
        "ambiguous_identity_rows": int((~merged.mapping_unique).sum()),
        "source_id_equals_query_id": int(merged.id.eq(merged.query_id).sum()),
    }
    return merged, corpus, audit

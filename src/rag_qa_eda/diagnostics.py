"""Untuned lexical probe to understand data difficulty, not a final RAG model."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - half), float(center + half)


def retrieval_probe(
    qa: pd.DataFrame, corpus: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Evaluate only project validation; no project test results are computed.

    Both strict raw IDs and equivalent normalized passages are scored on exactly
    the same raw-corpus ranking to expose duplicate-ID evaluation artefacts.
    """
    candidates = qa.loc[qa.project_split.eq("validation") & qa.label_available]
    sample = candidates.sample(n=min(1000, len(candidates)), random_state=seed).sort_values(
        "row_key"
    )
    documents = corpus.sort_values("corpus_id")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=100000, sublinear_tf=True, dtype=np.float32
    )
    passages = vectorizer.fit_transform(documents.context)
    questions = vectorizer.transform(sample.question)
    ids = documents.corpus_id.to_numpy()
    canonical_ids = documents.canonical_id.to_numpy()
    results = []
    for start in range(0, len(sample), 128):
        scores = (questions[start : start + 128] @ passages.T).toarray()
        for local, row in enumerate(sample.iloc[start : start + 128].itertuples(index=False)):
            ranking = np.lexsort((ids, -scores[local]))[:10]
            ranking = ranking[scores[local, ranking] > 0]
            hits_raw = np.flatnonzero(ids[ranking] == row.corpus_id)
            hits_equivalent = np.flatnonzero(canonical_ids[ranking] == row.canonical_id)
            strict_rank = int(hits_raw[0] + 1) if len(hits_raw) else 0
            equivalent_rank = int(hits_equivalent[0] + 1) if len(hits_equivalent) else 0
            results.append(
                {
                    "row_key": row.row_key,
                    "source_id": row.id,
                    "source_split": row.split,
                    "query_id": row.query_id,
                    "gold_corpus_id": row.corpus_id,
                    "canonical_id": row.canonical_id,
                    "strict_rank_at_10": strict_rank,
                    "equivalent_rank_at_10": equivalent_rank,
                    "recall_at_1": int(0 < equivalent_rank <= 1),
                    "recall_at_5": int(0 < equivalent_rank <= 5),
                    "recall_at_10": int(0 < equivalent_rank <= 10),
                    "reciprocal_rank_at_10": 1 / equivalent_rank if equivalent_rank else 0,
                    "q_context_coverage": row.q_context_coverage,
                    "q_words": row.q_words,
                    "a_words": row.a_words,
                    "question_type": row.question_type,
                    "context_reference_flag": row.context_reference_flag,
                    "top1_corpus_id": int(ids[ranking[0]]) if len(ranking) else -1,
                    "top1_score": float(scores[local, ranking[0]]) if len(ranking) else 0,
                    "question": row.question,
                    "answer_text": row.answer_text,
                }
            )
    frame = pd.DataFrame(results)
    recall = frame.recall_at_5
    rr = frame.reciprocal_rank_at_10.to_numpy()
    rng = np.random.default_rng(seed)
    cluster = frame.groupby("canonical_id").agg(
        n=("row_key", "size"), recall=("recall_at_5", "sum"), rr=("reciprocal_rank_at_10", "sum")
    )
    bootstrap = []
    for _ in range(1000):
        draw = cluster.iloc[rng.integers(0, len(cluster), size=len(cluster))]
        bootstrap.append([draw.recall.sum() / draw.n.sum(), draw.rr.sum() / draw.n.sum()])
    intervals = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    summary = {
        "scope": "fixed random sample of project validation only; project test never evaluated",
        "seed": seed,
        "n_questions": len(frame),
        "corpus_rows": len(documents),
        "indexed_text": "raw context only, no questions or answers",
        "vectorizer": {"ngram_range": [1, 2], "max_features": 100000, "sublinear_tf": True},
        "recall_at_1_equivalent": float(frame.recall_at_1.mean()),
        "recall_at_5_equivalent": float(recall.mean()),
        "recall_at_10_equivalent": float(frame.recall_at_10.mean()),
        "mrr_at_10_equivalent": float(rr.mean()),
        "recall_at_5_strict_id": float(frame.strict_rank_at_10.between(1, 5).mean()),
        "mrr_at_10_strict_id": float(
            frame.strict_rank_at_10.map(lambda rank: 1 / rank if rank else 0).mean()
        ),
        "recall_at_5_wilson_95": list(wilson_interval(int(recall.sum()), len(recall))),
        "recall_at_5_cluster_bootstrap_95": intervals[:, 0].tolist(),
        "mrr_at_10_cluster_bootstrap_95": intervals[:, 1].tolist(),
        "ci_note": "1000 bootstrap resamples of canonical passage clusters; this sample only",
        "warnings": "gold passage labels are not exhaustive relevance judgments; this is not answer correctness",
    }
    return frame, summary

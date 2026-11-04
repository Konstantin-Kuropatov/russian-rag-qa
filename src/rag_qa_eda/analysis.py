"""Reproducible full-source audit with machine-readable intermediate artefacts."""

import json
import logging
import platform
from collections import Counter
from importlib.metadata import version
from itertools import combinations

import pandas as pd
from scipy.stats import ks_2samp

from .data import SPLITS, flatten_labels, join_retrieval, load_sources
from .diagnostics import retrieval_probe
from .download import verify_raw_files
from .features import add_features
from .paths import PROCESSED, REPORTS, TABLES
from .splits import make_project_split, near_duplicate_candidates
from .text import content_words

LOGGER = logging.getLogger(__name__)


def save_table(name: str, frame: pd.DataFrame, index: bool = False) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=index)


def missingness_table(raw_qa, raw_corpus, queries, qrels) -> pd.DataFrame:
    records = []
    for name, frame in {
        "qa": raw_qa,
        "corpus": raw_corpus,
        "queries": queries,
        "qrels": qrels,
    }.items():
        groups = frame.groupby("split", sort=False) if "split" in frame else [("all", frame)]
        for split, part in groups:
            for field in part.columns:
                nulls = int(part[field].isna().sum())
                empty = int(part[field].map(lambda x: isinstance(x, str) and not x.strip()).sum())
                records.append(
                    {
                        "source": name,
                        "split": split,
                        "field": field,
                        "rows": len(part),
                        "null_count": nulls,
                        "empty_string_count": empty,
                        "missing_fraction": (nulls + empty) / len(part),
                    }
                )
    return pd.DataFrame(records)


def overlap_table(qa: pd.DataFrame) -> pd.DataFrame:
    records = []
    for left, right in combinations(SPLITS, 2):
        a, b = qa.loc[qa.split.eq(left)], qa.loc[qa.split.eq(right)]
        for key in (
            "id",
            "context_hash",
            "context_normalized_hash",
            "question_normalized",
            "pair_hash",
        ):
            shared = set(a[key]) & set(b[key])
            records.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "key": key,
                    "shared_unique": len(shared),
                    "left_rows_affected": int(a[key].isin(shared).sum()),
                    "right_rows_affected": int(b[key].isin(shared).sum()),
                    "right_fraction": float(b[key].isin(shared).mean()),
                }
            )
    return pd.DataFrame(records)


def make_tables(
    qa: pd.DataFrame,
    corpus: pd.DataFrame,
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    raw_qa: pd.DataFrame,
    near: pd.DataFrame,
    probe: pd.DataFrame,
) -> dict:
    labelled = qa.loc[qa.label_available].copy()
    overview = (
        qa.groupby("split", sort=False)
        .agg(
            rows=("row_key", "size"),
            unique_original_ids=("id", "nunique"),
            contexts=("context_hash", "nunique"),
            labelled_rows=("label_available", "sum"),
            reader_eligible=("reader_eligible", "sum"),
            median_question_words=("q_words", "median"),
            median_context_words=("c_words", "median"),
        )
        .reset_index()
    )
    save_table("source_overview", overview)
    describe = raw_qa.assign(answers=raw_qa.answers.map(repr)).describe(include="all").T
    save_table("describe_all_source_fields", describe, index=True)
    numeric = labelled.select_dtypes(include=["number"]).drop(
        columns=["id", "source_row", "query_id", "corpus_id", "canonical_id", "split_group"],
        errors="ignore",
    )
    save_table(
        "describe_numeric_features",
        numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T,
        index=True,
    )
    missing = missingness_table(raw_qa, corpus, queries, qrels)
    # Nested answers are missing semantically even though the outer dict is non-null.
    nested = []
    for split, group in qa.groupby("split", sort=False):
        for field, mask in {
            "answers.text (blank)": ~group.label_available,
            "answers.answer_start (negative)": group.answer_start_raw.lt(0),
        }.items():
            nested.append(
                {
                    "source": "qa_nested",
                    "split": split,
                    "field": field,
                    "rows": len(group),
                    "null_count": 0,
                    "empty_string_count": int(mask.sum()),
                    "missing_fraction": float(mask.mean()),
                }
            )
    missing = pd.concat([missing, pd.DataFrame(nested)], ignore_index=True)
    save_table("missingness", missing)
    span = qa.groupby(["split", "span_status"]).size().rename("rows").reset_index()
    span["fraction_within_split"] = span.rows / span.split.map(qa.split.value_counts())
    save_table("span_status", span)
    overlaps = overlap_table(qa)
    save_table("split_overlap", overlaps)
    save_table("near_duplicate_candidates", near)
    duplicates = labelled.loc[
        labelled.duplicated("question_normalized", keep=False),
        ["row_key", "id", "split", "question", "answer_text", "canonical_id"],
    ]
    save_table("repeated_questions", duplicates)
    raw_duplicate_ids = qa.loc[
        qa.duplicated("id", keep=False), ["row_key", "id", "split", "question"]
    ]
    save_table("repeated_original_ids", raw_duplicate_ids)
    outlier_rows = []
    train = labelled.loc[labelled.split.eq("train")]
    for field in ("q_words", "c_words", "a_words", "c_e5_tokens", "a_context_char_ratio"):
        q1, q3 = train[field].quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        for split, group in labelled.groupby("split", sort=False):
            flagged = group[field].lt(lower) | group[field].gt(upper)
            outlier_rows.append(
                {
                    "feature": field,
                    "split": split,
                    "lower_train_iqr": lower,
                    "upper_train_iqr": upper,
                    "flagged": int(flagged.sum()),
                    "fraction": float(flagged.mean()),
                    "action": "retain_and_flag",
                }
            )
    save_table("length_outliers", pd.DataFrame(outlier_rows))
    for column in ("q_words", "c_words", "a_words"):
        save_table(
            f"extreme_{column}",
            labelled.nlargest(8, column)[
                ["row_key", "id", "split", "question", "answer_text", column, "span_status"]
            ],
        )
    qtypes = (
        labelled.groupby("question_type")
        .agg(
            rows=("row_key", "size"),
            median_q_words=("q_words", "median"),
            median_a_words=("a_words", "median"),
            reader_eligible_fraction=("reader_eligible", "mean"),
            median_lexical_coverage=("q_context_coverage", "median"),
        )
        .sort_values("rows", ascending=False)
        .reset_index()
    )
    save_table("question_types", qtypes)
    feature_columns = [
        "q_words",
        "c_words",
        "a_words",
        "c_sentences",
        "c_e5_tokens",
        "q_context_coverage",
        "q_context_jaccard",
        "q_answer_sentence_jaccard",
        "a_context_char_ratio",
        "a_position_relative",
    ]
    save_table(
        "spearman_correlations", labelled[feature_columns].corr(method="spearman"), index=True
    )
    for field in ("question", "context"):
        texts = (
            labelled[field]
            if field == "question"
            else corpus.drop_duplicates("canonical_id").context
        )
        count = Counter(word for text in texts for word in content_words(text))
        save_table(
            f"frequent_{field}_terms",
            pd.DataFrame(
                sorted(count.items(), key=lambda item: (-item[1], item[0]))[:30],
                columns=["term", "document_frequency"],
            ),
        )
    save_table(
        "answer_forms",
        labelled.answer_form.value_counts().rename_axis("answer_form").reset_index(name="rows"),
    )
    per_context = labelled.groupby("canonical_id").size().rename("questions").reset_index()
    save_table("questions_per_context", per_context)
    shift = []
    for field in ("q_words", "c_words", "c_e5_tokens", "q_context_coverage"):
        for split in ("validation", "test"):
            left = qa.loc[qa.split.eq("train"), field]
            right = qa.loc[qa.split.eq(split), field]
            shift.append(
                {
                    "feature": field,
                    "compared_to_train": split,
                    "train_median": float(left.median()),
                    "other_median": float(right.median()),
                    "ks_distance": float(ks_2samp(left, right, method="asymp").statistic),
                }
            )
    save_table("source_distribution_shift", pd.DataFrame(shift))
    quality = []
    for name, flag in {
        "no_final_question_mark": ~labelled.ends_question_mark,
        "question_at_most_two_words": labelled.q_words.le(2),
        "wiki_citation_in_context": labelled.c_wiki_citation_flag,
        "context_reference_heuristic": labelled.context_reference_flag,
        "answer_ratio_at_least_09": labelled.a_context_char_ratio.ge(0.9),
    }.items():
        quality.append(
            {
                "flag": name,
                "rows": int(flag.sum()),
                "fraction_of_labelled": float(flag.mean()),
                "action": "review_flag_not_automatic_deletion",
            }
        )
    save_table("text_review_flags", pd.DataFrame(quality))
    repeated = labelled.groupby("question_normalized").agg(
        rows=("row_key", "size"),
        contexts=("canonical_id", "nunique"),
        distinct_answer_strings=("answer_text", "nunique"),
    )
    save_table("repeated_question_summary", repeated.loc[repeated.rows.gt(1)].reset_index())
    chunk_rows = []
    valid = labelled.loc[labelled.reader_eligible]
    for size in (128, 256, 512):
        chunk_rows.append(
            {
                "method": "prefix_truncation",
                "content_window": size,
                "overlap": 0,
                "evaluated": len(valid),
                "answer_retained_fraction": float(valid[f"answer_retained_at_{size}"].mean()),
            }
        )
    for size in (128, 256):
        for overlap in (0, 32):
            chunk_rows.append(
                {
                    "method": "all_token_windows",
                    "content_window": size,
                    "overlap": overlap,
                    "evaluated": len(valid),
                    "answer_retained_fraction": float(
                        valid[f"answer_window_{size}_overlap_{overlap}"].mean()
                    ),
                }
            )
    save_table("chunking_coverage", pd.DataFrame(chunk_rows))
    project = (
        labelled.groupby("project_split")
        .agg(
            rows=("row_key", "size"),
            contexts=("canonical_id", "nunique"),
            groups=("split_group", "nunique"),
            reader_eligible=("reader_eligible", "sum"),
        )
        .reset_index()
    )
    save_table("project_split_summary", project)
    project_overlap = []
    for left, right in combinations(("train", "validation", "test"), 2):
        a, b = labelled[labelled.project_split.eq(left)], labelled[labelled.project_split.eq(right)]
        for key in ("context_normalized_hash", "question_normalized", "split_group"):
            project_overlap.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "key": key,
                    "shared_unique": len(set(a[key]) & set(b[key])),
                }
            )
    save_table("project_split_overlap", pd.DataFrame(project_overlap))
    save_table("retrieval_probe", probe)
    probe = probe.copy()
    probe["coverage_bin"] = pd.cut(
        probe.q_context_coverage, [-0.001, 0.25, 0.5, 0.75, 1], include_lowest=True
    )
    stratified = (
        probe.groupby("coverage_bin", observed=False)
        .agg(
            questions=("row_key", "size"),
            recall_at_5=("recall_at_5", "mean"),
            mrr_at_10=("reciprocal_rank_at_10", "mean"),
        )
        .reset_index()
    )
    save_table("retrieval_by_lexical_coverage", stratified)
    save_table(
        "retrieval_by_question_type",
        probe.groupby("question_type")
        .agg(
            questions=("row_key", "size"),
            recall_at_5=("recall_at_5", "mean"),
            mrr_at_10=("reciprocal_rank_at_10", "mean"),
        )
        .reset_index(),
    )
    make_review_sample(labelled).to_csv(PROCESSED / "review_sample.csv", index=False)
    return {"overview": overview.to_dict("records"), "project_split": project.to_dict("records")}


def make_review_sample(labelled: pd.DataFrame) -> pd.DataFrame:
    """Targeted review sample, explicitly not an unbiased prevalence estimate."""
    parts = []
    for name, group in labelled.groupby("span_status", sort=True):
        parts.append(group.sample(min(5, len(group)), random_state=42).assign(review_reason=name))
    parts.append(
        labelled.loc[labelled.context_reference_flag]
        .sample(10, random_state=42)
        .assign(review_reason="context_reference_flag")
    )
    parts.append(labelled.nlargest(5, "a_words").assign(review_reason="long_answer"))
    parts.append(labelled.sample(15, random_state=43).assign(review_reason="random_labelled"))
    review = pd.concat(parts).drop_duplicates("row_key")
    columns = [
        "row_key",
        "id",
        "split",
        "review_reason",
        "question",
        "context",
        "answer_text",
        "span_status",
    ]
    review = review[columns].copy()
    review["human_answerable"] = ""
    review["human_reference_correct"] = ""
    review["human_comment"] = ""
    return review


def run_analysis() -> dict:
    for path in (PROCESSED, REPORTS, TABLES):
        path.mkdir(parents=True, exist_ok=True)
    verify_raw_files()
    LOGGER.info("Loading QA, corpus, queries and qrels")
    raw_qa, raw_corpus, queries, qrels = load_sources()
    qa = flatten_labels(raw_qa)
    qa, corpus, join_audit = join_retrieval(qa, raw_corpus, queries, qrels)
    qa, corpus = add_features(qa, corpus)
    near = near_duplicate_candidates(corpus)
    assignments = make_project_split(qa, near)
    qa = qa.merge(assignments, on="row_key", how="left", validate="one_to_one")
    qa["project_split"] = qa.project_split.fillna("unlabeled_source_test")
    qa["retrieval_train_eligible"] = (
        qa.label_available & qa.question.str.strip().ne("") & qa.context.str.strip().ne("")
    )
    LOGGER.info("Running fixed TF-IDF diagnostic on project validation (no test evaluation)")
    probe, probe_summary = retrieval_probe(qa, corpus)
    statistics = make_tables(qa, corpus, queries, qrels, raw_qa, near, probe)
    qa.to_parquet(PROCESSED / "qa_features.parquet", index=False)
    corpus.to_parquet(PROCESSED / "corpus_features.parquet", index=False)
    corpus.drop_duplicates("canonical_id").sort_values("canonical_id").to_parquet(
        PROCESSED / "corpus_canonical.parquet", index=False
    )
    canonical_qrels = qa[
        [
            "row_key",
            "query_id",
            "corpus_id",
            "canonical_id",
            "split",
            "project_split",
            "label_available",
        ]
    ].copy()
    canonical_qrels["score"] = 1
    canonical_qrels.to_parquet(PROCESSED / "qrels_canonical.parquet", index=False)
    qa[
        ["row_key", "id", "query_id", "corpus_id", "canonical_id", "project_split", "split_group"]
    ].to_parquet(PROCESSED / "split_assignments.parquet", index=False)
    summary = {
        "seed": 42,
        "join_audit": join_audit,
        **statistics,
        "labelled_rows": int(qa.label_available.sum()),
        "unlabelled_rows": int((~qa.label_available).sum()),
        "raw_negative_offsets_labelled": int(
            (qa.label_available & qa.answer_start_raw.lt(0)).sum()
        ),
        "raw_exact_spans": int(qa.span_status.eq("exact").sum()),
        "unique_exact_relocations": int(qa.span_status.eq("relocated_exact").sum()),
        "reader_eligible_rows": int(qa.reader_eligible.sum()),
        "span_boundary_violations": int(
            (qa.answer_start_fixed.notna() & ~qa.word_boundary_ok).sum()
        ),
        "pronoun_review_flags": int((qa.label_available & qa.context_reference_flag).sum()),
        "full_context_answers": int(qa.answer_equals_context.sum()),
        "near_full_context_answers": int(qa.a_context_char_ratio.ge(0.9).sum()),
        "source_id_collision_rows": int(qa.duplicated("id", keep=False).sum()),
        "near_duplicate_candidate_pairs": len(near),
        "near_pairs_ge_097": int(near.char_cosine.ge(0.97).sum()),
        "corpus_over_512_e5": int(corpus.drop_duplicates("canonical_id").c_e5_tokens.gt(512).sum()),
        "diagnostic": probe_summary,
        "environment": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["pandas", "numpy", "scikit-learn", "tokenizers", "razdel"]
            },
        },
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    from .report import write_findings

    write_findings(summary)
    LOGGER.info("Analysis complete: %s", REPORTS / "summary.json")
    return summary

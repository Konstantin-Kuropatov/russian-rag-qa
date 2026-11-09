"""Offline integration checks; skipped when the public snapshot is not downloaded."""

import pandas as pd
import pytest

from rag_qa_eda.paths import PROCESSED

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "qa_features.parquet").exists(),
    reason="Run rag-eda analyze for snapshot checks",
)


@pytest.fixture(scope="module")
def qa_snapshot():
    return pd.read_parquet(PROCESSED / "qa_features.parquet")


def test_all_accepted_reader_offsets_reconstruct_original_label(qa_snapshot):
    qa = qa_snapshot.loc[qa_snapshot.reader_eligible]
    assert all(
        row.context[
            int(row.answer_start_fixed) : int(row.answer_start_fixed) + len(row.answer_text)
        ]
        == row.answer_text
        for row in qa.itertuples(index=False)
    )


def test_unpublished_labels_stay_unknown(qa_snapshot):
    source_test = qa_snapshot.loc[qa_snapshot.split.eq("test")]
    assert not source_test.label_available.any()
    assert not source_test.reader_eligible.any()
    assert source_test.answer_start_fixed.isna().all()
    assert source_test.a_words.isna().all()
    assert source_test.a_chars.isna().all()
    assert source_test.iqr_flag_a_words.isna().all()
    assert source_test.project_split.eq("unlabeled_source_test").all()


def test_project_partitions_are_disjoint(qa_snapshot):
    labelled = qa_snapshot.loc[qa_snapshot.label_available]
    assert labelled.groupby("context_normalized_hash").project_split.nunique().max() == 1
    assert labelled.groupby("question_normalized").project_split.nunique().max() == 1
    assert labelled.groupby("split_group").project_split.nunique().max() == 1
    assert set(labelled.project_split) == {"train", "validation", "test"}


def test_original_id_collisions_are_resolved_with_unique_row_keys(qa_snapshot):
    assert qa_snapshot.id.duplicated().any()
    assert not qa_snapshot.row_key.duplicated().any()

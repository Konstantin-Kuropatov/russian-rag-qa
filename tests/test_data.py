import numpy as np
import pandas as pd
import pytest

from rag_qa_eda.data import flatten_labels, join_retrieval
from rag_qa_eda.splits import make_project_split


def fixture_qa():
    return pd.DataFrame(
        [
            {
                "id": 7,
                "split": "train",
                "source_row": 0,
                "row_key": "train:0",
                "title": "x",
                "context": "Ответ один.",
                "question": "Какой ответ?",
                "answers": {"text": np.array(["один"]), "answer_start": np.array([6])},
            },
            {
                "id": 7,
                "split": "test",
                "source_row": 0,
                "row_key": "test:0",
                "title": "x",
                "context": "Другой текст.",
                "question": "Что здесь?",
                "answers": {"text": np.array([""]), "answer_start": np.array([-1])},
            },
        ]
    )


def test_join_uses_question_context_not_original_id():
    qa = flatten_labels(fixture_qa())
    corpus = pd.DataFrame({"corpus_id": [21, 22], "context": ["Ответ один.", "Другой текст."]})
    queries = pd.DataFrame({"query_id": [100, 101], "question": ["Какой ответ?", "Что здесь?"]})
    qrels = pd.DataFrame(
        {"query_id": [100, 101], "corpus_id": [21, 22], "score": [1, 1], "split": ["train", "test"]}
    )
    joined, _, audit = join_retrieval(qa, corpus, queries, qrels)
    assert joined.query_id.tolist() == [100, 101]
    assert joined.id.tolist() == [7, 7]
    assert joined.label_available.tolist() == [True, False]
    assert audit["unmapped_qa_rows"] == 0


def test_dangling_qrels_fail_loudly():
    qa = flatten_labels(fixture_qa())
    with pytest.raises(ValueError, match="Dangling"):
        join_retrieval(
            qa,
            pd.DataFrame({"corpus_id": [1], "context": ["x"]}),
            pd.DataFrame({"query_id": [1], "question": ["q"]}),
            pd.DataFrame({"query_id": [99], "corpus_id": [1], "score": [1], "split": ["train"]}),
        )


def test_multianswer_schema_cannot_be_silently_flattened():
    qa = fixture_qa()
    qa.at[0, "answers"] = {"text": ["a", "b"], "answer_start": [0, 1]}
    with pytest.raises(ValueError, match="multi-answer"):
        flatten_labels(qa)


def test_group_split_keeps_shared_questions_and_passages_together():
    qa = pd.DataFrame(
        [
            {
                "row_key": f"train:{i}",
                "canonical_id": i // 2,
                "label_available": True,
                "question_normalized": f"q{i}",
                "context_normalized_hash": f"c{i // 2}",
            }
            for i in range(100)
        ]
    )
    qa.loc[2, "question_normalized"] = qa.loc[0, "question_normalized"]
    near = pd.DataFrame([{"left_id": 2, "right_id": 3, "char_cosine": 0.99}])
    result = qa.merge(make_project_split(qa, near), on="row_key")
    assert result.groupby("canonical_id").project_split.nunique().max() == 1
    assert result.groupby("question_normalized").project_split.nunique().max() == 1
    assert result.loc[result.canonical_id.isin([2, 3]), "project_split"].nunique() == 1
    pd.testing.assert_frame_equal(make_project_split(qa, near), make_project_split(qa, near))

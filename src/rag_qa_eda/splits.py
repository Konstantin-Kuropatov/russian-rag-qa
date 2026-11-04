"""Conservative split grouping and high-similarity candidate discovery."""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

LOGGER = logging.getLogger(__name__)


class UnionFind:
    def __init__(self, values):
        self.parent = {value: value for value in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def near_duplicate_candidates(corpus: pd.DataFrame) -> pd.DataFrame:
    """Nearest lexical neighbour per canonical passage, not semantic duplicate labels.

    Full-corpus sparse similarity is evaluated in bounded batches. Only candidates
    >= .90 are exported; >= .97 are conservatively co-grouped for project splits.
    """
    canonical = corpus.drop_duplicates("canonical_id").sort_values("canonical_id")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(4, 5), min_df=3, max_features=40000, dtype=np.float32
    )
    matrix = vectorizer.fit_transform(canonical.context)
    ids = canonical.canonical_id.to_numpy()
    pairs = {}
    for start in range(0, len(canonical), 128):
        scores = (matrix[start : start + 128] @ matrix.T).tocsr()
        for local in range(scores.shape[0]):
            i = start + local
            row = scores.getrow(local)
            mask = row.indices != i
            indices, values = row.indices[mask], row.data[mask]
            if not len(values):
                continue
            top = int(np.argmax(values))
            j, score = indices[top], float(values[top])
            if score >= 0.90:
                key = tuple(sorted([int(ids[i]), int(ids[j])]))
                pairs[key] = max(score, pairs.get(key, 0))
        if start % 2560 == 0:
            LOGGER.info("Near-duplicate scan %s / %s", start, len(canonical))
    return pd.DataFrame(
        [
            {"left_id": a, "right_id": b, "char_cosine": score}
            for (a, b), score in sorted(pairs.items())
        ],
        columns=["left_id", "right_id", "char_cosine"],
    )


def make_project_split(qa: pd.DataFrame, near: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Group normalized passages, repeated questions and >=.97 near candidates.

    The source test has no reference answers and is never converted into negative
    examples. A new project test is carved out of the labelled pool and is not
    used in the diagnostic retrieval experiment.
    """
    labelled = qa.loc[qa.label_available].copy()
    ids = sorted(labelled.canonical_id.unique())
    groups = UnionFind(ids)
    for row in near.itertuples(index=False):
        if (
            row.char_cosine >= 0.97
            and row.left_id in groups.parent
            and row.right_id in groups.parent
        ):
            groups.union(row.left_id, row.right_id)
    for _, group in labelled.groupby("question_normalized", sort=False):
        cids = group.canonical_id.unique()
        for cid in cids[1:]:
            groups.union(cids[0], cid)
    labelled["split_group"] = labelled.canonical_id.map(groups.find)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, remainder_idx = next(splitter.split(labelled, groups=labelled.split_group))
    labelled["project_split"] = "train"
    remainder = labelled.iloc[remainder_idx]
    val_part, test_part = next(
        GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1).split(
            remainder, groups=remainder.split_group
        )
    )
    labelled.loc[remainder.index[val_part], "project_split"] = "validation"
    labelled.loc[remainder.index[test_part], "project_split"] = "test"
    for key in ("context_normalized_hash", "question_normalized", "split_group"):
        if labelled.groupby(key).project_split.nunique().max() > 1:
            raise ValueError(f"Project split leaks {key}")
    assert len(train_idx) + len(remainder_idx) == len(labelled)
    return labelled[["row_key", "project_split", "split_group"]]

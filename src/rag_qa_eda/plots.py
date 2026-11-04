"""Static scientific figures for the notebook and standalone review."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

from .paths import FIGURES, TABLES

COLORS = ["#31688e", "#35b779", "#e6a23c", "#b45f6a"]


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", palette=COLORS)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelpad": 8,
        }
    )


def save_figure(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight", facecolor="white")
    return fig


def overview(summary: dict):
    frame = (
        pd.DataFrame(summary["overview"])
        .set_index("split")
        .reindex(["train", "validation", "test"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    x = np.arange(3)
    axes[0].bar(x, frame.labelled_rows, color=COLORS[0], label="Эталон опубликован")
    axes[0].bar(
        x,
        frame.rows - frame.labelled_rows,
        bottom=frame.labelled_rows,
        color=COLORS[2],
        label="Эталон не опубликован",
    )
    axes[0].set(xticks=x, xticklabels=frame.index, ylabel="Вопросов", title="Разделы QA-версии")
    for i, n in enumerate(frame.rows):
        axes[0].text(i, n + 800, f"{n:,}".replace(",", " "), ha="center", fontsize=10)
    axes[0].set_ylim(0, 51000)
    axes[0].legend(fontsize=9)
    audit = summary["join_audit"]
    counts = [
        audit["corpus_rows"],
        audit["unique_exact_contexts"],
        audit["unique_normalized_contexts"],
    ]
    axes[1].barh(
        ["Строки корпуса", "Уникальные тексты", "После нормализации"],
        counts,
        color=[COLORS[2], COLORS[0], COLORS[1]],
    )
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Фрагментов", title="Повторы в retrieval-корпусе")
    for y, n in enumerate(counts):
        axes[1].text(n + 120, y, f"{n:,}".replace(",", " "), va="center")
    axes[1].set_xlim(0, 20000)
    return save_figure(fig, "01_sources_and_duplicates")


def missingness(qa: pd.DataFrame):
    values = (
        qa.assign(
            context_blank=qa.context.str.strip().eq(""),
            question_blank=qa.question.str.strip().eq(""),
            answer_blank=~qa.label_available,
            offset_negative=qa.answer_start_raw.lt(0),
        )
        .groupby("split")[["context_blank", "question_blank", "answer_blank", "offset_negative"]]
        .mean()
    )
    values = values.reindex(["train", "validation", "test"])
    values.columns = ["Пустой контекст", "Пустой вопрос", "Пустой ответ", "Отрицательная позиция"]
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    sns.heatmap(
        values * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrBr",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% строк"},
        ax=ax,
    )
    ax.set(
        title="Семантические пропуски: отрицательная позиция не равна отсутствию ответа",
        xlabel="",
        ylabel="",
    )
    return save_figure(fig, "02_missing_labels")


def lengths(qa: pd.DataFrame, corpus: pd.DataFrame):
    labelled = qa.loc[qa.label_available]
    values = [labelled.q_words, corpus.drop_duplicates("canonical_id").c_words, labelled.a_words]
    titles = [
        "Вопросы · QA-строки",
        "Контексты · уникальные фрагменты",
        "Ответы · размеченные QA-строки",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for ax, data, title, color in zip(axes, values, titles, COLORS, strict=False):
        sns.histplot(
            data, bins=np.geomspace(max(data.min(), 1), data.max() + 1, 45), color=color, ax=ax
        )
        ax.set_xscale("log")
        ticks = [
            x for x in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] if data.min() <= x <= data.max()
        ]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.axvline(
            data.median(),
            color="#333333",
            linestyle="--",
            linewidth=1,
            label=f"Медиана {data.median():.0f}",
        )
        ax.set(title=title, xlabel="Слов (логарифмическая шкала)", ylabel="Объектов")
        ax.legend(fontsize=9)
    return save_figure(fig, "03_length_distributions")


def spans(qa: pd.DataFrame):
    counts = qa.loc[qa.label_available].span_status.value_counts()
    labels = {
        "exact": "Точный исходный span",
        "relocated_exact": "Единственное точное вхождение · исправлена позиция",
        "ambiguous_exact": "Несколько вхождений · позиция неоднозначна",
        "case_or_unicode_mismatch": "Различия регистра / Unicode / пробелов",
        "punctuation_or_space_mismatch": "Различия пунктуации / пробелов",
        "case_and_punctuation_mismatch": "Различия регистра вместе с пунктуацией",
        "not_found": "Строка ответа не найдена",
    }
    fig, ax = plt.subplots(figsize=(12, 4.2), layout="constrained")
    ax.barh(
        [labels[x] for x in counts.index],
        counts,
        color=[COLORS[0] if x == "exact" else COLORS[2] for x in counts.index],
    )
    ax.invert_yaxis()
    for i, n in enumerate(counts):
        ax.text(
            n + 220, i, f"{n:,} · {n / counts.sum():.1%}".replace(",", " "), va="center", fontsize=9
        )
    ax.set(
        xlim=(0, 46000),
        xlabel="Размеченных вопросов",
        title="Проверка разметки на train + validation",
    )
    return save_figure(fig, "04_answer_span_audit")


def source_overlap(qa: pd.DataFrame):
    splits = ["train", "validation", "test"]
    values = np.zeros((3, 3))
    for i, left in enumerate(splits):
        for j, right in enumerate(splits):
            a, b = (
                set(qa.loc[qa.split.eq(left), "context_hash"]),
                set(qa.loc[qa.split.eq(right), "context_hash"]),
            )
            values[i, j] = len(a & b) / len(b) * 100
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    sns.heatmap(
        pd.DataFrame(values, index=splits, columns=splits),
        cmap="Blues",
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% уникальных контекстов столбца"},
        ax=ax,
    )
    ax.set(
        xlabel="Раздел, который оцениваем",
        ylabel="Раздел, где ищем совпадение",
        title="Все контексты исходной validation присутствуют в train",
    )
    return save_figure(fig, "05_source_context_overlap")


def question_types(qa: pd.DataFrame):
    frame = (
        qa.loc[qa.label_available]
        .groupby("question_type")
        .agg(rows=("row_key", "size"), a_words=("a_words", "median"))
        .sort_values("rows", ascending=False)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    axes[0].barh(frame.index, frame.rows, color=COLORS[0])
    axes[0].invert_yaxis()
    axes[0].set(title="Типы вопросов · эвристика по первым трём словам", xlabel="Вопросов")
    axes[1].barh(frame.index, frame.a_words, color=COLORS[1])
    axes[1].invert_yaxis()
    axes[1].set(title="Связь типа вопроса с длиной эталона", xlabel="Медиана длины ответа · слов")
    return save_figure(fig, "06_question_types_and_answers")


def correlations(qa: pd.DataFrame):
    columns = [
        "q_words",
        "c_words",
        "a_words",
        "c_sentences",
        "c_e5_tokens",
        "q_context_coverage",
        "q_answer_sentence_jaccard",
        "a_context_char_ratio",
        "a_position_relative",
    ]
    labels = [
        "Слов в вопросе",
        "Слов в контексте",
        "Слов в ответе",
        "Предложений",
        "Токенов E5",
        "Покрытие вопроса",
        "Jaccard с gold-предложением",
        "Ответ / контекст",
        "Позиция ответа",
    ]
    corr = qa.loc[qa.label_available, columns].corr(method="spearman")
    corr.index = corr.columns = labels
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    sns.heatmap(
        corr,
        mask=np.triu(np.ones(corr.shape, dtype=bool), k=1),
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        ax=ax,
    )
    ax.set(title="Spearman · связи характеристик текста и эталона")
    return save_figure(fig, "07_spearman_features")


def pairplot(qa: pd.DataFrame):
    sample = qa.loc[qa.label_available].sample(
        min(1500, int(qa.label_available.sum())), random_state=42
    )
    data = pd.DataFrame(
        {
            "log1p слов в вопросе": np.log1p(sample.q_words),
            "log1p слов в контексте": np.log1p(sample.c_words),
            "log1p слов в ответе": np.log1p(sample.a_words),
            "Покрытие вопроса": sample.q_context_coverage,
            "Span для reader": sample.reader_eligible.map(
                {True: "Пригоден", False: "Нужна проверка"}
            ),
        }
    )
    grid = sns.pairplot(
        data,
        hue="Span для reader",
        diag_kind="hist",
        corner=True,
        height=2.5,
        plot_kws={"s": 12, "alpha": 0.28},
        diag_kws={"bins": 25, "alpha": 0.5},
    )
    grid.figure.suptitle("Парные распределения · 1 500 размеченных строк, seed=42", y=1.02)
    return save_figure(grid.figure, "08_pairplot")


def token_budgets(corpus: pd.DataFrame):
    unique = corpus.drop_duplicates("canonical_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    sns.histplot(unique.c_e5_tokens, bins=50, ax=axes[0], color=COLORS[0])
    for cutoff, color in zip((128, 256, 512), COLORS[1:], strict=False):
        axes[0].axvline(cutoff, color=color, linestyle="--", label=f"{cutoff} токенов")
    axes[0].set(
        xlabel="Токенов полного входа E5",
        ylabel="Уникальных фрагментов",
        title="Большинство исходных абзацев помещаются в encoder",
    )
    axes[0].legend(fontsize=9)
    chunks = pd.read_csv(TABLES / "chunking_coverage.csv")
    labels = [
        f"Обрезка {int(r.content_window)}"
        if r.method == "prefix_truncation"
        else f"Окна {int(r.content_window)}, overlap {int(r.overlap)}"
        for r in chunks.itertuples()
    ]
    axes[1].barh(labels, chunks.answer_retained_fraction * 100, color=COLORS[1])
    axes[1].invert_yaxis()
    for i, value in enumerate(chunks.answer_retained_fraction):
        axes[1].text(value * 100 + 0.4, i, f"{value:.3%}", va="center", fontsize=8)
    axes[1].set(
        xlim=(0, 115),
        xlabel="% валидных gold spans, полностью сохранённых",
        title="Сохранность ответа ≠ качество поиска",
    )
    return save_figure(fig, "09_token_budgets_and_answer_coverage")


def lexical_and_position(qa: pd.DataFrame):
    labelled = qa.loc[qa.label_available]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    sns.histplot(labelled.q_context_coverage, bins=20, ax=axes[0], color=COLORS[0])
    axes[0].set(
        title="Лексическое покрытие вопроса",
        xlabel="Доля содержательных слов вопроса в тексте",
        ylabel="Вопросов",
    )
    sns.histplot(labelled.q_answer_sentence_jaccard.dropna(), bins=25, ax=axes[1], color=COLORS[1])
    axes[1].set(
        title="Вопрос и предложение с gold span",
        xlabel="Jaccard · без лемматизации",
        ylabel="Вопросов",
    )
    sns.histplot(labelled.a_position_relative.dropna(), bins=25, ax=axes[2], color=COLORS[2])
    axes[2].set(title="Где начинается ответ", xlabel="Позиция / длина контекста", ylabel="Вопросов")
    return save_figure(fig, "10_lexical_overlap_and_position")


def retrieval_diagnostic(summary: dict):
    p = summary["diagnostic"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    values = [
        p["recall_at_5_strict_id"],
        p["recall_at_5_equivalent"],
        p["mrr_at_10_strict_id"],
        p["mrr_at_10_equivalent"],
    ]
    axes[0].barh(
        [
            "Recall@5 · строгий ID",
            "Recall@5 · эквивалентные тексты",
            "MRR@10 · строгий ID",
            "MRR@10 · эквивалентные тексты",
        ],
        values,
        color=[COLORS[2], COLORS[0], COLORS[2], COLORS[0]],
    )
    axes[0].invert_yaxis()
    for i, value in enumerate(values):
        axes[0].text(value + 0.012, i, f"{value:.3f}", va="center")
    axes[0].set(xlim=(0, 1), title="Одно ранжирование · разные правила проверки", xlabel="Метрика")
    table = pd.read_csv(TABLES / "retrieval_by_lexical_coverage.csv")
    axes[1].bar(table.coverage_bin, table.recall_at_5, color=COLORS[1])
    for i, row in table.iterrows():
        if pd.notna(row.recall_at_5):
            axes[1].text(i, row.recall_at_5 + 0.025, f"n={row.questions}", ha="center", fontsize=9)
    axes[1].set(
        ylim=(0, 1.12),
        title="Связь лексического покрытия с поиском",
        xlabel="Покрытие вопроса словами контекста",
        ylabel="Recall@5",
    )
    axes[1].tick_params(axis="x", labelsize=8)
    return save_figure(fig, "11_retrieval_diagnostic")


def frequent_terms():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, field, title, color in zip(
        axes, ["question", "context"], ["Вопросы", "Уникальные контексты"], COLORS, strict=False
    ):
        table = pd.read_csv(TABLES / f"frequent_{field}_terms.csv").head(18)
        ax.barh(table.term, table.document_frequency, color=color)
        ax.invert_yaxis()
        ax.set(title=title, xlabel="Документов, где встречается слово")
    fig.suptitle("Частые содержательные слова · без лемматизации")
    return save_figure(fig, "12_frequent_terms")


def answer_forms_and_question_counts():
    forms = pd.read_csv(TABLES / "answer_forms.csv")
    per_context = pd.read_csv(TABLES / "questions_per_context.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].barh(forms.answer_form, forms.rows, color=COLORS[: len(forms)])
    axes[0].invert_yaxis()
    for i, n in enumerate(forms.rows):
        axes[0].text(n + 300, i, f"{n:,}".replace(",", " "), va="center")
    axes[0].set(
        xlim=(0, forms.rows.max() * 1.18),
        title="Форма эталона · регулярное выражение",
        xlabel="Размеченных вопросов",
    )
    sns.histplot(per_context.questions, discrete=True, color=COLORS[0], ax=axes[1])
    axes[1].set(
        title="Сколько вопросов приходится на один контекст",
        xlabel="Вопросов к уникальному размеченному контексту",
        ylabel="Контекстов",
    )
    return save_figure(fig, "13_answer_forms_and_group_sizes")

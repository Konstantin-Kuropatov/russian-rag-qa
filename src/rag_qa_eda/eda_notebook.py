# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # SberQuAD: разведочный анализ
#
# Чекпойнт 2 · Команда №14 · Куропатов Константин · Куратор Ляпин Данила
#
# Проверим, как устроены вопросы, тексты и ответы, где есть ошибки разметки
# и как разделить данные для дальнейших экспериментов.
# Первый поисковый бейзлайн — TF-IDF; его реализация и оценка будут в чекпойнте 3.

# %%
import json
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython
from IPython.display import Markdown, display

from rag_qa_eda import plots
from rag_qa_eda.data import load_sources
from rag_qa_eda.paths import PROCESSED, REPORTS, ROOT, TABLES

get_ipython().run_line_magic("matplotlib", "inline")

if not (PROCESSED / "qa_features.parquet").exists():
    raise FileNotFoundError("Сначала выполните из корня проекта: uv run rag-eda all")

qa = pd.read_parquet(PROCESSED / "qa_features.parquet")
corpus = pd.read_parquet(PROCESSED / "corpus_features.parquet")
labelled = qa.loc[qa.label_available].copy()
summary = json.loads((REPORTS / "summary.json").read_text())
plots.set_style()
pd.set_option("display.max_colwidth", 95)
pd.set_option("display.max_columns", 16)


def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def show_plot(figure):
    display(figure)
    plt.close(figure)


print("Python:", sys.version.split()[0])
display(pd.Series(summary["environment"], name="Версия"))

# %% [markdown]
# ## 1. Данные
#
# [QA-версия SberQuAD](https://huggingface.co/datasets/kuznetsoffandrey/sberquad)
# содержит вопросы, контексты и ответы.
# [Retrieval-версия](https://huggingface.co/datasets/kaengreg/sberquad-retrieval)
# и [qrels](https://huggingface.co/datasets/kaengreg/sberquad-retrieval-qrels)
# связывают вопросы с документами корпуса. Версии закреплены в `config/data_sources.json`.
#
# Один абзац может относиться к нескольким вопросам. Поэтому длины текстов
# считаем по уникальным фрагментам, а вопросы и ответы — по строкам QA.

# %%
manifest = json.loads((REPORTS / "data_manifest.json").read_text())
manifest_df = pd.DataFrame(manifest["files"])
display(
    manifest_df.assign(
        revision=manifest_df.revision.str.slice(0, 12) + "…",
        sha256=manifest_df.sha256.str.slice(0, 16) + "…",
    )[["path", "revision", "size_bytes", "sha256"]]
)
display(table("source_overview"))
show_plot(plots.overview(summary))
audit = summary["join_audit"]
display(
    Markdown(
        f"Всего **{audit['queries']:,} вопросов**, из них **{summary['labelled_rows']:,}** "
        f"с ответами. Строк в корпусе — {audit['corpus_rows']:,}; после удаления "
        f"дублей остаётся **{audit['unique_normalized_contexts']:,} текстов**."
    )
)

# %% [markdown]
# ## 2. Поля и основные статистики
#
# Для QA эталон — текст ответа, для поиска — связь «вопрос → документ».
# Все qrels положительные (`score=1`); готовых отрицательных примеров здесь нет.
# В таблице `describe` словарь `answers` представлен строкой, чтобы pandas мог
# посчитать частоты. В обработке данных текст и позиция ответа хранятся отдельно.

# %%
raw_qa, raw_corpus, raw_queries, raw_qrels = load_sources()
display(raw_qa[["id", "title", "context", "question", "answers", "split"]].head(3))
display(raw_qa.dtypes.astype(str).rename("dtype").to_frame())
display(
    raw_qa.drop(columns=["row_key", "source_row"])
    .assign(answers=raw_qa.answers.map(repr))
    .describe(include="all")
    .T
)
display(
    labelled[
        ["q_chars", "q_words", "c_chars", "c_words", "a_chars", "a_words", "a_position_relative"]
    ]
    .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    .T
)
display(raw_qrels.score.value_counts(dropna=False).rename("Число qrels"))

# %% [markdown]
# `title` в QA — константа `SberChallenge`, а в корпусе заголовки пустые.
# Название и URL статьи из этого поля не получить. ID используются как ключи.

# %% [markdown]
# ## 3. Пропуски
#
# Проверяем `null`, пустые строки и отрицательные позиции ответа.
# Пустой ответ и неверная позиция при непустом ответе требуют разной обработки.

# %%
missing = table("missingness")
display(
    missing.loc[
        (missing.source.isin(["qa", "qa_nested"]))
        & missing.field.isin(
            [
                "id",
                "title",
                "context",
                "question",
                "answers",
                "answers.text (blank)",
                "answers.answer_start (negative)",
            ]
        )
    ]
)
fields = ["id", "title", "context", "question", "answers"]
display(
    pd.DataFrame(
        {
            "null_cells": [raw_qa[fields].isna().sum().sum()],
            "total_cells": [raw_qa[fields].size],
            "null_fraction": [raw_qa[fields].isna().to_numpy().mean()],
            "unpublished_answer_fraction": [(~qa.label_available).mean()],
        }
    )
)
show_plot(plots.missingness(qa))
display(
    qa.loc[
        qa.row_key.eq("test:0"),
        ["row_key", "question", "answer_text", "label_available"],
    ]
)
display(
    Markdown(
        f"У всех **{summary['unlabelled_rows']:,}** вопросов исходного test ответ пустой, "
        "позиция — −1. Эталоны не опубликованы, поэтому эти строки отмечены "
        "`label_available=False`. Их нельзя использовать как вопросы без ответа. "
        f"При этом в размеченных данных есть **{summary['raw_negative_offsets_labelled']:,}** "
        "отрицательных позиций — их проверяем по тексту. Пропуски не заполняем."
    )
)

# %% [markdown]
# ## 4. Позиции ответов
#
# По указанной позиции должна находиться ровно строка ответа. Если позиция
# неверная, исправляем её только при единственном точном вхождении в контекст.
# Дополнительно проверяем, что ответ не начинается и не заканчивается внутри слова.
# Исходные текст и позиция сохраняются рядом с результатом проверки.

# %%
display(table("span_status"))
show_plot(plots.spans(qa))
display(
    qa.loc[
        qa.span_status.eq("relocated_exact"),
        [
            "row_key",
            "question",
            "answer_text",
            "answer_start_raw",
            "answer_start_fixed",
            "word_boundary_ok",
        ],
    ].head(5)
)
display(
    Markdown(
        f"Исходная позиция верна у **{summary['raw_exact_spans']:,}** ответов; "
        f"у **{summary['unique_exact_relocations']:,}** она восстановлена. "
        f"Нарушения границ слова найдены в **{summary['span_boundary_violations']}** случаях. "
        f"Число строк для обучения извлечению ответа — **{summary['reader_eligible_rows']:,}** "
        "(`reader_eligible=True`). Для поиска документа позиция ответа не нужна."
    )
)

# %% [markdown]
# ## 5. Явные дефекты эталонов
#
# Ниже — оборванное начало ответа и почти весь абзац вместо числа.
# Примеры отобраны с помощью ассистента; по ним не оценивается доля ошибок.

# %%
cases = pd.DataFrame(json.loads((ROOT / "config/illustrative_cases.json").read_text())["cases"])
case_data = cases.merge(
    qa[["row_key", "question", "answer_text"]],
    on="row_key",
    validate="one_to_one",
)
display(
    case_data.assign(answer_excerpt=case_data.answer_text.str.slice(0, 100))[
        ["row_key", "question", "answer_excerpt", "issue"]
    ]
)
for row in case_data.itertuples(index=False):
    display(Markdown(f"**{row.row_key}: {row.issue}.** {row.interpretation}"))

# %% [markdown]
# Исходные эталоны сохранены без перезаписи. Возможные исправления требуют
# отдельной проверки и учёта изменений в разметке.

# %% [markdown]
# ## 6. Длины и выбросы
#
# Длины слов считаются регулярным выражением. Для длин контекстов берём один
# текст на группу дублей. Логарифмическая шкала помогает увидеть длинный хвост.

# %%
show_plot(plots.lengths(qa, corpus))
display(table("length_outliers"))
display(table("extreme_a_words")[["row_key", "question", "a_words", "span_status"]])
display(
    Markdown(
        f"Медианный вопрос — **{labelled.q_words.median():.0f} слов**, "
        f"ответ — **{labelled.a_words.median():.0f}**. "
        "Для контекстов и ответов виден длинный хвост."
    )
)

# %% [markdown]
# Пороги IQR рассчитаны на исходном train. Выбросы помечены, но сохранены:
# длинный ответ может быть правильным. Случаи, где ответ занимает почти весь
# абзац, стоит просмотреть перед обучением, а не обрезать автоматически.

# %% [markdown]
# ## 7. Вопросы и ответы
#
# Тип вопроса определяется по вопросительному слову среди первых трёх токенов;
# вопросы с частицей «ли» выделены отдельно. В частотных списках показано число
# объектов со словом. Лемматизации нет, поэтому формы одного слова считаются отдельно.

# %%
display(table("question_types"))
show_plot(plots.question_types(qa))
show_plot(plots.frequent_terms())
display(table("answer_forms"))
display(table("text_review_flags"))
show_plot(plots.answer_forms_and_question_counts())

# %% [markdown]
# Короткий запрос или отсутствие «?» сами по себе не означают ошибку.
# Больше всего вопросов «какой» и «что». Ответы на «почему» в медиане длиннее,
# чем на «кто»; при дальнейшем разборе ошибок эти группы стоит сравнивать отдельно.
# Число вопросов на один абзац различается: при будущей оценке это нужно учитывать,
# иначе тексты с большим числом вопросов получат больший вес.

# %%
display(table("source_distribution_shift"))
display(table("repeated_question_summary").head(10))

# %% [markdown]
# KS-distance показывает различие распределений длин между выборками.
# Повторяющиеся вопросы с разными ответами оставлены для отдельного просмотра:
# правильный ответ может зависеть от контекста.

# %% [markdown]
# ## 8. Корреляции и парные распределения
#
# Используем Spearman для длин и долей. Числовые ID и константный `score`
# исключены. Вместо корреляции с текстовым ответом рассматриваем его длину и позицию.
# Pairplot построен по 1 500 строкам с seed=42; для длин взят `log1p`.

# %%
show_plot(plots.correlations(qa))
show_plot(plots.pairplot(qa))
corr = pd.read_csv(TABLES / "spearman_correlations.csv", index_col=0)
display(
    Markdown(
        f"Связь длины контекста с длиной ответа слабая: "
        f"Spearman = **{corr.loc['c_words', 'a_words']:.3f}**. "
        "Длинный абзац не обязательно требует длинного ответа."
    )
)

# %% [markdown]
# ## 9. Дубли и пересечения выборок
#
# Сравним ID, тексты и вопросы между исходными train, validation и test.

# %%
display(table("split_overlap"))
show_plot(plots.source_overlap(qa))
display(table("repeated_original_ids").head(12))
display(pd.Series(summary["join_audit"], name="Сопоставление источников"))

# %% [markdown]
# Все контексты исходной validation уже встречаются в train. На таком разбиении
# можно проверять новые вопросы к знакомой базе, но не перенос на новые документы.
#
# QA ID и retrieval ID различаются, поэтому версии соединены по вопросу и контексту
# внутри выборки. Для повторяющихся пар сохранён флаг `mapping_unique=False`.
# Для одинаковых текстов введён `canonical_id`: он позволит не штрафовать будущий
# поиск за возвращение копии правильного документа под другим ID.

# %% [markdown]
# ## 10. Разбиение для проекта
#
# Почти одинаковые тексты найдены по символьному TF-IDF: 4–5-граммы, до 40 тыс.
# признаков, ближайший сосед каждого текста. Пары с cosine ≥0.90 сохранены для
# просмотра, с cosine ≥0.97 объединены в одну группу разбиения. Это проверка
# дублей; поисковый бейзлайн будет строиться отдельно по словам.

# %%
display(table("near_duplicate_candidates").sort_values("char_cosine", ascending=False).head(10))
display(table("project_split_summary"))
display(table("project_split_overlap"))

# %% [markdown]
# Размеченные строки разделены примерно в пропорции 80/10/10, seed=42.
# Одинаковые нормализованные тексты, повторные вопросы и найденные почти дубли
# остаются в одной выборке. Исходный test без ответов хранится отдельно.
# Таблица соответствия — `data/processed/split_assignments.parquet`.
#
# Разделение сделано по группам фрагментов, а не по статьям: ID исходных статей
# неизвестны. Проектный test оставляем для итоговой оценки.

# %% [markdown]
# ## 11. Что происходит с ответом при обрезке текста
#
# Для этой проверки взят токенизатор `intfloat/multilingual-e5-small`:
# E5 — компактная мультиязычная модель поиска, которую можно сравнить с
# лексическим поиском на следующих этапах. Сейчас измеряем только длины
# текстов в её токенах. Лимит модели — 512 токенов, включая префикс
# `query:` / `passage:` и специальные токены ([карточка модели](https://huggingface.co/intfloat/multilingual-e5-small)).
# Это пример ограничения для будущего нейросетевого этапа; веса E5 не загружаются.
# Первый бейзлайн TF-IDF использует слова и не зависит от этого токенизатора.
#
# Сравниваем сохранность целого ответа при обрезке начала текста и при нарезке
# на окна. В первом случае бюджет включает префикс и спецтокены, во втором —
# только содержимое окна. Сохранность ответа ещё не означает, что поиск найдёт окно.

# %%
display(
    corpus.drop_duplicates("canonical_id").c_e5_tokens.describe(percentiles=[0.5, 0.9, 0.95, 0.99])
)
display(table("chunking_coverage"))
show_plot(plots.token_budgets(corpus))

# %% [markdown]
# Для первого бейзлайна оставим исходные абзацы целиком. Нарезку длинных текстов
# и перекрытие окон проверим позже, когда появится ограничение нейросетевой модели.

# %% [markdown]
# ## 12. Общие слова и положение ответа
#
# Покрытие вопроса — доля его уникальных содержательных слов, встречающихся
# в контексте. Jaccard с предложением ответа рассчитан по эталонной позиции
# с помощью Razdel. Такой признак доступен только для анализа размеченных данных.

# %%
show_plot(plots.lexical_and_position(qa))
display(
    labelled[
        [
            "q_context_coverage",
            "q_context_jaccard",
            "q_answer_sentence_jaccard",
            "a_position_relative",
        ]
    ]
    .describe()
    .T
)
display(
    Markdown(
        f"Медианное покрытие вопроса словами контекста — "
        f"**{labelled.q_context_coverage.median():.0%}**. "
        "Это повод начать с TF-IDF и проверить качество лексического поиска "
        "на следующем чекпойнте."
    )
)

# %% [markdown]
# ## 13. Дальнейшие шаги
#
# - В чекпойнте 3 построить TF-IDF по словам с косинусной близостью. Индексировать
#   уникальные тексты корпуса; вопросы и эталоны использовать для оценки.
# - Подбирать настройки на проектной validation. Test использовать в конце.
# - Затем сравнить TF-IDF с BM25 и нейросетевым поиском на том же корпусе и разбиении.
# - При оценке ответов отдельно проверять смысл: единственный короткий эталон
#   может не учитывать другие правильные формулировки.
#
# SberQuAD подходит для учебного поиска и коротких ответов по Wikipedia.
# Для применения к рабочим документам понадобятся свои тексты и вопросы,
# включая вопросы, ответа на которые в базе нет.
#
# ## Материалы
#
# - [Выводы](../reports/eda_findings.md) и [словарь данных](../reports/data_dictionary.md).
# - [Efimov et al., SberQuAD: Description and Analysis](https://arxiv.org/abs/1912.09723).
#   Из статьи взяты идеи проверок пересечения слов и качества эталонов;
#   числа в ноутбуке посчитаны для закреплённой версии данных.
# - [RusBEIR](https://github.com/kaengreg/rusBEIR) — источник retrieval-версии.

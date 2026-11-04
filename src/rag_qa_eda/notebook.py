"""Build, execute and export the reviewable EDA notebook in the current uv Python."""

import hashlib
import logging
import re
import sys

import jupytext
import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .paths import REPORTS, ROOT

LOGGER = logging.getLogger(__name__)


def execute_notebook() -> None:
    source = ROOT / "notebooks/01_sberquad_eda.py"
    destination = source.with_suffix(".ipynb")
    notebook = jupytext.read(source)
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"{index}:{cell.source}".encode()).hexdigest()[:12]
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3.12 (uv)",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        record_timing=False,
    )
    client.km = KernelManager(kernel_name="python3")
    client.km.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    LOGGER.info("Executing %s with %s", source.name, sys.executable)
    client.execute()
    alt_text = {
        "overview": "Число вопросов с опубликованными и неопубликованными эталонами; число строк и уникальных текстов корпуса.",
        "missingness": "Доли пустых вопросов, контекстов, ответов и отрицательных позиций в исходных разделах.",
        "lengths": "Распределения числа слов в вопросах, уникальных контекстах и эталонах на логарифмической шкале.",
        "spans": "Распределение статусов проверки символьных позиций эталонных ответов.",
        "question_types": "Частоты типов вопросов и медианные длины соответствующих эталонов.",
        "source_overlap": "Направленные доли совпадающих контекстов между train, validation и test.",
        "correlations": "Матрица Spearman-корреляций числовых характеристик вопросов, текстов и эталонов.",
        "pairplot": "Парные распределения логарифмов длин и лексического покрытия на 1500 размеченных строках.",
        "token_budgets": "Длины входов tokenizer E5 и сохранность эталонного span при обрезке и нарезке на окна.",
        "lexical_and_position": "Распределения лексического покрытия, Jaccard и относительного положения ответа.",
        "retrieval_diagnostic": "Метрики фиксированного поиска с учётом эквивалентных ID и по группам лексического покрытия.",
        "frequent_terms": "Document frequency частых слов после ограниченной стоп-листовой фильтрации.",
        "answer_forms_and_question_counts": "Частоты поверхностных форм ответа и число вопросов на уникальный контекст.",
    }
    for cell in notebook.cells:
        names = re.findall(r"show_plot\(plots\.(\w+)\(", cell.source)
        image_outputs = [
            output for output in cell.get("outputs", []) if "image/png" in output.get("data", {})
        ]
        for name, output in zip(names, image_outputs, strict=True):
            output.setdefault("metadata", {}).setdefault("image/png", {})["alt"] = alt_text[name]
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    # The upstream lab template ignores PNG alt metadata; retain it explicitly.
    html_template = """{% extends 'lab/index.html.j2' %}
{% block data_png scoped %}
<div class="jp-RenderedImage jp-OutputArea-output">
<img src="data:image/png;base64,{{ output.data['image/png'] | escape_html }}"
     alt="{{ output | get_metadata('alt', 'image/png') | escape_html }}">
</div>
{% endblock data_png %}
"""
    exporter = HTMLExporter(template_name="lab", raw_template=html_template)
    exporter.embed_images = True
    html, _ = exporter.from_notebook_node(notebook)
    (REPORTS / "01_sberquad_eda.html").write_text(html, encoding="utf-8")
    LOGGER.info("Saved executed notebook and self-contained HTML")

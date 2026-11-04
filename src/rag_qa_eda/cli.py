"""Command-line entry points for a reproducible analysis."""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="SberQuAD EDA pipeline")
    parser.add_argument("command", choices=["download", "analyze", "notebook", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command in {"download", "all"}:
        from .download import download_all

        download_all()
    if args.command in {"analyze", "all"}:
        from .analysis import run_analysis

        run_analysis()
    if args.command in {"notebook", "all"}:
        from .notebook import execute_notebook

        execute_notebook()

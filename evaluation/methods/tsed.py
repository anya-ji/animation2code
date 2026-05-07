"""
METHOD: TSED (Tree Similarity of Edit Distance)

Measures structural code similarity between two HTML files by computing the
normalised tree-edit distance on their ASTs (via tree-sitter + APTED).

Score is in [0, 1]:  1 = identical structure, 0 = completely dissimilar.

No DTW — this method operates on full-document source code, not video frames.

TSED formula (from https://github.com/Etamin/TSED):
    score = (max_sexp_len - edit_distance) / max_sexp_len

Weights used: deletion=1.0, insertion=0.8, rename=1.0

Usage (standalone):
    uv run python evaluation/methods/tsed.py render.html gold.html
"""

import argparse
import json
import sys
from pathlib import Path

from apted import APTED, PerEditOperationConfig
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser, SupportedLanguage


# ─── AST utilities (from TSED reference implementation) ──────────────────────

class Node:
    def __init__(self, name, path):
        self.id = path
        self.name = name
        self.children = []
        self.len_sexp = 0

    def add_child(self, node):
        self.children.append(node)


def _parse_tree_string(tree_string: str) -> Node:
    stack = []
    current_node = None
    tokens = tree_string.replace("(", " ( ").replace(")", " ) ").split()
    for token in tokens:
        if token == "(":
            if current_node is not None:
                stack.append(current_node)
            current_node = Node("", "")
        elif token == ")":
            if stack:
                parent_node = stack.pop()
                parent_node.add_child(current_node)
                current_node = parent_node
        else:
            name = token.strip("()")
            if current_node is None:
                current_node = Node("", "")
            current_node.name = name if name else " "
    return current_node


def _get_tree(language: SupportedLanguage, source: str) -> Node:
    parser = get_parser(language)
    tree: Tree = parser.parse(bytes(source, "utf-8"))
    sexp = str(tree.root_node)
    node = _parse_tree_string(sexp)
    node.len_sexp = sexp.count(")")
    return node


def _tsed(language: SupportedLanguage, origin: str, target: str) -> float:
    """Compute TSED score in [0, 1] between two source strings."""
    tree1 = _get_tree(language, origin)
    tree2 = _get_tree(language, target)
    max_len = max(tree1.len_sexp, tree2.len_sexp)
    if max_len == 0:
        return 1.0
    apted = APTED(tree1, tree2, PerEditOperationConfig(1.0, 0.8, 1.0))
    dist = apted.compute_edit_distance()
    return 0.0 if dist > max_len else (max_len - dist) / max_len


# ─── Public API ───────────────────────────────────────────────────────────────


def run(render_path: str, gold_path: str) -> dict:
    """Compute TSED code similarity between a rendered HTML file and a gold HTML file.

    Args:
        render_path: path to the generated HTML file.
        gold_path:   path to the gold/reference HTML file.

    Returns:
        score  – TSED similarity in [0, 1]
        n_render_nodes – AST node count for the render file (sexp closing parens)
        n_gold_nodes   – AST node count for the gold file
    """
    render_html = Path(render_path).read_text(encoding="utf-8", errors="replace")
    gold_html = Path(gold_path).read_text(encoding="utf-8", errors="replace")

    tree_r = _get_tree("html", render_html)
    tree_g = _get_tree("html", gold_html)

    max_len = max(tree_r.len_sexp, tree_g.len_sexp)
    if max_len == 0:
        score = 1.0
    else:
        apted = APTED(tree_r, tree_g, PerEditOperationConfig(1.0, 0.8, 1.0))
        dist = apted.compute_edit_distance()
        score = 0.0 if dist > max_len else (max_len - dist) / max_len

    return {
        "score": score,
        "n_render_nodes": tree_r.len_sexp,
        "n_gold_nodes": tree_g.len_sexp,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="TSED code similarity between two HTML files"
    )
    parser.add_argument("render_html", help="Path to the generated HTML file")
    parser.add_argument("gold_html", help="Path to the gold/reference HTML file")
    args = parser.parse_args()

    result = run(args.render_html, args.gold_html)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

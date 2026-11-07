"""TSED computed separately on the HTML, CSS and JavaScript of each file.

Plain TSED treats <style> and <script> contents as single leaf nodes, so 500 lines of CSS look the
same as none. This parses each part with its own grammar, then averages the three scores:

    weight_i = nodes_i / (html_nodes + css_nodes + js_nodes)
    score = sum(weight_i * tsed_i)

Scores range from 0 to 1. APTED gets very slow on large trees (about 55 s at 4000 nodes), so a part
over its node cap falls back to token Jaccard similarity, which the result reports.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from apted import APTED, PerEditOperationConfig
from tree_sitter import Tree
from tree_sitter_language_pack import get_parser, SupportedLanguage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methods.tsed import _parse_tree_string, Node

MAX_NODES_CSS = 800   # APTED cap for CSS subtrees
MAX_NODES_JS  = 800   # APTED cap for JS subtrees
MAX_NODES_HTML = 400  # APTED cap for stripped HTML skeleton

_APTED_CFG = PerEditOperationConfig(1.0, 0.8, 1.0)


def _extract_css(html: str) -> str:
    return "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE))


def _extract_js(html: str) -> str:
    return "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE))


def _parse_lang(source: str, lang: SupportedLanguage) -> tuple[Node, int]:
    """Parse source string with given language, return (Node, node_count)."""
    parser = get_parser(lang)
    tree: Tree = parser.parse(bytes(source, "utf-8"))
    sexp = str(tree.root_node)
    node = _parse_tree_string(sexp)
    n = sexp.count(")")
    node.len_sexp = n
    return node, n


def _strip_raw_text(node: Node) -> Node | None:
    """Recursively remove raw_text nodes (CSS/JS opaque content)."""
    if node.name == "raw_text":
        return None
    new_node = Node(node.name, node.id)
    for child in node.children:
        stripped = _strip_raw_text(child)
        if stripped is not None:
            new_node.add_child(stripped)
    return new_node


def _sexp_len(node: Node | None) -> int:
    if node is None:
        return 0
    return 1 + sum(_sexp_len(c) for c in node.children)


def _parse_html_skeleton(html: str) -> tuple[Node | None, int]:
    """HTML tree with all raw_text content nodes removed."""
    parser = get_parser("html")
    tree: Tree = parser.parse(bytes(html, "utf-8"))
    sexp = str(tree.root_node)
    raw_root = _parse_tree_string(sexp)
    stripped = _strip_raw_text(raw_root)
    n = _sexp_len(stripped)
    if stripped:
        stripped.len_sexp = n
    return stripped, n


_TOKEN_RE = re.compile(r"[a-zA-Z_][\w-]*|[0-9]+(?:\.[0-9]+)?(?:%|px|em|rem|vh|vw|s|ms)?")


def _token_jaccard(a: str, b: str) -> float:
    toks_a = set(_TOKEN_RE.findall(a.lower()))
    toks_b = set(_TOKEN_RE.findall(b.lower()))
    if not toks_a and not toks_b:
        return 1.0
    if not toks_a or not toks_b:
        return 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def _layer_score(
    node_r: Node | None,
    len_r: int,
    node_g: Node | None,
    len_g: int,
    fallback_r: str,
    fallback_g: str,
    max_nodes: int,
) -> tuple[float, bool]:
    """Compute TSED for one part of the file. Returns (score, used_fallback)."""
    if len_r == 0 and len_g == 0:
        return 1.0, False
    if len_r == 0 or len_g == 0:
        return 0.0, False

    if node_r is None or node_g is None or max(len_r, len_g) > max_nodes:
        return _token_jaccard(fallback_r, fallback_g), True

    max_len = max(len_r, len_g)
    dist = APTED(node_r, node_g, _APTED_CFG).compute_edit_distance()
    score = max(0.0, (max_len - dist) / max_len)
    return score, False


def run(render_path: str, gold_path: str) -> dict:
    """Compute tripartite TSED between a generated HTML file and a gold HTML file.

    Args:
        render_path: Path to the generated HTML file.
        gold_path: Path to the gold HTML file.

    Returns a dict with:
        score: Weighted TSED in [0, 1].
        html_score, css_score, js_score: Score of each part.
        weight_html, weight_css, weight_js: Weight of each part.
        n_html_r, n_css_r, n_js_r: Render AST node counts.
        n_html_g, n_css_g, n_js_g: Gold AST node counts.
        html_fallback, css_fallback, js_fallback: True if token Jaccard was used.
    """
    render_html = Path(render_path).read_text(encoding="utf-8", errors="replace")
    gold_html   = Path(gold_path).read_text(encoding="utf-8", errors="replace")

    return _score(render_html, gold_html)


def _score(render_html: str, gold_html: str) -> dict:
    html_r, len_html_r = _parse_html_skeleton(render_html)
    html_g, len_html_g = _parse_html_skeleton(gold_html)

    css_src_r = _extract_css(render_html)
    css_src_g = _extract_css(gold_html)
    css_r, len_css_r = _parse_lang(css_src_r, "css") if css_src_r.strip() else (None, 0)
    css_g, len_css_g = _parse_lang(css_src_g, "css") if css_src_g.strip() else (None, 0)

    js_src_r = _extract_js(render_html)
    js_src_g = _extract_js(gold_html)
    js_r, len_js_r = _parse_lang(js_src_r, "javascript") if js_src_r.strip() else (None, 0)
    js_g, len_js_g = _parse_lang(js_src_g, "javascript") if js_src_g.strip() else (None, 0)

    html_score, html_fb = _layer_score(
        html_r, len_html_r, html_g, len_html_g,
        render_html, gold_html, MAX_NODES_HTML,
    )
    css_score, css_fb = _layer_score(
        css_r, len_css_r, css_g, len_css_g,
        css_src_r, css_src_g, MAX_NODES_CSS,
    )
    js_score, js_fb = _layer_score(
        js_r, len_js_r, js_g, len_js_g,
        js_src_r, js_src_g, MAX_NODES_JS,
    )

    w_html = max(len_html_r, len_html_g)
    w_css  = max(len_css_r,  len_css_g)
    w_js   = max(len_js_r,   len_js_g)
    total_w = w_html + w_css + w_js
    if total_w == 0:
        total_w = 1  # avoid div-by-zero for empty files

    weight_html = w_html / total_w
    weight_css  = w_css  / total_w
    weight_js   = w_js   / total_w

    combined = weight_html * html_score + weight_css * css_score + weight_js * js_score

    return {
        "score":          combined,
        "html_score":     html_score,
        "css_score":      css_score,
        "js_score":       js_score,
        "weight_html":    round(weight_html, 4),
        "weight_css":     round(weight_css,  4),
        "weight_js":      round(weight_js,   4),
        "n_html_r":       len_html_r,
        "n_html_g":       len_html_g,
        "n_css_r":        len_css_r,
        "n_css_g":        len_css_g,
        "n_js_r":         len_js_r,
        "n_js_g":         len_js_g,
        "html_fallback":  html_fb,
        "css_fallback":   css_fb,
        "js_fallback":    js_fb,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tripartite TSED: separate HTML/CSS/JS AST comparison"
    )
    parser.add_argument("render_html", help="Path to generated HTML")
    parser.add_argument("gold_html",   help="Path to gold HTML")
    args = parser.parse_args()

    result = run(args.render_html, args.gold_html)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

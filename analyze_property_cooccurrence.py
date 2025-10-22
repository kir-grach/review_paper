#!/usr/bin/env python3
import argparse
import itertools
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# Global containers for compiled patterns
PROPERTY_TERM_PATTERNS: Dict[str, List[Tuple[str, List[Pattern]]]] = {}
PROPERTY_PATTERNS: Dict[str, List[Pattern]] = {}

SUBSTRINGS_FOR_TEXT = ["abstract", "summary", "text", "описание", "аннотация"]
DOC_MIN_SUPPORT = 3
SENT_MIN_SUPPORT = 5
TERM_DOC_MIN_SUPPORT = 5
TOP_N = 15
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+")

UNICODE_DASH_RANGE = "\u2010\u2011\u2012\u2013\u2014\u2015"
WORD_SEPARATOR_PATTERN = re.compile(rf"[-\\s{UNICODE_DASH_RANGE}]+")
WORD_SEPARATOR_CLASS = rf"[-\\s{UNICODE_DASH_RANGE}]+"


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file while attempting to auto-detect the delimiter."""

    try:
        return pd.read_csv(path, sep=None, engine="python", **kwargs)
    except Exception as primary_error:
        try:
            return pd.read_csv(path, **kwargs)
        except Exception:
            raise primary_error


def resolve_input_path(
    filename: str,
    provided_path: str,
    search_locations: Iterable[Path],
) -> Path:
    """Resolve the path to an input file with optional overrides and fallbacks."""

    attempted: List[Path] = []

    if provided_path:
        candidate = Path(provided_path).expanduser()
        if candidate.is_dir():
            candidate = candidate / filename
        attempted.append(candidate)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"Provided path for {filename!r} not found. Tried: {candidate}"
        )

    for location in search_locations:
        if location is None:
            continue
        base = Path(location).expanduser()
        if base.is_dir():
            candidate = base / filename
        else:
            candidate = base
        if candidate in attempted:
            continue
        attempted.append(candidate)
        if candidate.is_file():
            return candidate

    attempted_str = ", ".join(str(path) for path in attempted if path)
    raise FileNotFoundError(
        f"Could not locate required file {filename!r}. Tried: {attempted_str}"
    )


def determine_output_base(output_dir: str, script_dir: Path) -> Path:
    """Determine where result files should be written, preferring /mnt/data when available."""

    candidates: List[Path] = []
    if output_dir:
        candidates.append(Path(output_dir).expanduser())
    else:
        candidates.extend([
            script_dir,
            script_dir / "data",
            Path("/mnt/data"),
        ])

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            last_error = error
            continue
        else:
            return candidate

    if last_error is not None:
        raise last_error
    raise OSError("Unable to establish output directory.")


def detect_text_column(df: pd.DataFrame) -> str:
    """Detect the most appropriate text column in a dataframe."""
    lower_columns = {col: col.lower() for col in df.columns}
    for col, lower in lower_columns.items():
        if any(substr in lower for substr in SUBSTRINGS_FOR_TEXT):
            return col

    best_col = None
    best_avg_len = -1.0
    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        cleaned = series.dropna().astype(str).str.strip()
        if cleaned.empty:
            continue
        lengths = cleaned.str.len()
        avg_len = lengths.mean()
        if avg_len > best_avg_len:
            best_avg_len = avg_len
            best_col = col

    if best_col is None:
        raise ValueError("Unable to detect text column in dataframe.")
    return best_col


def _create_phrase_pattern(phrase: str) -> Pattern:
    normalized = phrase.strip()
    tokens_raw = [token for token in WORD_SEPARATOR_PATTERN.split(normalized) if token.strip()]
    pattern_tokens: List[str] = []
    for token in tokens_raw:
        token = token.strip()
        if not token:
            continue
        pattern_tokens.append(re.escape(token))
    if not pattern_tokens:
        raise ValueError(f"Cannot build pattern from phrase: {phrase}")
    base_pattern = WORD_SEPARATOR_CLASS.join(pattern_tokens)
    last_token = tokens_raw[-1]
    plural_suffix = ""
    if re.search(r"[A-Za-z]$", last_token):
        plural_suffix = r"(?:s|es)?"
    pattern = rf"\b{base_pattern}{plural_suffix}\b"
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def _create_abbreviation_pattern(abbreviation: str) -> Pattern:
    abbrev = abbreviation.strip()
    if not abbrev:
        raise ValueError("Abbreviation is empty")
    pattern = rf"\b{re.escape(abbrev)}\b"
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def term_to_patterns(term: str) -> List[Pattern]:
    term = term.strip()
    if not term:
        return []
    patterns: List[Pattern] = []
    paren_match = re.match(r"^(.*?)\\s*\\(([^()]+)\\)\\s*$", term)
    if paren_match:
        phrase = paren_match.group(1).strip()
        abbreviation = paren_match.group(2).strip()
        if phrase:
            patterns.append(_create_phrase_pattern(phrase))
        if abbreviation:
            patterns.append(_create_abbreviation_pattern(abbreviation))
        return patterns

    patterns.append(_create_phrase_pattern(term))
    return patterns


def property_presence_in_text(text: str) -> Set[str]:
    if not isinstance(text, str):
        return set()
    present: Set[str] = set()
    for prop, patterns in PROPERTY_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(text):
                present.add(prop)
                break
    return present


def property_presence_in_sentences(text: str) -> List[Set[str]]:
    if not isinstance(text, str):
        return []
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_REGEX.split(text) if segment.strip()]
    return [property_presence_in_text(sentence) for sentence in sentences]


def detect_terms_in_text(text: str) -> Dict[str, Set[str]]:
    if not isinstance(text, str):
        return {}
    found: Dict[str, Set[str]] = {}
    for prop, term_patterns in PROPERTY_TERM_PATTERNS.items():
        for term_name, patterns in term_patterns:
            for pattern in patterns:
                if pattern.search(text):
                    found.setdefault(prop, set()).add(term_name)
                    break
            if prop in found and term_name in found[prop]:
                continue
    return found


def compute_pair_metrics(count_A: int, count_B: int, count_AB: int, N: int) -> Dict[str, float]:
    metrics = {
        "count_A": int(count_A),
        "count_B": int(count_B),
        "count_AB": int(count_AB),
        "jaccard": math.nan,
        "dice": math.nan,
        "pmi": math.nan,
        "npmi": math.nan,
        "llr": math.nan,
    }

    union = count_A + count_B - count_AB
    if union > 0:
        metrics["jaccard"] = count_AB / union

    denom_dice = count_A + count_B
    if denom_dice > 0:
        metrics["dice"] = 2 * count_AB / denom_dice

    if N > 0 and count_A > 0 and count_B > 0 and count_AB > 0:
        p_a = count_A / N
        p_b = count_B / N
        p_ab = count_AB / N
        denom = p_a * p_b
        if denom > 0 and p_ab > 0:
            pmi = math.log2(p_ab / denom)
            metrics["pmi"] = pmi
            normalizer = -math.log2(p_ab)
            if normalizer > 0:
                metrics["npmi"] = pmi / normalizer

    k11 = count_AB
    k12 = count_A - count_AB
    k21 = count_B - count_AB
    k22 = N - k11 - k12 - k21

    if min(k11, k12, k21, k22) >= 0 and N > 0:
        metrics["llr"] = _llr_2x2(k11, k12, k21, k22)

    return metrics


def _xlogx(x: int) -> float:
    return x * math.log(x) if x > 0 else 0.0


def _llr_2x2(k11: int, k12: int, k21: int, k22: int) -> float:
    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    total = row1 + row2
    if total == 0:
        return math.nan
    ll = (
        _xlogx(k11)
        + _xlogx(k12)
        + _xlogx(k21)
        + _xlogx(k22)
        + _xlogx(total)
        - _xlogx(row1)
        - _xlogx(row2)
        - _xlogx(col1)
        - _xlogx(col2)
    )
    return 2.0 * ll


def build_network_and_export(matrix: pd.DataFrame, path_gexf: Path, weight_name: str) -> None:
    G = nx.Graph()
    nodes = list(matrix.index)
    G.add_nodes_from(nodes)
    llr_matrix = matrix.attrs.get("llr_matrix")
    for i, prop_a in enumerate(nodes):
        for j, prop_b in enumerate(nodes):
            if j <= i:
                continue
            weight = matrix.loc[prop_a, prop_b]
            if pd.isna(weight):
                continue
            edge_data = {
                "weight": float(weight),
                weight_name: float(weight),
            }
            if llr_matrix is not None:
                llr_value = llr_matrix.loc[prop_a, prop_b]
                if not pd.isna(llr_value):
                    edge_data["llr"] = float(llr_value)
            G.add_edge(prop_a, prop_b, **edge_data)
    nx.write_gexf(G, path_gexf)


def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [segment.strip() for segment in SENTENCE_SPLIT_REGEX.split(text) if segment.strip()]


def prepare_texts(series: pd.Series) -> List[str]:
    texts: List[str] = []
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            texts.append(text)
    return texts


def analyze_corpus(texts: Iterable[str]) -> Dict[str, Dict]:
    doc_property_counts: Counter = Counter()
    doc_pair_counts: Counter = Counter()
    doc_term_counts: Counter = Counter()
    doc_term_pair_counts: Counter = Counter()
    sent_property_counts: Counter = Counter()
    sent_pair_counts: Counter = Counter()
    N_docs = 0
    N_sents = 0

    for text in texts:
        if not isinstance(text, str) or not text.strip():
            continue
        N_docs += 1
        term_presence = detect_terms_in_text(text)
        properties_in_doc = set(term_presence.keys())
        for prop in properties_in_doc:
            doc_property_counts[prop] += 1
        for prop_a, prop_b in itertools.combinations(sorted(properties_in_doc), 2):
            doc_pair_counts[(prop_a, prop_b)] += 1

        terms_in_doc: List[Tuple[str, str]] = []
        for prop, term_set in term_presence.items():
            for term_name in term_set:
                key = (prop, term_name)
                doc_term_counts[key] += 1
                terms_in_doc.append(key)
        for term_a, term_b in itertools.combinations(sorted(terms_in_doc), 2):
            if term_a[0] == term_b[0]:
                continue
            ordered_pair = tuple(sorted((term_a, term_b)))
            doc_term_pair_counts[ordered_pair] += 1

        sentences = split_sentences(text)
        for sentence in sentences:
            N_sents += 1
            sentence_props = property_presence_in_text(sentence)
            if not sentence_props:
                continue
            for prop in sentence_props:
                sent_property_counts[prop] += 1
            for prop_a, prop_b in itertools.combinations(sorted(sentence_props), 2):
                sent_pair_counts[(prop_a, prop_b)] += 1

    return {
        "doc": {
            "property_counts": doc_property_counts,
            "pair_counts": doc_pair_counts,
            "term_counts": doc_term_counts,
            "term_pair_counts": doc_term_pair_counts,
            "N": N_docs,
        },
        "sent": {
            "property_counts": sent_property_counts,
            "pair_counts": sent_pair_counts,
            "N": N_sents,
        },
    }


def generate_pair_metrics(property_counts: Counter, pair_counts: Counter, N: int, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (prop_a, prop_b), count_ab in pair_counts.items():
        if count_ab < min_support:
            continue
        count_a = property_counts.get(prop_a, 0)
        count_b = property_counts.get(prop_b, 0)
        metrics = compute_pair_metrics(count_a, count_b, count_ab, N)
        row = {"property_A": prop_a, "property_B": prop_b}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_term_pair_metrics(term_counts: Counter, term_pair_counts: Counter, N: int, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (term_a, term_b), count_ab in term_pair_counts.items():
        if count_ab < min_support:
            continue
        count_a = term_counts.get(term_a, 0)
        count_b = term_counts.get(term_b, 0)
        metrics = compute_pair_metrics(count_a, count_b, count_ab, N)
        row = {
            "property_A": term_a[0],
            "term_A": term_a[1],
            "property_B": term_b[0],
            "term_B": term_b[1],
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def ensure_output_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_heatmap(matrix: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    data = matrix.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    ensure_output_directory(output_path)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze co-occurrence of soil properties and terms across corpora."
    )
    parser.add_argument("--soil-terms", type=str, help="Path to soil terms CSV file.")
    parser.add_argument("--geoeco", type=str, help="Path to GEOECO corpus CSV file.")
    parser.add_argument(
        "--geomorpho", type=str, help="Path to GEOMORPHO corpus CSV file."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory that contains all required CSV input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where analysis outputs will be written.",
    )
    parser.add_argument(
        "--doc-min-support",
        type=int,
        default=None,
        help=(
            "Minimum joint occurrences required for property pairs at the document level. "
            "Defaults to 3 when not provided."
        ),
    )
    parser.add_argument(
        "--sent-min-support",
        type=int,
        default=None,
        help=(
            "Minimum joint occurrences required for property pairs at the sentence level. "
            "Defaults to 5 when not provided."
        ),
    )
    parser.add_argument(
        "--term-doc-min-support",
        type=int,
        default=None,
        help=(
            "Minimum joint occurrences required for term pairs (document level). "
            "Defaults to 5 when not provided."
        ),
    )
    parser.add_argument(
        "--focus-properties",
        type=str,
        help=(
            "Optional list of soil properties (separated by commas, semicolons, or pipes) "
            "to highlight in a dedicated output file."
        ),
    )
    parser.add_argument(
        "--focus-top-n",
        type=int,
        default=5,
        help=(
            "When --focus-properties is not supplied, automatically pick the top N properties "
            "by combined document coverage (default: 5)."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    search_locations: List[Path] = []
    if args.data_dir:
        search_locations.append(Path(args.data_dir))
    search_locations.extend([script_dir, script_dir / "data", Path.cwd(), Path("/mnt/data")])

    doc_min_support = args.doc_min_support if args.doc_min_support is not None else DOC_MIN_SUPPORT
    sent_min_support = (
        args.sent_min_support if args.sent_min_support is not None else SENT_MIN_SUPPORT
    )
    term_doc_min_support = (
        args.term_doc_min_support
        if args.term_doc_min_support is not None
        else TERM_DOC_MIN_SUPPORT
    )

    doc_min_support = max(1, doc_min_support)
    sent_min_support = max(1, sent_min_support)
    term_doc_min_support = max(1, term_doc_min_support)
    focus_top_n = max(1, args.focus_top_n)
    focus_properties_input = args.focus_properties

    manual_doc_support = args.doc_min_support is not None
    manual_sent_support = args.sent_min_support is not None
    manual_term_support = args.term_doc_min_support is not None

    soil_terms_path = resolve_input_path("soil_terms_25.csv", args.soil_terms, search_locations)
    geoeco_path = resolve_input_path("geoeco_results.csv", args.geoeco, search_locations)
    geomorpho_path = resolve_input_path(
        "geomorpho_results.csv", args.geomorpho, search_locations
    )

    print(f"Using soil terms file: {soil_terms_path}")
    print(f"Using GEOECO corpus file: {geoeco_path}")
    print(f"Using GEOMORPHO corpus file: {geomorpho_path}")

    output_base = determine_output_base(args.output_dir, script_dir)
    print(f"Outputs will be saved under: {output_base}")

    soil_df = read_csv_flexible(soil_terms_path)

    properties = soil_df["Soil Property"].tolist()
    property_terms_series = soil_df["Key Terms"].tolist()

    unique_properties = sorted(set(properties))
    property_to_terms: Dict[str, List[str]] = {prop: [] for prop in unique_properties}

    for prop, term_string in zip(properties, property_terms_series):
        if pd.isna(term_string):
            continue
        terms = [term.strip() for term in str(term_string).split(",")]
        for term in terms:
            if not term:
                continue
            property_to_terms[prop].append(term)

    global PROPERTY_TERM_PATTERNS, PROPERTY_PATTERNS
    PROPERTY_TERM_PATTERNS = {}
    PROPERTY_PATTERNS = {}

    for prop, terms in property_to_terms.items():
        compiled_terms: List[Tuple[str, List[Pattern]]] = []
        compiled_property_patterns: List[Pattern] = []
        for term in terms:
            patterns = term_to_patterns(term)
            if not patterns:
                continue
            compiled_terms.append((term, patterns))
            compiled_property_patterns.extend(patterns)
        PROPERTY_TERM_PATTERNS[prop] = compiled_terms
        PROPERTY_PATTERNS[prop] = compiled_property_patterns

    geoeco_df = read_csv_flexible(geoeco_path)
    geomorpho_df = read_csv_flexible(geomorpho_path)

    geoeco_text_col = detect_text_column(geoeco_df)
    geomorpho_text_col = detect_text_column(geomorpho_df)

    print(f"Detected text column for GEOECO: {geoeco_text_col}")
    print(f"Detected text column for GEOMORPHO: {geomorpho_text_col}")

    geoeco_texts = prepare_texts(geoeco_df[geoeco_text_col])
    geomorpho_texts = prepare_texts(geomorpho_df[geomorpho_text_col])
    combined_texts = geoeco_texts + geomorpho_texts

    corpora = {
        "GEOECO": analyze_corpus(geoeco_texts),
        "GEOMORPHO": analyze_corpus(geomorpho_texts),
        "COMBINED": analyze_corpus(combined_texts),
    }

    combined_doc_analysis = corpora["COMBINED"]["doc"]
    focus_properties: List[str] = []
    focus_candidates: List[str] = []
    if focus_properties_input:
        focus_candidates = [
            token.strip()
            for token in re.split(r"[;,|]", focus_properties_input)
            if token.strip()
        ]
        unknown = [prop for prop in focus_candidates if prop not in PROPERTY_PATTERNS]
        if unknown:
            print(
                "Warning: the following focus properties are not recognized and will be ignored: "
                + ", ".join(sorted(unknown))
            )
        focus_properties = [prop for prop in focus_candidates if prop in PROPERTY_PATTERNS]
    if not focus_properties:
        ranked_props = sorted(
            (
                (prop, combined_doc_analysis["property_counts"].get(prop, 0))
                for prop in PROPERTY_PATTERNS.keys()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        focus_properties = [prop for prop, _ in ranked_props[:focus_top_n]]
    if not focus_properties:
        focus_properties = sorted(PROPERTY_PATTERNS.keys())[:focus_top_n]
    focus_properties = list(dict.fromkeys(focus_properties))
    if len(focus_properties) > focus_top_n and not focus_properties_input:
        focus_properties = focus_properties[:focus_top_n]
    print(
        "Focus properties for dedicated output: "
        + (", ".join(focus_properties) if focus_properties else "<none>")
    )
    focus_property_set = set(focus_properties)

    doc_level_frames = []
    sent_level_frames = []
    term_level_frames = []

    for corpus_name, analysis in corpora.items():
        doc_analysis = analysis["doc"]
        sent_analysis = analysis["sent"]

        doc_pair_counts = doc_analysis["pair_counts"]
        doc_support = doc_min_support
        max_doc_pair = max(doc_pair_counts.values(), default=0)
        if not manual_doc_support and max_doc_pair > 0 and doc_support > max_doc_pair:
            print(
                f"Lowering document-level support for {corpus_name} from {doc_min_support} to {max_doc_pair} "
                "to include available property pairs."
            )
            doc_support = max_doc_pair

        doc_df = generate_pair_metrics(
            doc_analysis["property_counts"],
            doc_pair_counts,
            doc_analysis["N"],
            doc_support,
        )
        if not doc_df.empty:
            doc_df = doc_df.assign(corpus=corpus_name)
            doc_level_frames.append(doc_df)
        elif doc_pair_counts and manual_doc_support:
            print(
                f"No document-level property pairs for {corpus_name} satisfied the specified support threshold "
                f"({doc_support}). Consider using --doc-min-support with a smaller value."
            )

        term_pair_counts = doc_analysis["term_pair_counts"]
        term_support = term_doc_min_support
        max_term_pair = max(term_pair_counts.values(), default=0)
        if not manual_term_support and max_term_pair > 0 and term_support > max_term_pair:
            print(
                f"Lowering term-level document support for {corpus_name} from {term_doc_min_support} to {max_term_pair} "
                "to include available term pairs."
            )
            term_support = max_term_pair
        term_df = generate_term_pair_metrics(
            doc_analysis["term_counts"],
            term_pair_counts,
            doc_analysis["N"],
            term_support,
        )
        if not term_df.empty:
            term_df = term_df.assign(corpus=corpus_name)
            term_level_frames.append(term_df)
        elif term_pair_counts and manual_term_support:
            print(
                f"No term-level document pairs for {corpus_name} satisfied the specified support threshold "
                f"({term_support}). Consider reducing --term-doc-min-support."
            )

        sent_pair_counts = sent_analysis["pair_counts"]
        sent_support = sent_min_support
        max_sent_pair = max(sent_pair_counts.values(), default=0)
        if not manual_sent_support and max_sent_pair > 0 and sent_support > max_sent_pair:
            print(
                f"Lowering sentence-level support for {corpus_name} from {sent_min_support} to {max_sent_pair} "
                "to include available property pairs."
            )
            sent_support = max_sent_pair
        sent_df = generate_pair_metrics(
            sent_analysis["property_counts"],
            sent_pair_counts,
            sent_analysis["N"],
            sent_support,
        )
        if not sent_df.empty:
            sent_df = sent_df.assign(corpus=corpus_name)
            sent_level_frames.append(sent_df)
        elif sent_pair_counts and manual_sent_support:
            print(
                f"No sentence-level property pairs for {corpus_name} satisfied the specified support threshold "
                f"({sent_support}). Consider using --sent-min-support with a smaller value."
            )

    doc_output = pd.concat(doc_level_frames, ignore_index=True) if doc_level_frames else pd.DataFrame()
    sent_output = pd.concat(sent_level_frames, ignore_index=True) if sent_level_frames else pd.DataFrame()
    term_output = pd.concat(term_level_frames, ignore_index=True) if term_level_frames else pd.DataFrame()

    doc_output_path = output_base / "cooccurrence_properties_doclevel.csv"
    sent_output_path = output_base / "cooccurrence_properties_sentlevel.csv"
    term_output_path = output_base / "cooccurrence_terms_doclevel.csv"
    focus_output_path = output_base / "focus_properties_relationships.csv"

    doc_columns = [
        "property_A",
        "property_B",
        "corpus",
        "count_A",
        "count_B",
        "count_AB",
        "jaccard",
        "dice",
        "pmi",
        "npmi",
        "llr",
    ]
    if doc_output.empty:
        doc_output = pd.DataFrame(columns=doc_columns)
    else:
        doc_output = doc_output[doc_columns]
    ensure_output_directory(doc_output_path)
    doc_output.to_csv(doc_output_path, index=False)

    sent_columns = [
        "property_A",
        "property_B",
        "corpus",
        "count_A",
        "count_B",
        "count_AB",
        "jaccard",
        "dice",
        "pmi",
        "npmi",
        "llr",
    ]
    if sent_output.empty:
        sent_output = pd.DataFrame(columns=sent_columns)
    else:
        sent_output = sent_output[sent_columns]
    ensure_output_directory(sent_output_path)
    sent_output.to_csv(sent_output_path, index=False)

    term_columns = [
        "property_A",
        "term_A",
        "property_B",
        "term_B",
        "corpus",
        "count_A",
        "count_B",
        "count_AB",
        "jaccard",
        "dice",
        "pmi",
        "npmi",
        "llr",
    ]
    if term_output.empty:
        term_output = pd.DataFrame(columns=term_columns)
    else:
        term_output = term_output[term_columns]
    ensure_output_directory(term_output_path)
    term_output.to_csv(term_output_path, index=False)

    focus_rows: List[Dict[str, object]] = []
    combined_pair_counts = combined_doc_analysis["pair_counts"]
    combined_property_counts = combined_doc_analysis["property_counts"]
    combined_N = combined_doc_analysis["N"]
    if len(focus_properties) >= 2:
        for prop_a, prop_b in itertools.combinations(sorted(focus_property_set), 2):
            key = tuple(sorted((prop_a, prop_b)))
            count_a = combined_property_counts.get(prop_a, 0)
            count_b = combined_property_counts.get(prop_b, 0)
            count_ab = combined_pair_counts.get(key, 0)
            metrics = compute_pair_metrics(count_a, count_b, count_ab, combined_N)
            row = {"property_A": prop_a, "property_B": prop_b, "corpus": "COMBINED"}
            row.update(metrics)
            focus_rows.append(row)
    focus_columns = [
        "property_A",
        "property_B",
        "corpus",
        "count_A",
        "count_B",
        "count_AB",
        "jaccard",
        "dice",
        "pmi",
        "npmi",
        "llr",
    ]
    if len(focus_properties) < 2:
        print(
            "Not enough focus properties to compute pairwise relationships. "
            "The dedicated output will be empty."
        )
    focus_output = pd.DataFrame(focus_rows)
    if focus_output.empty:
        focus_output = pd.DataFrame(columns=focus_columns)
    else:
        focus_output = focus_output[focus_columns]
    ensure_output_directory(focus_output_path)
    focus_output.to_csv(focus_output_path, index=False)

    top_rows: List[Dict[str, object]] = []
    metrics_for_ranking = ["jaccard", "dice", "npmi", "llr"]

    for level_name, df in [("DOC", doc_output), ("SENT", sent_output)]:
        if df.empty:
            continue
        for corpus_name in sorted(df["corpus"].unique()):
            subset = df[df["corpus"] == corpus_name]
            for metric in metrics_for_ranking:
                sub = subset.dropna(subset=[metric])
                if sub.empty:
                    continue
                top = sub.sort_values(metric, ascending=False).head(TOP_N)
                for rank, (_, row) in enumerate(top.iterrows(), start=1):
                    top_rows.append(
                        {
                            "level": level_name,
                            "corpus": corpus_name,
                            "metric": metric,
                            "rank": rank,
                            "property_A": row["property_A"],
                            "property_B": row["property_B"],
                            "value": row[metric],
                            "count_AB": row["count_AB"],
                        }
                    )

    top_columns = [
        "level",
        "corpus",
        "metric",
        "rank",
        "property_A",
        "property_B",
        "value",
        "count_AB",
    ]
    top_output = pd.DataFrame(top_rows)
    if top_output.empty:
        top_output = pd.DataFrame(columns=top_columns)
    else:
        top_output = top_output[top_columns]
    top_output_path = output_base / "top_property_pairs.csv"
    ensure_output_directory(top_output_path)
    top_output.to_csv(top_output_path, index=False)

    combined_doc_df = doc_output[doc_output["corpus"] == "COMBINED"] if not doc_output.empty else pd.DataFrame()
    combined_sent_df = sent_output[sent_output["corpus"] == "COMBINED"] if not sent_output.empty else pd.DataFrame()

    heatmap_paths: List[Path] = []
    graph_paths: List[Path] = []

    if not combined_doc_df.empty:
        properties_sorted = sorted(set(combined_doc_df["property_A"]) | set(combined_doc_df["property_B"]))
        matrix_npmi = pd.DataFrame(np.nan, index=properties_sorted, columns=properties_sorted)
        matrix_llr = pd.DataFrame(np.nan, index=properties_sorted, columns=properties_sorted)
        for _, row in combined_doc_df.iterrows():
            a = row["property_A"]
            b = row["property_B"]
            npmi_val = row["npmi"]
            llr_val = row["llr"]
            matrix_npmi.loc[a, b] = npmi_val
            matrix_npmi.loc[b, a] = npmi_val
            matrix_llr.loc[a, b] = llr_val
            matrix_llr.loc[b, a] = llr_val
        np.fill_diagonal(matrix_npmi.values, 0.0)
        np.fill_diagonal(matrix_llr.values, 0.0)
        matrix_npmi.attrs["llr_matrix"] = matrix_llr

        doc_graph_path = output_base / "property_network_doclevel.gexf"
        ensure_output_directory(doc_graph_path)
        build_network_and_export(matrix_npmi, doc_graph_path, "npmi")
        graph_paths.append(doc_graph_path)

        heatmap_doc_path = output_base / "heatmap_properties_doclevel.png"
        save_heatmap(matrix_npmi.fillna(0.0), "Property NPMI (Doc Level)", heatmap_doc_path)
        heatmap_paths.append(heatmap_doc_path)

    if not combined_sent_df.empty:
        properties_sorted = sorted(set(combined_sent_df["property_A"]) | set(combined_sent_df["property_B"]))
        matrix_npmi = pd.DataFrame(np.nan, index=properties_sorted, columns=properties_sorted)
        matrix_llr = pd.DataFrame(np.nan, index=properties_sorted, columns=properties_sorted)
        for _, row in combined_sent_df.iterrows():
            a = row["property_A"]
            b = row["property_B"]
            npmi_val = row["npmi"]
            llr_val = row["llr"]
            matrix_npmi.loc[a, b] = npmi_val
            matrix_npmi.loc[b, a] = npmi_val
            matrix_llr.loc[a, b] = llr_val
            matrix_llr.loc[b, a] = llr_val
        np.fill_diagonal(matrix_npmi.values, 0.0)
        np.fill_diagonal(matrix_llr.values, 0.0)
        matrix_npmi.attrs["llr_matrix"] = matrix_llr

        sent_graph_path = output_base / "property_network_sentlevel.gexf"
        ensure_output_directory(sent_graph_path)
        build_network_and_export(matrix_npmi, sent_graph_path, "npmi")
        graph_paths.append(sent_graph_path)

        heatmap_sent_path = output_base / "heatmap_properties_sentlevel.png"
        save_heatmap(matrix_npmi.fillna(0.0), "Property NPMI (Sentence Level)", heatmap_sent_path)
        heatmap_paths.append(heatmap_sent_path)

    # Print top 10 pairs for GEOECO and GEOMORPHO (doc level)
    for corpus_name in ["GEOECO", "GEOMORPHO"]:
        corpus_df = doc_output[doc_output["corpus"] == corpus_name] if not doc_output.empty else pd.DataFrame()
        if corpus_df.empty:
            print(f"No document-level data for corpus {corpus_name} to report.")
            continue
        for metric in ["npmi", "llr"]:
            subset = corpus_df.dropna(subset=[metric])
            if subset.empty:
                print(f"No {metric.upper()} data for corpus {corpus_name}.")
                continue
            top10 = subset.sort_values(metric, ascending=False).head(10)
            print(
                f"Top 10 property pairs by {metric.upper()} for {corpus_name} (CSV format):"
            )
            print("property_A,property_B,metric,value,count_AB")
            for _, row in top10.iterrows():
                value = row[metric]
                value_str = "" if pd.isna(value) else f"{value:.6f}"
                print(
                    f"{row['property_A']},{row['property_B']},{metric.upper()},{value_str},{row['count_AB']}"
                )

    saved_paths = [
        doc_output_path,
        sent_output_path,
        term_output_path,
        focus_output_path,
        top_output_path,
        *graph_paths,
        *heatmap_paths,
    ]
    print("Saved files:")
    for path in saved_paths:
        if path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()

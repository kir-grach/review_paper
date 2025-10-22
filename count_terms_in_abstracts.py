from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

CANDIDATE_SUBSTRINGS = ["abstract", "summary", "text", "описание", "аннотация"]
DASH_REPLACEMENTS: Dict[str, str] = {
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
}


@dataclass
class TermRecord:
    soil_property: str
    term: str
    patterns: Sequence[re.Pattern]


def normalize_dashes(value: str) -> str:
    normalized = value
    for original, replacement in DASH_REPLACEMENTS.items():
        normalized = normalized.replace(original, replacement)
    return normalized


def detect_text_column(df: pd.DataFrame, source_name: str) -> str:
    for substring in CANDIDATE_SUBSTRINGS:
        for column in df.columns:
            if substring in column.lower():
                print(f"Detected abstract column for {source_name}: {column}")
                return column

    text_like_columns: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            text_like_columns.append(column)
    if not text_like_columns:
        text_like_columns = list(df.columns)

    best_column = text_like_columns[0]
    best_score = -1.0
    for column in text_like_columns:
        series = df[column].dropna()
        if series.empty:
            average_length = 0.0
        else:
            series_as_str = series.astype(str)
            average_length = series_as_str.map(len).mean()
        if average_length > best_score:
            best_score = average_length
            best_column = column

    print(f"Detected abstract column for {source_name}: {best_column}")
    return best_column


def safe_str_lower(series: pd.Series) -> pd.Series:
    string_methods = series.str
    lower = getattr(string_methods, "lower")
    try:
        return lower(errors="ignore")
    except TypeError:
        return lower()


def load_abstracts(file_path: Path, source_name: str) -> List[str]:
    df = pd.read_csv(file_path)
    column_name = detect_text_column(df, source_name)
    series = df[column_name]

    if not pd.api.types.is_string_dtype(series):
        series = series.astype(str)

    series = series.dropna()
    series = series.astype(str)
    series = series.str.strip()
    series = series[series.str.len() > 0]

    for original, replacement in DASH_REPLACEMENTS.items():
        series = series.str.replace(original, replacement, regex=False)

    series = safe_str_lower(series)
    series = series[series.str.len() > 0]
    return series.tolist()


def parse_key_terms(raw_terms: str) -> List[str]:
    parsed_terms: List[str] = []
    if not raw_terms:
        return parsed_terms

    for row in csv.reader([raw_terms], skipinitialspace=True):
        for term in row:
            clean_term = term.strip()
            if clean_term:
                parsed_terms.append(clean_term)
    return parsed_terms


def load_soil_terms(file_path: Path) -> List[Tuple[str, List[str]]]:
    df = pd.read_csv(file_path)
    records: List[Tuple[str, List[str]]] = []
    for _, row in df.iterrows():
        soil_property = str(row.get("Soil Property", "")).strip()
        raw_terms = row.get("Key Terms", "")
        if pd.isna(raw_terms):
            terms = []
        else:
            terms = parse_key_terms(str(raw_terms))
        records.append((soil_property, terms))
    return records


def should_use_word_boundary(token: str, position: str) -> bool:
    if not token:
        return False
    char = token[0] if position == "start" else token[-1]
    return bool(re.match(r"\w", char))


def is_abbreviation_token(token: str) -> bool:
    if not token:
        return False
    alnum_only = re.sub(r"[^A-Za-z0-9]", "", token)
    if not alnum_only:
        return False
    return alnum_only.upper() == alnum_only and alnum_only.lower() != alnum_only


def pluralize_token_pattern(raw_token: str) -> str:
    if not re.fullmatch(r"[a-z]+", raw_token):
        return re.escape(raw_token)

    if raw_token.endswith(("s", "x", "z")) or raw_token.endswith(("ch", "sh")):
        return f"{re.escape(raw_token)}(?:es)?"

    if raw_token.endswith("y") and len(raw_token) > 1 and raw_token[-2] not in "aeiou":
        base = re.escape(raw_token[:-1])
        return f"{base}(?:y|ies)"

    return f"{re.escape(raw_token)}s?"


def phrase_to_pattern(phrase: str, original_phrase: str) -> str | None:
    normalized_phrase = normalize_dashes(phrase.strip().lower())
    original_phrase_clean = normalize_dashes(original_phrase.strip())
    if not normalized_phrase:
        return None

    tokens = [token for token in re.split(r"[\s\-]+", normalized_phrase) if token]
    original_tokens = [token for token in re.split(r"[\s\-]+", original_phrase_clean) if token]
    if not tokens:
        return None

    token_patterns: List[str] = []
    for index, token in enumerate(tokens):
        original_token = original_tokens[index] if index < len(original_tokens) else token
        if index == len(tokens) - 1 and not is_abbreviation_token(original_token):
            token_patterns.append(pluralize_token_pattern(token))
        else:
            token_patterns.append(re.escape(token))

    body = r"[\s\-]+".join(token_patterns)

    start_boundary = "\b" if should_use_word_boundary(tokens[0], "start") else ""
    end_boundary = "\b" if should_use_word_boundary(tokens[-1], "end") else ""

    return f"{start_boundary}{body}{end_boundary}"


def abbreviation_to_pattern(abbreviation: str) -> str | None:
    normalized = abbreviation.strip().lower()
    if not normalized:
        return None
    normalized = normalize_dashes(normalized)
    if not normalized:
        return None

    start_boundary = "\b" if should_use_word_boundary(normalized, "start") else ""
    end_boundary = "\b" if should_use_word_boundary(normalized, "end") else ""
    escaped = re.escape(normalized)
    return f"{start_boundary}{escaped}{end_boundary}"


def term_to_patterns(term: str) -> Sequence[re.Pattern]:
    patterns: List[re.Pattern] = []
    normalized_term = normalize_dashes(term.strip())
    if not normalized_term:
        return patterns

    match = re.search(r"\(([^()]+)\)\s*$", normalized_term)
    abbreviations: List[str] = []
    if match:
        abbr_content = match.group(1)
        normalized_term = normalized_term[: match.start()].strip()
        abbreviations = [abbr.strip() for abbr in re.split(r"[;,/]+", abbr_content) if abbr.strip()]

    phrase_pattern = phrase_to_pattern(normalized_term, normalized_term)
    if phrase_pattern:
        patterns.append(re.compile(phrase_pattern))

    for abbreviation in abbreviations:
        if not abbreviation:
            continue
        if not re.match(r"^[A-Za-z0-9\-]+$", abbreviation):
            continue
        if abbreviation.upper() != abbreviation:
            continue
        abbr_pattern = abbreviation_to_pattern(abbreviation)
        if abbr_pattern:
            patterns.append(re.compile(abbr_pattern))

    if not patterns and normalized_term:
        fallback_lower = normalize_dashes(normalized_term.lower())
        start_boundary = "\b" if should_use_word_boundary(fallback_lower, "start") else ""
        end_boundary = "\b" if should_use_word_boundary(fallback_lower, "end") else ""
        fallback_pattern = f"{start_boundary}{re.escape(fallback_lower)}{end_boundary}"
        patterns.append(re.compile(fallback_pattern))

    return patterns


def count_occurrences(patterns: Sequence[re.Pattern], texts: Sequence[str]) -> int:
    total = 0
    for text in texts:
        if not text:
            continue
        for pattern in patterns:
            total += len(pattern.findall(text))
    return total


def build_term_records(soil_terms: List[Tuple[str, List[str]]]) -> List[TermRecord]:
    records: List[TermRecord] = []
    for soil_property, terms in soil_terms:
        for term in terms:
            patterns = term_to_patterns(term)
            records.append(TermRecord(soil_property=soil_property, term=term, patterns=patterns))
    return records


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    soil_terms_path = base_dir / "soil_terms_25.csv"
    geoeco_path = base_dir / "geoeco_results.csv"
    geomorpho_path = base_dir / "geomorpho_results.csv"

    soil_terms = load_soil_terms(soil_terms_path)
    term_records = build_term_records(soil_terms)

    geoeco_abstracts = load_abstracts(geoeco_path, "geoeco_results.csv")
    geomorpho_abstracts = load_abstracts(geomorpho_path, "geomorpho_results.csv")

    results: List[Dict[str, object]] = []
    for record in term_records:
        geoeco_count = count_occurrences(record.patterns, geoeco_abstracts)
        geomorpho_count = count_occurrences(record.patterns, geomorpho_abstracts)
        results.append(
            {
                "soil properties": record.soil_property,
                "key term": record.term,
                "частота в геоэкологии": geoeco_count,
                "частота в геоморфологии": geomorpho_count,
            }
        )

    result_df = pd.DataFrame(results)
    output_path = Path("/mnt/data/term_frequencies.csv")
    result_df.to_csv(output_path, index=False, encoding="utf-8")

    print(result_df.head(10))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

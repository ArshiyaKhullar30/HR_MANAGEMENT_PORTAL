"""Text and skill-name canonicalisation (Step 03).

The Build Notes call out the problem directly: 'AWS', 'Amazon Web Services' and
'AWS Cloud' are the same skill spelled three ways. Left alone, the skill-gap
engine treats them as three separate gaps and the org-wide rollup is wrong.

Canonicalisation is deliberately conservative — an alias table for the cases
that matter plus mechanical normalisation. Aggressive fuzzy merging would
collapse genuinely different tools.
"""

from __future__ import annotations

import re

import pandas as pd

# Explicit aliases. Keys are matched after `normalise_token`, so they are
# already lowercase and punctuation-free.
SKILL_ALIASES: dict[str, str] = {
    "amazon web services": "AWS",
    "amazon web services aws": "AWS",
    "aws": "AWS",
    "aws cloud": "AWS",
    "amazon aws": "AWS",
    "microsoft azure": "Azure",
    "azure": "Azure",
    "google cloud platform": "Google Cloud",
    "gcp": "Google Cloud",
    "structured query language sql": "SQL",
    "sql": "SQL",
    "structured query language": "SQL",
    "microsoft sql server": "Microsoft SQL Server",
    "python": "Python",
    "the python programming language": "Python",
    "r": "R",
    "the r programming language": "R",
    "microsoft excel": "Microsoft Excel",
    "ms excel": "Microsoft Excel",
    "microsoft office software": "Microsoft Office",
    "microsoft office": "Microsoft Office",
    "microsoft word": "Microsoft Word",
    "microsoft powerpoint": "Microsoft PowerPoint",
    "microsoft outlook": "Microsoft Outlook",
    "microsoft access": "Microsoft Access",
    "tableau": "Tableau",
    "tableau software": "Tableau",
    "microsoft power bi": "Power BI",
    "power bi": "Power BI",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "git": "Git",
    "github": "GitHub",
    "linux": "Linux",
    "unix": "Unix",
    "javascript": "JavaScript",
    "oracle java": "Java",
    "java": "Java",
    "c": "C",
    "c language": "C",
    "sap software": "SAP",
    "sap": "SAP",
    "salesforce software": "Salesforce",
    "salesforce": "Salesforce",
}

# Trailing version/edition noise: "Microsoft Excel 2016", "Tableau v10.2"
_VERSION = re.compile(r"\b(v?\d+(\.\d+)*|20\d{2})\b\s*$", re.I)
_PUNCT = re.compile(r"[^\w\s]+")
_SPACE = re.compile(r"\s+")


def is_constant(series: pd.Series) -> bool:
    """True when a column carries no information.

    Cheaper than `nunique() <= 1`, which builds the full set of distinct values
    just to discover there is one.
    """
    non_null = series.dropna()
    return non_null.empty or bool((non_null == non_null.iloc[0]).all())


def normalise_token(value: object) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop trailing versions."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    text = _VERSION.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def canonical_skill(value: object) -> str:
    """Map a raw skill/tool string to its canonical name.

    Unknown values keep their original spelling with whitespace tidied, so
    nothing is silently discarded.
    """
    token = normalise_token(value)
    if not token:
        return ""
    if token in SKILL_ALIASES:
        return SKILL_ALIASES[token]
    return _SPACE.sub(" ", str(value).strip())


def canonical_skill_series(series: pd.Series) -> pd.Series:
    """Vectorised :func:`canonical_skill` — cached per distinct value."""
    mapping = {v: canonical_skill(v) for v in series.dropna().unique()}
    return series.map(mapping).fillna("")


def clean_category(series: pd.Series) -> pd.Series:
    """Strip stray whitespace from a categorical column.

    Fixes the `'Production       '` padding in the engagement file (F6), which
    would otherwise split one department into two categories.
    """
    return series.astype("string").str.strip().str.replace(_SPACE, " ", regex=True)


def parse_dates(series: pd.Series, formats: list[str]) -> pd.Series:
    """Parse a mixed-format date column, trying each configured format in turn.

    The engagement file mixes `20-Sep-19` and `07-10-1969` across columns, so a
    single format string silently produces NaT for most of the data.
    """
    remaining = series.astype("string").str.strip().replace({"": None})
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in formats:
        missing = result.isna() & remaining.notna()
        if not missing.any():
            break
        result.loc[missing] = pd.to_datetime(remaining[missing], format=fmt, errors="coerce")
    missing = result.isna() & remaining.notna()
    if missing.any():  # last resort, still deterministic
        result.loc[missing] = pd.to_datetime(remaining[missing], errors="coerce", dayfirst=True)
    return result

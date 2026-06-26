import csv
import io
import logging
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRosterEntry:
    """One agent's canonical roster entry with domain skills."""

    name: str
    skills: Tuple[str, ...]  # e.g., ("LI", "GI")
    normalized_name: str
    tokens: Tuple[str, ...]


def _parse_bool(value: object) -> bool:
    s = str(value or "").strip().lower()
    return s in {"true", "t", "yes", "y", "1", "x"}


_NAME_CLEAN_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_person_name(name: str) -> str:
    """
    Normalize a name for fuzzy matching:
    - lower case
    - remove punctuation
    - collapse whitespace
    - remove common prefixes
    """
    s = str(name or "").strip().lower()
    if not s:
        return ""

    # Remove common prefixes that sometimes appear in transcripts
    for prefix in ("cso", "agent", "advisor", "officer", "csr"):
        s = re.sub(rf"^\s*{re.escape(prefix)}\s*[:\-]?\s*", "", s, flags=re.IGNORECASE)

    s = _NAME_CLEAN_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _tokenize(name_norm: str) -> Tuple[str, ...]:
    tokens = [t for t in (name_norm or "").split(" ") if t]
    # Drop ultra-common honorifics/titles
    drop = {"mr", "mrs", "ms", "miss", "dr", "sir", "madam"}
    tokens = [t for t in tokens if t not in drop]
    return tuple(tokens)


def load_agent_roster(path: Optional[str] = None) -> List[AgentRosterEntry]:
    """
    Load a roster CSV with columns:
      - Name
      - LI (TRUE/FALSE or 1/0)
      - GI
      - HI

    Set via env var AGENT_ROSTER_PATH, default: ./agent_roster.csv
    - Supports local paths and S3 URIs: s3://bucket/key/to/roster.xlsx
    - For Excel files, you may optionally set AGENT_ROSTER_SHEET (sheet name or 0-based index).
    """
    roster_path_raw = (path or os.getenv("AGENT_ROSTER_PATH", "agent_roster.csv")).strip()
    if not roster_path_raw:
        roster_path_raw = "agent_roster.csv"

    # Support "local-first, S3-fallback" by allowing multiple candidates via "|"
    # Example:
    #   AGENT_ROSTER_PATH="rosterstaff.xlsx|s3://my-bucket/rosters/rosterstaff.xlsx"
    candidates: List[str] = [p.strip() for p in roster_path_raw.split("|") if p.strip()]

    # Optional extra fallback env vars (useful when you don't want to overload AGENT_ROSTER_PATH)
    fallback = (
        str(os.getenv("AGENT_ROSTER_S3_FALLBACK", "") or "").strip()
        or str(os.getenv("AGENT_ROSTER_FALLBACK_PATH", "") or "").strip()
        or str(os.getenv("AGENT_ROSTER_S3_PATH", "") or "").strip()
    )
    if fallback and fallback not in candidates:
        candidates.append(fallback)

    # Basic load diagnostics (helps confirm whether Excel/CSV is being used)
    try:
        logger.info("Agent roster: requested path='%s' cwd='%s'", roster_path_raw, os.getcwd())
    except Exception:
        pass

    def _resolve_local_path(p: str) -> Optional[str]:
        if not p:
            return None
        if os.path.exists(p):
            return p
        # Try alongside this module (helps when running from a different CWD)
        try:
            here = os.path.dirname(__file__)
            alt = os.path.join(here, p)
            if os.path.exists(alt):
                return alt
            # Try one level up (repo root vs package dir)
            alt2 = os.path.join(os.path.dirname(here), p)
            if os.path.exists(alt2):
                return alt2
        except Exception:
            pass
        return None

    def _load_one(roster_path: str) -> List[AgentRosterEntry]:
        # Avoid accidental selection of Excel's temporary lock files (e.g. "~$Skillset.xlsx")
        try:
            if os.path.basename(roster_path).startswith("~$"):
                logger.warning(
                    f"Agent roster: '{roster_path}' looks like an Excel temp/lock file (~$...). "
                    "Please point AGENT_ROSTER_PATH to the actual .xlsx file instead."
                )
                return []
        except Exception:
            pass

        is_s3 = roster_path.lower().startswith("s3://")
        resolved_local = None if is_s3 else _resolve_local_path(roster_path)
        if not is_s3:
            if not resolved_local:
                try:
                    logger.warning(
                        "Agent roster: file not found. path='%s' abs='%s' cwd='%s'",
                        roster_path,
                        os.path.abspath(roster_path),
                        os.getcwd(),
                    )
                except Exception:
                    logger.warning("Agent roster: file not found. path='%s'", roster_path)
                return []
            roster_path = resolved_local

        try:
            logger.info("Agent roster: loading from %s '%s'", "S3" if is_s3 else "local", roster_path)
        except Exception:
            pass

        entries: List[AgentRosterEntry] = []
        _, ext = os.path.splitext(roster_path.lower())

        def _download_s3_bytes(s3_uri: str) -> Optional[bytes]:
        try:
            import boto3
        except Exception as e:
            logger.warning(f"Agent roster: boto3 not available to read S3 roster {s3_uri}: {e}")
            return None

        parsed = urlparse(s3_uri)
        bucket = (parsed.netloc or "").strip()
        key = (parsed.path or "").lstrip("/").strip()
        if not bucket or not key:
            logger.warning(f"Agent roster: invalid S3 URI (expected s3://bucket/key): {s3_uri}")
            return None

        profile = os.getenv("AWS_PROFILE") or None
        region = os.getenv("AWS_REGION") or None

        try:
            session = boto3.Session(profile_name=profile, region_name=region)
        except Exception:
            # Fallback to default credential chain if the profile isn't available
            session = boto3.Session(region_name=region)

        try:
            s3 = session.client("s3", verify=False)
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj.get("Body")
            return body.read() if body is not None else None
        except Exception as e:
            logger.warning(f"Agent roster: failed to download s3://{bucket}/{key}: {e}")
            return None

        def _iter_rows() -> Iterable[Dict[str, object]]:
            data_bytes: Optional[bytes] = None
            if is_s3:
                data_bytes = _download_s3_bytes(roster_path)
                if not data_bytes:
                    return []

            if ext in {".xlsx", ".xls"}:
                try:
                    import pandas as pd  # already used elsewhere in the app
                except Exception as e:
                    logger.warning(f"Agent roster: pandas not available to read Excel file {roster_path}: {e}")
                    return []

                sheet_raw = str(os.getenv("AGENT_ROSTER_SHEET", "") or "").strip()
                sheet_name = 0
                if sheet_raw:
                    try:
                        sheet_name = int(sheet_raw)
                    except Exception:
                        sheet_name = sheet_raw
                try:
                    logger.info("Agent roster: reading Excel sheet=%s from '%s'", sheet_name, roster_path)
                except Exception:
                    pass

                try:
                    if is_s3:
                        df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet_name)
                    else:
                        df = pd.read_excel(roster_path, sheet_name=sheet_name)
                except Exception as e:
                    logger.warning(
                        f"Agent roster: failed to read Excel roster {roster_path}. "
                        f"If this is an .xlsx file, ensure 'openpyxl' is installed. Error: {e}"
                    )
                    return []

                try:
                    # If caller passed sheet_name=None, pandas can return dict of DataFrames.
                    if isinstance(df, dict) and df:
                        df = next(iter(df.values()))
                    try:
                        cols = list(getattr(df, "columns", []) or [])
                        logger.info(
                            "Agent roster: Excel loaded rows=%s cols=%s",
                            int(getattr(df, "shape", (0, 0))[0] or 0),
                            cols,
                        )
                    except Exception:
                        pass
                    return df.to_dict(orient="records") or []
                except Exception:
                    return []

            # Default: CSV
            try:
                if is_s3:
                    text = (data_bytes or b"").decode("utf-8-sig", errors="replace")
                    rows = list(csv.DictReader(io.StringIO(text)))
                    logger.info("Agent roster: CSV loaded rows=%d from '%s'", len(rows), roster_path)
                    return rows
                with open(roster_path, "r", newline="", encoding="utf-8-sig") as f:
                    rows = list(csv.DictReader(f))
                logger.info("Agent roster: CSV loaded rows=%d from '%s'", len(rows), roster_path)
                return rows
            except Exception as e:
                logger.warning(f"Agent roster: failed to read CSV roster {roster_path}: {e}")
                return []

        for row in _iter_rows() or []:
            if not isinstance(row, dict):
                continue

            # Flexible column names
            name = (row.get("Name") or row.get("name") or "").strip()
            if not name:
                continue

            li = _parse_bool(row.get("LI") or row.get("li") or row.get("Life Insurance"))
            gi = _parse_bool(row.get("GI") or row.get("gi") or row.get("General Insurance"))
            hi = _parse_bool(row.get("HI") or row.get("hi") or row.get("Health Insurance"))

            skills: List[str] = []
            if li:
                skills.append("LI")
            if gi:
                skills.append("GI")
            if hi:
                skills.append("HI")

            norm = normalize_person_name(name)
            if not norm:
                continue

            entries.append(
                AgentRosterEntry(
                    name=name,
                    skills=tuple(skills),
                    normalized_name=norm,
                    tokens=_tokenize(norm),
                )
            )

        # Summary diagnostics
        try:
            li_cnt = sum(1 for e in entries if "LI" in e.skills)
            gi_cnt = sum(1 for e in entries if "GI" in e.skills)
            hi_cnt = sum(1 for e in entries if "HI" in e.skills)
            logger.info(
                "Agent roster: parsed entries=%d (LI=%d GI=%d HI=%d) from '%s'",
                len(entries),
                li_cnt,
                gi_cnt,
                hi_cnt,
                roster_path,
            )
        except Exception:
            pass

        return entries

    for cand in candidates:
        loaded = _load_one(cand)
        if loaded:
            return loaded

    # Nothing loaded
    return []


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _similarity(name_norm: str, tokens: Tuple[str, ...], entry: AgentRosterEntry) -> float:
    """
    Compute a robust similarity score for person names.
    We combine:
      - raw sequence match
      - token-sorted sequence match (handles reversed order)
      - token-set overlap (handles partial mentions)
    """
    raw = _ratio(name_norm, entry.normalized_name)
    sorted_a = " ".join(sorted(tokens))
    sorted_b = " ".join(sorted(entry.tokens))
    sorted_ratio = _ratio(sorted_a, sorted_b)
    token_overlap = _jaccard(tokens, entry.tokens)
    return max(raw, sorted_ratio, token_overlap)


def _single_token_similarity(token: str, entry: AgentRosterEntry) -> float:
    """
    Similarity for when we only have a single token from the transcript (often just a first name).
    We only return high scores for strong matches (exact token or clear prefix),
    and rely on the ambiguity check in `match_agent_name` to avoid wrong assignments.
    """
    t = str(token or "").strip().lower()
    if not t:
        return 0.0

    # Exact token match to any part of the roster name
    if t in entry.tokens:
        return 1.0

    # Prefix matching (e.g., "ben" -> "benja")
    if len(t) >= 3:
        for idx, et in enumerate(entry.tokens):
            et = str(et or "").strip().lower()
            if not et:
                continue
            if et.startswith(t):
                # Slightly prefer prefix match on the first token (first name)
                return 0.94 if idx == 0 else 0.92
            if t.startswith(et) and len(et) >= 3:
                return 0.92

    # Very weak signals are ignored (too risky / ambiguous)
    return 0.0


def match_agent_name(
    extracted_name: str,
    roster: List[AgentRosterEntry],
    *,
    min_score: float = 0.82,
    ambiguity_delta: float = 0.05,
) -> Tuple[Optional[AgentRosterEntry], float]:
    """
    Match an extracted (possibly messy) agent name to the roster.

    Returns: (best_entry_or_None, match_score)
    - If no roster is configured or match isn't confident, returns (None, score)
    """
    if not roster:
        return None, 0.0

    name_norm = normalize_person_name(extracted_name)
    if not name_norm:
        return None, 0.0

    tokens = _tokenize(name_norm)
    if not tokens:
        return None, 0.0

    # Fast path: exact normalized match
    for e in roster:
        if e.normalized_name == name_norm:
            return e, 1.0

    # Special handling: single-token names (often first-name only).
    # We only match when there is a strong, *unique* best candidate.
    if len(tokens) == 1:
        token = tokens[0]
        scored: List[Tuple[float, AgentRosterEntry]] = []
        for e in roster:
            s = _single_token_similarity(token, e)
            if s > 0.0:
                scored.append((s, e))
        if not scored:
            return None, 0.0

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0

        # Higher bar for single-token matches
        if best_score < 0.92:
            return None, best_score

        # Require a clearer gap to avoid mapping "Ben" to the wrong "Ben*"
        if (best_score - second_score) < 0.10 and best_score < 0.99:
            return None, best_score

        return best, best_score

    scored: List[Tuple[float, AgentRosterEntry]] = []
    for e in roster:
        scored.append((_similarity(name_norm, tokens, e), e))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if best_score < min_score:
        return None, best_score

    # If it's very close to the next best match, treat as ambiguous (avoid misassignment)
    if (best_score - second_score) < ambiguity_delta and best_score < 0.93:
        return None, best_score

    return best, best_score


def rank_agent_candidates(
    extracted_name: str,
    roster: List[AgentRosterEntry],
    *,
    top_n: int = 5,
) -> List[Tuple[float, AgentRosterEntry]]:
    """
    Return top candidate matches for debugging.
    Output is a list of (score, entry) sorted descending.
    """
    if not roster:
        return []

    name_norm = normalize_person_name(extracted_name)
    if not name_norm:
        return []

    tokens = _tokenize(name_norm)
    if not tokens:
        return []

    scored: List[Tuple[float, AgentRosterEntry]] = []
    if len(tokens) == 1:
        token = tokens[0]
        for e in roster:
            s = _single_token_similarity(token, e)
            if s > 0.0:
                scored.append((s, e))
    else:
        for e in roster:
            s = _similarity(name_norm, tokens, e)
            if s > 0.0:
                scored.append((s, e))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(0, int(top_n or 0))]


from ..connection import get_conn
from rapidfuzz import fuzz, process

def _all_providers() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT provider_id, name, specialty FROM providers").fetchall()
        return [dict(r) for r in rows]

def find_providers_by_name(name_query: str, limit: int  = 5) -> list[dict]:
    """
    Fuzzy match provider names using RapidFuzz (token_sort_ratio handles word-order
    and partial input like 'Sarah Lee' vs 'Dr. Sarah Lee' vs 'Lee, Sarah').
    Returns matches sorted by score, each annotated with match_score.
    """
    providers = _all_providers()
    if not providers:
        return []
    choices = {p["provider_id"]:p["name"] for p in providers}
    results = process.extract(
        name_query,
        choices=choices,
        scorer=fuzz.token_sort_ratio
    )
    by_id = {p["provider_id"]:p for p in providers}
    matched = []
    for matched_name, score, provider_id in results:
        p = dict(by_id[provider_id])
        p["match_score"] = round(score,1)
        matched.append(p)
    return matched

def get_provider_by_id(provider_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT provider_id, name, specialty FROM providers WHERE provider_id = ?",
            (provider_id,),
        ).fetchone()
        return dict(row) if row else None
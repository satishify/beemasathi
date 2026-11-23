"""Resolve the policy name a question mentions against catalog policy names.

One resolver is shared by the router and the retriever so that every insurer's
plans behave the same way when a customer uses a short or reordered name.
"""

import re
from typing import Dict, Iterable, List, Optional, Tuple

GENERIC_SUFFIXES = frozenset({"policy", "plan", "insurance"})

# Filename artifacts that got glued onto policy names during ingestion. They
# are never product words, so matching ignores them at the end of a name.
ARTIFACT_SUFFIXES = frozenset(
    {
        "myhealth", "healthplus", "medicare", "criticalillnessinsurance",
        "prospectus", "prospectuse", "website", "pdf206", "cum", "wef",
        "document", "print", "super", "charge",
    }
)

# Words that describe a product class, not one specific product.
GENERIC_SINGLE = frozenset(
    {
        "health", "insurance", "policy", "plan", "care", "cover", "covers",
        "coverage", "general", "motor", "vehicle", "travel", "home", "life",
        "group", "family", "personal", "accident", "critical", "illness",
        "hospital", "cashless", "benefit", "benefits", "senior", "wellness",
        "medical", "protection", "shield", "plus", "my", "new", "the",
        "and", "for", "with", "one",
    }
)

# Whole names that are only a product class ("car", "two wheeler"), never one
# specific plan. Questions about a class must list or retrieve across insurers.
PRODUCT_CLASS_NAMES = frozenset(
    {
        "two wheeler", "two wheelers", "tw", "car", "cars", "motor",
        "motor insurance", "vehicle", "vehicles", "health", "health insurance",
        "bike", "bikes", "two wheeler insurance", "car insurance",
        "policy", "insurance",
    }
)

# Score bonuses by match kind. Character count dominates, then how strict the
# match was. A longer name the customer described well beats a shorter one.
_BONUS = {"exact": 40, "cover": 35, "any_order": 30, "partial": 10}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokens_for(name: str) -> List[str]:
    """Name words without generic suffixes and ingestion artifacts."""
    tokens = normalize(name).split()
    while tokens and (tokens[-1] in GENERIC_SUFFIXES or tokens[-1] in ARTIFACT_SUFFIXES):
        tokens.pop()
    return tokens


def is_subsequence(needle: List[str], haystack: List[str]) -> bool:
    position = 0
    for token in haystack:
        if position < len(needle) and token == needle[position]:
            position += 1
    return position == len(needle)


def _insurer_words(insurer: Optional[str]) -> set:
    return set(normalize(insurer or "").split())


def _has_distinctive_token(tokens: List[str], insurer: Optional[str]) -> bool:
    insurer_words = _insurer_words(insurer)
    return any(
        token not in GENERIC_SINGLE and token not in insurer_words
        for token in tokens
    )


def _meaningful(tokens: List[str], insurer: Optional[str]) -> bool:
    if len(tokens) == 1:
        return (
            bool(insurer)
            and len(tokens[0]) >= 3
            and tokens[0] not in GENERIC_SINGLE
            and tokens[0] not in _insurer_words(insurer)
        )
    # With a known insurer, short product names like "M Care" are unambiguous.
    minimum = 5 if insurer else 8
    return len(" ".join(tokens)) >= minimum


def _ordered_positions(tokens: List[str], query_tokens: List[str]) -> List[int]:
    return [query_tokens.index(token) for token in tokens]


def _inversions(positions: List[int]) -> int:
    return sum(
        1
        for i, first in enumerate(positions)
        for second in positions[i + 1:]
        if first > second
    )


def _cover(tokens: List[str], query_tokens: List[str]) -> Optional[Tuple[int, int, int]]:
    """Match name tokens against consecutive query words, joining split words.

    Returns (matched name tokens, matched characters, query end position) for
    the best start position, so "my health" also matches the name word
    "myhealth" and vice versa.
    """
    best = None
    for start in range(len(query_tokens)):
        position = start
        matched = 0
        characters = 0
        for token in tokens:
            joined = ""
            found = False
            while position < len(query_tokens):
                joined += query_tokens[position]
                position += 1
                if joined == token:
                    found = True
                    break
                if not token.startswith(joined):
                    break
            if not found:
                break
            matched += 1
            characters += len(token)
        if matched and (best is None or (matched, characters) > best[:2]):
            best = (matched, characters, position)
    return best


def _partial_allowed(
    tokens: List[str],
    query_tokens: List[str],
    matched: int,
    characters: int,
    end: int,
    insurer: Optional[str],
) -> bool:
    # Only accept a customer typing less than the full name, never a customer
    # naming a different product that merely starts the same way.
    if matched < 3 or characters < 12:
        return False
    if end < len(query_tokens):
        extra = query_tokens[end:]
        if any(
            token not in GENERIC_SINGLE and token not in _insurer_words(insurer)
            for token in extra
        ):
            return False
    return _has_distinctive_token(tokens[:matched], insurer)


def _score(
    tokens: List[str],
    query_tokens: List[str],
    insurer: Optional[str],
    dropped: bool,
) -> Optional[float]:
    chars = len("".join(tokens))
    if is_subsequence(tokens, query_tokens):
        return chars * 100 + _BONUS["exact"] + (0.5 if dropped else 0)
    if all(token in set(query_tokens) for token in tokens) and _has_distinctive_token(
        tokens, insurer
    ):
        inversions = _inversions(_ordered_positions(tokens, query_tokens))
        return chars * 100 + max(_BONUS["any_order"] - inversions, 0)
    covered = _cover(tokens, query_tokens)
    if covered:
        matched, matched_chars, end = covered
        if matched == len(tokens) and _has_distinctive_token(tokens, insurer):
            return chars * 100 + _BONUS["cover"]
        if _partial_allowed(tokens, query_tokens, matched, matched_chars, end, insurer):
            return matched_chars * 100 + _BONUS["partial"]
    return None


def _stripped(name: str) -> Tuple[str, ...]:
    return tuple(tokens_for(name))


def _pick_winner(winners: List[Tuple[str, str]]) -> Optional[str]:
    if len(winners) == 1:
        return winners[0][1]
    # Same product listed under a longer filename artifact (e.g. "cum",
    # "print"): keep the shorter original name.
    same_insurer = len({insurer for insurer, _ in winners}) == 1
    same_product = len({_stripped(name) for _, name in winners}) == 1
    if same_insurer and same_product:
        shortest = min(len(normalize(name).split()) for _, name in winners)
        final = [pair for pair in winners if len(normalize(pair[1]).split()) == shortest]
        if len(final) == 1:
            return final[0][1]
    return None


def match_policy(
    query_tokens: List[str],
    candidates: Iterable[Tuple[str, str]],
    insurer: Optional[str] = None,
) -> Optional[str]:
    """Return the single most specific policy named in the query."""
    scored: Dict[Tuple[str, str], float] = {}
    for candidate_insurer, name in candidates:
        if insurer and candidate_insurer != insurer:
            continue
        best = None
        for dropped in (False, True):
            tokens = tokens_for(name) if dropped else normalize(name).split()
            if not _meaningful(tokens, insurer):
                continue
            score = _score(tokens, query_tokens, insurer, dropped)
            if score is not None and (best is None or score > best):
                best = score
        if best is not None:
            scored[(candidate_insurer, name)] = best
    if not scored:
        return None
    top_score = max(scored.values())
    winners = [key for key, score in scored.items() if score == top_score]
    return _pick_winner(winners)

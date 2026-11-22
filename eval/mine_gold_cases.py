"""Mine candidate gold cases from the indexed policy wording.

Facts are copied verbatim from chunks that are already in ChromaDB, so every
``gold_notes`` string is grounded in a real clause and every case carries the
``expected_source`` needed for recall@k. The output still needs a human read:
run ``--write`` to produce ``eval/gold_cases.generated.json`` and merge only the
cases you have checked against the PDF.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import INSURERS

CHUNKS_DIR = ROOT / "data" / "chunks"
DEFAULT_OUT = Path(__file__).resolve().parent / "gold_cases.generated.json"

SKIP_POLICY = re.compile(
    r"aap_ke_liye|na_ka|"
    r"andaman|nicobar|arunachal|assam|bihar|chhattisgarh|goa|gujarat|haryana|"
    r"himachal|jammu|kashmir|jharkhand|karnataka|kerala|madhya|maharashtra|"
    r"manipur|meghalaya|mizoram|nagaland|odisha|punjab|rajasthan|sikkim|tamil|"
    r"telangana|tripura|uttar|uttarakhand|bengal|ladakh",
    re.I,
)

GENERIC_POLICY = {"car", "motor", "health", "comprehensive", "policy", "cover"}

COP_RE = re.compile(
    r"co[\s-]?pay(?:ment)?\s+(?:of|is|@|at)\s*(\d{1,2}(?:\.\d)?)\s*%", re.I
)
WAIT_RE = re.compile(
    r"waiting period[^.]{0,150}?(\d{1,3})\s*(months?|days?|years?)", re.I
)
ROOM_RE = re.compile(r"room rent[^.]{0,180}?(\d{1,2}(?:\.\d)?)\s*%", re.I)
DED_RE = re.compile(
    r"deductible[^.]{0,60}?(?:rs\.?|inr|₹)\s?([\d,]{3,})", re.I
)
GRACE_RE = re.compile(r"grace period[^.]{0,120}?(\d{1,3})\s*days?", re.I)
UIN_RE = re.compile(r"UIN[:\s]*([A-Z]{2,}[A-Z0-9]{6,})")


def _clause(text: str, start: int, end: int) -> str:
    """Return the single sentence that contains the fact."""
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    left = 0 if left < 0 else left + 1
    right = text.find(". ", end)
    right = len(text) if right < 0 else right + 1
    clause = re.sub(r"\s+", " ", text[left:right]).strip()
    if len(clause) > 460:
        clause = re.sub(r"\s+", " ", text[max(0, start - 150):min(len(text), end + 150)]).strip()
    return clause


SMALL_WORDS = {"of", "and", "the", "for", "to", "in", "on", "wef", "w", "e", "f"}


def _policy_display(policy: str, insurer: str) -> str:
    words = [w for w in re.split(r"[_\-\s]+", policy) if w]
    # Drop a duplicated insurer prefix such as "tata_aig_tata_aig_criti_medi".
    prefix = [w for w in insurer.split("_") if w]
    if len(words) > len(prefix) and [w.lower() for w in words[: len(prefix)]] == [
        w.lower() for w in prefix
    ]:
        words = words[len(prefix):]
    pretty = []
    for i, word in enumerate(words):
        if word.lower() in SMALL_WORDS and i > 0:
            pretty.append(word.lower())
        elif len(word) <= 3 and word.isalpha():
            pretty.append(word.upper())
        else:
            pretty.append(word.capitalize())
    return " ".join(pretty)


def _clean_number(value: str) -> str:
    return value.replace(",", "")


def _waiting_subtype(prefix: str) -> Optional[str]:
    text = prefix.lower()
    labels = {
        "pre_existing": ("pre-existing disease waiting", "pre existing disease waiting",
                         "ped waiting", "pre-existing disease"),
        "specific": ("specific waiting", "specified disease", "specified diseases"),
        "initial": ("initial waiting", "first 30 days waiting", "30 days waiting"),
    }
    best, best_pos = None, -1
    for subtype, needles in labels.items():
        for needle in needles:
            pos = text.rfind(needle)
            # The label must sit close to the number, otherwise a table row
            # from a different clause can win the match.
            if pos > best_pos and len(text) - pos <= 110:
                best, best_pos = subtype, pos
    if best == "initial" and (
        "relationship" in text or "criticare" in text or "critical illness" in text
    ):
        return None
    if "add-on" in text or "add on" in text:
        return None
    return best


def _number_variants(fact: str) -> List[str]:
    variants = [fact]
    match = re.fullmatch(r"(\d+)\s*months?", fact)
    if match and int(match.group(1)) % 12 == 0:
        years = int(match.group(1)) // 12
        variants += [f"{years} years", f"{years}-year", f"{years} year"]
    match = re.fullmatch(r"(\d+(?:\.\d)?)%", fact)
    if match:
        variants += [f"{match.group(1)} percent", f"{match.group(1)} per cent"]
    return list(dict.fromkeys(variants))


def _build_case(
    kind,
    subtype,
    meta,
    fact,
    window,
    insurer_name,
    policy_display,
    extra_facts=None,
    gold_fact=None,
):
    policy = policy_display
    base = {
        "category": "fact_check",
        "insurer": meta.get("insurer"),
        "line_of_business": meta.get("line_of_business"),
        "expected_source": meta.get("source"),
        "expected_page": meta.get("page"),
        "gold_fact": gold_fact or fact,
        "gold_notes": f"The wording says: {window}",
        "expect_refuse": False,
    }
    if kind == "uin":
        return {
            **base,
            "id": f"uin_{meta['insurer']}_{meta['policy_name'][:40]}",
            "question": f"What is the UIN printed on the {insurer_name} {policy} policy wording?",
            "must_include_any": [fact],
            "must_not_include": [],
        }
    if kind == "waiting":
        prompts = {
            "pre_existing": f"In {insurer_name} {policy}, what is the waiting period for pre-existing diseases?",
            "specific": f"In {insurer_name} {policy}, how long is the specified-disease waiting period?",
            "initial": f"In {insurer_name} {policy}, is there an initial waiting period for illness claims?",
        }
        variants = list(
            dict.fromkeys(
                _number_variants(fact)
                + [fact.replace(" ", "-")]
                + [value for value in (extra_facts or []) if value]
            )
        )
        return {
            **base,
            "id": f"waiting_{subtype}_{meta['insurer']}_{meta['policy_name'][:36]}",
            "question": prompts[subtype],
            "must_include_any": variants,
            "must_not_include": [],
        }
    if kind == "copay":
        return {
            **base,
            "id": f"copay_{meta['insurer']}_{meta['policy_name'][:40]}",
            "question": f"Under {insurer_name} {policy}, what co-payment applies to each and every claim?",
            "must_include_any": _number_variants(fact) + [fact.replace("%", " %")],
            "must_not_include": [],
        }
    if kind == "grace":
        return {
            **base,
            "id": f"grace_{meta['insurer']}_{meta['policy_name'][:40]}",
            "question": f"What grace period applies for renewal of {insurer_name} {policy}?",
            "must_include_any": [fact],
            "must_not_include": [],
        }
    if kind == "roomrent":
        question = f"What room rent limit does {insurer_name} {policy} state?"
        if "discount" in window.lower():
            question = (
                f"If the room rent capping discount is opted, what per-day room rent "
                f"limit applies under {insurer_name} {policy}?"
            )
        return {
            **base,
            "id": f"roomrent_{meta['insurer']}_{meta['policy_name'][:40]}",
            "question": question,
            "must_include_any": _number_variants(fact) + [fact.replace("%", " %")],
            "must_not_include": [],
        }
    if kind == "deductible":
        return {
            **base,
            "id": f"deductible_{meta['insurer']}_{meta['policy_name'][:36]}",
            "question": f"What deductible applies per claim under {insurer_name} {policy}?",
            "must_include_any": [fact, _clean_number(fact)],
            "must_not_include": [],
        }
    raise ValueError(kind)


def collect_candidates() -> List[Dict]:
    candidates: List[Dict] = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        if path.name == "ingest_index.json":
            continue
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            if meta.get("document_type") != "policy_wording":
                continue
            if meta.get("line_of_business") not in ("health", "motor"):
                continue
            policy = str(meta.get("policy_name") or "")
            source = str(meta.get("source") or "")
            if (
                not policy
                or policy in GENERIC_POLICY
                or len(policy) > 42
                or SKIP_POLICY.search(policy)
                or "star_health_comprehensive" in source
            ):
                continue
            page = meta.get("page")
            if page in (None, "unknown", 0, "0"):
                continue
            text = chunk.get("text") or ""
            if len(text) < 80 or "sample" in text.lower():
                continue
            insurer_name = (INSURERS.get(meta.get("insurer")) or {}).get(
                "name", meta.get("insurer", "")
            )

            def add(kind, subtype, fact, window, extra_facts=None, gold_fact=None):
                candidates.append(
                    _build_case(
                        kind, subtype, meta, fact, window, insurer_name,
                        _policy_display(policy, meta.get("insurer") or ""),
                        extra_facts=extra_facts,
                        gold_fact=gold_fact,
                    )
                )

            match = UIN_RE.search(text)
            if match:
                add("uin", None, match.group(1), _clause(text, match.start(), match.end()))

            match = WAIT_RE.search(text)
            if match:
                clause = _clause(text, match.start(), match.end())
                prefix = text[max(0, match.start() - 170):match.end()]
                durations = set(
                    re.findall(r"\d{1,3}\s*(?:months?|years?|days?)", clause.lower())
                )
                unit = match.group(2).lower()
                verbatim = f"{match.group(1)} {unit}"
                if unit.startswith("year"):
                    fact = f"{int(match.group(1)) * 12} months"
                elif unit.startswith("day"):
                    fact = f"{match.group(1)} days"
                else:
                    fact = f"{match.group(1)} months"
                subtype = _waiting_subtype(prefix)
                if subtype == "initial" and (
                    not re.search(r"illness|treatment", clause, re.I)
                    or re.search(r"optional cover|add-?on", clause, re.I)
                ):
                    subtype = None
                if subtype and len(durations) == 1 and "dental" not in clause.lower():
                    add(
                        "waiting", subtype, fact, clause,
                        extra_facts=[verbatim], gold_fact=verbatim,
                    )

            match = COP_RE.search(text)
            if match:
                clause = _clause(text, match.start(), match.end())
                rates = set(re.findall(r"\d{1,2}(?:\.\d)?\s*%", clause))
                if (
                    "each and every claim" in clause.lower()
                    and "dental" not in clause.lower()
                    and len(rates) == 1
                ):
                    add("copay", None, f"{match.group(1)}%", clause)

            match = ROOM_RE.search(text)
            if match:
                clause = _clause(text, match.start(), match.end())
                if (
                    "sum insured" in clause.lower()
                    and len(re.findall(r"\d+(?:\.\d)?\s*%", clause)) == 1
                ):
                    add("roomrent", None, f"{match.group(1)}%", clause)

            match = DED_RE.search(text)
            if match:
                clause = _clause(text, match.start(), match.end())
                if (
                    "each and every claim" in clause.lower()
                    and "add-on" not in clause.lower()
                    and len(re.findall(r"(?:rs\.?|inr|₹)\s?[\d,]{3,}", clause, re.I)) == 1
                ):
                    money = re.search(
                        r"(?:rs\.?|inr|₹)\s?[\d,]{3,}", match.group(0), re.I
                    ).group(0)
                    add(
                        "deductible", None, f"Rs {match.group(1)}", clause,
                        gold_fact=money,
                    )

            match = GRACE_RE.search(text)
            if match:
                clause = _clause(text, match.start(), match.end())
                days = set(re.findall(r"\d{1,3}\s*days?", clause.lower()))
                if "due date of renewal" in clause.lower() and len(days) == 1:
                    add("grace", None, f"{match.group(1)} days", clause)

    deduped = {}
    seen_uins = set()
    for case in candidates:
        if case["id"].startswith("uin_"):
            key = (case["insurer"], case["must_include_any"][0])
            if key in seen_uins:
                continue
            seen_uins.add(key)
        key = (case["expected_source"], case["id"].split("_")[0], case["must_include_any"][0])
        deduped.setdefault(key, case)
    return list(deduped.values())


def select(candidates: List[Dict], per_insurer: int, per_policy: int) -> List[Dict]:
    quotas = {
        "waiting": 7,
        "copay": 5,
        "uin": 2,
        "grace": 2,
        "roomrent": 2,
        "deductible": 1,
    }
    subtype_cap = 3
    by_insurer: Dict[str, Dict[str, List[Dict]]] = {}
    for case in candidates:
        insurer = case["insurer"]
        kind = case["id"].split("_")[0]
        bucket = by_insurer.setdefault(insurer, {}).setdefault(kind, [])
        bucket.append(case)

    selected: List[Dict] = []
    for insurer in sorted(by_insurer):
        chosen: List[Dict] = []
        policies: Dict[str, int] = {}
        subtypes: Dict[str, int] = {}
        for kind in ("waiting", "copay", "uin", "grace", "roomrent", "deductible"):
            for case in by_insurer[insurer].get(kind, []):
                if len(chosen) >= per_insurer:
                    break
                if sum(1 for c in chosen if c["id"].split("_")[0] == kind) >= quotas[kind]:
                    break
                subtype = case["id"].split("_")[1] if kind == "waiting" else ""
                if subtype and subtypes.get(subtype, 0) >= subtype_cap:
                    continue
                policy = case["expected_source"]
                if policies.get(policy, 0) >= per_policy:
                    continue
                if any(c["id"] == case["id"] for c in chosen):
                    continue
                policies[policy] = policies.get(policy, 0) + 1
                if subtype:
                    subtypes[subtype] = subtypes.get(subtype, 0) + 1
                chosen.append(case)
        selected.extend(chosen)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine grounded gold cases from indexed wording.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-insurer", type=int, default=22)
    parser.add_argument("--per-policy", type=int, default=2)
    args = parser.parse_args()

    cases = select(collect_candidates(), args.per_insurer, args.per_policy)
    payload = {
        "version": "0.2-generated",
        "notes": (
            "Generated from indexed policy wording. Facts are verbatim clause windows. "
            "Read every case against the PDF before treating it as a gold label."
        ),
        "cases": cases,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} cases -> {args.out}")
    for case in cases:
        print(f"  {case['id']:<58} {case['must_include_any'][0]}")


if __name__ == "__main__":
    main()

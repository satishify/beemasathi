"""Discovery-driven scraper for official Indian insurance policy documents.

Collects health and motor policy documents (wordings, CIS, prospectus, brochures)
from insurer-owned domains only. Documents are saved for manual review and are
never auto-ingested into the RAG store.

Pipeline per insurer:
    1. Seed known document hubs and sitemap-discovered hub candidates.
    2. Crawl hubs (bounded depth) and harvest PDF links.
    3. Classify each link by line of business and document type.
    4. Keep the newest revision per policy, then download and validate.
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "data" / "raw_policies"
MANIFEST_PATH = SAVE_DIR / "manifest.json"
REPORT_PATH = SAVE_DIR / "discovery_report.json"

HEADERS = {
    "User-Agent": "BeemaSathiResearchBot/0.2 (local policy research; contact: local-admin)",
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-IN,en;q=0.8",
}

MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
REQUEST_TIMEOUT = 40
HOST_DELAY_SECONDS = 1.5
DISALLOW_ALL = ["User-agent: *", "Disallow: /"]


# ============================================================================
# INSURER REGISTRY
# ============================================================================
# hosts:    every domain that may serve this insurer's documents, including
#           asset/CDN domains. Anything outside this set is treated as
#           third-party and skipped unless --include-third-party is passed.
# hubs:     pages known to list downloadable documents.
# sitemaps: used to auto-discover new hubs when a hub URL changes.


@dataclass
class Insurer:
    name: str
    hosts: set
    hubs: list = field(default_factory=list)
    sitemaps: list = field(default_factory=list)
    lines: set = field(default_factory=lambda: {"health", "motor"})
    note: str = ""


INSURER_REGISTRY = {
    "tata_aig": Insurer(
        name="Tata AIG",
        hosts={"tataaig.com"},
        hubs=["https://www.tataaig.com/downloads"],
        sitemaps=["https://www.tataaig.com/sitemap.xml"],
    ),
    "cholamandalam": Insurer(
        name="Cholamandalam MS",
        hosts={"cholainsurance.com"},
        hubs=["https://www.cholainsurance.com/downloads"],
        sitemaps=["https://www.cholainsurance.com/sitemap.xml"],
    ),
    "digit": Insurer(
        name="Digit Insurance",
        hosts={"godigit.com"},
        hubs=["https://www.godigit.com/downloads"],
        sitemaps=[
            "https://www.godigit.com/sitemap-health.xml",
            "https://www.godigit.com/sitemap-motor.xml",
            "https://www.godigit.com/sitemap-homepage.xml",
        ],
    ),
    "acko": Insurer(
        name="Acko General",
        hosts={"acko.com", "ackoassets.com"},
        hubs=["https://www.acko.com/download/"],
        sitemaps=["https://www.acko.com/sitemap.xml"],
    ),
    "zuno": Insurer(
        name="Zuno General (Edelweiss)",
        hosts={"hizuno.com", "edelweissinsurance.com"},
        hubs=["https://www.hizuno.com/downloads"],
        sitemaps=["https://www.hizuno.com/sitemap.xml"],
    ),
    "hdfc_ergo": Insurer(
        name="HDFC Ergo",
        hosts={"hdfcergo.com"},
        hubs=[
            "https://www.hdfcergo.com/download/others",
            "https://www.hdfcergo.com/download/health-insurance",
            "https://www.hdfcergo.com/download/motor-insurance",
        ],
        sitemaps=["https://www.hdfcergo.com/sitemap.xml"],
    ),
    "bajaj_allianz": Insurer(
        name="Bajaj Allianz General",
        hosts={
            "bajajallianz.com",
            "bajajgeneralinsurance.com",
            "bajajfinserv.in",
        },
        hubs=[
            "https://www.bajajgeneralinsurance.com/policy-wording-and-brochure.html",
            "https://www.bajajgeneralinsurance.com/health-insurance-plans/health-insurance-documents.html",
            "https://www.bajajgeneralinsurance.com/motor-insurance/motor-insurance-documents.html",
        ],
        sitemaps=["https://www.bajajgeneralinsurance.com/general-insurance-sitemap.xml"],
    ),
    "raheja_qbe": Insurer(
        name="Raheja QBE",
        hosts={"rahejaqbe.com"},
        hubs=["https://www.rahejaqbe.com/downloads"],
    ),
    "aditya_birla": Insurer(
        name="Aditya Birla Health",
        hosts={"adityabirlacapital.com", "abhicl.in"},
        hubs=[
            "https://www.adityabirlacapital.com/healthinsurance/customer-service/downloads",
            "https://www.adityabirlacapital.com/healthinsurance/product-information",
        ],
        lines={"health"},
    ),
    "max_bupa": Insurer(
        name="Niva Bupa",
        hosts={"nivabupa.com"},
        hubs=[
            "https://www.nivabupa.com/health-insurance-plans.html",
            "https://transactions.nivabupa.com/pages/doc/policy_wording/",
        ],
        sitemaps=["https://www.nivabupa.com/sitemap.xml"],
        lines={"health"},
        note="Documents sit behind a transactions subdomain; hub discovery is unreliable.",
    ),
    "zurich_kotak": Insurer(
        name="Zurich Kotak General",
        hosts={"zurichkotak.com"},
        hubs=["https://www.zurichkotak.com/documents/customer-support/downloads"],
        sitemaps=["https://www.zurichkotak.com/sitemap.xml"],
        note="Download list is JavaScript-rendered; static HTML exposes no PDF links.",
    ),
    "manipal_cigna": Insurer(
        name="ManipalCigna Health",
        hosts={"manipalcigna.com"},
        hubs=[
            "https://www.manipalcigna.com/downloads/products",
            "https://www.manipalcigna.com/downloads",
        ],
        lines={"health"},
    ),
    "star_health": Insurer(
        name="Star Health",
        hosts={"starhealth.in"},
        hubs=["https://www.starhealth.in/downloads"],
        lines={"health"},
        note="Origin frequently times out for non-browser clients.",
    ),
    "care_health": Insurer(
        name="Care Health",
        hosts={"careinsurance.com"},
        hubs=["https://www.careinsurance.com/policy-documents.html"],
        lines={"health"},
        note="Edge WAF returns 403 for automated clients.",
    ),
    "icici_lombard": Insurer(
        name="ICICI Lombard",
        hosts={"icicilombard.com"},
        hubs=["https://www.icicilombard.com/downloads"],
        note="Edge WAF returns 403 for automated clients.",
    ),
    "irdai": Insurer(
        name="IRDAI",
        hosts={"irdai.gov.in"},
        hubs=["https://irdai.gov.in/consolidated-gazette-notified-regulations"],
        lines={"regulation"},
        note="robots.txt disallows all crawling; collect manually instead.",
    ),
}


# ============================================================================
# CLASSIFICATION
# ============================================================================
# Word-boundary matching matters here: "medicare" contains "car", so a naive
# substring match would label every health document as motor.

LINE_PATTERNS = (
    ("motor", r"\b(motor|vehicles?|private[\s_-]?cars?|two[\s_-]?wheelers?|bikes?|"
              r"auto[\s_-]?secure|commercial[\s_-]?vehicles?|goods[\s_-]?carrying|"
              r"passenger[\s_-]?carrying|tractors?|taxi|pay[\s_-]?as[\s_-]?you[\s_-]?drive|"
              r"own[\s_-]?damage|third[\s_-]?party[\s_-]?liability|cng|"
              r"miscellaneous[\s_-]?vehicles?)\b"),
    ("health", r"\b(health|medi[\s_-]?care|medicare|mediclaim|hospital|hospitalisation|"
               r"hospitalization|critical[\s_-]?illness|arogya|activ|reassure|"
               r"super[\s_-]?top[\s_-]?up|opd|maternity|personal[\s_-]?health)\b"),
    ("travel", r"\b(travel|overseas|student[\s_-]?travel|schengen|trip)\b"),
    ("personal_accident", r"\b(personal[\s_-]?accident|\bpa\b|janata[\s_-]?personal)\b"),
    ("home", r"\b(home|householder|bharat[\s_-]?griha|property|fire|shopkeeper|office[\s_-]?package)\b"),
    ("commercial", r"\b(marine|liability|engineering|workmen|employee[\s_-]?compensation|"
                   r"contractor|erection|burglary|money[\s_-]?insurance|fidelity|crop|cattle|pet)\b"),
)

DOC_TYPE_PATTERNS = (
    ("cis", r"\b(cis|customer[\s_-]?information[\s_-]?sheet)\b"),
    ("policy_wording", r"\b(policy[\s_-]?wording|policy[\s_-]?wordings|wording|wordings|"
                       r"policy[\s_-]?document|policy[\s_-]?terms|terms[\s_-]?and[\s_-]?conditions|"
                       r"\bpw\b|policy[\s_-]?clause)\b"),
    ("prospectus", r"\b(prospectus)\b"),
    ("brochure", r"\b(brochure|leaflet|sales[\s_-]?literature|product[\s_-]?flyer)\b"),
    ("endorsement", r"\b(endorsement|add[\s_-]?on|addon|rider)\b"),
    ("claim_form", r"\b(claim[\s_-]?form|claim[\s_-]?intimation|pre[\s_-]?auth)\b"),
    ("proposal_form", r"\b(proposal[\s_-]?form|application[\s_-]?form|enrolment[\s_-]?form)\b"),
)

# Documents that describe what a policy actually covers. Forms and reports are
# collected only when explicitly requested.
CONTRACTUAL_DOC_TYPES = {"policy_wording", "cis", "prospectus", "brochure", "endorsement"}

STALE_MARKERS = re.compile(
    r"\b(withdrawn|discontinued|obsolete|archive[d]?|superseded|expired|"
    r"old[\s_-]?version|previous[\s_-]?version|deprecated|closed[\s_-]?for[\s_-]?sale)\b",
    re.I,
)

# Indian product UIN formats seen in the wild:
#   TATHLIP21045V032021, IRDAN123RP0001V01202324, IRDAI/HLT/.../V.I/1/2020-21
UIN_PATTERNS = (
    re.compile(r"\b[A-Z]{3,8}[A-Z0-9]{2,24}V\d{2,9}\b"),
    re.compile(r"\bIRDA[I]?[/\\][A-Z]{2,5}[/\\][A-Z0-9/\\.\-]{4,40}\b", re.I),
)

# Anchor text that carries no product meaning, so it must not drive grouping.
GENERIC_LABEL = re.compile(
    r"^(downloads?|click\s*here|view(\s*more)?|read\s*more|know\s*more|"
    r"pdf|document|here|link|details?|policy|wording|brochure)$",
    re.I,
)

HUB_HINT = re.compile(
    r"download|policy-?wording|policy-?document|prospectus|brochure|"
    r"product-?information|customer-?service|customer-?care|documents?",
    re.I,
)
HUB_NOISE = re.compile(
    r"/blog|/blogs|articles?/|web-story|/news|/careers|visa|-tips|"
    r"how-to-|glossary|calculator|/claim/|login|sitemap",
    re.I,
)


def _normalize(text: str) -> str:
    """Lowercase and normalize separators so word-boundary rules behave."""
    text = unquote(text or "")
    text = re.sub(r"[%_\-+]+", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def classify_line(url: str, label: str = "") -> str:
    haystack = f"{_normalize(label)} {_normalize(url)}"
    for line, pattern in LINE_PATTERNS:
        if re.search(pattern, haystack, re.I):
            return line
    return "other"


def classify_doc_type(url: str, label: str = "") -> str:
    haystack = f"{_normalize(label)} {_normalize(url)}"
    for doc_type, pattern in DOC_TYPE_PATTERNS:
        if re.search(pattern, haystack, re.I):
            return doc_type
    return "other"


def extract_uin(text: str) -> str:
    for pattern in UIN_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return match.group(0)
    return ""


def revision_score(url: str, label: str = "") -> tuple:
    """Rank candidate revisions: newer year and higher revision win."""
    # Separators must become spaces first: "_R2_2024_" has no word boundary
    # before R or 2024 because underscore counts as a word character.
    haystack = _normalize(f"{label} {url}")
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", haystack)]
    revisions = [int(r) for r in re.findall(r"\b[Rr](\d{1,2})\b", haystack)]
    versions = [int(v) for v in re.findall(r"\b[Vv](?:er)?\.?\s?(\d{1,2})\b", haystack)]
    return (
        max(years) if years else 0,
        max(revisions) if revisions else 0,
        max(versions) if versions else 0,
    )


def document_stem(url: str) -> str:
    """Human-readable document name from a URL path.

    Some portals append an opaque id after the real filename, e.g.
    /documents/20121/o/Chola Motor Warranty (Policy Wording) - UIN.pdf/b49ac445...
    so the last path segment is not the name.
    """
    segments = [unquote(segment) for segment in urlparse(url).path.split("/") if segment]
    for segment in reversed(segments):
        if segment.lower().endswith(".pdf"):
            return segment[:-4]
    return Path(segments[-1]).stem if segments else ""


def policy_slug(url: str, label: str = "") -> str:
    """Stable grouping key for one product, shared across its revisions."""
    parts = []
    if label and not GENERIC_LABEL.match(label.strip()):
        parts.append(label)
    parts.append(document_stem(url))
    text = _normalize(" ".join(parts))

    for pattern in UIN_PATTERNS:
        text = pattern.sub(" ", text.upper()).lower()
    for _, pattern in DOC_TYPE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"\bpdf\b", " ", text)  # some CDNs embed ".pdf" mid-filename
    text = re.sub(r"\b[a-z]\d{3,}v\d{2,9}\b", " ", text)  # add-on codes, e.g. A0026V01202324
    text = re.sub(r"\b(20\d{2})\b", " ", text)
    text = re.sub(r"\b[rv]\d{1,2}\b", " ", text)
    text = re.sub(r"\b[0-9a-f]{8,}\b", " ", text)
    text = re.sub(r"\b\d{1,2}\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    # Anchor text and filename usually repeat the product name; keep first use.
    seen = set()
    text = "_".join(
        token for token in text.split("_") if not (token in seen or seen.add(token))
    )

    # A meaningless slug would merge unrelated products into one group.
    if len(text) < 3:
        return f"doc_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:10]}"
    # Plain truncation would merge long product names that differ only at the
    # end, so keep a digest of the full name.
    if len(text) > 70:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        return f"{text[:62].rstrip('_')}_{digest}"
    return text


def safe_filename(value: str, suffix: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return f"{stem[:120] or 'document'}{suffix}"


# ============================================================================
# HTTP SESSION, ROBOTS, RATE LIMITING
# ============================================================================


class Fetcher:
    """Polite HTTP client: per-host delay, robots.txt cache, shared session."""

    def __init__(self, respect_robots: bool = True):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.respect_robots = respect_robots
        self._robots = {}
        self._last_hit = defaultdict(float)
        self._lock = threading.Lock()

    def _throttle(self, url: str) -> None:
        host = urlparse(url).netloc
        with self._lock:
            wait = HOST_DELAY_SECONDS - (time.monotonic() - self._last_hit[host])
            if wait > 0:
                time.sleep(wait)
            self._last_hit[host] = time.monotonic()

    def robots_allows(self, url: str) -> bool:
        """Fail closed: if robots.txt cannot be read, do not fetch."""
        if not self.respect_robots:
            return True
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        with self._lock:
            if robots_url in self._robots:
                return self._robots[robots_url].can_fetch(HEADERS["User-Agent"], url)
        parser = RobotFileParser()
        try:
            response = self.session.get(robots_url, timeout=20)
            if 400 <= response.status_code < 500:
                # RFC 9309: an unavailable robots.txt imposes no restrictions.
                # Asset CDNs often answer 403 here even though the main site
                # serves a normal robots.txt that permits the document paths.
                parser.parse([])
            elif response.status_code >= 500:
                parser.parse(DISALLOW_ALL)
            else:
                parser.parse(response.text.splitlines())
        except requests.RequestException:
            parser.parse(DISALLOW_ALL)
        with self._lock:
            self._robots[robots_url] = parser
        return parser.can_fetch(HEADERS["User-Agent"], url)

    def get(self, url: str, stream: bool = False, timeout: int = REQUEST_TIMEOUT):
        self._throttle(url)
        return self.session.get(url, timeout=timeout, stream=stream, allow_redirects=True)


def is_official_url(url: str, insurer_key: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    allowed = INSURER_REGISTRY[insurer_key].hosts if insurer_key in INSURER_REGISTRY else set()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed)


# ============================================================================
# DISCOVERY
# ============================================================================


def discover_hubs_from_sitemap(fetcher: Fetcher, insurer_key: str, limit: int = 12) -> tuple:
    """Return (hub_candidates, direct_pdf_urls) parsed from declared sitemaps."""
    insurer = INSURER_REGISTRY[insurer_key]
    hubs, pdfs = [], []
    for sitemap_url in insurer.sitemaps:
        if not fetcher.robots_allows(sitemap_url):
            continue
        try:
            response = fetcher.get(sitemap_url)
            if response.status_code != 200:
                continue
            locations = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text, re.I)
        except requests.RequestException:
            continue

        # A sitemap index points at more sitemaps; follow one level.
        if "<sitemapindex" in response.text[:2000].lower():
            for nested in locations[:8]:
                try:
                    nested_response = fetcher.get(nested)
                    locations += re.findall(
                        r"<loc>\s*([^<\s]+)\s*</loc>", nested_response.text, re.I
                    )
                except requests.RequestException:
                    continue

        for location in locations:
            if not is_official_url(location, insurer_key):
                continue
            if location.lower().endswith(".pdf"):
                pdfs.append(location)
            elif HUB_HINT.search(location) and not HUB_NOISE.search(location):
                hubs.append(location)

    unique_hubs = list(dict.fromkeys(hubs))[:limit]
    return unique_hubs, list(dict.fromkeys(pdfs))


def harvest_page(fetcher: Fetcher, url: str, insurer_key: str) -> tuple:
    """Return (pdf_candidates, child_hubs, status) for one page."""
    if not fetcher.robots_allows(url):
        return [], [], "robots_denied"
    try:
        response = fetcher.get(url)
    except requests.RequestException as exc:
        return [], [], f"error:{type(exc).__name__}"
    if response.status_code != 200:
        return [], [], f"http_{response.status_code}"
    if "html" not in response.headers.get("Content-Type", ""):
        return [], [], "not_html"

    soup = BeautifulSoup(response.text, "html.parser")
    pdfs, children = [], []
    for anchor in soup.find_all("a", href=True):
        target = urljoin(response.url, anchor["href"].strip())
        label = anchor.get_text(" ", strip=True)[:180]
        if ".pdf" in target.lower():
            pdfs.append({"url": target.split("#")[0], "label": label})
        elif HUB_HINT.search(target) and not HUB_NOISE.search(target):
            children.append(target.split("#")[0])
    return pdfs, list(dict.fromkeys(children)), "ok"


def discover_documents(
    fetcher: Fetcher,
    insurer_key: str,
    depth: int = 1,
    use_sitemap: bool = True,
) -> dict:
    """Crawl an insurer's hubs and return classified PDF candidates."""
    insurer = INSURER_REGISTRY[insurer_key]
    queue = list(insurer.hubs)
    sitemap_pdfs = []
    if use_sitemap:
        sitemap_hubs, sitemap_pdfs = discover_hubs_from_sitemap(fetcher, insurer_key)
        queue += sitemap_hubs

    seen_pages, page_status, candidates = set(), {}, {}

    for candidate in sitemap_pdfs:
        candidates[candidate] = {"url": candidate, "label": ""}

    current_depth = 0
    while queue and current_depth <= depth:
        next_queue = []
        for page in queue:
            if page in seen_pages:
                continue
            seen_pages.add(page)
            pdfs, children, status = harvest_page(fetcher, page, insurer_key)
            page_status[page] = f"{status} (pdfs={len(pdfs)})"
            for item in pdfs:
                candidates.setdefault(item["url"], item)
            next_queue += children
        queue = next_queue
        current_depth += 1

    documents = []
    for item in candidates.values():
        if not is_official_url(item["url"], insurer_key):
            continue
        label = item.get("label", "")
        documents.append(
            {
                "insurer": insurer_key,
                "insurer_name": insurer.name,
                "url": item["url"],
                "label": label,
                "line": classify_line(item["url"], label),
                "doc_type": classify_doc_type(item["url"], label),
                "policy": policy_slug(item["url"], label),
                "uin": extract_uin(f"{label} {unquote(item['url'])}"),
                "revision": revision_score(item["url"], label),
                "stale_marker": bool(STALE_MARKERS.search(f"{label} {unquote(item['url'])}")),
            }
        )

    return {"insurer": insurer_key, "documents": documents, "pages": page_status}


def select_latest(documents: list, keep_all: bool = False) -> list:
    """Keep the newest revision per (line, policy, doc_type) group."""
    active = [doc for doc in documents if not doc["stale_marker"]]
    if keep_all:
        return active
    groups = defaultdict(list)
    for doc in active:
        groups[(doc["insurer"], doc["line"], doc["policy"], doc["doc_type"])].append(doc)
    latest = []
    for group in groups.values():
        group.sort(key=lambda doc: (doc["revision"], len(doc["url"])), reverse=True)
        latest.append(group[0])
    return latest


# ============================================================================
# DOWNLOAD
# ============================================================================


def download_document(fetcher: Fetcher, doc: dict, allow_third_party: bool = False) -> dict:
    """Stream a PDF to disk with validation, hashing, and atomic rename."""
    url, insurer_key = doc["url"], doc["insurer"]
    official = is_official_url(url, insurer_key)
    if not official and not allow_third_party:
        return {**doc, "status": "skipped_non_official"}
    if not fetcher.robots_allows(url):
        return {**doc, "status": "skipped_robots"}

    temp_path = None
    try:
        response = fetcher.get(url, stream=True, timeout=90)
        response.raise_for_status()
        if not is_official_url(response.url, insurer_key) and not allow_third_party:
            return {**doc, "status": "skipped_redirect_off_domain"}

        declared = int(response.headers.get("Content-Length") or 0)
        if declared > MAX_DOWNLOAD_BYTES:
            return {**doc, "status": "skipped_too_large"}

        target_dir = SAVE_DIR / insurer_key
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{doc['line']}__{doc['policy']}__{doc['doc_type']}"
        save_path = target_dir / safe_filename(stem, ".pdf")
        temp_path = save_path.with_suffix(".pdf.part")

        digest = hashlib.sha256()
        total, head = 0, b""
        with open(temp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                if not head:
                    head = chunk[:5]
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("exceeds size limit")
                digest.update(chunk)
                handle.write(chunk)

        if head != b"%PDF-":
            raise ValueError(
                f"not a PDF (Content-Type: {response.headers.get('Content-Type', 'unknown')})"
            )

        temp_path.replace(save_path)
        temp_path = None

        last_modified = response.headers.get("Last-Modified")
        modified_iso = ""
        if last_modified:
            try:
                modified_iso = parsedate_to_datetime(last_modified).isoformat()
            except (TypeError, ValueError):
                modified_iso = ""

        return {
            **doc,
            "status": "downloaded",
            "path": str(save_path.relative_to(BASE_DIR)),
            "sha256": digest.hexdigest(),
            "size_bytes": total,
            "official_source": official,
            "source_last_modified": modified_iso,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    except (requests.RequestException, ValueError, OSError) as exc:
        return {**doc, "status": f"failed:{type(exc).__name__}", "error": str(exc)[:200]}
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def write_sidecar(record: dict) -> None:
    """Write metadata that rag/sync_knowledge.py reads when indexing."""
    pdf_path = BASE_DIR / record["path"]
    # sync_knowledge accepts policy_wording, brochure, regulation. Summary
    # documents map to brochure so RAG keeps warning that they are not the
    # binding contract.
    summary_types = {"brochure", "prospectus", "cis"}
    document_type = "brochure" if record["doc_type"] in summary_types else "policy_wording"
    if record["line"] == "regulation":
        document_type = "regulation"
    sidecar = {
        "insurer": record["insurer"],
        "policy_name": record["policy"],
        "document_type": document_type,
        "line_of_business": record["line"],
        "source_doc_type": record["doc_type"],
        "source_url": record["url"],
        "uin": record.get("uin", ""),
    }
    pdf_path.with_name(pdf_path.stem + ".meta.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )


# ============================================================================
# MANIFEST AND REPORT
# ============================================================================


def update_manifest(records: list) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": [], "runs": []}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print("  Warning: manifest unreadable, rebuilding.")

    merged = {source.get("sha256") or source.get("url"): source
              for source in manifest.get("sources", [])}
    for record in records:
        merged[record.get("sha256") or record["url"]] = record

    manifest.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "review_required": True,
            "auto_ingested": False,
            "sources": list(merged.values()),
            "runs": manifest.get("runs", [])
            + [
                {
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "downloaded": len(records),
                }
            ],
        }
    )
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(discovery: dict, downloads: list, blocked: dict) -> None:
    print("\n" + "=" * 72)
    print("DISCOVERY SUMMARY")
    print("=" * 72)
    print(f"{'insurer':<18}{'found':>7}{'selected':>10}{'downloaded':>12}  note")
    for insurer_key, data in sorted(discovery.items()):
        done = sum(
            1
            for record in downloads
            if record["insurer"] == insurer_key and record["status"] == "downloaded"
        )
        note = blocked.get(insurer_key, "")
        print(
            f"{insurer_key:<18}{data['found']:>7}{data['selected']:>10}{done:>12}  {note[:30]}"
        )

    by_line = defaultdict(int)
    by_type = defaultdict(int)
    for record in downloads:
        if record["status"] == "downloaded":
            by_line[record["line"]] += 1
            by_type[record["doc_type"]] += 1
    if by_line:
        print("\nDownloaded by line:", dict(sorted(by_line.items())))
        print("Downloaded by type:", dict(sorted(by_type.items())))

    failures = [record for record in downloads if record["status"] != "downloaded"]
    if failures:
        print(f"\nSkipped or failed: {len(failures)}")
        reasons = defaultdict(int)
        for record in failures:
            reasons[record["status"]] += 1
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1])[:8]:
            print(f"  {count:>4}  {reason}")

    if blocked:
        print("\nInsurers needing manual collection:")
        for insurer_key, reason in sorted(blocked.items()):
            print(f"  - {INSURER_REGISTRY[insurer_key].name}: {reason}")


# ============================================================================
# RUNNER
# ============================================================================


def run(
    insurer_keys=None,
    lines=("health", "motor"),
    doc_types=None,
    limit=None,
    per_insurer_limit=None,
    depth=1,
    workers=4,
    dry_run=False,
    keep_all=False,
    allow_third_party=False,
    respect_robots=True,
    include_forms=False,
    use_sitemap=True,
):
    fetcher = Fetcher(respect_robots=respect_robots)
    selected = list(insurer_keys or INSURER_REGISTRY.keys())
    wanted_lines = set(lines) if lines else None
    wanted_types = set(doc_types) if doc_types else set(CONTRACTUAL_DOC_TYPES)
    if include_forms:
        wanted_types |= {"claim_form", "proposal_form", "other"}

    print("=" * 72)
    print("BeemaSathi Policy Document Scraper")
    print("=" * 72)
    print(f"Insurers : {', '.join(selected)}")
    print(f"Lines    : {', '.join(sorted(wanted_lines)) if wanted_lines else 'all'}")
    print(f"Doc types: {', '.join(sorted(wanted_types))}")
    print(f"Mode     : {'dry-run (no downloads)' if dry_run else 'download'}; "
          f"official-only={not allow_third_party}; robots={respect_robots}")

    discovery_stats, blocked, to_download = {}, {}, []

    print("\n--- Phase 1: discovery ---")
    with ThreadPoolExecutor(max_workers=min(workers, len(selected))) as pool:
        futures = {
            pool.submit(discover_documents, fetcher, key, depth, use_sitemap): key
            for key in selected
        }
        for future in as_completed(futures):
            insurer_key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                blocked[insurer_key] = f"discovery error: {type(exc).__name__}"
                discovery_stats[insurer_key] = {"found": 0, "selected": 0}
                continue

            documents = result["documents"]
            filtered = [
                doc
                for doc in documents
                if (not wanted_lines or doc["line"] in wanted_lines)
                and doc["doc_type"] in wanted_types
            ]
            chosen = select_latest(filtered, keep_all=keep_all)
            if per_insurer_limit:
                chosen = chosen[:per_insurer_limit]

            discovery_stats[insurer_key] = {"found": len(documents), "selected": len(chosen)}
            to_download += chosen

            if not documents:
                note = INSURER_REGISTRY[insurer_key].note or "no documents found"
                statuses = {
                    status.split(" ")[0]
                    for status in result["pages"].values()
                    if not status.startswith("ok")
                }
                if statuses:
                    note = f"{note} [{', '.join(sorted(statuses))}]"
                blocked[insurer_key] = note
            elif not chosen:
                blocked[insurer_key] = (
                    f"{len(documents)} files found but none are policy documents; "
                    "the hub lists service forms only"
                )
            print(
                f"  {insurer_key:<18} found={len(documents):<5} "
                f"match={len(filtered):<5} latest={len(chosen)}"
            )

    if limit:
        to_download = to_download[:limit]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "stats": discovery_stats,
                "blocked": blocked,
                "candidates": [
                    {key: value for key, value in doc.items() if key != "revision"}
                    for doc in to_download
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDiscovery report: {REPORT_PATH.relative_to(BASE_DIR)}")

    downloads = []
    if dry_run:
        print("\n--- Phase 2: skipped (dry run) ---")
        for doc in to_download[:40]:
            print(f"  would fetch [{doc['line']}/{doc['doc_type']}] {doc['url'][:95]}")
        if len(to_download) > 40:
            print(f"  ... and {len(to_download) - 40} more")
    else:
        print(f"\n--- Phase 2: downloading {len(to_download)} documents ---")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_document, fetcher, doc, allow_third_party)
                for doc in to_download
            ]
            for index, future in enumerate(as_completed(futures), 1):
                record = future.result()
                downloads.append(record)
                if record["status"] == "downloaded":
                    write_sidecar(record)
                    size_kb = record["size_bytes"] / 1024
                    print(f"  [{index}/{len(to_download)}] ok {size_kb:7.0f} KB  {record['path']}")
                else:
                    print(f"  [{index}/{len(to_download)}] {record['status']}  {record['url'][:80]}")

        saved = [record for record in downloads if record["status"] == "downloaded"]
        if saved:
            update_manifest(saved)
            print(f"\nManifest updated: {MANIFEST_PATH.relative_to(BASE_DIR)}")

    print_summary(discovery_stats, downloads, blocked)
    if not dry_run:
        print("\nNext step: review the files, then index them with:")
        print("  ./appctl.sh index")
    return downloads


def print_manual_guide():
    print(
        """
Manual collection guide (for WAF-blocked or JavaScript-only sites)
------------------------------------------------------------------
Blocked today: ICICI Lombard, Care Health, Star Health (bot protection),
Zurich Kotak (JavaScript-rendered list), IRDAI (robots.txt disallows all).

1. Open the insurer's download or policy-wording page in your browser.
2. Save the policy wording PDF yourself.
3. Place it in data/raw_policies/<insurer_key>/ using a clear name, e.g.
   data/raw_policies/icici_lombard/health__elevate__policy_wording.pdf
4. Optionally add <same-name>.meta.json:
   {"insurer": "icici_lombard", "policy_name": "elevate",
    "document_type": "policy_wording"}
5. Index it:  ./appctl.sh index

Only use insurer-owned domains. Third-party PDF mirrors are not valid
evidence for a claim decision.
"""
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Collect official Indian health and motor policy documents for review.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scrapers/hybrid_scraper.py --dry-run
  python scrapers/hybrid_scraper.py --insurer tata_aig --line health
  python scrapers/hybrid_scraper.py --line motor --limit 25
  python scrapers/hybrid_scraper.py --doc-type policy_wording --doc-type cis
  python scrapers/hybrid_scraper.py --list-insurers
""",
    )
    parser.add_argument(
        "--insurer", action="append", choices=sorted(INSURER_REGISTRY),
        help="Insurer key; repeat for several. Default: all.",
    )
    parser.add_argument(
        "--line", action="append",
        choices=["health", "motor", "travel", "personal_accident", "home",
                 "commercial", "regulation", "other", "all"],
        help="Line of business. Default: health and motor.",
    )
    parser.add_argument(
        "--doc-type", action="append",
        choices=sorted({name for name, _ in DOC_TYPE_PATTERNS} | {"other"}),
        help="Document types to keep. Default: policy wording, CIS, prospectus, brochure, endorsement.",
    )
    parser.add_argument("--limit", type=int, help="Maximum downloads for the whole run.")
    parser.add_argument("--per-insurer-limit", type=int, help="Maximum downloads per insurer.")
    parser.add_argument("--depth", type=int, default=1, help="Hub crawl depth (default 1).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4).")
    parser.add_argument("--dry-run", action="store_true", help="Discover and report only.")
    parser.add_argument("--keep-all", action="store_true",
                        help="Keep every revision instead of only the newest.")
    parser.add_argument("--include-forms", action="store_true",
                        help="Also collect claim and proposal forms.")
    parser.add_argument("--no-sitemap", action="store_true", help="Skip sitemap hub discovery.")
    parser.add_argument("--include-third-party", action="store_true",
                        help="Allow non-insurer domains (never authoritative).")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Bypass robots.txt. Not recommended.")
    parser.add_argument("--list-insurers", action="store_true", help="Show the registry and exit.")
    parser.add_argument("--manual-guide", action="store_true", help="Print the manual fallback guide.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.list_insurers:
        print(f"{'key':<18}{'lines':<22}{'hubs':>5}  name / note")
        for key, insurer in sorted(INSURER_REGISTRY.items()):
            lines = ",".join(sorted(insurer.lines))
            print(f"{key:<18}{lines:<22}{len(insurer.hubs):>5}  {insurer.name}"
                  f"{' — ' + insurer.note if insurer.note else ''}")
        return

    if args.manual_guide:
        print_manual_guide()
        return

    lines = None if args.line and "all" in args.line else (args.line or ["health", "motor"])
    run(
        insurer_keys=args.insurer,
        lines=lines,
        doc_types=args.doc_type,
        limit=args.limit,
        per_insurer_limit=args.per_insurer_limit,
        depth=args.depth,
        workers=args.workers,
        dry_run=args.dry_run,
        keep_all=args.keep_all,
        allow_third_party=args.include_third_party,
        respect_robots=not args.ignore_robots,
        include_forms=args.include_forms,
        use_sitemap=not args.no_sitemap,
    )


if __name__ == "__main__":
    main()

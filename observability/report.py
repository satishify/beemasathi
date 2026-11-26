"""Render docs/PROJECT.md to the submission PDF with headless Chrome."""

import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE = BASE_DIR / "docs" / "PROJECT.md"
OUTPUT = BASE_DIR / "docs" / "BeemaSathi_Technical_Report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

REPO = "https://github.com/satishify/beemasathi"
SITE = "https://satishify.github.io/beemasathi"
DASHBOARD = SITE + "/trace_dashboard.html"
VIDEO = "https://drive.google.com/file/d/11ixHKGa3MAflYDsbrNPQeQBucvjYm0uF/view?usp=share_link"
VIDEO_LABEL = "Ten-minute demo (Google Drive)"

CSS = """
@page { size: A4; margin: 15mm 14mm 14mm; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font: 9.6pt/1.34 -apple-system, "Helvetica Neue", Arial, sans-serif;
       color: #1d2733; margin: 0; }
p { margin: 0 0 4.8pt; }
h1, h2, h3, h4 { font-family: "Helvetica Neue", Arial, sans-serif; color: #0d3b40;
       break-after: avoid; }
h1 { font-size: 19pt; margin: 0 0 10pt; padding-bottom: 5pt;
     border-bottom: 2.5pt solid #0f766e; letter-spacing: -0.01em; }
h2 { font-size: 12.2pt; margin: 10pt 0 4pt; }
h3 { font-size: 10.4pt; margin: 9pt 0 3pt; }
ul, ol { margin: 0 0 6pt; padding-left: 15pt; }
li { margin: 0 0 2.5pt; }
strong { color: #0d3b40; }
code { font: 8.4pt/1.35 ui-monospace, "SF Mono", Menlo, monospace;
       background: #eef3f4; border-radius: 2.5pt; padding: 0.5pt 2.5pt; }
pre { background: #f4f7f8; border: 0.5pt solid #d8e2e4; border-radius: 4pt;
      padding: 7pt 9pt; margin: 0 0 8pt; overflow: hidden; }
pre code { background: none; padding: 0; font-size: 7.6pt; }
table { width: 100%; border-collapse: collapse; margin: 2pt 0 8pt;
        font-size: 8.2pt; line-height: 1.3; }
th { background: #eaf2f3; color: #0d3b40; text-align: left; font-weight: 600; }
th, td { border: 0.5pt solid #ccd8da; padding: 3pt 4.5pt; vertical-align: top;
         break-inside: avoid; }
a { color: #0f766e; text-decoration: none; }
hr { border: 0; border-top: 0.5pt solid #d8e2e4; margin: 9pt 0; }
.cover { height: 252mm; display: flex; flex-direction: column; }
.cover .kicker { font-size: 9pt; letter-spacing: 0.14em; text-transform: uppercase;
                 color: #0f766e; font-weight: 600; margin-top: 8mm; }
.cover h1.title { font-size: 30pt; line-height: 1.12; border: 0; margin: 4mm 0 3mm;
                  letter-spacing: -0.015em; }
.cover .subtitle { font-size: 13pt; color: #33555b; margin: 0 0 8mm; line-height: 1.4; }
.cover .meta { font-size: 9.5pt; color: #4a6570; margin-bottom: 10mm; }
.cover .meta b { color: #1d2733; font-weight: 600; }
.abstract { background: #f2f7f7; border-left: 3pt solid #0f766e; border-radius: 3pt;
            padding: 8pt 11pt; font-size: 9.8pt; line-height: 1.5; margin-bottom: 10mm; }
.links { border: 0.75pt solid #ccd8da; border-radius: 5pt; padding: 8pt 11pt;
         font-size: 9.2pt; }
.links h2 { margin: 0 0 5pt; font-size: 10.5pt; }
.links table { margin: 0; font-size: 8.8pt; }
.links th, .links td { border: 0; padding: 2.2pt 4pt 2.2pt 0; }
.links td:first-child { color: #4a6570; width: 30%; }
.cover .foot { margin-top: auto; font-size: 8.5pt; color: #6b828a; }
.page-break { break-after: page; }
.toc td { border: 0; padding: 1.4pt 4pt 1.4pt 0; font-size: 8.9pt; }
.toc td:first-child, .toc td:nth-child(3) { width: 16pt; color: #0f766e; font-weight: 600; }
.sections { margin-top: 6mm; font-size: 9pt; }
.sections h2 { margin: 0 0 4pt; font-size: 10.5pt; }
.small { font-size: 8.8pt; color: #5b7580; }
"""

COVER = f"""
<div class="cover">
  <div class="kicker">Track B · API-side service · Technical report</div>
  <h1 class="title">BeemaSathi</h1>
  <p class="subtitle">An evidence-grounded copilot for Indian health and motor
  insurance — retrieval, agentic routing and verification over official insurer
  and IRDAI documents.</p>
  <p class="meta">
    Executive Post Graduate Programme in Generative AI and Agentic AI,
    IIT Kharagpur<br>
    <b>Author:</b> Satish Singh &nbsp;·&nbsp; <b>Date:</b> 26 November 2026
    &nbsp;·&nbsp; <b>Version:</b> submission build
  </p>
  <div class="abstract">
    <b>Abstract.</b> Indian retail insurance is sold as a short brochure and governed
    by a long policy wording. BeemaSathi answers coverage questions only from official
    insurer and IRDAI documents, cites the clause, page and document type behind every
    statement, ranks binding wordings above marketing summaries, and refuses when a
    product is not indexed. The system combines hybrid dense and lexical retrieval over
    48,572 chunks with a deterministic agent layer — intent routing, per-insurer
    comparison, a number/UIN critic and a pre-model safety guard — and a single
    Qwen3-8B generation call. On a frozen 100-question gold set it passes 83 cases,
    finds the expected document in 97.5% of labelled cases and the exact supporting
    clause in 91.2%, and refuses all nine prompt-injection probes before the model is
    called. This report covers the problem, architecture, data governance, guardrails,
    model card, evaluation with its failure taxonomy, and reproduction steps.
  </div>
  <div class="links">
    <h2>Deliverables</h2>
    <table>
      <tr><td>Code repository</td><td><a href="{REPO}">{REPO}</a></td></tr>
      <tr><td>Trace dashboard (Track B)</td><td><a href="{DASHBOARD}">{DASHBOARD}</a></td></tr>
      <tr><td>Project pages</td><td><a href="{SITE}">{SITE}</a></td></tr>
      <tr><td>Ten-minute demo</td><td><a href="{VIDEO}">Google Drive video</a></td></tr>
      <tr><td>Model card</td><td>Section 6.1 of this report</td></tr>
    </table>
  </div>
  <div class="sections">
    <h2>Contents</h2>
    <table class="toc">
      <tr><td>1</td><td>The problem</td><td>7</td><td>Evaluation</td></tr>
      <tr><td>2</td><td>What the system does</td><td>8</td><td>Engineering review</td></tr>
      <tr><td>3</td><td>Architecture</td><td>9</td><td>Running locally</td></tr>
      <tr><td>4</td><td>Data and governance</td><td>10</td><td>Repository layout</td></tr>
      <tr><td>5</td><td>Retrieval and generation</td><td>11</td><td>Project plan</td></tr>
      <tr><td>6</td><td>Guardrails and model card</td><td>12</td><td>Distribution and non-goals</td></tr>
    </table>
  </div>
  <p class="foot">BeemaSathi · informational only, not legal or financial advice.
  Insurer PDFs are not redistributed; the corpus is rebuilt locally from official
  sources.</p>
</div>
<div class="page-break"></div>
"""

CONTENTS = """
<h1>Submission at a glance</h1>
<table>
  <tr><th>Deliverable</th><th>Where it lives</th><th>How it is produced</th></tr>
  <tr><td>Reproducible Git repository</td>
      <td><a href="%s">github.com/satishify/beemasathi</a></td>
      <td>Public repo; <code>venv/bin/python -m pytest</code> runs 154 tests;
          GitHub Actions runs compile and tests on every push</td></tr>
  <tr><td>Dashboard (Track B: traces)</td>
      <td><a href="%s">%s</a></td>
      <td><code>venv/bin/python observability/dashboard.py</code> renders
          <code>logs/traces/*.jsonl</code> to a static page — requests, p50/p95
          latency, intent mix, refusals, verification pass rate, recent requests</td></tr>
  <tr><td>Model card / Responsible AI</td>
      <td>Section 6.1 of this report</td>
      <td>Intended use, out of scope, failure modes, refusals, adversarial tests,
          limitations and monitoring</td></tr>
  <tr><td>Technical report</td>
      <td>This PDF</td>
      <td><code>venv/bin/python observability/report.py</code> renders
          <code>docs/PROJECT.md</code> through headless Chrome</td></tr>
  <tr><td>Ten-minute demo</td>
      <td><a href="%s">Google Drive video</a> (also
          <code>demo/BeemaSathi-live-demo-questions.pdf</code>)</td>
      <td>Recorded against the running Streamlit client and API on the local index</td></tr>
  <tr><td>Working system</td>
      <td><code>./beemasathi.sh start</code> → http://127.0.0.1:8501</td>
      <td>Streamlit client plus FastAPI service; <code>docker compose up --build</code>
          runs both. The 1.1 GB corpus and ~1 GB vector store are rebuilt locally (§9.3)</td></tr>
</table>
<p class="small">Measured numbers in this report come from the frozen run
<code>eval/reports/eval_20261126T173042Z.json</code> and the trace log of the demo
session. Latency and token figures depend on the hosted model endpoint.</p>
""" % (REPO, DASHBOARD, DASHBOARD, VIDEO)


def build_html(markdown_text: str) -> str:
    marker = "## 1. The problem"
    if marker in markdown_text:
        markdown_text = markdown_text[markdown_text.index(marker):]
    tail = "\n---\n\n*BeemaSathi · informational only, not legal or financial advice.*"
    if tail in markdown_text:
        markdown_text = markdown_text.replace(tail, "")
    body = MarkdownIt("gfm-like", {"linkify": False}).render(markdown_text)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>BeemaSathi — Technical Report</title>"
        f"<style>{CSS}</style></head><body>"
        f"{COVER}{CONTENTS}{body}</body></html>"
    )


def main() -> int:
    markdown_text = SOURCE.read_text(encoding="utf-8")
    html_text = build_html(markdown_text)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html_text, encoding="utf-8")
        chrome = Path(CHROME)
        if not chrome.exists():
            print(f"Chrome not found at {CHROME}", file=sys.stderr)
            return 1
        result = subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                "--virtual-time-budget=3000",
                f"--print-to-pdf={OUTPUT}",
                page.as_uri(),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not OUTPUT.exists():
            print(result.stderr[-2000:], file=sys.stderr)
            return 1
    print(f"Report written to {OUTPUT}")
    try:
        import pypdfium2

        pages = len(pypdfium2.PdfDocument(str(OUTPUT)))
        print(f"Pages: {pages}")
    except Exception as exc:  # page count is a nicety, not a gate
        print(f"Page count unavailable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

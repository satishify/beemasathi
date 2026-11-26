"""Render the per-request trace log as a static HTML dashboard (Track B).

Re-run after a demo or an evaluation to refresh the page:

    venv/bin/python observability/dashboard.py

The dashboard is written to docs/trace_dashboard.html and needs no server.
"""

import argparse
import html
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from observability.trace import TRACE_DIR, _percentile, summarize
except ModuleNotFoundError:  # run directly as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from observability.trace import TRACE_DIR, _percentile, summarize

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT = BASE_DIR / "docs" / "trace_dashboard.html"


def _load_records(directory: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.sort(key=lambda row: row.get("ts") or "")
    return records


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "n/a"


def _cards(summary: Dict[str, Any]) -> str:
    verified = summary.get("unverified") or 0
    requests = summary["requests"]
    rows = [
        ("Requests traced", f"{requests:,}"),
        ("p50 latency", f"{summary['p50_ms']} ms" if summary["p50_ms"] else "n/a"),
        ("p95 latency", f"{summary['p95_ms']} ms" if summary["p95_ms"] else "n/a"),
        ("Guard refusals", f"{summary['guard_refusals']} ({_pct(summary['guard_refusals'], requests)})"),
        ("Unverified answers", f"{verified} ({_pct(verified, requests)})"),
        ("Refused / blocked", f"{summary['unsafe']}"),
        ("Errors", f"{summary['errors']}"),
        ("Prompt tokens", f"{summary['prompt_tokens']:,}"),
    ]
    cells = "".join(
        f'<div class="card"><span class="k">{html.escape(label)}</span>'
        f'<span class="v">{html.escape(value)}</span></div>'
        for label, value in rows
    )
    return f'<div class="cards">{cells}</div>'


def _bars(counts: Counter, total: int, label: str) -> str:
    if not counts:
        return f"<h3>{label}</h3><p class=\"muted\">No data yet.</p>"
    peak = max(counts.values()) or 1
    rows = []
    for name, count in counts.most_common(12):
        width = max(2.0, 100.0 * count / peak)
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{html.escape(str(name))}</span>'
            f'<span class="bar"><span style="width:{width:.1f}%"></span></span>'
            f'<span class="bar-value">{count} <i>({_pct(count, total)})</i></span></div>'
        )
    return f"<h3>{label}</h3>" + "".join(rows)


def _recent(records: List[Dict[str, Any]], limit: int = 30) -> str:
    head = (
        "<tr><th>Time (UTC)</th><th>Question</th><th>Intent</th><th>Insurer</th>"
        "<th>Chunks</th><th>Latency</th><th>Verified</th></tr>"
    )
    rows = []
    for row in reversed(records[-limit:]):
        ts = (row.get("ts") or "").replace("T", " ")[:19]
        plan = row.get("plan") or {}
        insurers = ", ".join(plan.get("insurers") or []) or "—"
        verified = row.get("verified")
        mark = "Pass" if verified is True else ("Fail" if verified is False else "—")
        css = "ok" if verified is True else ("bad" if verified is False else "")
        rows.append(
            "<tr>"
            f"<td class=\"mono\">{html.escape(ts)}</td>"
            f"<td>{html.escape(row.get('question_preview') or '')}</td>"
            f"<td>{html.escape(row.get('intent') or '—')}</td>"
            f"<td>{html.escape(insurers)}</td>"
            f"<td>{row.get('retrieved') if row.get('retrieved') is not None else '—'}</td>"
            f"<td>{row.get('total_ms') if row.get('total_ms') is not None else '—'}</td>"
            f"<td class=\"{css}\">{mark}</td>"
            "</tr>"
        )
    return f'<table class="traces">{head}{"".join(rows)}</table>'


def _daily(records: List[Dict[str, Any]]) -> str:
    per_day: Dict[str, List[float]] = defaultdict(list)
    for row in records:
        day = (row.get("ts") or "")[:10]
        if day:
            per_day[day].append(row.get("total_ms") or 0)
    if not per_day:
        return ""
    peak = max(len(values) for values in per_day.values()) or 1
    rows = []
    for day in sorted(per_day):
        values = [value for value in per_day[day] if value]
        p95 = _percentile(values, 0.95)
        width = max(2.0, 100.0 * len(per_day[day]) / peak)
        rows.append(
            f'<div class="bar-row"><span class="bar-label mono">{day}</span>'
            f'<span class="bar"><span style="width:{width:.1f}%"></span></span>'
            f'<span class="bar-value">{len(per_day[day])} requests · p95 {p95} ms</span></div>'
        )
    return "<h3>Requests per day</h3>" + "".join(rows)


def _top_sources(records: List[Dict[str, Any]]) -> str:
    counts: Counter = Counter()
    for row in records:
        for source in row.get("sources") or []:
            name = source.get("policy_name") or source.get("source") or "unknown"
            counts[str(name)] += 1
    if not counts:
        return "<h3>Most cited plans</h3><p class=\"muted\">No sources recorded yet.</p>"
    items = "".join(
        f"<li>{html.escape(name)} <span class=\"muted\">· {count}</span></li>"
        for name, count in counts.most_common(10)
    )
    return f'<h3>Most cited plans</h3><ol class="sources">{items}</ol>'


def render(directory: Path = TRACE_DIR, out: Path = DEFAULT_OUT) -> Path:
    records = _load_records(directory)
    summary = summarize(directory)
    models = Counter(str(row.get("model") or "guard/rules") for row in records)
    intents = Counter(str(row.get("intent") or "unrouted") for row in records)
    checked = [row for row in records if row.get("verified") is not None]
    verified = sum(1 for row in checked if row.get("verified") is True)
    first = (records[0].get("ts") or "")[:10] if records else "—"
    last = (records[-1].get("ts") or "")[:10] if records else "—"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BeemaSathi — Trace dashboard</title>
<style>
  :root {{ --ink:#102a2a; --teal:#0f766e; --soft:#e6f2f0; --line:#cfe3df; --muted:#5b7370; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color:var(--ink); background:#f7fbfa; }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:28px 24px 60px; }}
  header {{ border-bottom:2px solid var(--teal); padding-bottom:14px; margin-bottom:22px; }}
  h1 {{ margin:0 0 4px; font-size:26px; }}
  h3 {{ margin:26px 0 10px; font-size:16px; color:var(--teal); }}
  .sub {{ color:var(--muted); font-size:13px; }}
  .cards {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin:18px 0 6px; }}
  .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
  .card .k {{ display:block; font-size:12px; color:var(--muted); }}
  .card .v {{ display:block; font-size:20px; font-weight:600; margin-top:2px; }}
  .bar-row {{ display:grid; grid-template-columns:190px 1fr 190px; gap:10px; align-items:center;
              margin:6px 0; font-size:13px; }}
  .bar {{ background:var(--soft); height:10px; border-radius:5px; overflow:hidden; }}
  .bar span {{ display:block; height:100%; background:var(--teal); }}
  .bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .bar-value i {{ color:var(--muted); font-style:normal; }}
  table.traces {{ width:100%; border-collapse:collapse; background:#fff; font-size:12.5px;
                  border:1px solid var(--line); }}
  table.traces th, table.traces td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line);
                  vertical-align:top; }}
  table.traces th {{ background:var(--soft); font-size:12px; }}
  .mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; }}
  .ok {{ color:#0f7a3d; font-weight:600; }} .bad {{ color:#b42318; font-weight:600; }}
  .muted {{ color:var(--muted); }}
  ol.sources {{ margin:0; padding-left:20px; font-size:13.5px; }}
  @media (max-width:760px) {{ .cards {{ grid-template-columns:repeat(2, 1fr); }}
    .bar-row {{ grid-template-columns:120px 1fr; }} .bar-value {{ grid-column:2; }} }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>BeemaSathi — trace dashboard</h1>
  <div class="sub">Track B · per-request records from <span class="mono">logs/traces/*.jsonl</span> ·
    {html.escape(first)} to {html.escape(last)} · generated {html.escape(generated)}</div>
</header>
{_cards(summary)}
{_daily(records)}
{_bars(intents, len(records), "Intent mix")}
{_bars(models, len(records), "Answering backend")}
<p class="sub">Verification pass rate: {_pct(verified, len(checked))} of the {len(checked)} answers the critic could check ({len(records) - len(checked)} guard-only or rule-based replies have nothing to verify).</p>
{_top_sources(records)}
<h3>Recent requests</h3>
{_recent(records)}
<p class="sub">Regenerate with <span class="mono">venv/bin/python observability/dashboard.py</span>.
Questions are redacted previews; no clause text, prompts or credentials are stored.</p>
</div>
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the trace dashboard.")
    parser.add_argument("--dir", type=Path, default=TRACE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    path = render(args.dir, args.out)
    print(f"Trace dashboard written to {path}")


if __name__ == "__main__":
    main()

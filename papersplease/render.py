"""Human-readable output: terminal text and Markdown.

The JSON shape (``Report.to_dict``) is the machine contract; these renderers are
for the person who has to read 20 abstracts' worth of verdicts at a glance.
Text output uses a single visual language: brand + rule, aligned labels, coloured
bucket badges, and wrapped prose. When stdout is not a TTY the colour disappears
but the layout stays, so logs remain readable.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import ui
from .models import Paper
from .pipeline import Finding, Report

BUCKET_TITLES = {
    "in_scope": "IN SCOPE",
    "adjacent": "ADJACENT - partial match, check before using",
    "out_of_scope": "OUT OF SCOPE - rejected by screening",
}


def _theme(color: bool | None) -> ui.Theme:
    if color is None:
        return ui.Theme.auto()
    return ui.Theme(enabled=color, unicode=ui.unicode_enabled())


def _header(subtitle: str, theme: ui.Theme, width: int) -> list[str]:
    return [
        theme.paint("PapersPlease", "bold", "cyan") + theme.dim(f"  {theme.sep}  {subtitle}"),
        ui.rule(width, theme),
    ]


def _meta_lines(paper: Paper, width: int, *, indent: str = "", sep: str = "·") -> list[str]:
    """Authors/year/citations, then the venue, each on its own wrapped line."""
    bits = [paper.author_line(limit=2), str(paper.year or "n.d.")]
    if paper.cited_by:
        bits.append(f"cited {paper.cited_by}")
    lines = ui.wrap(f" {sep} ".join(bits), width, initial=indent, subsequent=indent)
    if paper.journal:
        lines.extend(ui.wrap(paper.journal, width, initial=indent, subsequent=indent))
    return lines


def _access_lines(access, theme: ui.Theme, *, indent: str = "    ") -> list[str]:
    if access is None:
        return [theme.dim(f"{indent}no access resolved")]

    lines: list[str] = []
    for item in access.open_access[:3]:
        licence = theme.dim(f" ({item['license']})") if item.get("license") else ""
        lines.append(f"{indent}{theme.cyan('OA')}       {item['url']}{licence}")
    for link in access.scihub[:3]:
        state = link.get("state", "unchecked")
        glyph = {
            "ok": theme.green(theme.check),
            "blocked": theme.red(theme.cross),
            "unreachable": theme.yellow("!"),
        }.get(state, theme.dim("?"))
        label = theme.dim(f"[{state}]")
        lines.append(f"{indent}{theme.cyan('Sci-Hub')}  {link['url']}  {glyph} {label}")
    if not access.open_access and not access.scihub:
        lines.append(theme.dim(f"{indent}no link available (no DOI)"))
    return lines


def _finding_block(index: int, finding: Finding, theme: ui.Theme, width: int) -> list[str]:
    paper = finding.paper
    verdict = finding.verdict
    out: list[str] = []
    number = theme.dim(f"{index:>2}.")
    title = ui.truncate(paper.title, max(20, width - 4), unicode=theme.unicode)
    out.append(f"{number} {theme.bold(title)}")
    out.extend(_meta_lines(paper, width, indent="    ", sep=theme.sep))
    if paper.doi:
        out.append(f"    {theme.dim('doi')} {paper.doi}")
    elif paper.arxiv_id:
        out.append(f"    {theme.dim('arXiv')} {paper.arxiv_id}")
    else:
        out.append(f"    {theme.dim('doi')} (none)")

    marker = {
        "in_scope": theme.dot,
        "adjacent": theme.half,
        "out_of_scope": theme.ring,
    }.get(verdict.bucket, theme.sep)
    scope = f"{marker} {verdict.score:.2f}  coverage {verdict.coverage:.2f}"
    if verdict.matched:
        scope += f"  {theme.sep}  matched " + ", ".join(verdict.matched)
    out.extend(ui.wrap(scope, width, initial="    ", subsequent="      "))
    reason = verdict.excluded_reason or verdict.reason
    if reason:
        out.extend(ui.wrap(reason, width, initial=f"    {theme.dim('why')} ", subsequent="        "))
    out.extend(_access_lines(finding.access, theme))
    out.append("")
    return out


def _result_line(report: Report, sep: str = "·") -> str:
    total = report.in_scope_total if report.in_scope_total is not None else len(report.in_scope)
    shown = f" (showing {len(report.in_scope)})" if total > len(report.in_scope) else ""
    candidate_word = "candidate" if report.candidates == 1 else "candidates"
    return (
        f"{total} in scope{shown}  {sep}  {len(report.adjacent)} adjacent  {sep}  "
        f"{len(report.out_of_scope)} rejected  {sep}  from {report.candidates} {candidate_word}"
    )


def render_text(
    report: Report,
    *,
    max_rejected: int = 5,
    max_adjacent: int = 6,
    width: int | None = None,
    color: bool | None = None,
    origin: str | None = None,
) -> str:
    """The main terminal report: hierarchy first, prose second."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    spec = report.spec
    out = _header("literature scope report", theme, width)

    out.extend(ui.kv("Intent", f'"{spec.intent}"', theme, label_width=9, width=width))
    facets = " + ".join(f.name for f in spec.facets) or "(none)"
    out.extend(ui.kv("Scope", facets, theme, label_width=9, width=width))
    limits = [f"coverage {theme.ge} {spec.min_scope:.2f}"]
    if spec.year_from or spec.year_to:
        limits.append(f"years {spec.year_from or 'any'}-{spec.year_to or 'any'}")
    if spec.exclude:
        limits.append("excluding " + ", ".join(spec.exclude))
    if spec.anchors:
        limits.append("anchors " + ", ".join(spec.anchors))
    out.extend(ui.kv("Limits", f"  {theme.sep}  ".join(limits), theme, label_width=9, width=width))
    if origin is not None:
        out.extend(ui.kv("Input", origin, theme, label_width=9, width=width))
    else:
        counts = ", ".join(
            f"{name} {count}" for name, count in sorted(report.source_counts.items())
        )
        out.extend(ui.kv("Search", counts or "no sources answered", theme, label_width=9, width=width))
    out.extend(ui.kv("Result", _result_line(report, sep=theme.sep), theme, label_width=9, width=width))

    if report.warnings:
        out.append("")
        for warning in report.warnings[:4]:
            out.extend(ui.wrap(warning, width, initial=f"{theme.yellow('!')} ", subsequent="  "))
        if len(report.warnings) > 4:
            out.append(theme.dim(f"  (+{len(report.warnings) - 4} more notes; see --json)"))
    out.append("")

    rendered = 0
    for bucket, cap in (("in_scope", None), ("adjacent", max_adjacent)):
        findings = report.bucket(bucket)
        if not findings:
            continue
        shown = findings if cap is None else findings[:cap]
        line = f"{ui.badge(bucket, theme)}  {theme.bold(str(len(findings)))}"
        if cap is not None and len(findings) > len(shown):
            line += theme.dim(f"  showing {len(shown)}; use --format json for all")
        out.append(line)
        out.append(ui.rule(width, theme))
        for index, finding in enumerate(shown, start=1):
            out.extend(_finding_block(index, finding, theme, width))
        rendered += len(shown)

    rejected = report.out_of_scope
    if rejected:
        line = f"{ui.badge('out_of_scope', theme)}  {theme.bold(str(len(rejected)))}"
        if len(rejected) > max_rejected:
            line += theme.dim(f"  showing {max_rejected}")
        out.append(line)
        out.append(ui.rule(width, theme))
        for finding in rejected[:max_rejected]:
            reason = finding.verdict.excluded_reason or finding.verdict.reason
            out.append(f"{theme.red(theme.cross)} {ui.truncate(finding.paper.title, width - 2, unicode=theme.unicode)}")
            out.extend(ui.wrap(reason, width, initial="  ", subsequent="  "))
        out.append("")

    if not rendered and not rejected:
        out.append(theme.yellow("No papers matched."))
        out.extend(
            ui.wrap(
                "Try fewer concepts, a lower --min-scope (for example 0.55), or a "
                "different --sources subset. `papersplease explain` shows what was enforced.",
                width,
                initial="  ",
                subsequent="  ",
            )
        )
        out.append("")

    if report.adjacent or rejected:
        out.extend(
            ui.wrap(
                "treat adjacent papers as leads, not evidence. --require forces a "
                "concept to appear; --format json carries every result and reason.",
                width,
                initial="Tip  ",
                subsequent="     ",
            )
        )
    return "\n".join(out).rstrip() + "\n"


def render_explain(payload: dict[str, Any], *, width: int | None = None, color: bool | None = None) -> str:
    """Render `explain_intent` with the same visual language as the report."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    out = _header("scope contract (offline)", theme, width)

    out.extend(ui.kv("Intent", f'"{payload["intent"]}"', theme, label_width=11, width=width))
    out.extend(ui.kv("Search", payload.get("search_text") or "(same as intent)", theme, label_width=11, width=width))
    out.extend(ui.kv("Query", payload.get("query") or "(none)", theme, label_width=11, width=width))

    limits = [f"coverage {theme.ge} {payload['min_scope']:.2f}"]
    years = payload.get("years") or {}
    if years.get("from") or years.get("to"):
        limits.append(f"years {years.get('from') or 'any'}-{years.get('to') or 'any'}")
    guard = f"head {payload['head']}" if payload.get("head") else "no head concept"
    if payload.get("anchors"):
        guard += "; anchors " + ", ".join(payload["anchors"])
    limits.append(guard)
    out.extend(ui.kv("Guard", f"  {theme.sep}  ".join(limits), theme, label_width=11, width=width))
    out.append("")

    facets = payload.get("facets") or []
    out.append(theme.paint("FACETS", "bold", "cyan") + theme.dim(f"  {len(facets)} concepts"))
    out.append(ui.rule(width, theme))
    for facet in facets:
        name = facet["name"][:16].ljust(16)
        weight = f"{facet.get('weight', 1.0):.2f}"
        terms = ", ".join(facet.get("terms", [])[:8])
        first, *rest = ui.wrap(terms, max(20, width - 26))
        out.append(f"  {theme.cyan(name)} {theme.dim(weight)}  {first}")
        for line in rest:
            out.append(" " * 26 + line)
    if not facets:
        out.append(theme.dim("  (none)"))
    out.append("")

    if payload.get("anchors"):
        out.extend(
            ui.kv(
                "Anchors",
                ", ".join(payload["anchors"]) + "  (cover both, name at least one in the title)",
                theme,
                label_width=11,
                width=width,
            )
        )
    relation = payload.get("relation")
    if relation:
        src = ", ".join(relation.get("source") or [])
        dst = ", ".join(relation.get("target") or [])
        arrow = " -> " if not theme.unicode else " → "
        out.extend(ui.kv("Direction", f"{src}{arrow}{dst}", theme, label_width=11, width=width))
    if payload.get("exclude"):
        catches = payload.get("exclude_terms") or {}
        for term in payload["exclude"]:
            variants = catches.get(term) or []
            suffix = "  (catches " + ", ".join(variants[:6]) + ")" if variants else ""
            out.extend(ui.wrap(term + suffix, width, initial=f"{theme.times} ", subsequent="  "))
    else:
        out.extend(ui.kv("Excluded", "(none)", theme, label_width=11, width=width))

    if payload.get("required"):
        terms = ", ".join(term for term, _ in payload["required"])
        out.extend(ui.kv("Required", f"{terms}  (in {payload.get('require_field') or 'any field'})",
                         theme, label_width=11, width=width))

    if payload.get("notes") or payload.get("issues"):
        out.append("")
    for note in payload.get("notes") or []:
        out.extend(ui.wrap(note, width, initial=f"{theme.sep} ", subsequent="  "))
    for issue in payload.get("issues") or []:
        out.extend(ui.wrap(issue, width, initial=f"{theme.yellow('!')} ", subsequent="  "))
    return "\n".join(out).rstrip() + "\n"


def render_providers(rows: Sequence[dict[str, Any]], *, width: int | None = None, color: bool | None = None) -> str:
    """The `providers` table: enabled state, required key, contribution."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    out = _header("sources", theme, width)
    out.append(theme.dim(f"{'SOURCE':16}  {'STATE':5}  {'KEY':30}  ADDS"))
    out.append(ui.rule(width, theme))
    for row in rows:
        state_text = "on" if row["default"] else "off"
        state_text = ui.state(state_text, state_text, theme)
        key = row.get("key_env") or theme.dash
        if row.get("key_env") and not row.get("configured"):
            key = theme.yellow(key + " !")
        pad = " " * max(1, 5 - ui.visible_len(state_text))
        key_pad = " " * max(1, 30 - ui.visible_len(key))
        note_width = max(16, width - 16 - 5 - 30 - 6)
        out.append(
            f"{row['source'][:16]:16}  {state_text}{pad}  {key}{key_pad}  "
            f"{ui.truncate(row['note'], note_width, unicode=theme.unicode)}"
        )
    out.append("")
    out.append(theme.dim("Set PAPERSPLEASE_SOURCES to change the default set; keys live in the environment."))
    return "\n".join(out).rstrip() + "\n"


def render_mirrors(statuses: Sequence, *, width: int | None = None, color: bool | None = None) -> str:
    """The `mirrors` table: reachability, latency and the reason."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    out = _header("Sci-Hub mirrors", theme, width)
    out.append(theme.dim(f"{'MIRROR':28}  {'STATE':12}  {'LATENCY':>8}  NOTE"))
    out.append(ui.rule(width, theme))
    for status in statuses:
        latency = f"{status.latency_ms} ms" if status.latency_ms else theme.dash
        state_text = ui.state(status.state, status.state, theme)
        pad = " " * max(1, 12 - ui.visible_len(state_text))
        note_width = max(12, width - 28 - 12 - 8 - 8)
        out.append(
            f"{status.mirror[:28]:28}  {state_text}{pad}  {latency:>8}  "
            f"{ui.truncate(status.note, note_width, unicode=theme.unicode)}"
        )
    out.append("")
    if not any(item.usable for item in statuses):
        out.extend(
            ui.wrap(
                "No mirror answered from here. Links are still generated; your browser may reach them.",
                width,
                initial=f"{theme.yellow('!')} ",
                subsequent="  ",
            )
        )
    return "\n".join(out).rstrip() + "\n"


def render_read(
    paper: Paper,
    access,
    *,
    chars: int = 4000,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """One paper as a card: metadata, abstract, then the links."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    out = _header("paper", theme, width)
    out.extend(theme.bold(line) for line in ui.wrap(paper.title, width, initial="", subsequent=""))
    out.extend(theme.dim(line) for line in _meta_lines(paper, width, sep=theme.sep))
    if paper.doi:
        out.append(f"{theme.dim('DOI')} {paper.doi}")
    elif paper.arxiv_id:
        out.append(f"{theme.dim('arXiv')} {paper.arxiv_id}")
    else:
        out.append(f"{theme.dim('DOI')} (none)")
    if paper.read_note:
        out.append(theme.dim(f"read: {paper.read_note}"))
    out.append("")

    out.append(theme.paint("ABSTRACT", "bold", "cyan"))
    out.append(ui.rule(width, theme))
    abstract = " ".join((paper.abstract or "").split())
    if abstract:
        shown = abstract[:chars]
        if len(abstract) > chars:
            shown = shown.rstrip() + f" {theme.ellipsis}"
        out.extend(ui.wrap(shown, width))
    else:
        out.append(theme.dim("(no abstract available)"))
    out.append("")

    out.append(theme.paint("ACCESS", "bold", "cyan"))
    out.append(ui.rule(width, theme))
    out.extend(line.lstrip() for line in _access_lines(access, theme, indent=""))
    return "\n".join(out).rstrip() + "\n"


def render_downloads(results: Sequence[dict[str, Any]], *, width: int | None = None, color: bool | None = None) -> str:
    """Result cards for the `download` command."""
    theme = _theme(color)
    width = width or ui.terminal_width()
    out = _header("downloads", theme, width)
    saved = 0
    for item in results:
        ok = bool(item.get("saved"))
        saved += ok
        glyph = theme.green(theme.check) if ok else theme.red(theme.cross)
        label = theme.green("saved") if ok else theme.red("failed")
        title = item.get("title") or item.get("target") or "(unknown)"
        out.append(f"{glyph} {label}  {ui.truncate(title, width - 10, unicode=theme.unicode)}")
        if ok:
            out.append(f"    {theme.dim('path')} {item.get('path')}")
            out.append(f"    {theme.dim('from')} {item.get('source') or 'unknown'}")
        else:
            out.extend(ui.wrap(str(item.get("note") or "no downloadable source"), width,
                               initial="    ", subsequent="    "))
        out.append("")
    out.append(theme.dim(f"{saved} saved {theme.sep} {len(results) - saved} failed"))
    return "\n".join(out).rstrip() + "\n"


def render_markdown(report: Report, *, max_rejected: int = 10) -> str:
    spec = report.spec
    lines: list[str] = []
    lines.append(f"# Literature crawl: {spec.intent}")
    lines.append("")
    lines.append(f"**Scope facets:** {' + '.join(f.name for f in spec.facets)}")
    if spec.year_from or spec.year_to:
        lines.append(f"**Years:** {spec.year_from or 'any'}-{spec.year_to or 'any'}")
    if spec.exclude:
        lines.append(f"**Excluded concepts:** {', '.join(spec.exclude)}")
    lines.append(f"**Result:** {report.summary()}")
    lines.append("")

    def table(findings: list[Finding]) -> None:
        lines.append("| # | Paper | Year | DOI | Scope | Access |")
        lines.append("|---|-------|------|-----|-------|--------|")
        for index, finding in enumerate(findings, start=1):
            paper = finding.paper
            title = paper.title.replace("|", "\\|")
            doi = paper.doi or "-"
            links = []
            if finding.access:
                if finding.access.doi_url:
                    links.append(f"[DOI]({finding.access.doi_url})")
                if finding.access.open_access:
                    links.append(f"[OA]({finding.access.open_access[0]['url']})")
                for link in finding.access.scihub[:2]:
                    links.append(f"[{link['mirror'].replace('https://', '')}]({link['url']})")
            lines.append(
                f"| {index} | {title} | {paper.year or '-'} | {doi} | "
                f"{finding.verdict.score:.2f} | {' · '.join(links) or '-'} |"
            )
        lines.append("")

    for bucket in ("in_scope", "adjacent"):
        findings = report.bucket(bucket)
        if findings:
            lines.append(f"## {BUCKET_TITLES[bucket].title()} ({len(findings)})")
            lines.append("")
            table(findings)

    if report.out_of_scope:
        lines.append(f"## Rejected ({len(report.out_of_scope)})")
        lines.append("")
        for finding in report.out_of_scope[:max_rejected]:
            reason = finding.verdict.excluded_reason or finding.verdict.reason
            lines.append(f"- {finding.paper.title} - _{reason}_")
        lines.append("")

    if report.warnings:
        lines.append("## Notes")
        lines.append("")
        for warning in report.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("## Screening detail")
    lines.append("")
    for finding in report.findings:
        if finding.verdict.bucket == "out_of_scope":
            continue
        paper = finding.paper
        lines.append(f"### {paper.title}")
        lines.append("")
        lines.append(f"- {paper.author_line()} ({paper.year or 'n.d.'}) {paper.journal or ''}".rstrip())
        lines.append(f"- Verdict: **{finding.verdict.bucket}** at {finding.verdict.score:.2f}")
        lines.append(f"- {finding.verdict.reason}")
        if paper.abstract:
            lines.append("")
            lines.append(f"> {paper.abstract[:700].strip()}...")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

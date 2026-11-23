"""Human-readable output: terminal text and Markdown.

The JSON shape (``Report.to_dict``) is the machine contract; these renderers are
for the person who has to read 20 abstracts' worth of verdicts at a glance.
"""

from __future__ import annotations

from .pipeline import Finding, Report

BUCKET_TITLES = {
    "in_scope": "IN SCOPE",
    "adjacent": "ADJACENT - partial match, check before using",
    "out_of_scope": "OUT OF SCOPE - rejected by screening",
}


def _scope_line(finding: Finding) -> str:
    verdict = finding.verdict
    bits = [f"scope {verdict.score:.2f}", f"coverage {verdict.coverage:.2f}"]
    if verdict.reason:
        bits.append(verdict.reason)
    return " | ".join(bits)


def _access_line(finding: Finding) -> list[str]:
    access = finding.access
    if not access:
        return []
    lines: list[str] = []
    for item in access.open_access[:3]:
        licence = f" ({item['license']})" if item.get("license") else ""
        lines.append(f"   OA        {item['url']}{licence}")
    for link in access.scihub[:3]:
        state = link.get("state", "unchecked")
        marker = "" if state == "ok" else f"  [{state}]"
        lines.append(f"   sci-hub   {link['url']}{marker}")
    if not access.open_access and not access.scihub:
        lines.append("   no link available (no DOI)")
    return lines


def render_text(report: Report, *, max_rejected: int = 5, width: int = 96) -> str:
    spec = report.spec
    out: list[str] = []
    out.append(f'PaperHound - "{spec.intent}"')
    facets = " + ".join(f.name for f in spec.facets)
    dates = ""
    if spec.year_from or spec.year_to:
        dates = f"   years {spec.year_from or 'any'}-{spec.year_to or 'any'}"
    out.append(f"scope: {facets}   (need coverage >= {spec.min_scope:.2f}){dates}")
    if spec.exclude:
        out.append(f"excluding: {', '.join(spec.exclude)}")
    counts = ", ".join(f"{name} {count}" for name, count in sorted(report.source_counts.items()))
    out.append(f"search: {counts or 'no sources answered'}")
    out.append(f"result: {report.summary()}")
    if report.warnings:
        out.append("notes: " + "; ".join(report.warnings[:6]))
        if len(report.warnings) > 6:
            out.append(f"       (+{len(report.warnings) - 6} more)")
    out.append("")

    for bucket in ("in_scope", "adjacent"):
        findings = report.bucket(bucket)
        if not findings:
            continue
        out.append(f"{BUCKET_TITLES[bucket]} ({len(findings)})")
        out.append("-" * min(width, 96))
        for index, finding in enumerate(findings, start=1):
            paper = finding.paper
            out.append(f"{index}. {paper.title}")
            meta = " | ".join(
                str(x) for x in (paper.author_line(), paper.year, paper.journal) if x
            )
            if paper.cited_by:
                meta += f" | cited {paper.cited_by}"
            out.append(f"   {meta}")
            out.append(f"   DOI {paper.doi or '(none)'}")
            out.append(f"   {_scope_line(finding)}")
            out.extend(_access_line(finding))
            out.append("")

    rejected = report.out_of_scope
    if rejected:
        out.append(f"{BUCKET_TITLES['out_of_scope']} ({len(rejected)}, showing {min(len(rejected), max_rejected)})")
        out.append("-" * min(width, 96))
        for finding in rejected[:max_rejected]:
            reason = finding.verdict.excluded_reason or finding.verdict.reason
            out.append(f"- {finding.paper.title[:88]}")
            out.append(f"   {finding.paper.year or 'n.d.'} | {reason}")
        out.append("")

    out.append(
        "Tip: screen the adjacent bucket yourself, or re-run with --require to force "
        "a concept that must appear."
    )
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

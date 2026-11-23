"""Command-line interface.

Three useful commands for humans, plus `mcp` for agents:

    paperhound find "muscle strength in older adults since 2015"
    paperhound check "muscle strength in older adults" refs.txt
    paperhound read 10.1038/s41586-020-2649-2 --deep
    paperhound mirrors
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import __version__
from .config import Settings
from .intent import parse_intent
from .models import Paper
from .pipeline import Report, explain_intent, find, read_one, screen_only
from .render import render_markdown, render_text
from .resolver import access_for, mirror_health
from .sources import fetch_by_doi, find_by_title

_COMMANDS = {"find", "check", "read", "explain", "mirrors", "mcp"}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--min-scope", type=float, default=None,
                        help="coverage required to count as in scope (default 0.7)")
    parser.add_argument("--require", action="append", default=[],
                        help="concept that must be present (repeatable)")
    parser.add_argument("--exclude", action="append", default=[],
                        help="concept that disqualifies a paper (repeatable)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--out", help="write the report to this file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paperhound",
        description="Search the literature, read it, screen it for scope, and get the PDF links.",
    )
    parser.add_argument("--version", action="version", version=f"paperhound {__version__}")
    sub = parser.add_subparsers(dest="command")

    find_p = sub.add_parser("find", help="search -> read -> screen -> links")
    find_p.add_argument("intent", nargs="+", help="what you want, in words")
    find_p.add_argument("--limit", type=int, default=15,
                        help="roughly how many candidates to consider (each source is asked for ~2x)")
    find_p.add_argument("--sources", help="comma-separated subset of europepmc,crossref,arxiv,semanticscholar,openalex")
    find_p.add_argument("--since", type=int, help="earliest year")
    find_p.add_argument("--until", type=int, help="latest year")
    find_p.add_argument("--deep", action="store_true", help="read open-access full text and re-check scope")
    find_p.add_argument("--deep-limit", type=int, default=5, help="how many papers to deep-read (default 5)")
    find_p.add_argument("--llm", dest="use_llm", action="store_true", default=None,
                        help="use the LLM second opinion (needs PAPERHOUND_LLM_KEY)")
    find_p.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="lexical screening only")
    find_p.add_argument("--no-oa", action="store_true", help="skip open-access lookup (Unpaywall etc.)")
    find_p.add_argument("--no-mirror-probe", action="store_true", help="do not test mirror reachability")
    find_p.add_argument("--include-rejected", action="store_true", help="list rejected papers in JSON")
    find_p.add_argument("--download", metavar="DIR", help="download the best PDF for each in-scope paper")
    find_p.add_argument("--md", action="store_true", help="Markdown report")
    _add_common(find_p)

    check_p = sub.add_parser("check", help="screen a list of papers you already have")
    check_p.add_argument("intent", nargs="+", help="the research intent")
    check_p.add_argument("file", nargs="?", help="file with one DOI or title per line ('-' for stdin)")
    check_p.add_argument("--with-access", action="store_true", help="also resolve links")
    check_p.add_argument("--md", action="store_true")
    _add_common(check_p)

    read_p = sub.add_parser("read", help="read one paper by DOI or title")
    read_p.add_argument("target", nargs="+", help="DOI or title")
    read_p.add_argument("--deep", action="store_true", help="pull open-access full text")
    read_p.add_argument("--chars", type=int, default=4000)
    read_p.add_argument("--json", action="store_true")

    explain_p = sub.add_parser(
        "explain", help="show how an intent is parsed (offline, no network)"
    )
    explain_p.add_argument("intent", nargs="+", help="the research intent to parse")
    _add_common(explain_p)

    mirrors_p = sub.add_parser("mirrors", help="report Sci-Hub mirror health")
    mirrors_p.add_argument("--json", action="store_true")
    mirrors_p.add_argument("--refresh", action="store_true", help="ignore the 10-minute cache")

    sub.add_parser("mcp", help="run the MCP server on stdio (for AI agents)")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if getattr(args, "min_scope", None) is not None:
        settings.min_scope = args.min_scope
    return settings


def _require_list(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for item in getattr(args, "require", []) or []:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return values


def _exclude_list(args: argparse.Namespace) -> list[str]:
    values: list[str] = []
    for item in getattr(args, "exclude", []) or []:
        values.extend(part.strip() for part in item.split(",") if part.strip())
    return values


def _safe_name(paper: Paper) -> str:
    surname = ""
    if paper.authors:
        surname = re.split(r"[ ,]", paper.authors[0].strip())[0]
    tail = (paper.doi or paper.title)[-40:]
    stem = f"{paper.year or 'nd'}_{surname}_{tail}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")[:100] + ".pdf"


def _cmd_find(args: argparse.Namespace) -> int:
    settings = _settings(args)
    intent = " ".join(args.intent)
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    report = find(
        intent,
        settings=settings,
        limit=args.limit,
        source_names=sources,
        deep=args.deep,
        deep_limit=args.deep_limit,
        use_llm=args.use_llm,
        min_scope=args.min_scope,
        require=_require_list(args),
        exclude=_exclude_list(args),
        since=args.since,
        until=args.until,
        probe_mirrors=not args.no_mirror_probe,
        include_oa=not args.no_oa,
    )

    if args.download:
        from .resolver import download_any

        target_dir = Path(args.download).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        for finding in report.in_scope:
            if not finding.access:
                print(f"no PDF source for {finding.paper.doi or finding.paper.title[:50]}", file=sys.stderr)
                continue
            destination = target_dir / _safe_name(finding.paper)
            ok, note = download_any(finding.access, str(destination), settings)
            print(f"{'saved' if ok else 'failed'}: {note}", file=sys.stderr)

    if args.json:
        output = json.dumps(
            report.to_dict(abstract_chars=1000, include_rejected=args.include_rejected),
            ensure_ascii=False,
            indent=2,
        )
    elif args.md:
        output = render_markdown(report)
    else:
        output = render_text(report)

    if args.out:
        Path(args.out).expanduser().write_text(output, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


def _read_targets(raw: str | None) -> list[str]:
    raw = raw or "-"
    if raw == "-":
        text = sys.stdin.read()
    else:
        path = Path(raw).expanduser()
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        text = path.read_text("utf-8")
    text = text.strip()
    if text.startswith("["):
        try:
            data = json.loads(text)
            targets = []
            for item in data:
                if isinstance(item, dict):
                    targets.append(str(item.get("doi") or item.get("title") or ""))
                else:
                    targets.append(str(item))
            return [t for t in targets if t.strip()]
        except ValueError:
            pass
    targets = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not targets:
        raise SystemExit("no papers in the file (expected one DOI or title per line)")
    return targets


def _cmd_check(args: argparse.Namespace) -> int:
    settings = _settings(args)
    parts = list(args.intent)
    file_arg = args.file
    # argparse hands every positional to nargs="+" for `intent`, so the
    # documented `paperhound check "intent" refs.txt` arrives with file=None.
    # Treat a trailing path (or "-") as the file and keep the rest as intent.
    if file_arg is None and len(parts) > 1:
        candidate = parts[-1]
        if candidate == "-" or Path(candidate).expanduser().exists():
            file_arg = parts.pop()
    intent = " ".join(parts)
    targets = _read_targets(file_arg)
    papers: list[Paper] = []
    for target in targets:
        paper = None
        if target.lower().startswith("10.") or "doi.org/" in target.lower():
            paper = fetch_by_doi(target, settings, deep_meta=False)
        if paper is None:
            paper = Paper(title=target)
        papers.append(paper)
    findings = screen_only(
        intent,
        papers,
        settings=settings,
        min_scope=args.min_scope,
        require=_require_list(args),
        exclude=_exclude_list(args),
        with_access=args.with_access,
    )
    if args.json:
        payload = {
            "intent": intent,
            "counts": {bucket: sum(1 for f in findings if f.verdict.bucket == bucket)
                       for bucket in ("in_scope", "adjacent", "out_of_scope")},
            "results": [f.to_dict() for f in findings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.md:
        report = Report(
            spec=parse_intent(intent, min_scope=args.min_scope or 0.7),
            findings=findings,
            candidates=len(findings),
        )
        print(render_markdown(report))
        return 0
    print(f'Screening {len(findings)} papers against "{intent}"')
    print("-" * 78)
    for finding in findings:
        marker = {"in_scope": "IN ", "adjacent": "~  ", "out_of_scope": "OUT"}[finding.verdict.bucket]
        paper = finding.paper
        print(f"{marker} {finding.verdict.score:.2f}  {paper.title[:70]}")
        print(f"      {paper.doi or '(no doi)'} | {finding.verdict.reason}")
        if finding.access and (finding.access.open_access or finding.access.scihub):
            for item in finding.access.open_access[:1]:
                print(f"      OA      {item['url']}")
            for link in finding.access.scihub[:1]:
                print(f"      sci-hub {link['url']}")
    counts = {bucket: sum(1 for f in findings if f.verdict.bucket == bucket)
              for bucket in ("in_scope", "adjacent", "out_of_scope")}
    print("-" * 78)
    print(f"{counts['in_scope']} in scope, {counts['adjacent']} adjacent, {counts['out_of_scope']} rejected")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    target = " ".join(args.target)
    paper = read_one(target, settings=settings, deep=args.deep)
    if paper is None:
        print(f"could not resolve {target!r}", file=sys.stderr)
        return 1
    access = access_for(paper, settings)
    if args.json:
        print(json.dumps(
            {
                "title": paper.title,
                "doi": paper.doi,
                "year": paper.year,
                "authors": paper.authors,
                "journal": paper.journal,
                "cited_by": paper.cited_by,
                "abstract": paper.abstract,
                "fulltext_chars": paper.fulltext_chars,
                "read_note": paper.read_note,
                "access": access.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0
    print(paper.title)
    print(f"{paper.author_line()} | {paper.year or 'n.d.'} | {paper.journal or ''}")
    print(f"DOI: {paper.doi or '(none)'}   cited: {paper.cited_by}")
    if paper.read_note:
        print(f"read: {paper.read_note}")
    print("-" * 78)
    print((paper.abstract or "(no text available)")[: args.chars])
    print("-" * 78)
    for item in access.open_access[:3]:
        print(f"OA      {item['url']}")
    for link in access.scihub[:3]:
        print(f"sci-hub {link['url']}  [{link.get('state')}]")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    settings = _settings(args)
    payload = explain_intent(
        " ".join(args.intent),
        settings=settings,
        min_scope=args.min_scope,
        require=_require_list(args),
        exclude=_exclude_list(args),
    )
    if args.json:
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        lines = [f'PaperHound scope for "{payload["intent"]}"', ""]
        lines.append("facets (coverage is weighted over these):")
        for facet in payload["facets"]:
            terms = ", ".join(facet["terms"][:8])
            lines.append(f"  - {facet['name']}  weight {facet['weight']}  {terms}")
        lines.append(f"head concept: {payload['head'] or '(none)'}")
        lines.append(f"excluded: {', '.join(payload['exclude']) or '(none)'}")
        years = payload["years"]
        if years["from"] or years["to"]:
            lines.append(f"years: {years['from'] or 'any'}-{years['to'] or 'any'}")
        lines.append(f"threshold: coverage >= {payload['min_scope']:.2f}")
        lines.append(f"query: {payload['query'][:300]}")
        for note in payload["notes"]:
            lines.append(f"note: {note}")
        for issue in payload["issues"]:
            lines.append(f"warning: {issue}")
        output = "\n".join(lines)
    if args.out:
        Path(args.out).expanduser().write_text(output, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


def _cmd_mirrors(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    statuses = mirror_health(settings, force=args.refresh)
    if args.json:
        print(json.dumps([s.to_dict() for s in statuses], indent=2))
        return 0
    for status in statuses:
        latency = f"{status.latency_ms} ms" if status.latency_ms else "-"
        print(f"{status.mirror:28} {status.state:12} {latency:>8}  {status.note}")
    if not any(s.usable for s in statuses):
        print("\nNo mirror answered from here. Links are still generated; your own browser may reach them.")
    return 0


def _cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server import serve

    return serve()


HANDLERS = {
    "find": _cmd_find,
    "check": _cmd_check,
    "read": _cmd_read,
    "explain": _cmd_explain,
    "mirrors": _cmd_mirrors,
    "mcp": _cmd_mcp,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # Allow `paperhound "muscle strength in older adults"` without a subcommand.
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "find")
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # surface a clean message instead of a traceback
        if os.environ.get("PAPERHOUND_DEBUG"):
            raise
        print(f"paperhound: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

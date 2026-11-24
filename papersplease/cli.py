"""Command-line interface.

A small set of commands for humans, plus `mcp` for agents:

    papersplease find "muscle strength in older adults since 2015"
    papersplease check "muscle strength in older adults" refs.txt
    papersplease read 10.1038/s41586-020-2649-2 --deep
    papersplease download 10.1038/s41586-020-2649-2 2401.12345 -d papers/
    papersplease providers
    papersplease mirrors
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from . import __version__, export, ui
from .config import Settings
from .filters import TYPES as PAPER_TYPES
from .intent import parse_intent
from .models import Paper
from .pipeline import Report, explain_intent, find, read_one, screen_only
from .render import (
    render_downloads,
    render_explain,
    render_markdown,
    render_mirrors,
    render_providers,
    render_read,
    render_text,
)
from .resolver import access_for, download_best, filename_for, mirror_health
from .sources import SOURCE_INFO, arxiv_id, fetch_by_arxiv, fetch_by_doi, resolve_identifier

_COMMANDS = {"find", "check", "read", "download", "explain", "providers", "mirrors", "tui", "mcp"}


def _configure_stdio() -> None:
    """Use UTF-8 on piped stdio streams so Windows cp1252 pipes do not fail."""
    streams: list[object] = [sys.stdout, sys.stderr]
    if not getattr(sys.stdin, "isatty", lambda: False)():
        streams.append(sys.stdin)
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _scope_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not (0.0 <= parsed <= 1.0):
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _non_neg_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _add_common(
    parser: argparse.ArgumentParser,
    *,
    screen_target: argparse._ActionsContainer | None = None,
    output_target: argparse._ActionsContainer | None = None,
) -> None:
    screen = screen_target or parser
    output = output_target or parser
    screen.add_argument("--min-scope", type=_scope_float, default=None,
                        help="coverage required to count as in scope (0.0-1.0, default 0.7)")
    screen.add_argument("--require", action="append", default=[],
                        help="concept that must be present (repeatable)")
    screen.add_argument("--require-in", choices=("title", "abstract", "title_or_abstract"),
                        help="where --require terms must appear")
    screen.add_argument("--exclude", action="append", default=[],
                        help="concept that disqualifies a paper (repeatable)")
    output.add_argument("--json", action="store_true", help="machine-readable output")
    output.add_argument("--out", help="write the report to this file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papersplease",
        description="Search the literature, read it, screen it for scope, and get the PDF links.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  papersplease find \"muscle strength in older adults since 2015\"\n"
            "  papersplease check \"muscle strength in older adults\" refs.txt\n"
            "  papersplease explain \"sets and frequency for hypertrophy\"\n"
            "  papersplease download 10.7717/peerj.12958 2401.12345 -d papers/\n"
            "  papersplease mcp\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"papersplease {__version__}")
    sub = parser.add_subparsers(dest="command")

    find_p = sub.add_parser(
        "find",
        help="search -> read -> screen -> links",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  papersplease find \"muscle strength in older adults since 2015\"\n"
            "  papersplease find \"resistance training volume\" --author Schoenfeld --min-citations 50\n"
            "  papersplease find \"creatine\" --exclude rodents --format json\n"
        ),
    )
    find_p.add_argument("intent", nargs="+", help="what you want, in words")

    search = find_p.add_argument_group("search")
    search.add_argument("--limit", type=_positive_int, default=15,
                        help="show at most N in-scope papers; also sizes the candidate pool (~2x per source)")
    search.add_argument("--sources", metavar="LIST",
                        help="comma-separated subset of europepmc,crossref,arxiv,semanticscholar,openalex")
    search.add_argument("--since", type=int, help="earliest year")
    search.add_argument("--until", type=int, help="latest year")
    search.add_argument("--author", action="append", default=[],
                        help="author name must contain this (repeatable)")
    search.add_argument("--venue", action="append", default=[],
                        help="journal/venue must contain this (repeatable)")
    search.add_argument("--min-citations", type=_non_neg_int, help="drop papers with fewer citations")
    search.add_argument("--type", action="append", choices=PAPER_TYPES,
                        help="publication type (repeatable)")
    search.add_argument("--peer-reviewed", action="store_true",
                        help="shortcut for --type journal,conference,book")
    search.add_argument("--preprints-only", action="store_true",
                        help="shortcut for --type preprint")
    search.add_argument("--deep", action="store_true", help="read open-access full text and re-check scope")
    search.add_argument("--deep-limit", type=_positive_int, default=5, help="how many papers to deep-read (default 5)")
    search.add_argument("--llm", dest="use_llm", action="store_true", default=None,
                        help="use the LLM second opinion (needs PAPERSPLEASE_LLM_KEY)")
    search.add_argument("--no-llm", dest="use_llm", action="store_false",
                        help="lexical screening only")
    search.add_argument("--expand", choices=("citations", "references", "both"), default="none",
                        help="walk one citation-graph hop from the best matches and screen the result")
    search.add_argument("--expand-limit", type=_positive_int, default=25,
                        help="candidates to add when expanding (default 25)")

    screening = find_p.add_argument_group("screening")
    access = find_p.add_argument_group("access and downloads")
    access.add_argument("--no-oa", action="store_true", help="skip open-access lookup (Unpaywall etc.)")
    access.add_argument("--no-mirror-probe", action="store_true", help="do not test mirror reachability")
    access.add_argument("--download", metavar="DIR", help="download the best PDF for each in-scope paper")

    output = find_p.add_argument_group("output")
    output.add_argument("-i", "--interactive", action="store_true",
                        help="open results in the interactive in-place terminal UI")
    output.add_argument("--format", choices=export.FORMATS, help="output format (default: text)")
    output.add_argument("--md", action="store_true", help="Markdown report")
    output.add_argument("--include-rejected", action="store_true", help="list rejected papers in JSON")
    output.add_argument("--refresh", action="store_true", help="ignore cached verdicts and access lookups")
    _add_common(find_p, screen_target=screening, output_target=output)

    check_p = sub.add_parser("check", help="screen a list of papers you already have")
    check_p.add_argument("intent", nargs="+", help="the research intent")
    check_p.add_argument("file", nargs="?", help="file with one DOI or title per line ('-' for stdin)")
    check_p.add_argument("--with-access", action="store_true", help="also resolve links")
    check_p.add_argument("--format", choices=export.FORMATS, help="output format (default: text)")
    check_p.add_argument("--refresh", action="store_true", help="ignore cached verdicts")
    check_p.add_argument("--md", action="store_true")
    _add_common(check_p)

    read_p = sub.add_parser("read", help="read one paper by DOI, arXiv id or title")
    read_p.add_argument("target", nargs="+", help="DOI, arXiv id, URL or title")
    read_p.add_argument("--deep", action="store_true", help="pull open-access full text")
    read_p.add_argument("--chars", type=_positive_int, default=4000)
    read_p.add_argument("--json", action="store_true")

    download_p = sub.add_parser(
        "download", help="download PDFs for DOIs, arXiv ids or URLs"
    )
    download_p.add_argument("targets", nargs="+", help="DOI, arXiv id, URL or title")
    download_p.add_argument("-d", "--dir", dest="directory", default="papers",
                            help="output directory (default: ./papers)")
    download_p.add_argument("--no-scihub", action="store_true",
                            help="only try open-access copies")
    download_p.add_argument("--no-probe", action="store_true",
                            help="do not test mirror reachability first")
    download_p.add_argument("--json", action="store_true")

    providers_p = sub.add_parser("providers", help="list search sources and their setup")
    providers_p.add_argument("--json", action="store_true")

    explain_p = sub.add_parser(
        "explain", help="show how an intent is parsed (offline, no network)"
    )
    explain_p.add_argument("intent", nargs="+", help="the research intent to parse")
    _add_common(explain_p)

    mirrors_p = sub.add_parser("mirrors", help="report Sci-Hub mirror health")
    mirrors_p.add_argument("--json", action="store_true")
    mirrors_p.add_argument("--refresh", action="store_true", help="ignore the 10-minute cache")

    tui_p = sub.add_parser("tui", help="launch the interactive in-place terminal UI")
    tui_p.add_argument("intent", nargs="*", help="optional starting research intent")

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


def _output(cmd: argparse.Namespace, report) -> str:
    """Pick the renderer: explicit --format, then the legacy --json/--md."""
    fmt = cmd.format or ("json" if getattr(cmd, "json", False) else "md" if getattr(cmd, "md", False) else "text")
    if fmt == "json":
        return json.dumps(
            report.to_dict(
                abstract_chars=1000,
                include_rejected=getattr(cmd, "include_rejected", False),
            ),
            ensure_ascii=False,
            indent=2,
        )
    if fmt == "md":
        return render_markdown(report)
    if fmt in ("bibtex", "ris", "csljson"):
        return export.render(report.in_scope, fmt)
    return render_text(
        report,
        width=96 if getattr(cmd, "out", None) else None,
        color=False if getattr(cmd, "out", None) else None,
    )


def _cmd_find(args: argparse.Namespace) -> int:
    settings = _settings(args)
    intent = " ".join(args.intent)
    sources = [s.strip() for s in args.sources.split(",")] if args.sources else None
    find_kwargs = {
        "limit": args.limit,
        "source_names": sources,
        "deep": args.deep,
        "deep_limit": args.deep_limit,
        "use_llm": args.use_llm,
        "min_scope": args.min_scope,
        "require": _require_list(args),
        "exclude": _exclude_list(args),
        "since": args.since,
        "until": args.until,
        "authors": args.author,
        "venues": args.venue,
        "min_citations": args.min_citations,
        "types": args.type or (),
        "peer_reviewed": args.peer_reviewed,
        "preprints_only": args.preprints_only,
        "probe_mirrors": not args.no_mirror_probe,
        "include_oa": not args.no_oa,
        "require_in": args.require_in or "",
        "expand": args.expand,
        "expand_limit": args.expand_limit,
        "use_cache": False if args.refresh else None,
    }

    wants_tui = (
        getattr(args, "interactive", False)
        and not args.json
        and not args.md
        and not args.format
        and not args.out
        and not args.download
    )
    if wants_tui and ui.vt_enabled(sys.stdout) and getattr(sys.stdin, "isatty", lambda: False)():
        from .tui import run_tui

        return run_tui(settings=settings, initial_intent=intent, find_kwargs=find_kwargs)

    with ui.live_panel("live pipeline", f'"{intent}"') as panel:
        report = find(
            intent,
            settings=settings,
            on_progress=panel.event,
            **find_kwargs,
        )

    if args.download:
        from .resolver import download_any

        target_dir = Path(args.download).expanduser()
        target_dir.mkdir(parents=True, exist_ok=True)
        for finding in report.in_scope:
            if not finding.access:
                print(f"no PDF source for {finding.paper.doi or finding.paper.title[:50]}", file=sys.stderr)
                continue
            destination = target_dir / filename_for(finding.paper)
            ok, note = download_any(finding.access, str(destination), settings)
            print(f"{'saved' if ok else 'failed'}: {note}", file=sys.stderr)

    output = _output(args, report)

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
    # documented `papersplease check "intent" refs.txt` arrives with file=None.
    # Treat a trailing path (or "-") as the file and keep the rest as intent.
    if file_arg is None and len(parts) > 1:
        candidate = parts[-1]
        if candidate == "-" or Path(candidate).expanduser().exists():
            file_arg = parts.pop()
    intent = " ".join(parts)
    targets = _read_targets(file_arg)
    papers: list[Paper] = []
    with ui.progress("resolving and screening"):
        for target in targets:
            paper = None
            if target.lower().startswith("10.") or "doi.org/" in target.lower():
                paper = fetch_by_doi(target, settings, deep_meta=False)
            elif arxiv_id(target):
                paper = fetch_by_arxiv(target, settings)
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
            require_in=args.require_in or "",
            with_access=args.with_access,
            use_cache=False if args.refresh else None,
        )
    fmt = args.format or ("json" if args.json else "md" if args.md else "text")
    if fmt in ("bibtex", "ris", "csljson"):
        kept = [f for f in findings if f.verdict.bucket == "in_scope"]
        print(export.render(kept, fmt), end="")
        return 0
    if fmt == "json":
        payload = {
            "intent": intent,
            "counts": {bucket: sum(1 for f in findings if f.verdict.bucket == bucket)
                       for bucket in ("in_scope", "adjacent", "out_of_scope")},
            "results": [f.to_dict() for f in findings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    min_scope = args.min_scope if args.min_scope is not None else settings.min_scope
    spec = parse_intent(
        intent,
        min_scope=min_scope,
        require=_require_list(args),
        exclude=_exclude_list(args),
        require_in=args.require_in or "",
    )
    if fmt == "md":
        report = Report(spec=spec, findings=findings, candidates=len(findings))
        print(render_markdown(report))
        return 0
    report = Report(spec=spec, findings=findings, candidates=len(findings))
    print(render_text(report, origin="supplied list"), end="")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    target = " ".join(args.target)
    with ui.progress("resolving and reading"):
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
    print(render_read(paper, access, chars=args.chars), end="")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    target_dir = Path(args.directory).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for target in args.targets:
        with ui.progress(f"downloading {ui.truncate(target, 40)}"):
            paper = resolve_identifier(target, settings)
        if paper is None:
            results.append({"target": target, "saved": False, "note": "could not resolve to a paper"})
            continue
        with ui.progress(f"fetching {ui.truncate(paper.title, 40)}"):
            access = access_for(paper, settings, probe=not args.no_probe)
            outcome = download_best(
                access,
                str(target_dir / filename_for(paper)),
                settings,
                include_scihub=not args.no_scihub,
            )
        results.append(
            {
                "target": target,
                "title": paper.title,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                **outcome,
            }
        )
    if args.json:
        print(json.dumps({"schema_version": 1, "downloads": results}, ensure_ascii=False, indent=2))
    else:
        print(render_downloads(results), end="")
    return 0 if any(item["saved"] for item in results) else 1


def _cmd_providers(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    rows = []
    for name, info in SOURCE_INFO.items():
        env = info["key_env"]
        rows.append(
            {
                "source": name,
                "label": info["label"],
                "default": name in settings.sources,
                "key_env": env or None,
                "configured": not env or bool(os.environ.get(env)),
                "note": info["note"],
            }
        )
    if args.json:
        print(json.dumps({"sources": rows}, ensure_ascii=False, indent=2))
        return 0
    print(render_providers(rows), end="")
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    settings = _settings(args)
    payload = explain_intent(
        " ".join(args.intent),
        settings=settings,
        min_scope=args.min_scope,
        require=_require_list(args),
        exclude=_exclude_list(args),
        require_in=getattr(args, "require_in", "") or "",
    )
    if args.json:
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        output = render_explain(payload, color=False if args.out else None)
    if args.out:
        Path(args.out).expanduser().write_text(output, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(output)
    return 0


def _cmd_mirrors(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    with ui.progress("probing mirrors"):
        statuses = mirror_health(settings, force=args.refresh)
    if args.json:
        print(json.dumps([s.to_dict() for s in statuses], indent=2))
        return 0
    print(render_mirrors(statuses), end="")
    return 0


def _cmd_tui(args: argparse.Namespace) -> int:
    from .tui import run_tui

    if not ui.vt_enabled(sys.stdout) or not getattr(sys.stdin, "isatty", lambda: False)():
        print("papersplease tui requires an interactive terminal", file=sys.stderr)
        return 1
    intent = " ".join(args.intent) if getattr(args, "intent", None) else ""
    return run_tui(settings=Settings.from_env(), initial_intent=intent)


def _cmd_mcp(_: argparse.Namespace) -> int:
    from .mcp_server import serve

    return serve()


HANDLERS = {
    "find": _cmd_find,
    "check": _cmd_check,
    "read": _cmd_read,
    "download": _cmd_download,
    "explain": _cmd_explain,
    "providers": _cmd_providers,
    "mirrors": _cmd_mirrors,
    "tui": _cmd_tui,
    "mcp": _cmd_mcp,
}


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # Allow `papersplease "muscle strength in older adults"` without a subcommand.
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "find")
    if not argv:
        if ui.vt_enabled(sys.stdout) and getattr(sys.stdin, "isatty", lambda: False)():
            from .tui import run_tui

            return run_tui(settings=Settings.from_env())
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
        if os.environ.get("PAPERSPLEASE_DEBUG"):
            raise
        print(f"papersplease: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

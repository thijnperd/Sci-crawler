# Prior art, comparison and attribution

PapersPlease is not the first tool that searches for papers, downloads PDFs, or
manages a bibliography. This page records the projects we compared against,
what each one does better, what this repository borrowed from them, and where
the honest differences are. It is also the attribution file: if an idea here
came from another project, that project is named.

Checked on **2026-11-23** against the versions and pages linked below. These
projects move; verify current behaviour before quoting a row.

## The projects

| project | license | what it is |
|---|---|---|
| [paperhound](https://pypi.org/project/paperhound/) / [site](https://alexfdez1010.github.io/paperhound/) ([repo](https://github.com/alexfdez1010/paperhound)) | MIT | Alejandro Fernández Camello's eight-provider search CLI: docling PDF→Markdown, SQLite FTS5 local library, embedding rerank, `refs`/`cited-by`, BibTeX/RIS/CSL-JSON, JSON everywhere, agent skill. **Unrelated project; its name is why this tool is now called PapersPlease.** |
| [paper-search-cli](https://github.com/openags/paper-search-cli) | MIT | CLI + MCP + Claude skill over 20+ sources, with a multi-stage download fallback (source → OA → Unpaywall → Sci-Hub) and PDF text extraction. |
| [paper-find](https://mcpmarket.com/server/paper-find) ([repo](https://github.com/h-lu/paper-find-mcp)) | see repo | MCP server over 10+ platforms (arXiv, S2, PubMed, bioRxiv, medRxiv, CrossRef, IACR, Google Scholar, RePEc), PDF download incl. Sci-Hub, PDF→Markdown, economics-specific filters. |
| [arxiv-dl](https://github.com/MarkHershey/arxiv-dl) | MIT | Downloader/naming/cataloguing tool for arXiv and major proceedings (NeurIPS, ICLR, CVPR, ICCV, WACV, ECCV, HF Papers), with notes, a JSON index and optional aria2 parallel downloads. |
| [papis](https://alternativeto.net/software/papis/) ([repo](https://github.com/papis/papis)) | GPL-3.0 | Mature CLI document and bibliography manager: add/search/sync, YAML per entry, BibTeX import/export, TUI, web app, editor plugins. |
| [cobib](https://github.com/mrossinek/cobib) (compared on [LibHunt](https://www.libhunt.com/compare-cobib-vs-papis?ref=compare)) | MIT | Console bibliography manager with a TUI, import/export and a local database. |

The [AlternativeTo](https://alternativeto.net/software/papis/) and
[LibHunt](https://www.libhunt.com/compare-cobib-vs-papis?ref=compare) pages are
directories, not code we used; they are how papis and cobib were found.

## Capability matrix

“This repo” is version 0.4.0. A dash means the project does not advertise the
capability; it does not mean the capability is impossible with a plugin.

| capability | this repo | paperhound | paper-search-cli | paper-find | arxiv-dl | papis / cobib |
|---|---|---|---|---|---|---|
| Sources | 5 | 8 | 20+ | 10+ | arXiv + proceedings | n/a (local) |
| **Per-paper scope screening** | **yes: facets, head, anchors, exclusions, reasons** | no | no | no | no | no |
| Verdict buckets (`in_scope` / `adjacent` / `out_of_scope`) | yes | no | no | no | no | no |
| Native MCP server | yes, stdlib, 8 tools, `structuredContent` | no (argues CLI > MCP) | yes | yes | no | no |
| Zero runtime dependencies | yes | no | no | no | no | no |
| Identifier input | DOI, DOI/arXiv URL, arXiv id, title | all common ids + URLs | arXiv id, DOI | DOI/URL | arXiv id/URL | DOI |
| Search filters | year, author, venue, citations, type | year, author, venue, citations, type | year (S2) | economics filters | — | library query |
| Exports | BibTeX, RIS, CSL-JSON | BibTeX, RIS, CSL-JSON | JSON | — | BibTeX | BibTeX + more |
| Local library / full-text grep | no (by design) | SQLite FTS5 | no | no | JSON index | yes |
| PDF→Markdown | text extraction only | docling (figures, equations, tables) | text extraction | yes | notes | no |
| Citation graph | 1 hop, Europe PMC | depth-controlled, OpenAlex + S2 | no | no | no | no |
| Bulk PDF download | yes (`download` command) | yes | yes | yes | yes | papis (not cobib) |
| Verifies `%PDF` before writing | yes | likely | likely | likely | yes | n/a |

## Why this tool exists next to them

The other tools are **retrieval** tools: they answer “find papers matching these
words” and stop. PapersPlease is a **screening** tool that also retrieves: it
answers “which of these papers is actually about the question, and why?”

That difference is the reason for the rest of the design:

* **The verdict is the product.** Every paper carries its bucket, matched and
  missing concepts, and a sentence explaining the decision. A creatine trial
  that mentions “3 sets” in its methods is demoted to `adjacent`, not silently
  returned as a training-volume paper.
* **The guards are explicit.** Head concept for single-topic intents; anchors
  from the leading variables for list-style intents; hard exclusions; date
  filters; an optional LLM second opinion. `explain` prints all of it before a
  search is spent.
* **Agent-first, not agent-only.** The MCP server speaks the protocol directly,
  returns typed `structuredContent`, marks the one writing tool, and never
  invents a DOI or a working link. The CLI and Python library share the same
  pipeline.
* **Failure is reported.** A source that 429s becomes a warning; a mirror that
  is blocked is listed as blocked; a captcha page is not written as a PDF. The
  access cache and the verdict cache both say when they were used.
* **Reproducible screening.** The verdict cache key includes
  `SCREENING_VERSION`, so changing the rules invalidates old decisions instead
  of serving them silently.
* **One dependency: Python.** It installs and runs inside restricted agent
  sandboxes where a docling or SQLite-FTS stack is not available.

### Evidence, not a head-to-head

On the same Dutch gym intent and the same 71-candidate pool, tightening the
list-intent anchors changed the result from **16 in scope** to **4 in scope**,
demoting the vegan-protein, creatine-in-older-men and regional-hypertrophy
trials to `adjacent` with a stated reason. That is an internal before/after
measurement of the screening rule; it is **not** a benchmark against the other
projects, which do not expose a scope verdict to compare.

## Where the other projects are better

Being honest about this is more useful than a feature-count victory lap.

| their advantage | status here |
|---|---|
| **Breadth** — 20+ sources, or 8 providers with DBLP/CORE/HF Papers | We ship 5 verified adapters. DBLP is not included: its API returned an Anubis bot challenge (HTTP 202, HTML) from this network, and an unverified adapter would fail silently for users behind the same protection. |
| **PDF→Markdown quality** — docling preserves figures, equations and tables | We extract text (and `pypdf` is optional); we do not reimplement docling. For reading a paper as Markdown, use paperhound or paper-search-cli. |
| **Library management** — papis, cobib and paperhound own this | Deliberately out of scope. We export BibTeX/RIS/CSL-JSON so papis, Zotero or cobib can own the library. |
| **Citation-graph depth** — `refs`/`cited-by` with depth control | We do one hop through Europe PMC, enough to widen a search, not to map a field. |
| **Embedding rerank** | Lexical screening plus an optional LLM pass; no model download. Dependency-free is the constraint. |
| **Maturity and community** | Those projects have years of releases and users. This one is 0.4.0. |

## What was borrowed, and from where

No source code was copied. The items below are interface and behaviour
inspiration, reimplemented against this pipeline and covered by this repo's
tests.

| recent change | inspired by | license note |
|---|---|---|
| `arxiv_id` parsing (modern, legacy, `arXiv:`, `/abs/`, `/pdf/` URLs) and `fetch_by_arxiv` | **arxiv-dl** and **paperhound** both accept every arXiv identifier form | MIT / MIT; no code copied |
| `papersplease download ID... -d DIR` for bulk identifier download | **arxiv-dl**'s core workflow, **paper-search-cli**'s fallback chain, **paperhound**'s `download`/`get` | MIT / MIT / MIT; our resolver and `%PDF` verification are original |
| `--author`, `--venue`, `--min-citations`, `--type`, `--peer-reviewed`, `--preprints-only`, applied after merging | **paperhound**'s filter set and its “unknown citation count is excluded” rule | MIT; no code copied |
| CSL-JSON export | **paperhound** and **papis** both expose CSL-JSON; the format is the Citation Style Language standard | MIT / GPL-3.0; format only, no papis code |
| `papersplease providers` (which sources are on, which key they need) | **paperhound**'s `providers` command | MIT; no code copied |
| MCP tool shape: search, screen, read, download, export | **paper-find** and **paper-search-cli** validated that agents want these verbs | our JSON-RPC server is stdlib and original |
| Comparison framing for papis/cobib | **AlternativeTo** and **LibHunt** directory pages | directory data, not code |

Papis is **GPL-3.0**; cobib is **MIT**. Only papis's public CLI concepts
(subcommands, BibTeX as the interchange format) were studied. If code is ever
adapted from any of these projects, the license and attribution must be added
to the file header and to this table before merging.

## Why the name changed

PapersPlease was renamed in **0.4.0**: its working name was too close to
[paperhound](https://pypi.org/project/paperhound/), and `pip install paperhound`
installs **Alejandro Fernández Camello's project**, not this one. The
distribution and CLI are now both `papersplease`; the two projects are not
affiliated.

* Want eight providers, a local FTS5 library, docling conversion and rerank?
  Use [paperhound](https://pypi.org/project/paperhound/).
* Want scope verdicts, reasons, an MCP server and zero dependencies? Use
  PapersPlease.

The similarity is why `SOURCES.md` exists and why the README says which project
you are reading.

## Reproducing this comparison

```bash
papersplease providers --json        # what this install can actually query
papersplease explain "sets and frequency for hypertrophy"   # the screening contract
python -m unittest discover -s tests                      # 179 tests, no network
```

For the other projects, their own READMEs are the source of truth:

* <https://pypi.org/project/paperhound/>
* <https://github.com/openags/paper-search-cli>
* <https://github.com/h-lu/paper-find-mcp>
* <https://github.com/MarkHershey/arxiv-dl>
* <https://github.com/papis/papis>
* <https://github.com/mrossinek/cobib>

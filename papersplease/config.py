"""Runtime configuration, all of it environment-variable driven."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SOURCES = ("europepmc", "crossref", "semanticscholar", "arxiv", "openalex")

DEFAULT_MIRRORS = (
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
)


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def default_cache_dir() -> Path:
    override = os.environ.get("PAPERSPLEASE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "papersplease"


@dataclass
class Settings:
    mailto: str = ""
    openalex_key: str = ""
    s2_key: str = ""
    unpaywall_email: str = ""
    mirrors: list[str] = field(default_factory=lambda: list(DEFAULT_MIRRORS))
    sources: list[str] = field(default_factory=lambda: list(DEFAULT_SOURCES))
    cache_dir: Path = field(default_factory=default_cache_dir)
    cache: bool = True
    timeout: float = 25.0
    user_agent: str = ""
    min_scope: float = 0.7
    # Optional second-opinion screening by an LLM (OpenAI-compatible endpoint).
    llm_base: str = ""
    llm_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_json_mode: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls()
        settings.mailto = os.environ.get("PAPERSPLEASE_MAILTO", "")
        settings.openalex_key = os.environ.get("PAPERSPLEASE_OPENALEX_KEY", "")
        settings.s2_key = os.environ.get("PAPERSPLEASE_S2_KEY", "")
        settings.unpaywall_email = (
            os.environ.get("PAPERSPLEASE_UNPAYWALL_EMAIL") or settings.mailto
        )
        mirrors = _split(os.environ.get("PAPERSPLEASE_SCIHUB_MIRRORS"))
        if mirrors:
            settings.mirrors = [m.rstrip("/") for m in mirrors]
        else:
            settings.mirrors = [m.rstrip("/") for m in settings.mirrors]
        sources = _split(os.environ.get("PAPERSPLEASE_SOURCES"))
        if sources:
            settings.sources = sources
        if os.environ.get("PAPERSPLEASE_NO_CACHE"):
            settings.cache = False
        if os.environ.get("PAPERSPLEASE_CACHE_DIR"):
            settings.cache_dir = default_cache_dir()
        try:
            settings.min_scope = float(os.environ.get("PAPERSPLEASE_MIN_SCOPE", "0.7"))
        except ValueError:
            settings.min_scope = 0.7
        # Accept either a dedicated key or the conventional OPENAI_API_KEY.
        settings.llm_base = (
            os.environ.get("PAPERSPLEASE_LLM_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or ("https://api.openai.com/v1" if os.environ.get("PAPERSPLEASE_LLM_KEY") or os.environ.get("OPENAI_API_KEY") else "")
        ).rstrip("/")
        settings.llm_key = os.environ.get("PAPERSPLEASE_LLM_KEY") or os.environ.get("OPENAI_API_KEY", "")
        settings.llm_model = os.environ.get("PAPERSPLEASE_LLM_MODEL", settings.llm_model)
        if os.environ.get("PAPERSPLEASE_LLM_JSON") in {"0", "false", "no"}:
            settings.llm_json_mode = False
        from . import __version__

        contact = settings.mailto or settings.unpaywall_email or "anonymous"
        settings.user_agent = os.environ.get(
            "PAPERSPLEASE_USER_AGENT",
            f"papersplease/{__version__} (+https://github.com/thijnperd/Sci-crawler; mailto:{contact})",
        )
        return settings

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base and self.llm_key)

    @property
    def polite_email(self) -> str:
        # No invented fallback: both Crossref's polite pool and Unpaywall want a
        # real contact address, so when the user has not set one we skip the
        # lookup and say so in the report rather than sending a fake email.
        return self.unpaywall_email or self.mailto

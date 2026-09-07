from __future__ import annotations

import re

import requests

from ..models import ResearchResult, Source


class WikipediaInformationProvider:
    """A replaceable web research provider backed by the MediaWiki API."""

    def __init__(self, language: str = "ru", max_results: int = 5, timeout_seconds: int = 20):
        self.language = language
        self.max_results = max_results
        self.timeout_seconds = timeout_seconds
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"

    def research(self, event: str) -> ResearchResult:
        search_response = requests.get(
            self.api_url,
            params={
                "action": "query",
                "list": "search",
                "srsearch": event,
                "srlimit": self.max_results,
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": "historical-panorama/0.1 (educational project)"},
            timeout=self.timeout_seconds,
        )
        search_response.raise_for_status()
        titles = [row["title"] for row in search_response.json()["query"]["search"]]
        if not titles:
            raise LookupError(f"No web sources found for event: {event}")

        extract_response = requests.get(
            self.api_url,
            params={
                "action": "query",
                "prop": "extracts|info",
                "explaintext": 1,
                "inprop": "url",
                "redirects": 1,
                "titles": "|".join(titles),
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": "historical-panorama/0.1 (educational project)"},
            timeout=self.timeout_seconds,
        )
        extract_response.raise_for_status()
        pages = extract_response.json()["query"]["pages"].values()
        sources = [Source(title=p["title"], url=p.get("fullurl", ""), text=p.get("extract", "").strip()[:6000])
                   for p in pages if p.get("extract", "").strip()]
        query_terms = self._terms(event)
        sources.sort(
            key=lambda source: self._relevance(source, query_terms),
            reverse=True,
        )
        if not sources:
            raise LookupError(f"Sources were found, but no readable extracts were returned for: {event}")
        return ResearchResult(query=event, sources=sources)

    @staticmethod
    def _terms(text: str) -> set[str]:
        # Five-character prefixes provide a small language-independent approximation
        # of stemming (e.g. Russian крепости/крепость -> крепо).
        return {word[:5] for word in re.findall(r"[\w-]+", text.lower()) if len(word) >= 4}

    @classmethod
    def _relevance(cls, source: Source, query_terms: set[str]) -> tuple[int, int]:
        title_terms = cls._terms(source.title)
        text_terms = cls._terms(source.text)
        return (3 * len(query_terms & title_terms) + len(query_terms & text_terms), -len(source.text))

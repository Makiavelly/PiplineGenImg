from __future__ import annotations

import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup, Tag

from ..models import ResearchResult, Source, WikipediaEventMetadata

EXCLUDED_SECTIONS = {
    "references", "bibliography", "literature", "sources", "external links",
    "see also", "notes", "further reading", "примечания", "литература",
    "источники", "ссылки", "внешние ссылки", "см. также", "библиография",
}
VISUAL_TERMS = {
    "architecture", "building", "clothing", "costume", "weapon", "army",
    "transport", "ship", "horse", "nature", "geography", "people", "culture",
    "архитект", "строител", "одежд", "костюм", "оруж", "войск", "транспорт",
    "кораб", "лошад", "природ", "географ", "народ", "культур", "быт",
}


@dataclass(frozen=True)
class _Article:
    title: str
    url: str
    sections: list[Source]
    links: list[str]


class WikipediaInformationProvider:
    """Wikipedia-only research with redirects, disambiguation and ranked related pages."""

    def __init__(
        self,
        language: str = "ru",
        max_results: int = 5,
        timeout_seconds: int = 20,
        max_related_articles: int = 6,
        session: requests.Session | None = None,
    ):
        if not 0 <= max_related_articles <= 8:
            raise ValueError("max_related_articles must be between 0 and 8")
        self.language = language
        self.max_results = max(1, max_results)
        self.max_related_articles = max_related_articles
        self.timeout_seconds = timeout_seconds
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        self.session = session or requests.Session()
        self.headers = {"User-Agent": "historical-panorama/0.2 (educational project)"}

    def research(self, event: str) -> ResearchResult:
        limitations: list[str] = []
        try:
            candidates = self._search_titles(event, self.max_results)
        except requests.RequestException as exc:
            return ResearchResult(
                query=event,
                limitations=[f"Wikipedia search unavailable: {type(exc).__name__}: {exc}"],
            )
        if not candidates:
            return ResearchResult(query=event, limitations=["No Wikipedia article found for the event"])

        try:
            primary_title = self._choose_primary(candidates, event, limitations)
        except (requests.RequestException, ValueError, KeyError) as exc:
            return ResearchResult(
                query=event,
                limitations=[f"Wikipedia candidates unavailable: {type(exc).__name__}: {exc}"],
            )
        if not primary_title:
            return ResearchResult(
                query=event,
                limitations=limitations or ["No non-disambiguation Wikipedia article found"],
            )
        try:
            primary = self._parse_article(primary_title, is_primary=True, relevance=1.0)
        except (requests.RequestException, ValueError, KeyError) as exc:
            return ResearchResult(
                query=event,
                primary_article=primary_title,
                limitations=limitations
                + [f"Primary Wikipedia article could not be parsed: {type(exc).__name__}: {exc}"],
            )

        related_titles = self._select_related_titles(event, primary, limitations)
        articles = [primary]
        for rank, title in enumerate(related_titles, start=1):
            try:
                articles.append(
                    self._parse_article(
                        title,
                        is_primary=False,
                        relevance=max(0.1, 1.0 - rank / (len(related_titles) + 1)),
                    )
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                limitations.append(
                    f"Related Wikipedia article {title!r} unavailable: {type(exc).__name__}: {exc}"
                )

        return ResearchResult(
            query=event,
            sources=[source for article in articles for source in article.sections],
            primary_article=primary.title,
            related_articles=[article.title for article in articles[1:]],
            metadata=self._extract_metadata(event, primary),
            limitations=limitations,
        )

    def _get(self, **params: object) -> dict:
        response = self.session.get(
            self.api_url,
            params={"format": "json", "formatversion": 2, "utf8": 1, **params},
            headers=self.headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _search_titles(self, query: str, limit: int) -> list[str]:
        data = self._get(action="query", list="search", srsearch=query, srlimit=limit)
        return [row["title"] for row in data.get("query", {}).get("search", []) if row.get("title")]

    def _choose_primary(self, titles: list[str], event: str, limitations: list[str]) -> str:
        data = self._get(
            action="query", prop="extracts|info|pageprops", exintro=1, explaintext=1,
            inprop="url", redirects=1, titles="|".join(titles),
        )
        pages = data.get("query", {}).get("pages", [])
        valid = [page for page in pages if not page.get("missing")]
        ordinary = [page for page in valid if "disambiguation" not in page.get("pageprops", {})]
        if ordinary:
            terms = self._terms(event)
            best = max(
                ordinary,
                key=lambda page: self._text_relevance(
                    f"{page.get('title', '')} {page.get('extract', '')}", terms
                ),
            )
            return str(best.get("title", ""))
        if valid:
            title = str(valid[0].get("title", ""))
            limitations.append(f"Search resolved to disambiguation page: {title}")
            ranked = sorted(
                self._parse_links(title),
                key=lambda candidate: self._text_relevance(candidate, self._terms(event)),
                reverse=True,
            )
            if ranked:
                limitations.append(f"Selected {ranked[0]!r} from disambiguation candidates")
                return ranked[0]
        limitations.append("All candidate Wikipedia pages were missing")
        return ""

    def _parse_links(self, title: str) -> list[str]:
        data = self._get(action="parse", page=title, prop="links", redirects=1)
        return [
            str(link["title"])
            for link in data.get("parse", {}).get("links", [])
            if link.get("ns") == 0 and not link.get("missing", False)
        ]

    def _parse_article(self, title: str, *, is_primary: bool, relevance: float) -> _Article:
        data = self._get(action="parse", page=title, prop="text|links", redirects=1)
        parsed = data.get("parse")
        if not parsed:
            raise ValueError(f"Wikipedia parse response is empty for {title!r}")
        resolved = str(parsed.get("title", title))
        html = parsed.get("text", "")
        if isinstance(html, dict):
            html = html.get("*", "")
        sections = self._sections_from_html(str(html), resolved, is_primary, relevance)
        links = [
            str(link["title"])
            for link in parsed.get("links", [])
            if link.get("ns") == 0 and not link.get("missing", False)
        ]
        return _Article(resolved, self._article_url(resolved), sections, links)

    def _sections_from_html(
        self, html: str, article_title: str, is_primary: bool, relevance: float
    ) -> list[Source]:
        soup = BeautifulSoup(html, "html.parser")
        infobox_lines: list[str] = []
        for table in soup.select("table.infobox"):
            for row in table.select("tr"):
                heading = row.find("th")
                value = row.find("td")
                if heading and value:
                    key = self._clean_text(heading.get_text(" ", strip=True))
                    text = self._clean_text(value.get_text(" ", strip=True))
                    if key and text and self._is_visual_infobox_field(key):
                        infobox_lines.append(f"{key}: {text}")
        for selector in (
            "table", "style", "script", "sup.reference", ".mw-editsection", ".navbox",
            ".vertical-navbox", ".metadata", ".ambox", ".hatnote", ".magnify",
        ):
            for node in soup.select(selector):
                node.decompose()
        collected: dict[str, list[str]] = {"Lead": []}
        if infobox_lines:
            collected["Infobox"] = infobox_lines
        current = "Lead"
        excluded_depth: int | None = None
        for node in soup.find_all(["h2", "h3", "h4", "p", "li"]):
            if not isinstance(node, Tag):
                continue
            if node.name in {"h2", "h3", "h4"}:
                level = int(node.name[1])
                heading = self._clean_text(node.get_text(" ", strip=True))
                if self._excluded_heading(heading):
                    excluded_depth, current = level, ""
                elif excluded_depth is not None and level > excluded_depth:
                    current = ""
                else:
                    excluded_depth = None
                    current = heading or "Unnamed section"
                    collected.setdefault(current, [])
                continue
            if current:
                text = self._clean_text(node.get_text(" ", strip=True))
                if len(text) >= 20 and text not in collected[current]:
                    collected[current].append(text)
        url = self._article_url(article_title)
        return [
            Source(
                title=article_title, url=url, text="\n".join(chunks)[:8000], section=section,
                article_title=article_title, is_primary=is_primary, relevance=relevance,
            )
            for section, chunks in collected.items()
            if chunks
        ]

    def _select_related_titles(
        self, event: str, primary: _Article, limitations: list[str]
    ) -> list[str]:
        if self.max_related_articles == 0:
            return []
        pool = set(primary.links)
        suffixes = (
            ["место архитектура одежда оружие", "народы быт транспорт природа"]
            if self.language == "ru"
            else ["place architecture clothing weapons", "peoples daily life transport nature"]
        )
        for suffix in suffixes:
            try:
                pool.update(self._search_titles(f"{primary.title} {suffix}", self.max_results * 2))
            except requests.RequestException as exc:
                limitations.append(f"Related-article search partially unavailable: {type(exc).__name__}")
        pool.discard(primary.title)
        terms = self._terms(f"{event} {primary.title}")
        ranked = sorted(pool, key=lambda title: self._related_score(title, terms), reverse=True)
        relevant = [title for title in ranked if self._related_score(title, terms) > 0]
        if not relevant:
            limitations.append("No sufficiently relevant related Wikipedia articles were found")
        return relevant[: self.max_related_articles]

    def _extract_metadata(self, event: str, primary: _Article) -> WikipediaEventMetadata:
        lead = next((source.text for source in primary.sections if source.section == "Lead"), "")
        all_text = "\n".join(source.text for source in primary.sections)
        match = re.search(
            r"\b(?:1[0-9]{3}|20[0-9]{2})(?:\s*[–—-]\s*(?:1[0-9]{3}|20[0-9]{2}))?\b",
            all_text,
        )
        participants: list[str] = []
        place = ""
        visual: list[str] = []
        for source in primary.sections:
            name = source.section.lower()
            if name == "infobox":
                for line in source.text.splitlines():
                    key, _, value = line.partition(":")
                    normalized = key.lower().strip()
                    if normalized in {"место", "location", "place"} and not place:
                        place = value.strip()
                    if normalized in {
                        "участники", "стороны", "противники", "participants", "belligerents"
                    }:
                        participants.append(value.strip())
            if any(term in name for term in ("участ", "сторон", "forces", "participants")):
                participants.extend(self._sentences(source.text)[:4])
            if not place and any(term in name for term in ("место", "географ", "location", "place")):
                place = " ".join(self._sentences(source.text)[:2])[:500]
            if any(term in name for term in VISUAL_TERMS):
                visual.extend(self._sentences(source.text)[:3])
        return WikipediaEventMetadata(
            title=primary.title or event,
            date_or_period=match.group(0) if match else "",
            place=place,
            participants=participants,
            summary=" ".join(self._sentences(lead)[:3])[:1200],
            visual_reconstruction_notes=visual[:12],
        )

    def _article_url(self, title: str) -> str:
        slug = requests.utils.quote(title.replace(" ", "_"))
        return f"https://{self.language}.wikipedia.org/wiki/{slug}"

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\[[0-9, ]+]", "", text)).strip()

    @staticmethod
    def _excluded_heading(heading: str) -> bool:
        lowered = heading.lower().strip().rstrip(":")
        return lowered in EXCLUDED_SECTIONS or any(lowered.startswith(f"{x} ") for x in EXCLUDED_SECTIONS)

    @staticmethod
    def _is_visual_infobox_field(field: str) -> bool:
        normalized = field.lower().strip().rstrip(":")
        terms = (
            "дата", "период", "место", "участник", "сторон", "противник", "командир",
            "date", "period", "location", "place", "participant", "belligerent", "commander",
        )
        return any(term in normalized for term in terms)

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {word[:5] for word in re.findall(r"[\w-]+", text.lower()) if len(word) >= 4}

    @classmethod
    def _text_relevance(cls, text: str, terms: set[str]) -> int:
        return len(cls._terms(text) & terms)

    @classmethod
    def _related_score(cls, title: str, terms: set[str]) -> int:
        lowered = title.lower()
        return 3 * cls._text_relevance(title, terms) + sum(2 for word in VISUAL_TERMS if word in lowered)

    @classmethod
    def _relevance(cls, source: Source, query_terms: set[str]) -> tuple[int, int]:
        return (
            3 * len(query_terms & cls._terms(source.title)) + len(query_terms & cls._terms(source.text)),
            -len(source.text),
        )

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

from types import SimpleNamespace

import requests

from historical_panorama.models import Source
from historical_panorama.providers.wikipedia import WikipediaInformationProvider


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class Session:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, _url, *, params, **_kwargs):
        self.calls.append(params)
        value = self.handler(params)
        if isinstance(value, Exception):
            raise value
        return Response(value)


def test_wikipedia_provider_resolves_redirect_and_preserves_sections():
    def handler(params):
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Old event title"}]}}
        if params.get("prop") == "extracts|info|pageprops":
            return {"query": {"pages": [{"title": "Event", "extract": "Event in 1551"}]}}
        return {
            "parse": {
                "title": "Event",
                "text": "<p>Facts about the event in 1551 and its location.</p>"
                "<h2>Architecture</h2><p>Timber walls surrounded the settlement.</p>"
                "<h2>References</h2><p>This material must be removed.</p>",
                "links": [],
            }
        }

    result = WikipediaInformationProvider(max_related_articles=0, session=Session(handler)).research("event")

    assert result.primary_article == "Event"
    assert [source.section for source in result.sources] == ["Lead", "Architecture"]
    assert result.sources[0].url.endswith("/Event")
    assert "This material must be removed" not in result.context
    assert result.metadata.date_or_period == "1551"


def test_wikipedia_extracts_place_and_participants_from_infobox():
    provider = WikipediaInformationProvider(max_related_articles=0)
    html = (
        '<table class="infobox"><tr><th>Место</th><td>Высокий берег реки</td></tr>'
        '<tr><th>Участники</th><td>Строители и военные</td></tr></table>'
        '<p>Событие состоялось в 1551 году и подробно описано летописью.</p>'
    )
    sections = provider._sections_from_html(html, "Событие", True, 1.0)
    article = SimpleNamespace(title="Событие", sections=sections)
    metadata = provider._extract_metadata("Событие", article)
    assert metadata.place == "Высокий берег реки"
    assert metadata.participants == ["Строители и военные"]


def test_wikipedia_disambiguation_selects_a_candidate():
    def handler(params):
        if params.get("list") == "search":
            return {"query": {"search": [{"title": "Event (disambiguation)"}]}}
        if params.get("prop") == "extracts|info|pageprops":
            return {"query": {"pages": [{"title": "Event (disambiguation)", "pageprops": {"disambiguation": ""}}]}}
        if params.get("prop") == "links":
            return {"parse": {"links": [{"ns": 0, "title": "Event battle"}]}}
        return {"parse": {"title": "Event battle", "text": "<p>A documented historical battle occurred here.</p>", "links": []}}

    result = WikipediaInformationProvider(max_related_articles=0, session=Session(handler)).research("event battle")

    assert result.primary_article == "Event battle"
    assert any("disambiguation" in item for item in result.limitations)


def test_wikipedia_failure_is_recorded_instead_of_raised():
    result = WikipediaInformationProvider(
        session=Session(lambda _params: requests.ConnectionError("offline"))
    ).research("missing")
    assert result.sources == []
    assert "unavailable" in result.limitations[0]


def test_related_articles_are_ranked_and_limited():
    provider = WikipediaInformationProvider(
        max_related_articles=2, session=Session(lambda _p: {"query": {"search": []}})
    )
    primary = SimpleNamespace(
        title="Свияжская крепость",
        links=["Русская архитектура", "Футбол", "Русское оружие", "Повседневность"],
    )
    selected = provider._select_related_titles("Свияжская крепость", primary, [])
    assert len(selected) <= 2
    assert "Футбол" not in selected


def test_relevance_prefers_event_specific_source():
    provider = WikipediaInformationProvider()
    terms = provider._terms("Строительство крепости Свияжск в 1551 году")
    exact = provider._relevance(
        Source("Свияжская крепость", "", "Деревянная крепость возведена в 1551 году"), terms
    )
    generic = provider._relevance(Source("1551 год", "", "Календарный год"), terms)
    assert exact > generic

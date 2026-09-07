from unittest.mock import Mock, patch

from historical_panorama.models import Source
from historical_panorama.providers.wikipedia import WikipediaInformationProvider


def response(data):
    result = Mock()
    result.json.return_value = data
    result.raise_for_status.return_value = None
    return result


@patch("historical_panorama.providers.wikipedia.requests.get")
def test_wikipedia_provider_returns_sources(get):
    get.side_effect = [
        response({"query": {"search": [{"title": "Event"}]}}),
        response(
            {
                "query": {
                    "pages": {
                        "1": {
                            "title": "Event",
                            "fullurl": "https://example.test/event",
                            "extract": "Facts about the event.",
                        }
                    }
                }
            }
        ),
    ]

    result = WikipediaInformationProvider().research("event")

    assert result.sources[0].title == "Event"
    assert "Facts" in result.context
    assert get.call_count == 2


def test_relevance_prefers_event_specific_source():
    provider = WikipediaInformationProvider()
    terms = provider._terms("Строительство крепости Свияжск в 1551 году")
    exact = provider._relevance(
        Source("Свияжская крепость", "", "Деревянная крепость возведена в 1551 году"),
        terms,
    )
    generic = provider._relevance(
        Source("1551 год", "", "Календарный год"),
        terms,
    )
    assert exact > generic

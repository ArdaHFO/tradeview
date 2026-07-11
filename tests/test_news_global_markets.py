from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from forecaster.config import Config
from forecaster.news import fetch


def test_strip_corporate_suffix_turkish():
    assert fetch._strip_corporate_suffix(
        "ASELSAN Elektronik Sanayi ve Ticaret Anonim Sirketi"
    ) == "ASELSAN Elektronik Sanayi ve Ticaret"


def test_strip_corporate_suffix_multiple_forms():
    assert fetch._strip_corporate_suffix("Turk Hava Yollari Anonim Ortakligi") == "Turk Hava Yollari"
    assert fetch._strip_corporate_suffix("LVMH Moet Hennessy Louis Vuitton SE") == "LVMH Moet Hennessy Louis Vuitton"
    assert fetch._strip_corporate_suffix("Apple Inc.") == "Apple"
    assert fetch._strip_corporate_suffix("SAP SE") == "SAP"


def test_strip_corporate_suffix_no_suffix_is_unchanged():
    assert fetch._strip_corporate_suffix("Tesla") == "Tesla"


def test_locale_for_symbol_by_exchange_suffix():
    assert fetch._locale_for_symbol("ASELS.IS") == ("tr-TR", "TR", "TR:tr")
    assert fetch._locale_for_symbol("SAP.DE") == ("de-DE", "DE", "DE:de")
    assert fetch._locale_for_symbol("MC.PA") == ("fr-FR", "FR", "FR:fr")


def test_locale_for_symbol_defaults_to_us_for_unsuffixed():
    assert fetch._locale_for_symbol("AAPL") == fetch._DEFAULT_LOCALE


def test_locale_for_symbol_defaults_for_unknown_suffix():
    assert fetch._locale_for_symbol("XYZ.ZZ") == fetch._DEFAULT_LOCALE


def _entry(title: str, hours_ago: float):
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return SimpleNamespace(get=lambda k, d=None: {
        "title": title,
        "link": "http://x",
        "summary": "",
        "published_parsed": ts.timetuple(),
    }.get(k, d))


def test_fetch_articles_widens_lookback_when_nothing_recent(monkeypatch):
    # All matches are 200h old — outside the default 24h window but inside
    # the 720h (30-day) widened window.
    old_entries = [_entry("Some old headline about Foo", 200)]

    def fake_parse(url):
        return SimpleNamespace(entries=old_entries)

    import feedparser as real_feedparser
    monkeypatch.setattr(real_feedparser, "parse", fake_parse)

    cfg = Config(groq_api_key="", news_lookback_hours=24)
    articles = fetch.fetch_articles("FOO", "Foo Inc.", cfg, ["google"])
    assert len(articles) == 1
    assert articles[0].title == "Some old headline about Foo"


def test_fetch_articles_returns_empty_when_nothing_found_at_any_window(monkeypatch):
    import feedparser as real_feedparser
    monkeypatch.setattr(real_feedparser, "parse", lambda url: SimpleNamespace(entries=[]))

    cfg = Config(groq_api_key="", news_lookback_hours=24)
    articles = fetch.fetch_articles("FOO", "Foo Inc.", cfg, ["google"])
    assert articles == []

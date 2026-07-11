from types import SimpleNamespace

from forecaster import symbols_search


def test_filters_foreign_and_non_equity(monkeypatch):
    fake_quotes = [
        {"symbol": "AAPL", "quoteType": "EQUITY", "longname": "Apple Inc."},
        {"symbol": "AAPL.SW", "quoteType": "EQUITY", "longname": "Apple Inc. (Swiss)"},
        {"symbol": "BRK-B", "quoteType": "EQUITY", "longname": "Berkshire Hathaway"},
        {"symbol": "APLEBOND", "quoteType": "BOND", "longname": "Some Bond"},
    ]
    monkeypatch.setattr(
        symbols_search.yf, "Search",
        lambda q, max_results: SimpleNamespace(quotes=fake_quotes),
    )
    results = symbols_search.search_symbols("apple")
    symbols = [r["symbol"] for r in results]
    assert symbols == ["AAPL", "AAPL.SW", "BRK-B"]


def test_empty_query_returns_empty_list():
    assert symbols_search.search_symbols("") == []
    assert symbols_search.search_symbols("   ") == []


def test_search_error_returns_empty_list(monkeypatch):
    def boom(q, max_results):
        raise RuntimeError("network down")

    monkeypatch.setattr(symbols_search.yf, "Search", boom)
    assert symbols_search.search_symbols("apple") == []

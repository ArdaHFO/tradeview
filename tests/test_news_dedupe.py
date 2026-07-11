from forecaster.news.fetch import _is_duplicate, _normalize_title


def test_normalize_strips_punctuation_and_case():
    assert _normalize_title("HELLO World!") == "hello world"
    assert _normalize_title("Apple's Q3 Earnings: Beat!") == "apples q3 earnings beat"


def test_near_duplicate_titles_detected():
    seen = [_normalize_title("Apple stock rises on strong iPhone sales")]
    assert _is_duplicate("Apple stock rises on strong iPhone sales today", seen, threshold=0.85)


def test_distinct_titles_not_duplicate():
    seen = [_normalize_title("Apple stock rises on strong iPhone sales")]
    assert not _is_duplicate("Tesla recalls vehicles over battery issue", seen, threshold=0.85)

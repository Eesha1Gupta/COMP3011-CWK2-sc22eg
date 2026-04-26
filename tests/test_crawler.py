"""
test_crawler.py - Unit and integration tests for the Crawler module.

Run with:
    pytest tests/test_crawler.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crawler import Crawler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML = """
<html>
<body>
  <a href="/page1">Page 1</a>
  <a href="/page2">Page 2</a>
  <a href="https://external.com/other">External</a>
  <a href="#section">Fragment only</a>
</body>
</html>
"""

SAMPLE_HTML_2 = """
<html>
<body>
  <p>Page two content</p>
  <a href="/">Home</a>
</body>
</html>
"""


@pytest.fixture
def crawler():
    """Return a Crawler instance pointed at the test site."""
    return Crawler("https://quotes.toscrape.com", politeness_window=0)


# ---------------------------------------------------------------------------
# URL Utilities
# ---------------------------------------------------------------------------

class TestNormaliseUrl:
    def test_strips_trailing_slash(self, crawler):
        assert crawler._normalise_url("https://example.com/page/") == "https://example.com/page"

    def test_strips_fragment(self, crawler):
        assert crawler._normalise_url("https://example.com/page#section") == "https://example.com/page"

    def test_preserves_query_string(self, crawler):
        url = "https://example.com/search?q=hello"
        assert crawler._normalise_url(url) == url

    def test_no_change_for_clean_url(self, crawler):
        url = "https://example.com/page"
        assert crawler._normalise_url(url) == url


class TestIsSameDomain:
    def test_same_domain_returns_true(self, crawler):
        assert crawler._is_same_domain("https://quotes.toscrape.com/page1") is True

    def test_different_domain_returns_false(self, crawler):
        assert crawler._is_same_domain("https://external.com/page") is False

    def test_subdomain_returns_false(self, crawler):
        assert crawler._is_same_domain("https://sub.quotes.toscrape.com/") is False


# ---------------------------------------------------------------------------
# Link extraction
# ---------------------------------------------------------------------------

class TestExtractLinks:
    def test_extracts_internal_links(self, crawler):
        links = crawler._extract_links(SAMPLE_HTML, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/page1" in links
        assert "https://quotes.toscrape.com/page2" in links

    def test_excludes_external_links(self, crawler):
        links = crawler._extract_links(SAMPLE_HTML, "https://quotes.toscrape.com/")
        assert all("external.com" not in link for link in links)

    def test_handles_fragment_only_links(self, crawler):
        """Fragment-only <a href="#section"> should resolve to current page."""
        links = crawler._extract_links(SAMPLE_HTML, "https://quotes.toscrape.com/")
        # After normalisation the base URL (without fragment) may appear
        # but external links must not appear
        assert all("external.com" not in link for link in links)

    def test_empty_page_returns_no_links(self, crawler):
        links = crawler._extract_links("<html><body></body></html>", "https://quotes.toscrape.com/")
        assert links == []

    def test_relative_links_are_resolved(self, crawler):
        html = '<a href="author/einstein">Einstein</a>'
        links = crawler._extract_links(html, "https://quotes.toscrape.com/")
        assert "https://quotes.toscrape.com/author/einstein" in links


# ---------------------------------------------------------------------------
# Fetch page
# ---------------------------------------------------------------------------

class TestFetchPage:
    def test_returns_html_on_success(self, crawler):
        mock_response = MagicMock()
        mock_response.text = "<html>Hello</html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(crawler.session, "get", return_value=mock_response):
            result = crawler.fetch_page("https://quotes.toscrape.com/")

        assert result == "<html>Hello</html>"

    def test_returns_none_on_http_error(self, crawler):
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404")

        with patch.object(crawler.session, "get", return_value=mock_response):
            result = crawler.fetch_page("https://quotes.toscrape.com/nonexistent")

        assert result is None

    def test_returns_none_on_connection_error(self, crawler):
        import requests
        with patch.object(
            crawler.session, "get", side_effect=requests.ConnectionError("connection refused")
        ):
            result = crawler.fetch_page("https://quotes.toscrape.com/")

        assert result is None

    def test_returns_none_on_timeout(self, crawler):
        import requests
        with patch.object(
            crawler.session, "get", side_effect=requests.Timeout("timed out")
        ):
            result = crawler.fetch_page("https://quotes.toscrape.com/")

        assert result is None


# ---------------------------------------------------------------------------
# Full crawl (mocked)
# ---------------------------------------------------------------------------

class TestCrawl:
    def test_crawls_pages_via_bfs(self, crawler):
        """Crawler should follow links and index all reachable pages."""
        html_home = '<a href="/page1">P1</a><a href="/page2">P2</a>'
        html_p1 = "<p>Page one</p>"
        html_p2 = "<p>Page two</p>"

        url_map = {
            "https://quotes.toscrape.com": html_home,
            "https://quotes.toscrape.com/page1": html_p1,
            "https://quotes.toscrape.com/page2": html_p2,
        }

        def fake_fetch(url):
            return url_map.get(url)

        crawler.fetch_page = fake_fetch
        pages = crawler.crawl()

        assert len(pages) == 3
        assert "https://quotes.toscrape.com" in pages
        assert "https://quotes.toscrape.com/page1" in pages
        assert "https://quotes.toscrape.com/page2" in pages

    def test_does_not_revisit_pages(self, crawler):
        """Each URL should be fetched exactly once."""
        call_counts: dict[str, int] = {}
        html = '<a href="/loop">Loop</a>'

        def fake_fetch(url):
            call_counts[url] = call_counts.get(url, 0) + 1
            return html

        crawler.fetch_page = fake_fetch
        crawler.crawl()

        assert all(count == 1 for count in call_counts.values())

    def test_handles_fetch_failure_gracefully(self, crawler):
        """A failed page should not crash the crawl; other pages still fetched."""
        def fake_fetch(url):
            if "bad" in url:
                return None
            return '<a href="/bad">Bad</a><p>OK</p>'

        crawler.fetch_page = fake_fetch
        pages = crawler.crawl()

        # Home page fetched successfully; bad page returned None and was skipped
        assert "https://quotes.toscrape.com" in pages
        assert "https://quotes.toscrape.com/bad" not in pages

    def test_visited_set_populated_after_crawl(self, crawler):
        crawler.fetch_page = lambda url: "<p>content</p>"
        crawler.crawl()
        assert "https://quotes.toscrape.com" in crawler.visited

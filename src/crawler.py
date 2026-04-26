"""
crawler.py - Web Crawler for the Search Engine

Crawls all pages of a target website, respecting a politeness window
between requests to avoid overloading the server.
"""

import time
import logging
from urllib.parse import urljoin, urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class Crawler:
    """
    A polite web crawler that fetches pages from a target website.

    Attributes:
        base_url (str): The root URL of the website to crawl.
        politeness_window (float): Seconds to wait between requests.
        session (requests.Session): Persistent HTTP session for efficiency.
        visited (set): URLs that have already been crawled.
        pages (dict): Mapping of URL -> raw page content (HTML text).
    """

    def __init__(self, base_url: str, politeness_window: float = 6.0):
        """
        Initialise the crawler.

        Args:
            base_url: Root URL to crawl (e.g. 'https://quotes.toscrape.com/').
            politeness_window: Minimum seconds between HTTP requests (default 6).
        """
        self.base_url = base_url.rstrip("/")
        self.politeness_window = politeness_window
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "COMP3011-SearchEngine/1.0 (University of Leeds coursework)"}
        )
        self.visited: set[str] = set()
        self.pages: dict[str, str] = {}

    def _is_same_domain(self, url: str) -> bool:
        """Return True if *url* belongs to the same domain as base_url."""
        return urlparse(url).netloc == urlparse(self.base_url).netloc

    def _normalise_url(self, url: str) -> str:
        """Strip trailing slashes and fragments for consistent deduplication."""
        parsed = urlparse(url)
        # Rebuild without fragment; keep scheme + netloc + path + params + query
        normalised = parsed._replace(fragment="").geturl()
        return normalised.rstrip("/")

    def _extract_links(self, html: str, current_url: str) -> list[str]:
        """
        Parse HTML and return all internal links found on the page.

        Args:
            html: Raw HTML content of the page.
            current_url: URL the HTML was fetched from (used to resolve relative links).

        Returns:
            List of absolute, normalised URLs belonging to the same domain.
        """
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            absolute = urljoin(current_url, href)
            normalised = self._normalise_url(absolute)
            if self._is_same_domain(normalised):
                links.append(normalised)
        return links

    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a single page and return its HTML content.

        Returns None on any network or HTTP error (graceful failure).
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    def crawl(self) -> dict[str, str]:
        """
        Crawl the entire website starting from base_url using BFS.

        Respects the politeness window between requests.

        Returns:
            dict mapping URL -> HTML content for every successfully fetched page.
        """
        queue: list[str] = [self._normalise_url(self.base_url)]
        self.visited.clear()
        self.pages.clear()

        while queue:
            url = queue.pop(0)

            if url in self.visited:
                continue

            self.visited.add(url)
            logger.info("Crawling: %s", url)
            print(f"  Crawling: {url}")

            html = self.fetch_page(url)
            if html is None:
                continue

            self.pages[url] = html

            new_links = self._extract_links(html, url)
            for link in new_links:
                if link not in self.visited:
                    queue.append(link)

            # Politeness window – only sleep if there are more pages to fetch
            if queue:
                time.sleep(self.politeness_window)

        logger.info("Crawl complete. %d pages fetched.", len(self.pages))
        return self.pages

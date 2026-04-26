"""
indexer.py - Inverted Index Builder for the Search Engine

Parses raw HTML pages and constructs an inverted index mapping each word
to a list of postings (documents it appears in, with rich statistics).

Design choice: the index is stored as a plain Python dict (JSON-serialisable),
making it easy to save/load without third-party serialisation libraries.

Index structure:
    {
        "word": {
            "doc_freq": int,          # number of documents containing the word
            "postings": {
                "url": {
                    "term_freq": int,          # raw count in this document
                    "positions": [int, ...],   # 0-based word positions
                    "tf_idf": float,           # TF-IDF score (computed post-build)
                }
            }
        }
    }
"""

import re
import math
import logging
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stop-words (optional, kept minimal to preserve recall for the coursework)
# ---------------------------------------------------------------------------
STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "it", "as", "be",
}


def tokenise(text: str) -> list[str]:
    """
    Lowercase the text and split into alphabetic tokens.

    Non-alphabetic characters (punctuation, numbers) are treated as
    delimiters. This makes the search case-insensitive as required.

    Args:
        text: Plain text (not HTML) to tokenise.

    Returns:
        Ordered list of lowercase word tokens (stop-words included).
    """
    return re.findall(r"[a-z]+", text.lower())


def extract_text(html: str) -> str:
    """
    Strip HTML tags and return visible text content only.

    Args:
        html: Raw HTML string.

    Returns:
        Plain text with whitespace normalised to single spaces.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Remove script and style elements – not visible text
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    # Collapse excess whitespace
    return re.sub(r"\s+", " ", text).strip()


class Indexer:
    """
    Builds and manages an inverted index over a collection of HTML pages.

    Usage::

        indexer = Indexer()
        indexer.build(pages)   # pages: dict[url, html]
        indexer.compute_tf_idf()
    """

    def __init__(self):
        """Initialise an empty index."""
        # index[word] = {"doc_freq": int, "postings": {url: {...}}}
        self.index: dict[str, Any] = {}
        # Total number of documents in the collection
        self.num_docs: int = 0
        # Per-document metadata (useful for TF-IDF denominator)
        self.doc_lengths: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, pages: dict[str, str]) -> None:
        """
        Index all pages in *pages*.

        Args:
            pages: Mapping of URL -> raw HTML content.
        """
        self.index.clear()
        self.doc_lengths.clear()
        self.num_docs = len(pages)

        for url, html in pages.items():
            self._index_page(url, html)
        self.compute_tf_idf()
        logger.info("Index built: %d terms across %d documents.", len(self.index), self.num_docs)

    def _index_page(self, url: str, html: str) -> None:
        """
        Extract text from *html* and add all words to the inverted index.

        Args:
            url: The URL of the page (used as the document identifier).
            html: Raw HTML content of the page.
        """
        text = extract_text(html)
        tokens = tokenise(text)
        self.doc_lengths[url] = len(tokens)

        for position, word in enumerate(tokens):
            if word not in self.index:
                self.index[word] = {"doc_freq": 0, "postings": {}}

            entry = self.index[word]

            if url not in entry["postings"]:
                entry["doc_freq"] += 1
                entry["postings"][url] = {
                    "term_freq": 0,
                    "positions": [],
                    "tf_idf": 0.0,
                }

            posting = entry["postings"][url]
            posting["term_freq"] += 1
            posting["positions"].append(position)


    # ------------------------------------------------------------------
    # TF-IDF scoring 
    # ------------------------------------------------------------------

    def compute_tf_idf(self) -> None:
        """
        Compute TF-IDF scores for every (term, document) pair.

        TF  = term_freq / doc_length   (normalised term frequency)
        IDF = log(N / doc_freq)        (inverse document frequency)
        TF-IDF = TF * IDF
        """
        if self.num_docs == 0:
            return

        for word, entry in self.index.items():
            idf = math.log(self.num_docs / entry["doc_freq"])
            for url, posting in entry["postings"].items():
                tf = posting["term_freq"] / max(self.doc_lengths.get(url, 1), 1)
                posting["tf_idf"] = round(tf * idf, 6)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_postings(self, word: str) -> dict:
        """
        Return the postings dict for *word*, or an empty dict if not found.

        The search is case-insensitive: 'Good' and 'good' resolve to the same entry.
        """
        return self.index.get(word.lower(), {}).get("postings", {})

    def find(self, query: str) -> list[tuple[str, float]]:
        """
        Return pages containing **all** words in *query*, ranked by TF-IDF.

        For a multi-word query, a page must contain every query word.
        Results are sorted by the sum of TF-IDF scores (descending).

        Args:
            query: One or more space-separated search terms.

        Returns:
            List of (url, score) tuples, best match first.
            Returns an empty list if no pages match or query is empty.
        """
        words = tokenise(query)
        if not words:
            return []

        # Start with the set of pages for the first word
        candidate_sets = [set(self.get_postings(w).keys()) for w in words]
        matching_urls = candidate_sets[0]
        for s in candidate_sets[1:]:
            matching_urls = matching_urls.intersection(s)

        if not matching_urls:
            return []

        # Score each candidate by summing TF-IDF across all query terms
        scored: list[tuple[str, float]] = []
        for url in matching_urls:
            score = sum(
                self.index[w]["postings"][url]["tf_idf"]
                for w in words
                if w in self.index and url in self.index[w]["postings"]
            )
            scored.append((url, round(score, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def print_postings(self, word: str) -> str:
        """
        Return a human-readable string of the inverted index for *word*.

        Args:
            word: The term to look up.

        Returns:
            Formatted multi-line string, or a 'not found' message.
        """
        word_lower = word.lower()
        if word_lower not in self.index:
            return f"Word '{word}' not found in index."

        entry = self.index[word_lower]
        lines = [
            f"Inverted index for '{word_lower}':",
            f"  Document frequency: {entry['doc_freq']}",
            "",
        ]
        for url, posting in entry["postings"].items():
            lines.append(f"  URL:       {url}")
            lines.append(f"  Term freq: {posting['term_freq']}")
            lines.append(f"  TF-IDF:    {posting['tf_idf']}")
            lines.append(f"  Positions: {posting['positions'][:20]}"
                         + (" ..." if len(posting["positions"]) > 20 else ""))
            lines.append("")
        return "\n".join(lines)

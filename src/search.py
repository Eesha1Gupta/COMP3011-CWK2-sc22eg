"""
search.py - Index Storage & Retrieval for the Search Engine

Handles serialisation of the inverted index to disk (JSON format) and
provides the high-level search interface used by main.py.

Design choice: JSON was chosen over pickle for portability and human
readability. The index file can be inspected in any text editor, making
debugging straightforward.
"""

import json
import logging
import os
from typing import Optional

from indexer import Indexer

logger = logging.getLogger(__name__)

# Default path for the persisted index file
DEFAULT_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "index.json"
)


class SearchEngine:
    """
    High-level search engine facade.

    Wraps the Indexer and handles persistence so that callers
    (e.g. main.py) can use a simple, stable interface.
    """

    def __init__(self, index_path: str = DEFAULT_INDEX_PATH):
        """
        Initialise the search engine.

        Args:
            index_path: Path to the JSON file where the index is stored.
        """
        self.index_path = index_path
        self.indexer = Indexer()
        self._loaded = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """
        Serialise the index to a JSON file at *index_path*.

        The directory is created automatically if it does not exist.

        Raises:
            IOError: If the file cannot be written.
        """
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        payload = {
            "num_docs": self.indexer.num_docs,
            "doc_lengths": self.indexer.doc_lengths,
            "index": self.indexer.index,
        }

        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        size_kb = os.path.getsize(self.index_path) / 1024
        logger.info("Index saved to %s (%.1f KB).", self.index_path, size_kb)
        print(f"Index saved to '{self.index_path}' ({size_kb:.1f} KB, "
              f"{len(self.indexer.index):,} terms, "
              f"{self.indexer.num_docs} documents).")
        self._loaded = True

    def load(self) -> bool:
        """
        Deserialise the index from *index_path*.

        Returns:
            True on success, False if the file does not exist or is malformed.
        """
        if not os.path.exists(self.index_path):
            print(f"Index file not found at '{self.index_path}'. "
                  "Run 'build' first.")
            return False

        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)

            self.indexer.num_docs = payload["num_docs"]
            self.indexer.doc_lengths = payload["doc_lengths"]
            self.indexer.index = payload["index"]
            self._loaded = True

            size_kb = os.path.getsize(self.index_path) / 1024
            logger.info("Index loaded from %s.", self.index_path)
            print(f"Index loaded from '{self.index_path}' ({size_kb:.1f} KB, "
                  f"{len(self.indexer.index):,} terms, "
                  f"{self.indexer.num_docs} documents).")
            return True

        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to load index: %s", exc)
            print(f"Error loading index: {exc}")
            return False

    # ------------------------------------------------------------------
    # Commands forwarded from main.py
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> str:
        """
        Return a formatted postings listing for *word*.

        Args:
            word: The search term to look up.
        """
        if not self._loaded:
            return "No index loaded. Run 'build' or 'load' first."
        if not word.strip():
            return "Usage: print <word>"
        return self.indexer.print_postings(word.strip())

    def find(self, query: str) -> str:
        """
        Search for pages matching all words in *query*.

        Returns a formatted results string ranked by TF-IDF score.

        Args:
            query: One or more space-separated search terms.
        """
        if not self._loaded:
            return "No index loaded. Run 'build' or 'load' first."

        query = query.strip()
        if not query:
            return "Usage: find <word> [word ...]"

        results = self.indexer.find(query)

        if not results:
            return f"No pages found containing: '{query}'"

        lines = [f"Found {len(results)} page(s) for query '{query}':\n"]
        for rank, (url, score) in enumerate(results, start=1):
            lines.append(f"  {rank}. {url}")
            lines.append(f"     TF-IDF score: {score:.6f}")
        return "\n".join(lines)

    def is_ready(self) -> bool:
        """Return True if an index is currently loaded in memory."""
        return self._loaded

"""
test_search.py - Unit tests for the SearchEngine / storage module.

Run with:
    pytest tests/test_search.py -v
"""

import pytest
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search import SearchEngine
from indexer import Indexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PAGES = {
    "http://example.com/a": "<p>The quick brown fox</p>",
    "http://example.com/b": "<p>The quick cat jumped</p>",
    "http://example.com/c": "<p>A lazy dog slept well</p>",
}


@pytest.fixture
def tmp_index_path(tmp_path):
    """Return a temporary path for the index file."""
    return str(tmp_path / "test_index.json")


@pytest.fixture
def built_engine(tmp_index_path):
    """Return a SearchEngine with an index already built and saved."""
    engine = SearchEngine(index_path=tmp_index_path)
    engine.indexer.build(SAMPLE_PAGES)
    engine._loaded = True
    engine.save()
    return engine


# ---------------------------------------------------------------------------
# Save & Load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_creates_file(self, built_engine, tmp_index_path):
        assert os.path.exists(tmp_index_path)

    def test_save_produces_valid_json(self, built_engine, tmp_index_path):
        with open(tmp_index_path) as fh:
            data = json.load(fh)
        assert "index" in data
        assert "num_docs" in data
        assert "doc_lengths" in data

    def test_load_restores_index(self, tmp_index_path, built_engine):
        # Create a fresh engine and load from disk
        fresh_engine = SearchEngine(index_path=tmp_index_path)
        result = fresh_engine.load()
        assert result is True
        assert fresh_engine.is_ready()
        assert len(fresh_engine.indexer.index) == len(built_engine.indexer.index)

    def test_load_returns_false_when_no_file(self, tmp_index_path):
        engine = SearchEngine(index_path=tmp_index_path)
        assert engine.load() is False

    def test_load_returns_false_for_corrupt_json(self, tmp_index_path):
        with open(tmp_index_path, "w") as fh:
            fh.write("not valid json {{{{")
        engine = SearchEngine(index_path=tmp_index_path)
        assert engine.load() is False

    def test_num_docs_preserved_after_round_trip(self, tmp_index_path, built_engine):
        fresh = SearchEngine(index_path=tmp_index_path)
        fresh.load()
        assert fresh.indexer.num_docs == built_engine.indexer.num_docs

    def test_tf_idf_preserved_after_round_trip(self, tmp_index_path, built_engine):
        original_score = built_engine.indexer.index["quick"]["postings"]["http://example.com/a"]["tf_idf"]
        fresh = SearchEngine(index_path=tmp_index_path)
        fresh.load()
        loaded_score = fresh.indexer.index["quick"]["postings"]["http://example.com/a"]["tf_idf"]
        assert abs(original_score - loaded_score) < 1e-9


# ---------------------------------------------------------------------------
# print_word command
# ---------------------------------------------------------------------------

class TestPrintWord:
    def test_returns_postings_info(self, built_engine):
        output = built_engine.print_word("quick")
        assert "quick" in output
        assert "Document frequency" in output

    def test_unknown_word_message(self, built_engine):
        output = built_engine.print_word("xyzzy")
        assert "not found" in output.lower()

    def test_empty_word_returns_usage(self, built_engine):
        output = built_engine.print_word("")
        assert "Usage" in output

    def test_no_index_loaded_message(self, tmp_index_path):
        engine = SearchEngine(index_path=tmp_index_path)
        output = engine.print_word("quick")
        assert "No index loaded" in output

    def test_case_insensitive(self, built_engine):
        lower = built_engine.print_word("fox")
        upper = built_engine.print_word("FOX")
        assert lower == upper


# ---------------------------------------------------------------------------
# find command
# ---------------------------------------------------------------------------

class TestFind:
    def test_single_word_finds_pages(self, built_engine):
        output = built_engine.find("fox")
        assert "http://example.com/a" in output

    def test_multi_word_finds_intersection(self, built_engine):
        output = built_engine.find("quick fox")
        assert "http://example.com/a" in output
        assert "http://example.com/b" not in output

    def test_no_match_message(self, built_engine):
        output = built_engine.find("unicorn")
        assert "No pages found" in output

    def test_empty_query_returns_usage(self, built_engine):
        output = built_engine.find("")
        assert "Usage" in output

    def test_whitespace_only_query(self, built_engine):
        output = built_engine.find("   ")
        assert "Usage" in output

    def test_no_index_loaded_message(self, tmp_index_path):
        engine = SearchEngine(index_path=tmp_index_path)
        output = engine.find("quick")
        assert "No index loaded" in output

    def test_results_include_rank(self, built_engine):
        output = built_engine.find("the")
        assert "1." in output

    def test_results_include_score(self, built_engine):
        output = built_engine.find("the")
        assert "TF-IDF" in output

    def test_case_insensitive_search(self, built_engine):
        """Results for 'quick' and 'QUICK' should contain the same URLs."""
        lower = built_engine.find("quick")
        upper = built_engine.find("QUICK")
        # Extract URLs from both outputs for comparison (query string differs in display)
        lower_urls = [line.strip() for line in lower.splitlines() if line.strip().startswith("http")]
        upper_urls = [line.strip() for line in upper.splitlines() if line.strip().startswith("http")]
        assert lower_urls == upper_urls


# ---------------------------------------------------------------------------
# is_ready()
# ---------------------------------------------------------------------------

class TestIsReady:
    def test_false_before_build_or_load(self, tmp_index_path):
        engine = SearchEngine(index_path=tmp_index_path)
        assert engine.is_ready() is False

    def test_true_after_build(self, built_engine):
        assert built_engine.is_ready() is True

    def test_true_after_successful_load(self, tmp_index_path, built_engine):
        fresh = SearchEngine(index_path=tmp_index_path)
        fresh.load()
        assert fresh.is_ready() is True

"""
test_indexer.py - Unit tests for the Indexer module.

Run with:
    pytest tests/test_indexer.py -v
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from indexer import Indexer, tokenise, extract_text


# ---------------------------------------------------------------------------
# tokenise()
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_text(self):
        assert tokenise("Hello World") == ["hello", "world"]

    def test_splits_on_punctuation(self):
        assert tokenise("well, it's fine!") == ["well", "it", "s", "fine"]

    def test_empty_string_returns_empty_list(self):
        assert tokenise("") == []

    def test_numbers_are_excluded(self):
        tokens = tokenise("3 blind mice")
        assert "3" not in tokens
        assert "blind" in tokens

    def test_preserves_order(self):
        assert tokenise("the quick brown fox") == ["the", "quick", "brown", "fox"]

    def test_handles_special_characters(self):
        assert tokenise("café au lait") == ["caf", "au", "lait"]


# ---------------------------------------------------------------------------
# extract_text()
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_strips_html_tags(self):
        text = extract_text("<p>Hello <strong>world</strong></p>")
        assert "<p>" not in text
        assert "Hello" in text
        assert "world" in text

    def test_removes_script_tags(self):
        html = "<script>var x = 1;</script><p>visible</p>"
        text = extract_text(html)
        assert "var x" not in text
        assert "visible" in text

    def test_removes_style_tags(self):
        html = "<style>body{color:red}</style><p>content</p>"
        text = extract_text(html)
        assert "color" not in text
        assert "content" in text

    def test_collapses_whitespace(self):
        html = "<p>too   many    spaces</p>"
        text = extract_text(html)
        assert "  " not in text


# ---------------------------------------------------------------------------
# Indexer.build()
# ---------------------------------------------------------------------------

PAGES = {
    "http://example.com/a": "<p>The quick brown fox</p>",
    "http://example.com/b": "<p>The quick cat jumped</p>",
    "http://example.com/c": "<p>A lazy dog slept</p>",
}


@pytest.fixture
def built_indexer():
    idx = Indexer()
    idx.build(PAGES)
    return idx


class TestIndexerBuild:
    def test_words_present_in_index(self, built_indexer):
        assert "quick" in built_indexer.index
        assert "fox" in built_indexer.index

    def test_doc_freq_counts_correctly(self, built_indexer):
        # 'quick' appears in pages a and b
        assert built_indexer.index["quick"]["doc_freq"] == 2

    def test_term_freq_counts_correctly(self, built_indexer):
        postings = built_indexer.index["quick"]["postings"]
        assert postings["http://example.com/a"]["term_freq"] == 1

    def test_positions_recorded(self, built_indexer):
        postings = built_indexer.index["fox"]["postings"]
        # "the quick brown fox" → fox is at position 3
        assert 3 in postings["http://example.com/a"]["positions"]

    def test_case_insensitive_indexing(self, built_indexer):
        # 'The' should be stored as 'the'
        assert "the" in built_indexer.index
        assert "The" not in built_indexer.index

    def test_num_docs_set_correctly(self, built_indexer):
        assert built_indexer.num_docs == 3

    def test_rebuild_clears_old_data(self):
        idx = Indexer()
        idx.build({"http://a.com": "<p>hello world</p>"})
        idx.build({"http://b.com": "<p>goodbye world</p>"})
        assert "hello" not in idx.index
        assert "goodbye" in idx.index


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TestTfIdf:
    def test_tf_idf_scores_are_non_negative(self, built_indexer):
        for word, entry in built_indexer.index.items():
            for url, posting in entry["postings"].items():
                assert posting["tf_idf"] >= 0, f"Negative TF-IDF for '{word}' in {url}"

    def test_rare_word_has_higher_idf(self, built_indexer):
        # 'fox' appears only in page a (df=1); 'quick' appears in a and b (df=2)
        fox_score = built_indexer.index["fox"]["postings"]["http://example.com/a"]["tf_idf"]
        quick_score = built_indexer.index["quick"]["postings"]["http://example.com/a"]["tf_idf"]
        assert fox_score > quick_score

    def test_no_division_by_zero_empty_doc(self):
        idx = Indexer()
        # Indexing an empty page should not raise
        idx.build({"http://empty.com": "<p></p>"})


# ---------------------------------------------------------------------------
# get_postings()
# ---------------------------------------------------------------------------

class TestGetPostings:
    def test_returns_postings_for_known_word(self, built_indexer):
        postings = built_indexer.get_postings("quick")
        assert "http://example.com/a" in postings

    def test_case_insensitive_lookup(self, built_indexer):
        assert built_indexer.get_postings("Quick") == built_indexer.get_postings("quick")

    def test_returns_empty_dict_for_unknown_word(self, built_indexer):
        assert built_indexer.get_postings("xyzzy") == {}


# ---------------------------------------------------------------------------
# find()
# ---------------------------------------------------------------------------

class TestFind:
    def test_finds_single_word(self, built_indexer):
        results = built_indexer.find("fox")
        urls = [r[0] for r in results]
        assert "http://example.com/a" in urls

    def test_find_multi_word_returns_intersection(self, built_indexer):
        # 'quick' is in a and b; 'fox' only in a → intersection = {a}
        results = built_indexer.find("quick fox")
        urls = [r[0] for r in results]
        assert urls == ["http://example.com/a"]

    def test_find_no_match_returns_empty(self, built_indexer):
        assert built_indexer.find("unicorn") == []

    def test_find_returns_sorted_by_tfidf(self, built_indexer):
        results = built_indexer.find("the")
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_find_empty_query_returns_empty(self, built_indexer):
        assert built_indexer.find("") == []

    def test_find_whitespace_only_query(self, built_indexer):
        assert built_indexer.find("   ") == []

    def test_find_case_insensitive(self, built_indexer):
        lower = built_indexer.find("quick")
        upper = built_indexer.find("QUICK")
        assert lower == upper


# ---------------------------------------------------------------------------
# print_postings()
# ---------------------------------------------------------------------------

class TestPrintPostings:
    def test_prints_word_info(self, built_indexer):
        output = built_indexer.print_postings("quick")
        assert "quick" in output
        assert "Document frequency" in output

    def test_not_found_message_for_unknown_word(self, built_indexer):
        output = built_indexer.print_postings("zythum")
        assert "not found" in output.lower()

    def test_case_insensitive_print(self, built_indexer):
        lower = built_indexer.print_postings("quick")
        upper = built_indexer.print_postings("QUICK")
        assert lower == upper

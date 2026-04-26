"""
main.py - Command-Line Interface for the Search Engine

Provides an interactive shell with four commands:
    build  - crawl the target website and create the inverted index
    load   - load a previously saved index from disk
    print  - display the inverted index entry for a given word
    find   - find pages containing one or more search terms

Usage:
    python main.py
"""

import sys
import logging
import os

# Allow running from the src/ directory directly
sys.path.insert(0, os.path.dirname(__file__))

from crawler import Crawler
from search import SearchEngine

# ---------------------------------------------------------------------------
# Logging – written to file so it doesn't clutter the interactive shell
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=os.path.join(os.path.dirname(os.path.dirname(__file__)), "search_engine.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_URL = "https://quotes.toscrape.com/"
POLITENESS_WINDOW = 6.0  # seconds between HTTP requests

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║         COMP3011 Search Engine  –  Leeds 2025/26     ║
╠══════════════════════════════════════════════════════╣
║  Commands:                                           ║
║    build         – crawl site and build index        ║
║    load          – load saved index from disk        ║
║    print <word>  – show inverted index for a word    ║
║    find <query>  – find pages matching query         ║
║    help          – show this help message            ║
║    quit / exit   – exit the shell                    ║
╚══════════════════════════════════════════════════════╝
"""


def run_shell(engine: SearchEngine) -> None:
    """
    Start the interactive command-line shell.

    Reads commands from stdin in a loop until the user types 'quit' or 'exit'.
    """
    print(BANNER)

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command == "build":
            _cmd_build(engine)

        elif command == "load":
            engine.load()

        elif command == "print":
            if not argument:
                print("Usage: print <word>")
            else:
                print(engine.print_word(argument))

        elif command == "find":
            if not argument:
                print("Usage: find <word> [word ...]")
            else:
                print(engine.find(argument))

        elif command in ("help", "?"):
            print(BANNER)

        elif command in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        else:
            print(f"Unknown command: '{command}'. Type 'help' for usage.")


def _cmd_build(engine: SearchEngine) -> None:
    """Crawl the target website, build the index, and save it to disk."""
    print(f"Starting crawl of {TARGET_URL}")
    print(f"Politeness window: {POLITENESS_WINDOW}s between requests")
    print("This may take several minutes – please wait...\n")

    crawler = Crawler(TARGET_URL, politeness_window=POLITENESS_WINDOW)
    pages = crawler.crawl()

    if not pages:
        print("No pages fetched. Check your internet connection.")
        return

    print(f"\nCrawl complete: {len(pages)} pages fetched.")
    print("Building inverted index...")

    engine.indexer.build(pages)
    engine.save()


def main() -> None:
    """Entry point – creates the engine and launches the shell."""
    engine = SearchEngine()
    run_shell(engine)


if __name__ == "__main__":
    main()

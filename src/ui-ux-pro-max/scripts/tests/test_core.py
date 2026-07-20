#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stdlib-only regression tests for core.py / design_system.py (unittest, not
pytest -- this project ships with zero external dependencies and the tests
shouldn't add one).

Run with:
    python -m unittest discover -s scripts/tests -v
or directly:
    python scripts/tests/test_core.py
"""

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from core import (
    AVAILABLE_STACKS,
    BM25,
    CSV_CONFIG,
    _resolve_rtl_level,
    _search_csv,
    detect_domain,
    search,
    search_stack,
)
from design_system import generate_design_system, persist_design_system, DesignSystemGenerator


class TestTokenizer(unittest.TestCase):
    def test_short_domain_terms_are_kept(self):
        bm25 = BM25()
        tokens = bm25.tokenize("UI and UX design with 3D and AI")
        self.assertIn("ui", tokens)
        self.assertIn("3d", tokens)
        self.assertIn("ai", tokens)

    def test_stopwords_removed(self):
        bm25 = BM25()
        tokens = bm25.tokenize("this is for the team to do")
        for stopword in ("is", "for", "the", "to", "do"):
            self.assertNotIn(stopword, tokens)

    def test_synonym_normalization(self):
        bm25 = BM25()
        self.assertEqual(bm25.tokenize("e-commerce store"), bm25.tokenize("ecommerce store"))
        self.assertEqual(bm25.tokenize("dark-mode toggle"), bm25.tokenize("dark toggle"))


class TestSearchDomains(unittest.TestCase):
    """Known query -> expected top-domain sanity checks (not exact-row pinning,
    since data can grow; these assert the engine still finds *something*
    relevant for each domain's core vocabulary)."""

    def test_ui_is_searchable_in_style_domain(self):
        result = search("ui minimalism", domain="style", max_results=1)
        self.assertGreater(result["count"], 0, "literal 'ui' token must be searchable, not filtered by tokenizer")

    def test_accessibility_query_hits_ux(self):
        result = search("accessibility contrast wcag keyboard", domain="ux", max_results=3)
        self.assertGreater(result["count"], 0)

    def test_zero_result_query_reports_suggestions_not_error(self):
        result = search("zzqqxx totally made up gibberish", domain="ux", max_results=2)
        self.assertEqual(result["count"], 0)
        self.assertIn("suggestions", result)
        self.assertNotIn("error", result)

    def test_every_configured_domain_file_exists_and_is_searchable(self):
        for domain, config in CSV_CONFIG.items():
            with self.subTest(domain=domain):
                result = search("design", domain=domain, max_results=1)
                self.assertNotIn("error", result, f"domain '{domain}' failed: {result.get('error')}")

    def test_every_stack_file_exists_and_is_searchable(self):
        for stack in AVAILABLE_STACKS:
            with self.subTest(stack=stack):
                result = search_stack("performance", stack, max_results=1)
                self.assertNotIn("error", result, f"stack '{stack}' failed: {result.get('error')}")


class TestDomainDetection(unittest.TestCase):
    def test_style_keywords_route_to_style(self):
        self.assertEqual(detect_domain("glassmorphism dark ui"), "style")

    def test_accessibility_keywords_route_to_ux(self):
        self.assertEqual(detect_domain("accessibility contrast wcag"), "ux")

    def test_ambiguous_query_returns_runner_up(self):
        domain, runner_up = detect_domain("font pairing elegant crypto", return_scores=True)
        self.assertIsNotNone(domain)
        # runner_up may be None if the winning domain has no close second --
        # this just verifies the call shape works without raising.

    def test_empty_query_falls_back_to_style(self):
        self.assertEqual(detect_domain("...!!!???"), "style")


class TestRTLFiltering(unittest.TestCase):
    def test_dataset_levels_match_maintainer_examples(self):
        data_dir = SCRIPTS_DIR.parent / "data"
        with (data_dir / "styles.csv").open(encoding="utf-8", newline="") as handle:
            styles = {row["Style Category"]: row["rtl_level"] for row in csv.DictReader(handle)}
        with (data_dir / "products.csv").open(encoding="utf-8", newline="") as handle:
            products = {row["Product Type"]: row["rtl_level"] for row in csv.DictReader(handle)}

        self.assertEqual(styles["Minimalism & Swiss Style"], "full")
        self.assertEqual(styles["Glassmorphism"], "partial")
        self.assertEqual(styles["Motion-Driven"], "caveats")
        self.assertEqual(styles["Brutalism"], "full")
        self.assertEqual(products["SaaS (General)"], "full")
        self.assertEqual(set(styles.values()), {"full", "partial", "caveats"})
        self.assertEqual(set(products.values()), {"full", "partial", "caveats"})

    def test_level_filter_scans_beyond_initial_rank_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "rtl.csv"
            data_file.write_text(
                "Name,rtl_level\nmatch first,partial\nmatch second,full\n",
                encoding="utf-8",
            )
            results, _ = _search_csv(
                data_file,
                ["Name"],
                ["Name", "rtl_level"],
                "match",
                1,
                rtl="full",
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["rtl_level"], "full")

    def test_legacy_boolean_metadata_is_supported(self):
        self.assertEqual(_resolve_rtl_level({"rtl_compatible": "TRUE"}), "full")
        self.assertEqual(_resolve_rtl_level({"rtl_compatible": "FALSE"}), "")

    def test_public_search_supports_bare_and_specific_filters(self):
        all_levels = search("saas", domain="product", max_results=3, rtl="all")
        self.assertGreater(all_levels["count"], 0)
        self.assertTrue(
            all(row["rtl_level"] in {"full", "partial", "caveats"} for row in all_levels["results"])
        )

        full_only = search("saas", domain="product", max_results=3, rtl="full")
        self.assertGreater(full_only["count"], 0)
        self.assertTrue(all(row["rtl_level"] == "full" for row in full_only["results"]))

    def test_cli_accepts_optional_rtl_value(self):
        search_script = SCRIPTS_DIR / "search.py"
        queries = {
            "full": "Minimalism",
            "partial": "Glassmorphism",
            "caveats": "Motion-Driven",
        }
        for level, query in queries.items():
            with self.subTest(level=level):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(search_script),
                        query,
                        "--domain",
                        "style",
                        f"--rtl={level}",
                        "--json",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = json.loads(completed.stdout)
                self.assertGreater(result["count"], 0)
                self.assertTrue(all(row["rtl_level"] == level for row in result["results"]))

    def test_design_system_propagates_rtl_filter(self):
        result = generate_design_system("saas dashboard", "RTL Project", rtl="full")
        self.assertTrue(result["design_system"]["rtl"]["enabled"])
        self.assertEqual(result["design_system"]["rtl"]["filter"], "full")
        self.assertIn("RTL GUIDELINES", result["text"])


class TestPersistence(unittest.TestCase):
    def test_persist_then_skip_then_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_design_system("saas dashboard", "Test Project", persist=True, output_dir=tmp)
            self.assertEqual(result["persistence"]["status"], "success")
            master = Path(result["persistence"]["master_file"])
            self.assertTrue(master.exists())
            original_content = master.read_text(encoding="utf-8")

            # Second persist without force must not overwrite.
            result2 = generate_design_system("saas dashboard", "Test Project", persist=True, output_dir=tmp)
            self.assertEqual(result2["persistence"]["status"], "skipped_exists")
            self.assertEqual(master.read_text(encoding="utf-8"), original_content)

            # With force=True it must overwrite.
            result3 = generate_design_system("ecommerce luxury", "Test Project", persist=True, output_dir=tmp, force=True)
            self.assertEqual(result3["persistence"]["status"], "success")

    def test_persist_writes_only_under_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_design_system("saas dashboard", "Scoped Project", persist=True, output_dir=tmp)
            expected = Path(tmp) / "design-system" / "scoped-project" / "MASTER.md"
            self.assertTrue(expected.exists())


class TestReasoningMatch(unittest.TestCase):
    def test_known_category_matches_exactly(self):
        gen = DesignSystemGenerator()
        rule = gen._find_reasoning_rule("SaaS (General)")
        self.assertTrue(rule, "exact-match category lookup should not fall through to fuzzy matching")

    def test_unknown_category_falls_back_gracefully(self):
        gen = DesignSystemGenerator()
        rule = gen._find_reasoning_rule("Totally Unknown Category XYZ")
        # Should not raise; may return {} which _apply_reasoning handles with defaults.
        self.assertIsInstance(rule, dict)


if __name__ == "__main__":
    unittest.main()

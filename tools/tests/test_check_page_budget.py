"""The page-budget gate has to be trustworthy before anything is compressed on
its say-so, so its page-finding logic is tested against synthetic documents
rather than only against the real manuscript.

An earlier session reported the paper as fitting in nine pages when the rendered
PDF said ten. That claim was made by reading rather than by measuring, which is
the failure this module exists to prevent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_page_budget import body_end_page  # noqa: E402

FF = "\f"


def _doc(pages):
    return FF.join(pages)


def test_body_ends_on_page_before_references():
    doc = _doc(["intro", "method", "results", "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 3


def test_statements_between_body_and_references_do_not_count():
    doc = _doc(["intro", "conclusion", "E THICS STATEMENT\ntext",
                "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 2


def test_a_page_mixing_body_and_statements_still_counts_as_body():
    # The Conclusion spilling four lines onto the statements page is exactly the
    # situation being measured; that page is body and must be counted.
    doc = _doc(["intro",
                "conclusion continues here\nE THICS STATEMENT\ntext",
                "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 2


def test_the_word_references_in_prose_is_not_the_bibliography():
    # Section 5 says "oracle-code references" and "memorisation references" in
    # running prose. Case-insensitively matching the bare word finds those and
    # reports the bibliography starting on page 2.
    doc = _doc(["intro",
                "we include them as memorisation references rather than peers",
                "conclusion",
                "R EFERENCES\nsmith et al"])
    assert body_end_page(doc) == 3


def test_letterspaced_references_heading_is_found():
    doc = _doc(["a", "b", "R E F E R E N C E S"])
    assert body_end_page(doc) == 2


def test_plain_capitalised_references_heading_is_found():
    # This template renders the bibliography heading as plain "References",
    # unlike the statement headings, which come through letterspaced.
    doc = _doc(["a", "b", "References\nsmith et al"])
    assert body_end_page(doc) == 2


def test_references_heading_may_begin_partway_down_a_page():
    # The statements run on, so the bibliography commonly starts below the tail
    # of the Reproducibility statement rather than at the top of a page.
    doc = _doc(["intro", "conclusion",
                "E THICS STATEMENT\ntail of the statements\nReferences\nsmith et al"])
    assert body_end_page(doc) == 2


def test_missing_references_raises():
    try:
        body_end_page(_doc(["a", "b"]))
    except ValueError:
        return
    raise AssertionError("expected ValueError when References is absent")

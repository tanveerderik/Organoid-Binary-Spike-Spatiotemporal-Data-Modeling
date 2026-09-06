"""The anonymity scrubber is a desk-reject guard, so it gets a real test.

The case that matters most is the one that already slipped through once: a
case-sensitive name rule passed the build while leaving `TanveerDerik` in the
exported LICENSE. A scrubber that reports success without scrubbing is worse
than no scrubber, so `verify_anonymous` is tested against the same strings.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from MAGVIT_project.tools.make_release import (  # noqa: E402
    scrub, verify_anonymous, SCRUB, TEXT_SUFFIX,
)


@pytest.mark.parametrize("raw", [
    "@author: derik",
    "# -*- @author: derik",
    "Copyright (c) 2026 TanveerDerik",
    "Copyright (c) 2026 tanveerderik",
    'ROOT = Path("/media/derik/Seagate Desktop Drive/organoid_data")',
    "# prevents /media/... nesting",
    "git@github.com:tanveerderik/Organoid-Binary-Spike.git",
    "https://github.com/tanveerderik/Organoid-Binary-Spike",
    "cd /media/derik/Seagate Desktop Drive/organoid_data/MAGVIT_project",
])
def test_scrub_removes_identity(raw):
    out = scrub(raw)
    low = out.lower()
    assert "derik" not in low
    assert "seagate" not in low
    assert "/media/" not in out
    assert "github.com" not in low


def test_scrub_leaves_ordinary_text_alone():
    src = "def fit(x):\n    return x  # ordinary comment\n"
    assert scrub(src) == src


def test_verify_catches_what_scrub_would_miss(tmp_path):
    """verify_anonymous must use the same rules, case-insensitively."""
    (tmp_path / "LICENSE").write_text("Copyright (c) 2026 TanveerDerik\n")
    bad = verify_anonymous(tmp_path)
    assert len(bad) == 1 and "LICENSE:1" in bad[0]


def test_verify_passes_a_scrubbed_tree(tmp_path):
    (tmp_path / "a.py").write_text(scrub(
        '# @author: derik\nROOT = "/media/derik/Seagate Desktop Drive/x"\n'))
    assert verify_anonymous(tmp_path) == []


def test_binary_suffixes_are_not_rewritten():
    """A regex over a .pt would corrupt it while appearing to succeed."""
    assert ".pt" not in TEXT_SUFFIX
    assert ".npz" not in TEXT_SUFFIX
    assert ".pdf" not in TEXT_SUFFIX


def test_every_rule_is_case_insensitive_or_deliberately_not():
    """Name rules must carry re.I; the path rules need not."""
    import re
    for pat, _ in SCRUB:
        if "derik" in pat.pattern.lower() or "seagate" in pat.pattern.lower():
            assert pat.flags & re.I, f"name rule is case-sensitive: {pat.pattern}"


def test_release_tooling_is_not_exported():
    """The scrubber must not ship: it carries the identity list verbatim.

    `make_release.py` holds the surname, the drive name and the remote URL as
    literal regex sources, and this test file asserts against the same strings.
    Both were exported once. The built-in verifier could not see it, because it
    waves through any line containing "anonymised" or "PROJECT_ROOT" and every
    one of those lines contains both the name and the word.
    """
    import MAGVIT_project.tools.make_release as MR
    from pathlib import Path
    assert Path("tools/make_release.py") in MR.SKIP_FILE
    assert Path("tools/tests/test_make_release.py") in MR.SKIP_FILE


def test_name_rules_have_no_exemption():
    """Any match of a name rule is a leak, in every context."""
    import MAGVIT_project.tools.make_release as MR
    for probe in ("TanveerDerik", "derik", "Seagate Desktop Drive"):
        assert any(p.search(probe) for p in MR.NAME_RULES), probe
    # And the words the general verifier exempts must not rescue a real name.
    leak = '(re.compile(r"\\btanveerderik\\b", re.I), "anonymised"),'
    assert any(p.search(leak) for p in MR.NAME_RULES), \
        "a line naming the author is exempted when it also says 'anonymised'"

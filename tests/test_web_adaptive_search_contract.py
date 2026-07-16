"""Static contracts for the reply-token backward-search UI.

The behavioral path is exercised with Playwright.  These inexpensive checks
keep wire-critical details from silently regressing when the single-file UI
is refactored.
"""

from pathlib import Path


HTML = (Path(__file__).parents[1] / "web" / "index.html").read_text()


def test_reply_search_replaces_the_composer_in_place():
    assert HTML.count('id="wish-pop"') == 1
    chat_input = HTML.index('<div id="chat-input">')
    wish = HTML.index('id="wish-pop"', chat_input)
    editor = HTML.index('id="chat-input-row"', wish)
    assert chat_input < wish < editor
    assert 'id="wish-go" type="button" class="ok" disabled' in HTML


def test_adaptive_request_uses_valid_evidence_and_resume_fields():
    assert "sourceHits.push({ layer, rank: hitRank + 1 })" in HTML
    assert "msg_idx: meta && Number.isInteger(meta.msgIdx) ? meta.msgIdx : null" in HTML
    assert "(isFrontier ? 'frontier' : 'template')" in HTML
    assert "'/api/intervention_search_adaptive'" in HTML
    assert "exclude_recipe_keys:" in HTML
    assert "prior_promising:" in HTML
    assert "similarity_to_desired:" in HTML


def test_only_exact_verified_candidates_can_be_applied():
    assert "obj.verified === true" in HTML
    assert "obj.exact_response_match === true" in HTML
    assert "scope: { type: 'at', pos: Number(cell.position) }" in HTML
    assert "rerunWithInterventions({ tempOverride: 0 })" in HTML


def test_legacy_scan_is_not_exposed_from_the_detail_editor():
    assert 'id="iv-scan" type="button" hidden aria-hidden="true"' in HTML
    assert "openWishModal('conclusion')" not in HTML

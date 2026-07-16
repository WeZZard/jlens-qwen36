"""Static contracts for the reply-token backward-search UI.

The behavioral path is exercised with Playwright.  These inexpensive checks
keep wire-critical details from silently regressing when the single-file UI
is refactored.
"""

from pathlib import Path


HTML = (Path(__file__).parents[1] / "web" / "index.html").read_text()


def test_reply_search_and_composer_share_one_continuous_panel():
    assert HTML.count('id="wish-pop"') == 1
    chat_input = HTML.index('<div id="chat-input">')
    wish = HTML.index('id="wish-pop"', chat_input)
    editor = HTML.index('id="chat-input-row"', wish)
    assert chat_input < wish < editor
    assert 'id="wish-go" type="button" class="ok" disabled' in HTML
    assert "#chat-input.wishing #chat-input-row { display: none; }" not in HTML
    assert "width: min(var(--center-panel-width), calc(100vw - 32px))" in HTML


def test_adaptive_request_uses_valid_evidence_and_resume_fields():
    assert "sourceHits.push({ layer, rank: hitRank + 1 })" in HTML
    assert "msg_idx: meta && Number.isInteger(meta.msgIdx) ? meta.msgIdx : null" in HTML
    assert "(isFrontier ? 'frontier' : 'template')" in HTML
    assert "'/api/intervention_search_adaptive'" in HTML
    assert "allow_continuation: true" in HTML
    assert "'/api/intervention_search_adaptive_control'" in HTML
    assert "action === 'extend' ? { additional_time_seconds: 120 }" in HTML
    assert "event === 'search_paused'" in HTML
    assert "event === 'search_resumed'" in HTML


def test_deeper_search_is_one_disabled_two_minute_extension():
    assert 'id="wish-search-deeper" type="button" disabled' in HTML
    assert "Search deeper · +2 min" in HTML
    assert "search.awaitingExtension" in HTML
    assert "search.elapsedSeconds = search.budgetSeconds" in HTML


def test_background_search_keeps_composer_and_guards_stale_apply():
    assert "const backgroundWishSearch = !!(_wish && _wish.search)" in HTML
    assert "(!backgroundWishSearch && (state.streaming || !selectedReply))" in HTML
    assert "state.streaming || !wishSearchContextIsCurrent(wish)" in HTML
    assert "enabledInterventions().length === 0" in HTML


def test_only_exact_verified_candidates_can_be_applied():
    assert "obj.verified === true" in HTML
    assert "obj.exact_response_match === true" in HTML
    assert "scope: { type: 'at', pos: Number(cell.position) }" in HTML
    assert "rerunWithInterventions({ tempOverride: 0 })" in HTML


def test_legacy_scan_is_not_exposed_from_the_detail_editor():
    assert 'id="iv-scan" type="button" hidden aria-hidden="true"' in HTML
    assert "openWishModal('conclusion')" not in HTML

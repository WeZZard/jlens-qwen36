"""Static contracts for the reply-token backward-search UI.

The behavioral path is exercised with Playwright.  These inexpensive checks
keep wire-critical details from silently regressing when the single-file UI
is refactored.
"""

from pathlib import Path


HTML = (Path(__file__).parents[1] / "web" / "index.html").read_text()
SERVE = (Path(__file__).parents[1] / "jlens_qwen" / "serve.py").read_text()


def test_reply_search_replaces_the_model_composer_in_one_panel():
    assert HTML.count('id="wish-pop"') == 1
    chat_input = HTML.index('<div id="chat-input">')
    wish = HTML.index('id="wish-pop"', chat_input)
    editor = HTML.index('id="chat-input-row"', wish)
    assert chat_input < wish < editor
    assert 'id="wish-go" type="button" class="ok" disabled' in HTML
    assert "#chat-input.wishing #chat-input-row { display: none; }" in HTML
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


def test_modal_search_still_guards_stale_recipe_runs():
    assert "const wishPinned = uiWishPinned()" in HTML
    assert "return phase === 'searching' || phase === 'done'" in HTML
    assert "function runIsRecipeBaseline(run, recipe)" in HTML
    assert "runAssistantText(run) === recipe.baselineResponse" in HTML
    assert "runContextKey(run) === recipe.contextKey" in HTML
    assert "enabledInterventions().length === 0" in HTML


def test_only_exact_verified_candidates_receive_verified_recipe_status():
    assert "obj.verified === true" in HTML
    assert "obj.exact_response_match === true" in HTML
    assert "upsertWishInterventionRecipe(obj, 'verified')" in HTML
    assert "scope: { type: 'at', pos: Number(cell.position) }" in HTML
    assert "await rerunWithSpecs(recipeToSpecs(recipe)" in HTML
    assert "tempOverride: 0" in HTML


def test_sidebar_separates_markups_recipes_and_manual_interventions():
    markups = HTML.index('id="markups" aria-label="Saved markups"')
    recipes = HTML.index('id="intervention-recipes" role="radiogroup" aria-label="Intervention recipes"')
    manual = HTML.index('id="interventions" aria-label="Manual interventions"')
    assert markups < recipes < manual
    assert "Intervention recipes · ${recipes.length}" in HTML
    assert "Manual interventions · ${specs.length}" in HTML


def test_recipe_preview_reuses_panel_and_minimap_without_editable_fields():
    panel = HTML.index('id="intervene-popup"')
    recipe = HTML.index('id="iv-recipe-view"', panel)
    panel_end = HTML.index('</div>\n  </div>\n</div>\n\n<script>', recipe)
    recipe_markup = HTML[recipe:panel_end]
    assert 'id="iv-recipe-cells"' in recipe_markup
    assert 'id="iv-recipe-run"' in recipe_markup
    assert '<input' not in recipe_markup
    assert "recipe.cells.map((cell)" in HTML
    assert "scope: { type: 'at', pos: Number(cell.position) }" in HTML
    assert 'id="minimap" hidden aria-label="Intervention minimap"' in HTML


def test_recipe_choice_exposes_every_cell_strength_and_is_immutable():
    assert "btn.setAttribute('role', 'radio')" in HTML
    assert "state.selectedInterventionRecipeId = recipe.id" in HTML
    assert "meta.textContent = recipeCellSummary(recipe)" in HTML
    assert "alpha.textContent = `α ${formatAlpha(cell.alpha)}`" in HTML
    assert "_ivDraft = null" in HTML
    assert "Duplicate as manual" in HTML


def test_saved_sessions_round_trip_recipe_state():
    assert "interventionRecipes: list[dict[str, Any]] = []" in SERVE
    assert "selectedInterventionRecipeId: str | None = None" in SERVE
    assert "interventionRecipes: state.interventionRecipes.map(persistRecipe)" in HTML


def test_search_grid_only_exposes_recipe_backed_cells():
    assert "Raw/untested/rejected blocks are inert" in HTML
    assert "const recipe = wishSearchRecipeForCell(+el.dataset.pos, +el.dataset.layer)" in HTML
    assert "if (recipe) openRecipePreview(recipe.id)" in HTML
    assert "recipeIds: new Map()" in HTML
    assert "registerWishRecipeCells(recipe)" in HTML
    assert "html.wish-search-active #grid td.cell.selectable { cursor: default; }" in HTML
    assert "#grid td.cell.wish-search-promising" in HTML


def test_empty_recipe_section_stays_visible_during_search():
    assert "No selectable recipe yet" in HTML
    assert "if (!recipes.length && !search) return" in HTML
    assert "`${search.tested || 0} tested · search ${search.paused ? 'paused' : 'running'}`" in HTML


def test_legacy_scan_is_not_exposed_from_the_detail_editor():
    assert 'id="iv-scan" type="button" hidden aria-hidden="true"' in HTML
    assert "openWishModal('conclusion')" not in HTML


def test_premise_stage_reaches_the_ui_as_a_labeled_recipe_kind():
    assert "enable_premise_search: true" in HTML
    assert "event === 'premise_proposal'" in HTML
    assert "obj.class === 'premise_redirect' ? 'premise'" in HTML
    assert "obj.premise_verified === true" in HTML
    assert "search.premiseRecipeId" in HTML
    assert "recipe.kind === 'premise' && recipe.premise" in HTML
    assert "scope: { type: 'from', pos: 0 }" in HTML
    assert ": recipe.status === 'premise' ? 'PREMISE' : 'PROMISING'" in HTML
    assert "enable_premise_search: bool = False" in SERVE
    assert '"premise_redirect"' in SERVE
    assert "_premise_search_candidate" in SERVE
    assert "Premise recipe found — no exact match yet" in HTML

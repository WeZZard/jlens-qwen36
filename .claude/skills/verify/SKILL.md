---
name: verify
description: Build/launch/drive recipe for verifying jlens-qwen36 UI and server changes end-to-end in a real headless browser.
---

# Verify jlens-qwen36 changes

## Launch the server

```bash
JLENS_PATH=data/lens/qwen36_27b_neuronpedia_n1000.npz \
  uv run python -m uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

Poll `GET /api/lens` until it answers and confirm `"n_prompts": 1000`
before interpreting any result (project rule; startup takes ~30–60 s).

## Drive the UI headlessly

The repo has no Node dependency tree. The Playwright MCP server needs
Google Chrome, which is not installed — use a Node driver with a cached
Chromium instead, following `scripts/e2e_intervention_playwright.cjs`:

```bash
# Find a usable playwright install (npx cache works):
for d in ~/.npm/_npx/*/node_modules; do [ -d "$d/playwright" ] && echo "$d"; done
# Pick one whose chromium exists:
NODE_PATH=<dir> node -e "console.log(require('playwright').chromium.executablePath())"

NODE_PATH=<dir> node your-driver.cjs   # chromium.launch({ headless: true })
```

Driver facts that hold for `web/index.html` (single classic script):

- Top-level bindings (`state`, `_wish`, `$`, `addIntervention`,
  `rerunWithInterventions`, `setActiveVariant`, …) are reachable from
  `page.evaluate` even when declared with `let`/`const`.
- Send a chat turn: click `#chat-editor`, type, press Enter; then poll
  `!state.streaming && state.messages.at(-1).tokens.length > 0`.
  One short turn takes ~60–90 s (27B model + per-token lens readouts).
- Backward-search (wish) flow: click a reply token
  (`#chat-log .tok`, e.g. hasText 'Paris'), fill `#wish-input`, press
  Enter. Search requests go to `/api/intervention_search_adaptive`;
  candidate replays advance roughly one per second.
- A real intervened A/B rerun: `addIntervention({... mode: 'swap',
  token: ' Paris', target: ' Beijing', alpha: 1, layers: [52],
  scope: { type: 'at', pos: firstGeneratedPos() - 1 } ...})` then
  `rerunWithInterventions()`. The server resolves token text to ids.

## Flows worth driving

- Clean baseline chat → wish search starts (progress track, stats,
  pause button, recipes pill "N tested · search running").
- Dirty baseline (applied or enabled interventions) → wish refusal
  modal, no search state, no `/api/intervention_search_adaptive` POST.
- A/B compare: variant toggle labels (`Baseline`/`Run A` + config
  label), switching variants with `setActiveVariant(false|true)`.

## Gotchas

- Check the uvicorn log for exactly the requests you expect; refusals
  and client-side guards send nothing.
- `state.appliedInterventions` is frozen per run; only a fresh clean
  run, `clearState()`, or switching to a clean variant empties it.

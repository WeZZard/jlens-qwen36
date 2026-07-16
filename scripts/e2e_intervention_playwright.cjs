#!/usr/bin/env node

/*
 * Real-browser intervention smoke test.
 *
 * The repository intentionally has no Node dependency tree. Run this with a
 * Playwright installation supplied through NODE_PATH, for example:
 *
 *   NODE_PATH=/path/to/node_modules \
 *     node scripts/e2e_intervention_playwright.cjs
 */

'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');

const baseURL = process.env.JLENS_E2E_URL || 'http://127.0.0.1:8765';
const artifactDir = process.env.JLENS_E2E_ARTIFACTS || '/tmp/jlens-intervention-e2e';
const trials = process.env.JLENS_E2E_TRIALS
  ? JSON.parse(process.env.JLENS_E2E_TRIALS)
  : [
      { prompt: 'Tell me the capital of France.', source: 'Paris', target: 'Beijing' },
      { prompt: 'Say Paris and nothing else.', source: 'Paris', target: 'Beijing' },
      { prompt: 'Say London and nothing else.', source: 'London', target: 'Tokyo' },
    ];
const searchProfile = process.env.JLENS_E2E_PROFILE || 'standard';
const exercisePauseResume = process.env.JLENS_E2E_PAUSE_RESUME === '1';

function phraseCount(text, phrase) {
  const haystack = String(text || '').normalize('NFKC').toLocaleLowerCase();
  const needle = String(phrase || '').normalize('NFKC').toLocaleLowerCase();
  if (!needle) return 0;
  let count = 0;
  let offset = 0;
  while ((offset = haystack.indexOf(needle, offset)) >= 0) {
    count += 1;
    offset += Math.max(needle.length, 1);
  }
  return count;
}

function repeatedSuffix(tokens) {
  if (tokens.length < 12) return false;
  for (let period = 1; period <= 4; period += 1) {
    let repeated = true;
    for (let i = tokens.length - 1; i >= tokens.length - 12 + period; i -= 1) {
      if (tokens[i] !== tokens[i - period]) {
        repeated = false;
        break;
      }
    }
    if (repeated) return true;
  }
  return false;
}

async function browserState(page) {
  return page.evaluate(() => {
    const last = state.messages[state.messages.length - 1];
    const tokens = last && last.role === 'assistant'
      ? last.tokens.map((token) => token.text)
      : [];
    return {
      streaming: state.streaming,
      compareCompleted: state.compare ? state.compare.completed : null,
      text: tokens.join(''),
      tokens,
      interventions: state.interventions.map((spec) => ({
        mode: spec.mode,
        token: spec.token,
        tokenId: spec.tokenId,
        target: spec.target,
        targetId: spec.targetId,
        alpha: spec.alpha,
        layers: spec.layers,
        scope: spec.scope,
        enabled: spec.enabled,
      })),
    };
  });
}

async function clearConversation(page) {
  const visible = await page.locator('#wish-pop:not([hidden])').count();
  if (visible) {
    const close = page.locator('#wish-search-close:visible');
    if (await close.count()) await close.click();
    else await page.keyboard.press('Escape');
  }
  if (await page.locator('#clear-btn:not([disabled])').count()) {
    await page.locator('#clear-btn').click();
    await page.waitForFunction(() => state.messages.length === 0);
  }
}

async function runTrial(page, trial, index) {
  await clearConversation(page);
  await page.locator('#chat-editor').fill(trial.prompt);
  await page.locator('#chat-editor').press('Enter');
  await page.waitForFunction(() => {
    const assistants = document.querySelectorAll('#chat-log .msg.assistant .tok');
    return assistants.length > 0 && !state.streaming;
  }, null, { timeout: 120_000 });

  const baseline = await browserState(page);
  const replyTokens = page.locator('#chat-log .msg.assistant .tok');
  const tokenIndex = await replyTokens.evaluateAll((nodes, source) => {
    const texts = nodes.map((node) => node.textContent || '');
    const exact = texts.findIndex((text) => text.trim() === source);
    if (exact >= 0) return exact;
    // Exercise the same first-fragment path as the UI for split BPE words
    // such as Tok+yo or R+ome.
    for (let start = 0; start < texts.length; start += 1) {
      if (!/^\s*[\p{L}\p{M}]+$/u.test(texts[start])) continue;
      let joined = '';
      for (let end = start; end < texts.length; end += 1) {
        if (!/^\s*[\p{L}\p{M}]+$/u.test(texts[end])) break;
        if (end > start && /^\s/.test(texts[end])) break;
        joined += texts[end];
        if (joined.trim() === source) return start;
        if (joined.trim().length >= source.length) break;
      }
    }
    return -1;
  }, trial.source);
  if (tokenIndex < 0) {
    return {
      index,
      ...trial,
      baseline: baseline.text,
      status: 'source-token-not-found',
      replyTokens: await replyTokens.allTextContents(),
    };
  }

  await replyTokens.nth(tokenIndex).click();
  await page.locator('#wish-pop:not([hidden])').waitFor({ timeout: 10_000 });
  if (await page.locator('#chat-input-row').isVisible()) {
    throw new Error('model composer remained visible during reply intervention');
  }
  await page.locator('#wish-input').fill(trial.target);
  await page.locator('#wish-go').click();
  await page.locator('#wish-search-view:not([hidden])').waitFor({ timeout: 10_000 });
  if (await page.locator('#chat-input-row').isVisible()) {
    throw new Error('model composer remained visible during intervention search');
  }
  if (exercisePauseResume) {
    await page.waitForFunction(() => _wish?.search?.searchId &&
      _wish.search.tested >= 1 && _wish.search.running, null, { timeout: 30_000 });
    await page.locator('#wish-search-stop').click();
    await page.waitForFunction(() => _wish?.search?.paused === true &&
      !_wish.search.awaitingExtension, null, { timeout: 30_000 });
    const pausedElapsed = await page.evaluate(() => _wish.search.elapsedSeconds);
    await page.waitForTimeout(1_000);
    const stillPausedElapsed = await page.evaluate(() => _wish.search.elapsedSeconds);
    if (Math.abs(stillPausedElapsed - pausedElapsed) > 0.01) {
      throw new Error('paused active-search clock advanced');
    }
    await page.locator('#wish-search-stop').click();
    await page.waitForFunction(() => _wish?.search?.running === true &&
      _wish.search.paused === false, null, { timeout: 30_000 });
  }
  await page.waitForFunction(() => _wish?.search?.done === true ||
    _wish?.search?.paused === true, null, { timeout: 100_000 });
  let search = await page.evaluate(() => ({
    profile: _wish.search.profile,
    extended: _wish.search.extended,
    paused: _wish.search.paused,
    awaitingExtension: _wish.search.awaitingExtension,
    tested: _wish.search.tested,
    partials: _wish.search.partials,
    verified: _wish.search.verified,
    error: _wish.search.error || null,
  }));
  if (!search.verified && searchProfile === 'thorough' &&
      search.awaitingExtension &&
      await page.locator('#wish-search-deeper:not([hidden]):not([disabled])').count()) {
    await page.locator('#wish-search-deeper').click();
    await page.waitForFunction(() => _wish?.search?.extended === true &&
      (_wish.search.done === true || _wish.search.verified), null, { timeout: 220_000 });
    search = await page.evaluate(() => ({
      profile: _wish.search.profile,
      extended: _wish.search.extended,
      paused: _wish.search.paused,
      awaitingExtension: _wish.search.awaitingExtension,
      tested: _wish.search.tested,
      partials: _wish.search.partials,
      verified: _wish.search.verified,
      error: _wish.search.error || null,
    }));
  }

  const cells = await page.locator('#grid td.cell').evaluateAll((nodes) =>
    nodes.filter((node) => [...node.classList].some((name) => name.startsWith('wish-search-')))
      .map((node) => ({
        position: Number(node.dataset.pos),
        layer: Number(node.dataset.layer),
        classes: [...node.classList],
        title: node.title,
      })));
  const progress = {
    title: await page.locator('#wish-search-title').textContent(),
    stage: await page.locator('#wish-search-stage').textContent(),
    stats: await page.locator('#wish-search-stats').textContent(),
  };
  await page.screenshot({
    path: path.join(artifactDir, `trial-${index}-search.png`),
    fullPage: true,
  });

  if (!search.verified) {
    return {
      index,
      ...trial,
      baseline: baseline.text,
      status: search.error ? 'search-error' : 'no-verified-recipe',
      search,
      progress,
      cells,
    };
  }

  const applyVisible = await page.locator('#wish-search-apply:not([hidden])').count() > 0;
  if (!applyVisible) {
    return {
      index,
      ...trial,
      baseline: baseline.text,
      status: 'verified-recipe-not-applicable',
      search,
      progress,
      cells,
    };
  }

  const chosen = search.verified;
  await page.locator('#wish-search-apply').click();
  await page.waitForTimeout(5_000);
  const afterFiveSeconds = await browserState(page);
  const infiniteLike = afterFiveSeconds.streaming &&
    (repeatedSuffix(afterFiveSeconds.tokens) ||
      phraseCount(afterFiveSeconds.text, trial.target) >= 4);

  if (afterFiveSeconds.streaming) {
    await page.waitForFunction(() => !state.streaming, null, { timeout: 120_000 })
      .catch(async () => page.evaluate(() => stopStreaming()));
  }
  const final = await browserState(page);
  const correct = !infiniteLike &&
    phraseCount(final.text, trial.target) === 1 &&
    phraseCount(final.text, trial.source) === 0;
  const streamEvents = await page.evaluate(() => window.__jlensE2EStreams || []);
  const latestDone = [...streamEvents].reverse().find((entry) =>
    entry.url.endsWith('/api/chat_stream') && entry.events.some((event) => event.event === 'done'));

  await page.screenshot({
    path: path.join(artifactDir, `trial-${index}-rerun.png`),
    fullPage: true,
  });
  return {
    index,
    ...trial,
    baseline: baseline.text,
    status: infiniteLike ? 'infinite-loop-generation'
      : correct ? 'correctly-intervened'
        : 'intervention-incorrect',
    chosen,
    search: {
      profile: search.profile, extended: search.extended,
      tested: search.tested, partials: search.partials,
    },
    afterFiveSeconds,
    final,
    done: latestDone ? latestDone.events.find((event) => event.event === 'done') : null,
  };
}

(async () => {
  fs.mkdirSync(artifactDir, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    localStorage.clear();
    window.__jlensE2EStreams = [];
    const nativeFetch = window.fetch.bind(window);
    window.fetch = async (...args) => {
      const request = args[0];
      const url = typeof request === 'string' ? request : request.url;
      const init = args[1] || {};
      // Keep a smoke test from replacing the user's server-side autosave.
      if (url.endsWith('/api/sessions') && String(init.method || 'GET').toUpperCase() === 'POST') {
        return new Response(JSON.stringify({ id: 'e2e-autosave-suppressed' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      const response = await nativeFetch(...args);
      if (url.endsWith('/api/chat_stream') || url.endsWith('/api/intervention_search_adaptive')) {
        const clone = response.clone();
        clone.text().then((body) => {
          const events = body.split('\n\n').map((frame) => {
            let event = 'message';
            let data = null;
            for (const line of frame.split('\n')) {
              if (line.startsWith('event: ')) event = line.slice(7).trim();
              if (line.startsWith('data: ')) {
                try { data = JSON.parse(line.slice(6)); } catch (_) { data = null; }
              }
            }
            return { event, data };
          }).filter((item) => [
            'search_start', 'position_ranking', 'baseline', 'stage',
            'search_paused', 'search_resumed',
            'candidate_start', 'candidate', 'search_end', 'done', 'error',
          ].includes(item.event));
          window.__jlensE2EStreams.push({ url, events });
        }).catch(() => {});
      }
      return response;
    };
  });
  const page = await context.newPage();
  page.on('dialog', (dialog) => dialog.accept());
  page.on('pageerror', (error) => console.error('[pageerror]', error));
  await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => state.lensInfo?.n_prompts === 1000, null, { timeout: 30_000 });
  await page.evaluate(() => {
    state.settings.maxTokens = 96;
    state.settings.temp = 0;
    state.settings.topN = 3;
  });

  const results = [];
  for (let index = 0; index < trials.length; index += 1) {
    const result = await runTrial(page, trials[index], index + 1);
    results.push(result);
    console.log(JSON.stringify(result));
  }
  fs.writeFileSync(path.join(artifactDir, 'results.json'), `${JSON.stringify(results, null, 2)}\n`);
  await browser.close();

  const hardFailures = results.filter((result) =>
    ['infinite-loop-generation', 'verified-recipe-not-applicable', 'search-error']
      .includes(result.status));
  if (hardFailures.length) process.exitCode = 1;
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

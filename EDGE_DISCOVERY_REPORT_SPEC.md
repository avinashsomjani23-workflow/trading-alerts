# EDGE DISCOVERY REPORT SPEC — for Opus

**Written 2026-07-05, from the Fable chat that audited the first discovery run
(`h1only_20080102_20251231`). Everything in "Verified facts" was checked against live
code and the committed run artifacts in that chat — none of it is guessed. Re-verify
line numbers before editing (file may have moved).**

Parent specs: `EDGE_ENGINE_SPEC.md`, `EDGE_ENGINE_STAGED_REVIEW_SPEC.md` (SPEC_STAGED).
Guardrails: `DECISION_GUARDRAILS.md` — read it first. Nothing in this spec touches
validation or holdout data.

---

## 1. Problem

The discovery email for `h1only_20080102_20251231` is a bare summary. The trader cannot
read any insight from it: no per-feature numbers, no near-misses, no per-pair/session
performance, nothing verifiable. Two root causes:

1. **The spec'd full-detail report was never built.** SPEC_STAGED §7 and §9.1 require a
   committed human-readable `edge_engine_discovery.md` ("summary only [in email]; the
   committed .md carries full detail"). `grep edge_engine_discovery backtest/diagnostics/`
   → zero hits. The email matches its spec exactly; its companion report is missing.
2. **A real dtype bug** locks 10 of 27 categorical features out of candidate status
   forever (see §2, Fact F3). It did NOT change this run's verdicts (proven, F4), but it
   corrupts the confirm phase for any categorical, so it must be fixed BEFORE the
   approval token is ever spent.

## 2. Verified facts (do not re-derive; re-verify only if code moved)

- **F1 — Run artifacts exist and are committed.** In
  `backtest/results/h1only_20080102_20251231/edge_engine/`:
  `stage0_gate.json` (pass:true, scope:verdict, census D/V/H = 11333/6765/5221),
  `stage1_discovery.json` (38 KB, all 43 feature records with Δ, CI, p, verdicts),
  `stage1_discovery_features.csv` (every bucket/level row: n, expR, ci, wr_pct, totR,
  pos_quarters — DISCOVERY split only). All the insight the trader wants already exists
  here; nothing renders it.
- **F2 — Runtime is legitimate.** stage0 generated 05:58:08 UTC, discovery JSON
  06:02:20 UTC → ~4 min. The engine reads the committed run's trades.csv (23,319
  eligible rows), it does not re-walk 18 years of bars. Stage-0 self-check replayed all
  23,319 trades against the frozen bar cache and reproduced committed `r_realised`
  exactly (mean_diff 0.0, per_trade_bad 0).
- **F3 — The dtype bug.** `_categorical_screen` (edge_engine.py ~line 900) keys levels
  by `str(lvl)` (~line 915) but `_lvl_vals` (~line 937) compares that string key against
  the NATIVE-dtype column: `frame[frame[feat] == lvl]`. For bool and int64 columns the
  comparison matches ZERO rows → `bootstrap_diff_ci` gets empty lists → CI is
  `(None, None)` → `_ci_excludes_zero` is False → `disc_signal` can never pass →
  verdict capped at `noise` regardless of data. Reproduced on the real trades.csv:
  `ob_in_killzone` (bool): native_match=6627, str_match=0; `fvg_present` (bool):
  6890 vs 0; `ob_walkback_depth` (int64): 7802 vs 0; `ob_touches` (int64): 8710 vs 0.
  Affected features (all show `ci=[null,null]` in the committed JSON):
  `reversed_from_extreme` (mixed-type object — handle it), `fvg_present`,
  `ob_in_killzone`, `fill_in_killzone`, `ob_touches`, `ob_walkback_depth`,
  `structure_ranging_at_alert`, `flip_pending_at_alert`, `dr_ceiling_broken_at_ob`,
  `dr_floor_broken_at_ob`.
  The same string-vs-native comparison also breaks the CONFIRM path: validation
  `_lvl_vals(v, best)` (~line 954) and `_pos_quarters(v[v[feat] == best])` (~line 957)
  — a categorical survivor's validation stats would be computed on zero rows.
- **F4 — Fixing the bug does NOT change this run's discovery verdicts.** Candidate
  requires `fdr_reject` AND CI-excludes-0. BH-FDR (q=0.10, m≈42 non-null p-values)
  rejected only p=1.8e-4 (`ob_range_atr`) and p=2.9e-4 (`alert_utc_hour`); the third
  smallest p (0.0214) needed ≤ ~0.007. Every affected feature's Kruskal p ≥ 0.0214 →
  none would newly reject → candidate set stays `{alert_utc_hour}`. State this in the
  PR: this is a correctness fix, not a result change (no D4 flavour).
- **F5 — Token mechanics handle the code change automatically.** The approval token =
  sha256(discovery_sha + code_sha)[:12] (`_compute_token`, ~line 1189). Editing
  edge_engine.py kills token `60f10397592f` by design; `approve()` will refuse it.
  After merge, discovery must be re-run (free, sanctioned — validation frames are never
  built in `stage1_discovery`, ~line 1322).
- **F6 — The Action already commits anything in engine_dir.**
  `.github/workflows/edge_engine.yml` commit step: `git add -f "$RUN_DIR/edge_engine/"`.
  Writing the report into engine_dir needs zero workflow changes.
- **F7 — Language rule is test-enforced.** `tests/test_staged_review.py` §11.3 step:
  discovery output must carry the language stamp and must NOT contain "survivor" or
  " edge". The report generator must obey the same rule and the test must be extended
  to scan the rendered .md.
- **F8 — Observation, no action:** `event` and `bos_tier` produced byte-identical
  screen stats (d=0.0448, p=0.05645) in this run — likely 1:1 level mapping in this
  population. Note it in the report's observations section; do NOT drop either feature
  (B6: manifest is frozen).

---

## 3. Work item 1 — fix the categorical dtype bug (P0)

- **Fix location:** normalise categorical level values to strings ONCE, at the top of
  `_categorical_screen` (both the disc and val frames), e.g. map non-null values
  through `str()` before `_merge_rare_levels` runs. Keys, groupby levels, `_lvl_vals`
  comparisons and `_pos_quarters` filters then all agree. Preserve NaN as NaN (do not
  create a "nan" level).
- One implementation, one place — do NOT patch `_lvl_vals` and the val-side separately
  (CLAUDE.md: one concept, one implementation). The interaction screen groups natively
  and compares natively (`_interaction_screen` ~line 1136-1142) — audit it once for the
  same class of mismatch (`v[(v[a] == av) ...]` uses groupby-produced native values, so
  it is likely fine — verify, don't assume).
- Mixed-type columns (`reversed_from_extreme` holds bools AND strings in the committed
  CSV read) must come out as one coherent string level set. Check its resulting levels
  ("True"/"False" only, plus whatever third level carried n=2312 in this run —
  investigate what that value actually is and say so in the PR).
- **Structural guard (required, truth-ledger pattern — kill the bug CLASS):** a
  parametrised regression test over level dtypes {bool, int64, str, mixed-object}:
  synthetic frame, two levels, n≥150 each, strong expR separation → assert
  `delta_disc_ci` is non-null and `ci_excludes_0` is True; run through BOTH
  `_categorical_screen(disc, None, ...)` (discovery mode) and
  `_categorical_screen(disc, val, ...)` (confirm mode), and assert the val-side
  `best_worst_n_val` is non-zero for bool/int dtypes.
- Also extend the existing fuzz/ladder test so `_apply_candidate_criteria` sees a
  record with a real CI from a bool feature.

## 4. Work item 2 — `edge_engine_discovery.md` writer (P0)

The missing SPEC_STAGED §7/§9.1 artifact. Requirements:

- **Inputs:** engine_dir only — `stage1_discovery.json` + `stage1_discovery_features.csv`
  (+ `stage0_gate.json` for the census). NEVER reads trades.csv, NEVER builds a
  validation frame. Pure rendering of committed discovery data.
- **Writer lives in** `edge_engine.py` (or a sibling `edge_report.py` if cleaner —
  Opus's call), invoked automatically at the end of `stage1_discovery`, plus a
  standalone CLI flag (e.g. `--render-discovery-report [--run-dir ...]`) that re-renders
  from committed JSON without recomputation (backfill for past runs).
- **Overall-population stats:** add a small `population_stats` block to the
  `stage1_discovery` result (discovery-split n, expR, bootstrap CI, wr_pct, totR,
  per-pair and per-session rows) so the report does not have to re-derive them. This
  changes the JSON → token changes → fine, discovery is re-run anyway (F5).
- **Failed-criteria transparency:** refactor `_apply_candidate_criteria` to return the
  verdict PLUS which criteria passed/failed (fdr_reject, ci_excludes_0, substance_n,
  substance_effect) as a dict stored on each feature record. Same refactor pattern for
  `_apply_survivor_criteria` is optional but preferred (shared shape). The report's
  near-miss section is derived from these flags, never re-derived from thresholds.

### Report structure (all tables carry N — blind-spot guard)

1. **Header** — language stamp verbatim, run_id, window, N(discovery), split census,
   scope, token block (same 4-line block as the email), and a 5-line plain-English
   "how to read this" guide (what candidate/noise/thin mean, what Δdisc is, why
   discovery-only numbers can still be luck).
2. **Verdict table** — ALL features, sorted by |Δdisc| desc: feature, type, timing,
   verdict, Δdisc, CI, N(top/bot), screen p, fdr_reject, failed-criteria flags.
3. **Candidate deep-dives** — per candidate: the full bucket/level table (from the
   CSV), favoured bucket, a 2-3 bullet plain-English readout (e.g. for
   `alert_utc_hour`: bucket 15-18 UTC expR +0.016 [CI −0.049,+0.080] n=1330 vs bucket
   4-7 UTC −0.086 [−0.131,−0.040] n=3016; 16/28 positive quarters vs 8/36), and what
   the confirm phase will test.
4. **Near-misses** — features whose criteria flags show exactly one failure. Label the
   section: "NOT candidates. Shown for transparency (C4). No action, no threshold
   renegotiation (F)." Known entries this run: `fill_session` (Δ 0.1505, CI excludes 0,
   fdr no), `trend_alignment` (Δ 0.0998, just under the 0.10R floor), `ob_range_atr`
   (fdr-rejected Spearman but flat top-vs-bottom Δ −0.005 — non-monotonic shape; show
   its quintile curve).
5. **Baseline context — "how did pairs/sessions/hours do"** — per-pair, per-session
   level tables (n, expR, CI, wr, totR, pos_quarters) + overall discovery-split stats.
   MUST carry two caveats: (a) this is the gates-off, all-scores population — NOT what
   live (score≥4, filtered) trading would produce; (b) Book B pairs (GBPUSD, AUDUSD,
   USDCAD, EURJPY) are pooled per SPEC §3.3 but are not in live trade scope.
6. **Sub-screens** — snapback bins table (with the existing bin-1-2 timing caveat),
   SL-anatomy rows, news confounder block. "interactions: deferred to confirm phase."
7. **Full appendix** — every feature's complete bucket/level table from the CSV. The
   report must render a table for EVERY feature record (count assert in tests — no
   silent truncation).
8. **Observations** — F8 (`event` ≡ `bos_tier` this run) and anything else stamped as
   observation-only.

### Language + guards

- The stamp sentence appears verbatim near the top. The strings "survivor" and " edge"
  must not appear anywhere in the rendered .md (match the §11.3 email test).
- Extend `tests/test_staged_review.py` `_step_language` to also render + scan the .md.
- Add a test asserting the report renders one table per feature record and that every
  rendered stat cell that shows expR also shows its n.

## 5. Work item 3 — email upgrade (P1, keep it a summary)

- Keep the current body (it matches §9.1). Add:
  - a near-miss count line ("near-misses: 3 — see report"),
  - a pointer line with the committed report path,
  - the overall discovery-split expR + N one-liner (context so "candidate=1" reads as
    "1 of 43 screened on 11,329 trades", not "the system found almost nothing").
- Language test must still pass unchanged.

## 6. Operational order (after merge — flag to the trader, do not do silently)

1. Merge fix + report writer. Old token `60f10397592f` is dead automatically (F5).
2. Re-run the discovery Action on `h1only_20080102_20251231`. New JSON + CSV + report +
   token get committed and emailed. Expected: candidate set unchanged (F4) — if it
   CHANGES, stop and report the diff before anything else; that would mean F4's
   analysis missed something.
3. Only after the trader reads the report: approve + confirm (one shot, unchanged
   mechanics).

## 7. Do NOT touch

- Thresholds, verdict ladder semantics, BH-FDR family, split dates, token/approval/
  ledger mechanics, the feature manifest (B6), `--force` semantics (C1).
- No validation or holdout frame may be built anywhere in the discovery path or the
  report writer (SPEC_STAGED §4.2 — the existing "never materialised" guarantee).
- The confirm/verdict emails and `edge_engine_confirm.md` / Phase B-C reports are OUT
  of scope here (build only what discovery needs; confirm report is its own item once
  discovery lands).

## 8. Acceptance checklist

- [ ] Dtype regression test (bool/int64/str/mixed) green; would have failed on current code.
- [ ] All 10 F3 features show real CIs in a fresh discovery run.
- [ ] Fresh discovery run on `h1only_20080102_20251231`: candidate set unchanged (F4) — or a stop-and-report diff.
- [ ] `edge_engine_discovery.md` committed by the Action with zero workflow edits (F6).
- [ ] Report: every feature has a table, every expR carries N, stamp present, "survivor"/" edge" absent.
- [ ] `--render-discovery-report` regenerates the .md from committed JSON alone.
- [ ] Email carries report path + near-miss count + overall-population line; §11.3 test green.
- [ ] TRUTH_LEDGER.md: no new trades.csv columns are created (engine-internal only) — confirm and say so in the PR.

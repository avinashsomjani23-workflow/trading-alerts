# EDGE ENGINE — STAGED HUMAN-IN-THE-LOOP REVIEW SPEC (v1)

**Author: Fable, 2026-07-04. Implementer: Opus. Problem statement:
`EDGE_ENGINE_STAGED_REVIEW_HANDOFF.md`. Parent spec: `EDGE_ENGINE_SPEC.md` (v1).
Guardrails: `DECISION_GUARDRAILS.md`.**

**Prime directive for the implementer: after your change, `edge_engine.py` must be a
complete, runnable engine. Every existing behaviour that this spec does not explicitly
change stays byte-identical. All existing tests stay green. This is a modification of a
frozen-phase tool — surgical, not creative.**

---

## 0. MISSION

- Split the engine's one-shot run into **three human-gated phases**:

```
PHASE A  --phase discovery   Stage 0 + discovery-only preview of Stage 1.
                             Output = CANDIDATES. Prints an approval TOKEN. STOPS.
         (human reads for as long as they want, then approves the token)
PHASE B  --phase confirm     Canonical Stage 1 (discovery + validation, unchanged math).
                             Output = SURVIVORS. Logs the validation spend. STOPS.
         (human reads survivors, sees what died)
PHASE C  --phase final       Stages 2, 3, 4. Holdout opens once, exactly as today.
```

- The dangerous pattern this must prevent: *tweak → re-run validation → repeat until it
  passes* (guardrail D4). The design makes that path **loud and permanently stamped**, not
  silent. It cannot be made physically impossible (the trader owns the machine); the
  mechanism is "for the trader, against the trader" — same philosophy as D4's generation
  stamp and Stage 4's `holdout_reopened`.

### 0.1 Hard constraints (from the handoff — violating any one = failed implementation)

1. Validation is a one-shot confirmation. Re-running it is stamped forever (§5).
2. Holdout behaviour untouched: `stage4()` is **not modified** except where §7 says.
3. One run, sliced. No new backtests. Read-only post-processing of the existing run.
4. No new `trades.csv` columns. No changes to live / simulator / exit code.
5. Discovery-only output NEVER uses the words "survivor" or "edge" (§4.5).
6. Short-range / exploratory mode (SPEC §15) is completely unaffected (§8).
7. Canonical Stage 1–4 outputs stay **byte-identical** for a given input when the
   approval gate passes (the only additions are the new stamp fields listed in §5.4).

---

## 1. FILES TOUCHED (complete list — nothing else)

| File | Change |
|---|---|
| `backtest/diagnostics/edge_engine.py` | All engine changes (§3–§8) |
| `backtest/diagnostics/edge_email.py` | NEW — phase emails (§9) |
| `.github/workflows/edge_engine.yml` | NEW — Action, 3 buttons (§10) |
| `tests/test_staged_review.py` | NEW — regression tests (§11) |
| `EDGE_ENGINE_SPEC.md` | New §18 summary + §17 change-log entry (§12) |
| `DECISION_GUARDRAILS.md` | Draft D5 prepared but NOT committed by Opus (§12.3) |

New artefacts written at runtime (all inside `<run_dir>/edge_engine/`, all committed by
the Action like other engine outputs):

| Artefact | Written by | Purpose |
|---|---|---|
| `stage1_discovery.json` | Phase A | Candidate records (discovery-only) |
| `stage1_discovery_features.csv` | Phase A | Discovery bucket table |
| `approval.json` | `--approve` | The signed one-time key |
| `validation_ledger.jsonl` | Phase B / burn | Append-only spend log |
| `edge_engine_discovery.md` | Phase A | Human-readable discovery report |
| `edge_engine_confirm.md` | Phase B | Human-readable confirm report |

`edge_engine_report.md` (Phase C) keeps its current name, format and writer
(`_write_report`), plus the staging header of §7.2.

---

## 2. CLI (exact argparse changes in `main()`)

Add to the existing parser (keep `--run-dir`, `--start`, `--end`, `--stage`, `--force`
exactly as they are):

```python
ap.add_argument("--phase", choices=["discovery", "confirm", "final"], default=None,
                help="staged review phase (SPEC_STAGED §2)")
ap.add_argument("--approve", metavar="TOKEN", default=None,
                help="sign the discovery approval token; writes approval.json and exits")
ap.add_argument("--burn-validation", metavar="REASON", default=None,
                help="explicitly re-open validation with a written reason. "
                     "Appends to validation_ledger.jsonl and stamps everything downstream.")
```

### 2.1 Dispatch rules (in `main()`, after `engine_dir` is resolved)

- `--approve TOKEN` → run §5.2 approval routine, print result, **exit**. Mutually
  exclusive with `--phase` and `--stage` (error out if combined).
- `--phase discovery` → run `stage0()` fresh, then `stage1_discovery()` (§4). Stop.
- `--phase confirm` → require existing Stage 0 JSON with `pass: true` (do NOT re-run
  Stage 0 — re-running it would change nothing but wastes nothing either; requiring the
  existing file keeps the phase cheap and the trust chain explicit). Then canonical
  `stage1()` behind the approval gate (§5.3). Stop.
- `--phase final` → require `stage1_features.json` with `pass: true` (via the existing
  `_prior_passed` mechanism), then run stages 2, 3, 4 in order exactly as today. Stop.
- `--phase` and `--stage` are mutually exclusive (error out if both given).
- **No `--phase` and no `--stage` (legacy full run):** keep the current `[0,1,2,3,4]`
  loop. Stage 1 will hit the approval gate (§5.3) and, in verdict scope with no valid
  approval, refuse and print the staged instructions. Net effect: a bare
  `python -m backtest.diagnostics.edge_engine` in verdict scope now stops after Stage 0
  with instructions instead of silently spending validation. **This is the intended
  behaviour change.**
- `--stage 1` alone: allowed, but subject to the same approval gate. `--force` never
  bypasses the gate (§5.5).

---

## 3. SPLIT-AWARE SCREEN REFACTOR (one implementation, two modes)

**Rule: do not duplicate the screen math. The discovery preview and canonical Stage 1
must flow through the SAME functions** (CLAUDE.md: one concept, one implementation —
divergent copies would let the preview and the confirm disagree by drift, which is worse
than no preview).

Change the signature of both screens:

```python
def _continuous_screen(disc, val: Optional[pd.DataFrame], feat, buckets_out): ...
def _categorical_screen(disc, val: Optional[pd.DataFrame], feat, buckets_out): ...
```

Behaviour when `val is None` (discovery-only mode):

- `_continuous_screen`:
  - Skip the `("VALIDATION", val_s)` bucket loop — emit DISCOVERY bucket rows only.
  - Skip `_delta(val_s)`, `_spearman(val, feat)`, and the favoured-quarter computation.
  - The record contains ONLY: `feature, type, timing, edges, n_buckets, delta_disc,
    delta_disc_ci, spearman_disc, spearman_p_disc, top_bottom_n_disc, favoured_bucket,
    _fdr_p`. No `delta_val`, no `*_val` keys at all (absent, not null — tests assert
    absence).
- `_categorical_screen`: same treatment — discovery bucket rows only; keep
  `best_level, worst_level, delta_disc, delta_disc_ci, best_worst_n_disc, _fdr_p`; no
  validation keys.
- Behaviour when `val` is a DataFrame: **identical to today, byte for byte.** The
  refactor must be pure addition of the `None` branch.

Sub-screens in discovery-only mode (used by §4):

- `_snapback_screen(disc)` — called with the discovery frame instead of `pooled`.
- `_sl_anatomy_screen(disc, None)` — same optional-val treatment: discovery stats only,
  no persistence verdicts, no promotions. In discovery mode `anatomy_promoted` is always
  `[]` (promotion is a validation concept).
- `_news_confounder(...)` — called with discovery-restricted frames.
- `_interaction_screen` — **not run** in discovery mode. Interaction cells are
  N-hungry (MIN_CELL_N) and their §5.6 semantics are persistence-based; a
  discovery-only interaction table invites exactly the thin-cell story-telling C3
  forbids. The discovery report notes: "interactions: deferred to confirm phase".

---

## 4. PHASE A — `stage1_discovery()` (new function)

### 4.1 Entry conditions

- Stage 0 has just been run by the same invocation and `pass: true`.
- `scope == "verdict"`. If scope is `exploratory`, print
  `"staged review requires verdict scope; use the short-range workflow (SPEC §15)"`
  and exit non-zero. (Staging a run that has no splits is meaningless.)

### 4.2 Computation

- Load population, pool, split — identical preamble to `stage1()`
  (`load_population` → `pooled_fx_gold` → `split_frame(pooled, "DISCOVERY")`).
- **The validation and holdout frames are never materialised in this function.** Only
  `disc` exists past the split line. (This is the code-level guarantee that Phase A
  cannot leak: the frame it would need is simply never built.)
- Run every feature through the screens with `val=None` (§3), same DECREED_OUT and
  column-presence skips as `stage1()`.
- BH-FDR across the family on `_fdr_p`, same as today.
- Verdict per feature via the NEW `_apply_candidate_criteria` (§4.3).
- Sub-screens per §3 (snapback/anatomy/news discovery-only; interactions deferred).
- Ranking: by `-abs(delta_disc)` (there is no `delta_val` yet).

### 4.3 `_apply_candidate_criteria(rec, fdr_reject) -> str` (new function)

Discovery-only verdict ladder — deliberately parallel to `_apply_survivor_criteria`, but
**criterion 2 (validation persistence) does not exist here and is not approximated**:

```
if rec verdict already "thin"                      -> "thin"
if delta_disc is None                              -> "thin"
disc_signal   = fdr_reject AND delta_disc_ci excludes 0
substance_n   = min(top_bottom_n_disc / best_worst_n_disc) >= MIN_BUCKET_N
substance_eff = abs(delta_disc) >= MIN_EFFECT_R
if not substance_n:   "candidate_thin" if disc_signal else "noise"
if not disc_signal:   "noise"
if not substance_eff: "candidate_thin"
else:                 "candidate"
```

- Verdict vocabulary of Phase A: `candidate`, `candidate_thin`, `noise`, `thin`.
  **The strings `survivor`, `hypothesis`, `inverted` must be impossible outputs of this
  function** (inverted needs validation; hypothesis is §15's word — using it here would
  blur two different modes).

### 4.4 Outputs

- `stage1_discovery.json`:

```json
{
  "phase": "discovery", "pass": true,
  "run_id": "...", "generated_utc": "...",
  "scope": "verdict", "window": "...",
  "n_discovery": 8939,
  "validation_untouched": true,
  "features": [ ...records from §3/§4.3... ],
  "candidates": ["feat_a", "feat_b"],
  "ranked_candidates": [ {"feature","verdict","delta_disc","timing"} ... ],
  "snapback": {...}, "sl_anatomy": {...}, "news_confounder": {...},
  "interactions": "deferred_to_confirm",
  "token": "<12-hex, §5.1>",
  "language_stamp": "CANDIDATE ONLY — discovery split only. Luck is NOT ruled out. A candidate becomes a survivor only if it repeats on validation years it has never seen."
}
```

- `stage1_discovery_features.csv` — the DISCOVERY bucket rows (same columns as the
  discovery rows of today's `stage1_features.csv`).
- **File naming is load-bearing:** the canonical Stage-1 artefact is
  `stage1_features.json` (`_stage_path`), and `_prior_passed(engine_dir, 2)` reads it.
  The preview writes to a DIFFERENT name, so Stage 2 can never mistake a discovery
  preview for a completed Stage 1. Do not add the preview to `_stage_path`'s map.
- `edge_engine_discovery.md` — human report (§9.1 structure).
- Console print: candidate table + the token, with the §4.5 language.

### 4.5 Mandatory output language (not optional phrasing — copy §15.3's pattern)

- Every candidate row in report/email/console carries `CANDIDATE — luck not ruled out`.
- The report header carries the `language_stamp` sentence verbatim.
- The words **survivor** and **edge** must not appear anywhere in Phase A output
  (add a unit test asserting this on the rendered report string).

---

## 5. THE APPROVAL TOKEN + VALIDATION LEDGER (the lock — the crux of the design)

### 5.1 Token definition

```
code_sha      = sha256(bytes of edge_engine.py on disk)
discovery_sha = sha256(bytes of stage1_discovery.json on disk)   # after §4.4 write
token         = sha256(discovery_sha + code_sha).hexdigest()[:12]
```

- Computed by a new helper `_compute_token(engine_dir) -> dict` returning
  `{token, discovery_sha, code_sha}`.
- Because `stage1_discovery.json` includes `generated_utc`, every discovery run mints a
  fresh token — a token is bound to ONE specific discovery output produced by ONE
  specific engine code state. Practical consequence: the token must be embedded in
  `stage1_discovery.json` **after** hashing — write the JSON without the `token` key,
  hash it, then rewrite with the token key added. `discovery_sha` is therefore defined
  as the sha256 of the token-less serialisation; `--approve` and Phase B recompute it
  by loading the JSON, dropping the `token` key, and re-serialising with the same
  `json.dump(..., indent=2, default=str)` call that `_write_json` uses.

### 5.2 `--approve TOKEN` routine

1. Recompute per §5.1. If recomputed token ≠ supplied TOKEN → print
   `"REFUSED: discovery output or engine code changed since this token was issued — re-run --phase discovery"`
   and exit non-zero. Do not write anything.
2. Else write `approval.json`:

```json
{ "token": "...", "discovery_sha": "...", "code_sha": "...",
  "approved_utc": "...", "consumed": false }
```

3. Print: `"Approved. Validation is now armed for ONE confirmation run (--phase confirm)."`

### 5.3 The approval gate inside `stage1()` (canonical)

At the top of `stage1()`, after scope is known and **only when `scope == "verdict"`**
(exploratory mode never hits the gate — §15 has no validation to protect):

1. Read `approval.json`. Recompute `_compute_token`. Gate passes iff the file exists,
   `token`/`discovery_sha`/`code_sha` all match the recomputation, and
   `consumed == false`.
2. If the gate fails and `--burn-validation` was NOT given → print the refusal block
   (which file/hash failed, plus the three-command staged workflow) and return a result
   `{"stage": 1, "pass": false, "refused": "approval_gate", ...}` **without touching the
   validation frame** — the refusal must happen before `split_frame(pooled,
   "VALIDATION")` is called.
3. If the gate passes → set `consumed: true` in `approval.json` (the token is
   single-use), then proceed.
4. Whether via approval or burn, append one line to `validation_ledger.jsonl`:

```json
{ "run_utc": "...", "token": "...or null", "code_sha": "...",
  "via": "approval" | "burn", "burn_reason": null | "<reason text>" }
```

5. `validation_runs` = number of lines now in the ledger.

### 5.4 Stamping (the loud part — mirrors `holdout_reopened`)

- `stage1()`'s result JSON gains: `validation_runs: N`, `validation_burned: bool`
  (true iff any ledger line has `via: "burn"` OR `N > 1`).
- `stage4()` gains ONE change: `_collect_caveats` also receives the ledger summary and,
  when `validation_burned`, prepends the caveat
  `"VALIDATION RE-OPENED (runs: N) — survivor list is no longer a one-shot confirmation; treat the verdict as exploratory (D4)."`
- `_write_report` prints `validation_runs` + the caveat in the report header.
- Every email subject for confirm/final phases is prefixed `[VALIDATION RE-RUN N]` when
  `N > 1` (§9).
- The ledger file is append-only in code (open mode `"a"`), and the Action commits it —
  git history preserves every spend even if the file is later hand-edited.

### 5.5 What `--force` does NOT do

- `--force` keeps its current meaning (run despite prior stage not passing) and
  **must not bypass the approval gate**. The only sanctioned bypass is
  `--burn-validation "<non-empty reason>"`, which is by construction loud (§5.3.4,
  §5.4). An empty/whitespace reason is rejected.

### 5.6 Honest limitation (state in SPEC §18, do not paper over)

- The trader can delete `approval.json`/the ledger or edit the code. The mechanism makes
  silent burning **impossible to do accidentally and impossible to hide from git
  history** — it does not make it physically impossible. That is the same trust model as
  C5/D4 today.

---

## 6. PHASE B — confirm (canonical Stage 1, unchanged math)

- `--phase confirm` runs `stage1()` exactly as it exists today — same screens (val is
  the real validation frame), same `_apply_survivor_criteria`, same FDR, same outputs to
  `stage1_features.json/.csv` — plus the gate/ledger/stamps of §5.
- After it completes, write `edge_engine_confirm.md` (§9.2) and send the confirm email.
- **No re-computation of discovery numbers is allowed to differ:** discovery quintile
  edges come from the discovery frame both times, so candidate Δ numbers in Phase A and
  discovery Δ numbers in Phase B are identical by construction (same function, same
  input). Add a test (§11.6).
- The confirm report must include a **died-in-validation table**: every Phase-A
  `candidate` whose final verdict is not `survivor`, with its verdict
  (`inverted` / `directional_thin` / `noise`). This table is the trader's main learning
  artefact and the anti-attachment device: candidates dying here is the system working,
  not a bug signal (rule F: luck doesn't repeat).

## 7. PHASE C — final (Stages 2–4, holdout untouched)

### 7.1 Behaviour

- `--phase final` = run stages 2, 3, 4 in order through the existing loop code paths,
  including `_prior_passed` chaining and all current stage-4 behaviour
  (`holdout_opened_utc`, `holdout_reopened`, verdict tree, recipe JSON).
- No pause between Stages 2/3 and 4 inside the phase (trader decision 2026-07-04: the
  three-button flow was approved; Stage 2/3 outputs are still individually readable
  afterwards via the committed report, and `--stage 2` / `--stage 3` remain available
  for stepping manually if ever wanted).

### 7.2 Only two additions

1. `stage4()` caveat propagation per §5.4.
2. `_write_report` header gains a `STAGED REVIEW` block: discovery token, approval
   timestamp, `validation_runs`, ledger `via` values. Pure additional lines — no
   existing report content changes.

## 8. EXPLORATORY / SHORT-RANGE SCOPE — UNTOUCHED

- When Stage 0 stamps `scope: exploratory`, nothing in this spec applies: no gate, no
  token, no phases (`--phase` errors out per §4.1), legacy full-run loop works exactly
  as today. §15's hypothesis language stays its own thing. Test §11.7 pins this.

---

## 9. REPORTS + EMAILS (`backtest/diagnostics/edge_email.py`, new)

- Copy the transport pattern of `backtest/diagnostics/sweep_email.py` (smtplib,
  `smtp.gmail.com:587`, env `GMAIL_APP_PASSWORD` + `BACKTEST_EMAIL`, fallback recipient
  logic identical). Email failure must never fail the engine run (try/except, print
  warning) — same convention as the existing mailers.
- Plain-text-first bodies (tables as monospace blocks), summary only; the committed
  `.md` file carries full detail (SPEC §16.3 division of labour).

### 9.1 Discovery email + `edge_engine_discovery.md`

- Subject: `EDGE ENGINE — DISCOVERY candidates — <run_id>`
- Body order: language stamp (§4.5) → candidate count by verdict → top-10 ranked
  candidates (feature, Δdisc, CI, N, timing) → snapback/anatomy one-liners → the token
  in its own block:

```
APPROVAL TOKEN: a1b2c3d4e5f6
To confirm on validation (ONE shot):  press CONFIRM in the Action with this token,
or locally:  python -m backtest.diagnostics.edge_engine --approve a1b2c3d4e5f6
             python -m backtest.diagnostics.edge_engine --phase confirm
```

### 9.2 Confirm email + `edge_engine_confirm.md`

- Subject: `EDGE ENGINE — VALIDATION confirm — <run_id> — <k> survivors of <m> candidates`
  (prefix per §5.4 when re-run).
- Body: survivors table (feature, Δdisc, Δval, quarters, timing, actionable_at) →
  **died-in-validation table** (§6) → interactions summary → next-step block ("read,
  sleep (E6), then press FINAL").

### 9.3 Verdict email

- Subject: `EDGE ENGINE — VERDICT <verdict> — <run_id>` (+ §5.4 prefix if burned).
- Body: verdict + holdout headline numbers + caveats list + recipe one-liner + pointer
  to `edge_engine_report.md`.

---

## 10. GITHUB ACTION — `.github/workflows/edge_engine.yml` (new)

- `workflow_dispatch` inputs:
  - `phase` — choice: `discovery | confirm | final` (required)
  - `run_id` — string, optional; empty ⇒ §14.3 latest-wins resolution
  - `approve_token` — string, optional; **required when phase=confirm** (validate in a
    script step: fail fast with a clear message if confirm + empty token)
- Engine-only, always: this workflow NEVER launches a backtest (handoff constraint 3).
  The §16.2 "FULL RUN" button stays future work — out of scope here.
- Steps:
  1. Checkout (full history not needed; default depth is fine — engine reads committed
     run folders).
  2. Setup Python 3.11 + `pip install pandas numpy scipy`.
  3. Run the phase:
     - discovery: `python -m backtest.diagnostics.edge_engine --phase discovery [--run-dir backtest/results/<run_id>]`
     - confirm: `python -m backtest.diagnostics.edge_engine --approve "$TOKEN"` then
       `... --phase confirm`
     - final: `... --phase final`
  4. Send the phase email (env: `GMAIL_APP_PASSWORD`, `BACKTEST_EMAIL` secrets — same
     names as `backtest.yml`).
  5. Commit + push `<run_dir>/edge_engine/` (including `approval.json`,
     `validation_ledger.jsonl`) and the phase `.md` report, `git add -f` (results dirs
     are gitignored — same path as the backtest logs pipeline), message
     `Edge engine <phase>: <run_id> [skip ci]`.
  6. Email step runs before push-failure can cancel it OR use `if: always()` on the
     email step — a failed push must not eat the report email (lesson from the
     knob-sweep pre-flight incident: verify remote reachability read-only first with
     `git ls-remote`, never a canary push).
- Ordering note: email AFTER commit is preferred (email can then link the committed
  file), but the email must fire even if the push fails (`if: always()` + step-level
  `continue-on-error: false` on the engine step so a failed ENGINE still skips email).
  Precise rule: engine step fails ⇒ no email, no commit. Engine succeeds ⇒ commit
  attempt, then email regardless of push outcome.

---

## 11. TESTS — `tests/test_staged_review.py` (new; existing 24 tests stay green)

Build a small synthetic trades frame fixture (same trick as `test_gates_off_proof.py`)
with enough rows in both splits. Required tests:

1. **No-leak:** `stage1_discovery` output JSON contains no `*_val` keys, no
   `VALIDATION` bucket rows in the CSV, and `validation_untouched: true`.
2. **Candidate ladder:** synthetic features engineered to hit each of
   `candidate / candidate_thin / noise / thin`; assert `_apply_candidate_criteria`
   verdicts; assert `survivor`/`hypothesis`/`inverted` are impossible outputs.
3. **Language:** rendered discovery report/email body contains the language stamp and
   does NOT contain `survivor` or ` edge` (word-boundary match).
4. **Gate refusal:** in verdict scope with no `approval.json`, `stage1()` returns
   `pass: false, refused: "approval_gate"` and writes no `stage1_features.json`… 
   (assert the canonical artefact is absent) and no ledger line.
5. **Token binding:** approve a token, then modify one byte of
   `stage1_discovery.json` (or the engine file hash input) → confirm refuses.
6. **Phase A/B agreement:** for the same fixture, `delta_disc` per feature from
   `stage1_discovery.json` equals `delta_disc` in `stage1_features.json` after confirm
   (same function, same input — this pins the no-drift guarantee of §3).
7. **Exploratory bypass:** scope=exploratory runs legacy full loop with no gate; 
   `--phase discovery` errors out.
8. **Single-use + ledger:** first confirm consumes the token (`consumed: true`,
   ledger N=1, `validation_burned: false`); second confirm without a new approval
   refuses; second confirm via `--burn-validation "reason"` runs with ledger N=2,
   `validation_burned: true`, and the stage-4 caveat string present after a final run.
9. **Force is not a bypass:** `--force` alone does not open the gate.

## 12. SPEC + GUARDRAIL BOOKKEEPING (part of DONE)

### 12.1 `EDGE_ENGINE_SPEC.md`

- Add **§18 STAGED HUMAN REVIEW** — a one-page summary of: the three phases, the token
  mechanism, the ledger, the stamp fields, the §5.6 honest limitation, and the rule
  that §15 exploratory mode is exempt.
- Amend §3.2's "welded" description with a pointer: *"Stages 1–3 consume
  discovery+validation as specified; ACTIVATION of Stage 1's validation half is gated
  by the §18 approval token."*
- §17 change-log entry (dated): what changed, why (trader participation + bug-catching
  before Stage 2 consumes survivors), and that canonical math is untouched.

### 12.2 What this does NOT change

- No DECISION_GUARDRAILS rule is violated: C5 (holdout) untouched; D4 is what §5
  enforces; B4 unaffected (engine stages remain the sanctioned way to read results);
  F thresholds unchanged.

### 12.3 D5 draft (Opus prepares the text in the PR description / a comment — the
TRADER commits it to `DECISION_GUARDRAILS.md` in a separate sitting, per that file's
own change procedure; a rule may not be added in the same sitting it first gates)

> **D5 — Validation is opened by token, once.** Discovery is read freely; validation
> runs only after `--approve <token>` and consumes the token. Re-opening validation
> requires `--burn-validation "<reason>"` and permanently stamps every downstream
> report (`validation_runs`, `validation_burned`). A pause is discipline; a re-run is
> arguing with the verdict (D4).

## 13. ACCEPTANCE CHECKLIST (Opus self-verifies before handing back)

- [ ] `python -m backtest.diagnostics.edge_engine --phase discovery --run-dir <18yr run>`
      runs end-to-end on the real run folder, prints candidates + token, writes all
      Phase-A artefacts, never materialises the validation frame.
- [ ] `--approve` + `--phase confirm` produces `stage1_features.json` identical in
      schema to today's (plus the §5.4 stamp fields) and the died-in-validation table.
- [ ] `--phase final` produces stage 2/3/4 outputs and `edge_engine_report.md` with the
      staging header; `holdout_opened_utc` / `holdout_reopened` logic untouched.
- [ ] Legacy: `--stage 0` works; exploratory scope full-run works with zero gating.
- [ ] Full test suite green (existing 24 + new file).
- [ ] No new `trades.csv` columns; no edits outside §1's file list.
- [ ] SPEC §18 + §17 entries written; D5 draft delivered but NOT committed to
      `DECISION_GUARDRAILS.md`.

## 14. TRUTH-LEDGER NOTE

- This change adds NO trades.csv columns and NO per-trade insights, so no
  `TRUTH_LEDGER.md` rows are required. The new artefacts (`approval.json`,
  `validation_ledger.jsonl`, stamp fields) are engine-state metadata; their structural
  guard is `tests/test_staged_review.py` (§11), which kills the bug class "validation
  spent without a ledger line / stamp".

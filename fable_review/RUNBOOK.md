# RUNBOOK — Brain-off steps for the Fable review

You make zero decisions. Follow in order. If Fable asks a question, you do NOT answer it —
you bring it back to Claude Code, Claude verifies against the real code and answers for you,
and you paste that answer into Fable.

## Files (all in `fable_review/`)
- `FABLE_REFERENCE_LIVE.md` — the ground-truth reference. Upload this to Fable.
- `PROMPT_1_TRADING_LOGIC.md` — first review prompt. Copy-paste its body.
- `PROMPT_2_ARCHITECTURE.md` — second review prompt. Copy-paste its body.

## Round 1 — Trading logic
1. Open a NEW Fable chat.
2. **Upload** `FABLE_REFERENCE_LIVE.md`.
3. **Paste** the entire body of `PROMPT_1_TRADING_LOGIC.md` (everything below its `---`).
4. Send. Let Fable answer fully.
5. **Copy Fable's whole reply back to Claude Code.** Say: "Fable trading-logic review —
   verify and tell me what's real."
6. If Fable's reply ends with **Questions for the engineer**: copy those questions to Claude
   Code first. Claude answers them against the code. Paste Claude's answers back into the SAME
   Fable chat and say "answers below, finalise your review." Then repeat step 5.

## Round 2 — Architecture
7. Open ANOTHER NEW Fable chat (fresh — don't reuse Round 1's).
8. **Upload** the same `FABLE_REFERENCE_LIVE.md`.
9. **Paste** the entire body of `PROMPT_2_ARCHITECTURE.md`.
10. Send. Let Fable answer fully.
11. **Copy Fable's whole reply back to Claude Code.** Say: "Fable architecture review —
    verify and tell me what's real."
12. Same question-handling loop as step 6 if Fable asks anything.

## What Claude Code does with each Fable reply
- Checks every claim against the actual code (file:line).
- Tells you: which findings are TRUE, which are WRONG (with the code that disproves them),
  and which are TRUE-but-not-worth-doing.
- Turns the survivors into a ranked change list. **No code is changed** until you approve each
  one — trading-logic changes always need your explicit "yes" per the project rules.

## The only rule you must hold
- Fable reasons. Claude verifies. You move paper between them.
- Never let a Fable claim become a change without Claude confirming it against the code first.
- If Fable and Claude disagree, Claude's file:line evidence wins — or Claude says "I need to
  re-check" and does.

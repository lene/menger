# Investigation: `--animate` without `--save-name` crashes at startup

Manual-test defect #9 (`ManualTestNeedFixing.md`), scene 115 "TesseractSponge L2, animated XW,
GPU update". Original report speculated this was 4D-sponge- or rotation-specific.

## Stage 0 — Context — 2026-08-20

**Command:**
```
./menger-app/target/universal/stage/bin/menger-app --headless -o --animate frames=60:rot-x-w=0-360 \
  --objects type=tesseract-sponge:level=2 --plane y:-2 -s /tmp/scene115.png
```
**Output/measurement:** exit 0, no crash.
**Hypothesis update:** headless single-frame mode (with `-s` added) does not reproduce. The
real manual-test invocation has no `-s` and is genuinely interactive (`manual-test.sh` runs
`$MENGER $args` directly, no headless/save flags injected).

**Command:**
```
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a ./menger-app/target/universal/stage/bin/menger-app \
  -o --animate frames=60:rot-x-w=0-360 --objects type=tesseract-sponge:level=2 --plane y:-2
```
**Output/measurement:** `Error: None.get`, exit 1. Reproduced.
**Hypothesis update:** crash happens under a real windowed run with the exact scene-115 args.

## Stage 0 — Root cause — 2026-08-20

Added a temporary `e.printStackTrace()` to `Main.main`'s catch-all (reverted after capture) and
re-ran the same command.

**Output/measurement:**
```
java.util.NoSuchElementException: None.get
  at scala.None$.get(Option.scala:628)
  at org.rogach.scallop.ScallopOption.apply(ScallopOption.scala:35)
  at Main$.createCliBasedOptiXEngine(Main.scala:221)
  at Main$.createEngine(Main.scala:89)
  at Main$.main(Main.scala:41)
```
**Hypothesis update:** not `MeshFactory`, not 4D-specific, not rotation-related. `Main.scala:221`
called `opts.saveName()` (Scallop's unwrapped `apply()`, throws `NoSuchElementException` when the
option has no value and no default) to build `CliAnimationEngine`. `saveName` has no default
(`MengerCLIOptions.scala:106-109`) and scene 115 passes no `-s`. The sibling call in
`buildExecutionConfig` (`Main.scala:229`, now 233) already used the safe `opts.saveName.toOption`
for the same field — one call site was fixed for optionality, the other wasn't. Any `--animate`
scene without `-s` hits this, not just 4D sponges; scene 115 is simply the only manual-test entry
combining `--animate` with no `-s`.

## Checkpoint 1 (description) — user confirmed, then decided the fix direction

Two possible fixes were presented:
1. `--animate` without `-s` is meant to work as live interactive playback (no files written) —
   requires `CliAnimationEngine` to accept `Option[String]`.
2. `--animate` always requires `-s` — one-line scene-arg fix plus a CLI validation.

**User decision: (1).** `--animate` without `-s` should work as live interactive playback.

## Stage 3 — Root cause and fix — 2026-08-20

`CliAnimationEngine.render()` already renders to screen unconditionally
(`rendererWrapper.renderScene` → `renderResources.renderToScreen`) regardless of saving;
`saveImage()` is `currentSaveName.foreach(...)`, already a no-op on `None`
(`SavesScreenshots.scala:5`). So live playback without saving was already structurally
supported — the only blocker was the unconditional unwrap of `saveName` before construction.

**Fix:**
- `CliAnimationEngine.savePattern`: `String` → `Option[String]`.
- Extracted the pure formatting logic into `CliAnimationEngine.formatSaveName(pattern, frame)`
  so it's testable without a GPU-backed engine instance.
- `currentSaveName` now delegates to `formatSaveName`, returning `None` when no pattern given.
- `Main.scala:221`: `opts.saveName()` → `opts.saveName.toOption`.

**Verification:**
```
__GL_THREADED_OPTIMIZATIONS=0 xvfb-run -a ./menger-app/target/universal/stage/bin/menger-app \
  -o --animate frames=60:rot-x-w=0-360 --objects type=tesseract-sponge:level=2 --plane y:-2
```
Result: 60/60 frames rendered, `Animation complete`, exit 0.

**Regression test:** `CliAnimationEngineSuite` (new) — `formatSaveName(None, n) == None`,
`formatSaveName(Some("frame-%d.png"), 3) == Some("frame-3.png")`.

## Minimum artefacts checklist

- [x] Regression test exists (`CliAnimationEngineSuite`, permanent)
- [x] Fix commit passes that test
- [x] This note records per-step commands, measurements, root-cause narrative
- [x] Prior incorrect description (4D/rotation-specific) corrected in `ManualTestNeedFixing.md`
- [x] No `CODE_IMPROVEMENTS.md` entry existed for this (nothing to delete)
- [x] No residual defects found during this investigation

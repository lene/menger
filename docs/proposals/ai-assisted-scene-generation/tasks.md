## 1. Module scaffolding & boundaries (D1)

- [ ] 1.1 Create new sbt subproject `menger-ai` (JVM-pure; no dependency on `optix-jni` or native/GPU code). Add to root `build.sbt` aggregate; wire scalafix + wartremover to match repo conventions.
- [ ] 1.2 Add an ArchUnit test asserting `menger-ai` depends on NO `io.github.lene.optix.*` or native-binding packages (enforces AD-31 boundary at compile/test time).
- [ ] 1.3 Verify the module compiles and its test lane runs in CI without a GPU.

## 2. Tool-catalog & parity fitness function (D3, spec: ai-scene-generation)

- [ ] 2.1 Define the catalog data model (object types, material presets, lights, camera/plane options, feature blocks; each parameter with name/type/default/range + natural-language hint).
- [ ] 2.2 Author the initial catalog (YAML) covering every current `menger.dsl.*` object type and material preset, derived from `docs/guide/dsl-reference.md` and reflection over `menger.dsl.*`.
- [ ] 2.3 Implement the catalog-parity fitness function: assert the catalog's type/preset set equals the set discoverable by reflection over `menger.dsl.*` (mirror `ScriptParitySuite`). Divergence fails the test.
- [ ] 2.4 Add an ArchUnit/reachability test that the catalog loader reads only from the catalog artifact (no inline DSL type literals drift in).

## 3. Static guard / safety (D4, spec: ai-scene-generation › "Static guard rejects unsafe generated code")

- [ ] 3.1 Implement the deterministic allowlist guard over generated `.scala` source: permitted imports are `menger.dsl.*`, `scala.language.*`, `scala.math`, `examples.dsl.common.*`; reject `Runtime`, `sys.process`, `java.io`, `java.net`, `System.exit`, reflection.
- [ ] 3.2 Unit-test the guard: safe scene passes; forbidden-import file rejected; `Runtime`/`System.exit` reference rejected; rejection produces a structured error message usable as a compile-style feedback.
- [ ] 3.3 Add an explicit "do not run the loop outside a container against untrusted input" safety note in module README and arc42 §11 (risk register).

## 4. LLM adapter interface & stub (D2)

- [ ] 4.1 Define the `SceneGenerator` interface: `clarify(description, catalog)`, `generate(description, catalog, answers)`, `refine(scene, feedback)`.
- [ ] 4.2 Implement a scripted **stub adapter** (deterministic, no network) capable of driving the full loop for tests.
- [ ] 4.3 Add unit tests for the loop using the stub adapter (no real LLM, no network).

## 5. GPU-free scene validation (D5, spec: scene-refinement-loop › "GPU-free scene validation")

- [ ] 5.1 Add `menger --validate-scene <file.scala>` CLI path: runs `SceneCompiler` + `SceneLoader` load-only, exits non-zero with the `Left` message on failure, exit zero on success, NO rendering, NO GPU.
- [ ] 5.2 Unit/integration tests: valid file validates without GPU; invalid file reports the compile error; no render is invoked.
- [ ] 5.3 Document `--validate-scene` in the CLI reference / user guide (additive, independently useful).

## 6. Generation & clarification (spec: ai-scene-generation)

- [ ] 6.1 Implement the generation step: description + catalog → `.scala` file text, using only catalog constructs, written to a temp file.
- [ ] 6.2 Implement the clarification protocol: detect ambiguity (missing material, contradictory lighting, undefined count, empty description) and emit targeted questions; only generate once sufficiently specified or defaults accepted.
- [ ] 6.3 Integrate the static guard (task 3) as the gate before any compile: rejected output returns to the generator as a compile-style error.
- [ ] 6.4 Tests: straightforward description yields a compiling scene; multi-object+feature description uses only catalog constructs; empty description returns a clarification rather than a file; non-existent preset is mapped/asked, never emitted literally.

## 7. Refinement loop & structured feedback (spec: scene-refinement-loop)

- [ ] 7.1 Define the sealed `Feedback` type: `CompileError(msg)`, `RenderStats(json, image)`, `UserText(text)`, `VisionCritique(image, analysis)`.
- [ ] 7.2 Implement the compile-error auto-correction loop: feed `--validate-scene`/`SceneCompiler` `Left` back to the generator; bounded by the compile-fix ceiling owned in task 9.1 (default ≤3 attempts); escalate to user on exhaustion.
- [ ] 7.3 Implement render invocation (subprocess `menger --scene`), success/failure handling, and collection of JSON stats (via the existing `--stats-json <file>` flag) + output PNG into `RenderStats`.
- [ ] 7.4 Implement refinement from `UserText` (targeted parameter changes, preserve rest of scene) and ensure the text path works with NO vision adapter configured.
- [ ] 7.5 Tests: compile error auto-fixed within bound; 3 failures escalate; successful render yields `RenderStats`; failed render yields failure feedback without crashing the loop; user-text feedback adjusts scene.

## 8. Preview-then-final rendering (D7, spec: scene-refinement-loop › "Preview-then-final rendering")

- [ ] 8.1 Define orchestrator-level preview and final render-setting presets (resolution, accumulation frames, denoise).
- [ ] 8.2 Ensure iteration renders use preview settings regardless of the scene's own quality settings; final high-quality render only on user acceptance. Assert via test that the LLM/refinement does not mutate render settings per iteration.

## 9. Bounded iteration & cost safety (spec: scene-refinement-loop › "Bounded iteration and cost safety")

- [ ] 9.1 Own the configurable bound constants consumed by the loop — per-generation compile-fix ceiling (default ≤3) and refinement-round ceiling; on exhaustion stop and report last scene + last error/render + ask user. (Task 7.2 consumes the compile-fix ceiling.)
- [ ] 9.2 Tests: refinement-round ceiling honored; bounded termination always yields a usable status (never an infinite loop).

## 10. First real LLM adapter (D2, open question 1)

- [ ] 10.1 Pick the initial provider (OpenAI / Anthropic / local) and implement an HTTPS adapter satisfying the `SceneGenerator` interface (no heavyweight SDK dependency).
- [ ] 10.2 Add configuration for endpoint + credentials via env/config (no secrets in repo); stub remains the default for tests.
- [ ] 10.3 Smoke test the real adapter behind a guarded integration flag (opt-in, not in the default CI lane).

## 11. Docs, integration tests, definition-of-done

- [ ] 11.1 Write user-facing doc for the AI authoring workflow (NL → DSL → render → refine); cross-link from the DSL reference.
- [ ] 11.2 Add an end-to-end integration test (stub-driven) covering: describe → clarify → generate → guard → validate → preview-render → user-text refine → accept → final-render.
- [ ] 11.3 Update arc42 §9 (decision: LLM is an external swappable authoring concern) and §11 (risk: untrusted-Scala execution mitigated by D4) per the Definition of Done.
- [ ] 11.4 Run the pre-push hook (`./.git_hooks/pre-push 2>&1 | tee /tmp/pre-push.log`) and mark the change complete in the sprint doc.

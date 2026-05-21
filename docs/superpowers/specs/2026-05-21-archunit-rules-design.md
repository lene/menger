# ArchUnit Rules Design

**Date:** 2026-05-21  
**Status:** Approved  
**Branch:** feature/tooling-static-analysis

## Context

Current `ArchitectureSpec` enforces three module boundary rules (menger.common and menger.optix must
not depend on application layers). This spec expands coverage to a full architectural rule suite,
organised in phases matching required prerequisite refactors. Extraction of `optix-jni` as a
separate artifact is a near-term goal; rules in Phase 1 and 3 are directly aimed at making that
extraction mechanical rather than risky.

## Architectural model

The codebase targets a hexagonal / clean-architecture shape with three sbt modules already
reflecting the intended layering:

```
menger-common   menger.common.*         domain primitives — most stable layer
optix-jni       menger.optix.*          JNI adapter — native code lives here only
menger-app      menger.{objects,dsl,config,input,engines,cli}.*
                menger.optix.*          thin wrapper (anti-corruption layer over optix-jni)
```

Onion layer ordering (inner → outer):

```
L0  menger.common                   shared primitives
L1  menger.objects, menger.dsl      geometry + scene language
L2  menger.config                   application configuration
L3  menger.optix (wrapper), menger.input    ports / adapters
L4  menger.engines, menger.engines.scene    orchestration
L5  menger.cli, Main                boundary (CLI entry)
    ─────────────────────────────────────
    optix-jni                       external adapter (separate artifact)
    LibGDX                          external adapter (separate artifact)
```

Dependencies must point inward only (outer may depend on inner, never the reverse).

## Current violations blocking layer rules

Two violations exist today that must be resolved before the corresponding rules can be enabled:

**V1 — `menger.dsl` → `menger.cli`**  
DSL imports `LightSpec`, `PlaneColorSpec`, `PlaneConfig`, `AxisSpec` from `menger.cli`. These are
shared domain shapes currently homed in the CLI package. Fix: move shared types to `menger.common`
or into `menger.dsl` itself; `menger.cli` adapts raw arg strings to those types.

**V2 — `menger.objects` → `menger.input`**  
A class in `menger.objects` imports from `menger.input` (geometry depending on input events).
Fix: extract the interactive-response behaviour to `menger.engines.scene` or a new sub-package;
`menger.objects` retains pure geometry data.

## Root-level `menger.*` types

`ObjectSpec`, `AnimationSpecification`, `RenderState`, `ProfilingConfig`, and similar sit in the
root `menger` package, bypassing layer rules. Triage target (migrated case-by-case, not en masse):

| Type | Target package |
|------|----------------|
| Domain shapes (`ObjectSpec`, `AnimationSpecification`) | `menger.dsl` or `menger.common` |
| Render state (`RenderState`, `OptiXRenderResources`) | `menger.engines` |
| Cross-cutting utilities (`ProfilingConfig`, `ColorConversions`) | `menger.common` |

Types are migrated as part of Phase 0 refactors or opportunistically during later work.

---

## Phase 0 — Prerequisite refactors

Must complete before Phase 2 rules can be added to `ArchitectureSpec`.

- **P0.A** Decouple `menger.dsl` from `menger.cli`:  
  Identify CLI-spec types imported by DSL. Move them to `menger.common`. Update DSL and CLI imports.
- **P0.B** Decouple `menger.objects` from `menger.input`:  
  Identify the coupling class. Extract interactive-response behaviour to `menger.engines`.

These are standard refactors (move type + fix imports). No behaviour changes.

---

## Phase 1 — Rules enabled immediately (no refactor needed)

Add to `ArchitectureSpec` (or a separate `ArchitecturePhase1Spec`) without code changes.

### 1.1 Acyclic dependencies

```scala
SlicesRuleDefinition.slices()
  .matching("menger.(*)..")
  .should().beFreeOfCycles()
  .check(allClasses)
```

*Rationale:* Cycles couple modules permanently; prevent artifact extraction and incremental
recompilation. Almost certainly already absent — cheap to lock in.

### 1.2 JNI boundary isolation

- Only classes in `menger.optix` (menger-app wrapper) may import from `optix-jni` classes.
- No `System.loadLibrary` or `System.load` calls outside `optix-jni` module.
- No `@native` method declarations outside `optix-jni` module.
- `optix-jni` must not depend on `com.badlogic.gdx.*` (two adapters must not couple).

*Rationale:* `optix-jni` extraction is a near-term goal. A single seam (`menger.optix` wrapper)
means extraction requires no changes to callers — only packaging changes.

### 1.3 Java-friendly API surface in `optix-jni`

Public method signatures in `optix-jni` use only `Array`, primitives (`int`, `long`, `float`,
`boolean`), `String`, and types from `menger.common`. No `scala.Option`, `scala.collection.*`,
`scala.util.Try` in public signatures.

*Rationale:* `optix-jni` is intended for any JVM language. Scala-specific types break Java and
Kotlin consumers. Fixing this before extraction avoids a breaking API change later.

### 1.4 Side-effect anti-patterns in production code

- No `println` / `System.out.print*` / `System.err.print*` outside `Main.scala`.
- No `System.exit` outside `Main.scala`.
- No `???` (unimplemented placeholder) in production source sets.

*Rationale:* `println` bypasses structured logging (no level, no timestamps, no filtering).
`System.exit` in library code breaks embeddability and makes tests hard to isolate. `???` in
production crashes with no actionable diagnostic.

### 1.5 Naming conventions

- Classes named `*Engine` reside in `menger.engines` or a sub-package.
- Classes named `*Config` reside in `menger.config` or `menger.common`.
- Classes in `optix-jni` whose name starts with `OptiX` reside in `menger.optix.*` (not scattered across other packages in that module).
- Test classes are suffixed `Suite` or `Spec` (not both on same class name prefix).

*Rationale:* When names predict location, navigation is fast and misplacements are immediately
visible.

### 1.6 Resource lifecycle

Classes that hold a JNI resource handle (heuristic: class name contains `Wrapper` and has a field
of `OptiXRenderer` type) must implement `java.lang.AutoCloseable`.

*Rationale:* JNI handles are native memory. Without `AutoCloseable`, callers have no standard
signal that disposal is required, leading to leaks in long-running render sessions.

### 1.7 Error handling baseline

- Domain layers (`menger.common`, `menger.dsl`, `menger.objects`) throw only subclasses of
  `MengerException` (already defined in `menger.common`). No `new RuntimeException("...")` directly.

*Rationale:* Typed exceptions allow catch blocks at application boundaries to distinguish domain
errors from programming errors.

---

## Phase 2 — Rules enabled after Phase 0 refactors

### 2.1 Onion layer ordering

```
menger.objects  → menger.common only
menger.dsl      → menger.common + menger.objects only
menger.config   → menger.common + menger.dsl only
menger.input    → menger.common + menger.optix (wrapper) only
menger.cli      → menger.common + menger.config + menger.dsl only
menger.engines  → any of the above (no restriction; outer layer)
```

*Rationale:* Onion layering is the structural guarantee that inner layers never see outer-layer
change. Without enforcement, any developer can reintroduce cross-layer imports silently.

### 2.2 Domain immutability

- No `var` fields in `menger.common` or `menger.dsl`.
- No `scala.collection.mutable.*` in `menger.common` or `menger.dsl`.
- All `*Config` types are case classes.

*Rationale:* Mutable domain state breaks structural equality, enables race conditions in parallel
rendering pipelines, and makes property-based tests unreliable.

### 2.3 Domain purity (no side effects in inner layers)

- No `java.io.*` or `java.nio.file.*` in `menger.common`, `menger.objects`, `menger.dsl`.
- No SLF4J / `LazyLogging` in `menger.common` or `menger.objects`.

*Rationale:* Pure inner layers need zero mocking in unit tests. File IO and logging imports pull
infrastructure into the core domain.

### 2.4 Sealed hierarchies (scoped)

- Scene node variant types in `menger.dsl` (the closed set of renderable node types) must use
  sealed traits/abstract classes.
- Error type hierarchy under `MengerException` in `menger.common` must be sealed per level.

*Rationale:* Sealed hierarchies give exhaustive pattern-match checking. Applied only where the set
of variants is genuinely closed — not as a blanket policy.

---

## Phase 3 — Rules enabled at optix-jni extraction

### 3.1 Wrapper API purity

Public method return types and parameter types of `menger.optix` wrapper classes (`OptiXRendererWrapper`,
`SceneConfigurator`, `CameraState`) use only `menger.common` types or primitives. No raw `Long`
handles, no `Array[Byte]` without a typed wrapper.

*Rationale:* If the wrapper leaks JNI handle types, callers become co-owners of native resources.
The wrapper is the seam; it must absorb all native-type ugliness and expose only domain types.

### 3.2 Module dependency post-extraction

After extraction, `menger-app` depends on the published `optix-jni` artifact. ArchUnit verifies
that only `menger.optix` wrapper imports from `optix-jni`'s published classes — no other package
in `menger-app` reaches across the artifact boundary.

*Rationale:* Same as Phase 1 rule 1.2, but now enforced at the Maven/sbt dependency level rather
than source-classpath level.

---

## Implementation notes

- All rules live in `ArchitectureSpec.scala` (or sibling files by phase, e.g. `ArchitecturePhase2Spec.scala`).
- Rules for violations not yet fixed are added as `@Ignore` tests with a comment referencing the
  blocking task, so intent is captured and CI does not fail.
- ArchUnit version already on classpath (added in Sprint 21 tooling work); no new dependency needed.
- `allClasses` import in `ArchitectureSpec` already imports the full `menger` package tree from
  all sbt modules on the test classpath; no changes to import setup needed for Phase 1.

## Out of scope

- Package-private visibility migration (`private[engines]` for `engines.scene`) — deferred; no
  ArchUnit rule depends on it.
- Test-class-to-production mirroring beyond naming conventions — too brittle to enforce via ArchUnit
  given Scala's class file naming.
- P9 thread-safety rules (no `Thread.sleep` in unit tests) — not enforceable via ArchUnit bytecode
  analysis without custom predicates; rely on code review.

## ADDED Requirements

### Requirement: Bounded compile-error auto-correction

When a generated scene fails compilation, the system SHALL feed the compiler's structured error
message (the `Left` value from `SceneCompiler`) back to the generator and re-attempt generation,
without surfacing the raw error to the user as a dead end. The auto-correction loop MUST be bounded
(≤3 attempts by default); if the bound is reached the system MUST escalate to the user rather than
loop indefinitely.

#### Scenario: Compile error is auto-fixed within the bound

- **WHEN** a generated file fails to compile with error E and the corrected file compiles on retry
- **THEN** the user receives the corrected, compiling scene and never sees error E as a final outcome

#### Scenario: Repeated compile errors escalate to the user

- **WHEN** compilation fails on 3 consecutive attempts
- **THEN** the system stops auto-retrying and presents the last error to the user with the failed
  file, requesting guidance

### Requirement: GPU-free scene validation

The system SHALL be able to validate that a `.scala` scene compiles and loads WITHOUT rendering and
WITHOUT requiring a GPU. This MUST be available as a `menger --validate-scene <file.scala>` CLI path
(exit non-zero with the error message on failure, exit zero on success) so the refine loop's
compile-check step is fast and CI-runnable.

#### Scenario: Valid scene validates without a GPU

- **WHEN** `--validate-scene` is run on a well-formed `.scala` file on a machine with no GPU
- **THEN** the command exits zero and performs no rendering

#### Scenario: Invalid scene reports the compile error without rendering

- **WHEN** `--validate-scene` is run on a file with a compile error
- **THEN** the command exits non-zero and prints the `SceneCompiler` error message
- **AND** performs no rendering

### Requirement: Structured feedback collection after a render

After a scene renders, the system SHALL collect structured feedback consisting of render success or
failure, the JSON render stats exported by menger, and the rendered output image. This feedback MUST
be represented as a sealed `Feedback` type with at least the cases `CompileError`, `RenderStats`,
`UserText`, and `VisionCritique`.

#### Scenario: Successful render produces stats and an image

- **WHEN** a preview render completes successfully
- **THEN** the system produces a `RenderStats` feedback carrying the exported JSON stats and a
  reference to the rendered PNG

#### Scenario: Failed render produces a failure feedback

- **WHEN** a render fails (e.g. out of memory, instance limit exceeded)
- **THEN** the system produces a feedback describing the failure without crashing the loop

### Requirement: Refinement from feedback

The system SHALL refine an existing scene in response to feedback. Compile errors are handled by the
auto-correction requirement; other feedback (`UserText`, optional `VisionCritique`, `RenderStats`)
MUST drive targeted parameter changes (object positions/sizes, materials, lighting, feature blocks)
rather than regenerating the scene from scratch. The text-feedback path MUST function without any
multimodal/vision capability.

#### Scenario: User text feedback adjusts the scene

- **WHEN** the user says "make the glass bluer and the sponge smaller" on a rendered scene
- **THEN** the system adjusts the relevant material color and object size and re-renders, preserving
  the rest of the scene

#### Scenario: Refinement works without a vision model

- **WHEN** no vision-capable adapter is configured
- **THEN** the system still refines scenes using `UserText` and `CompileError` feedback

#### Scenario: Vision critique is opt-in

- **WHEN** a vision-capable adapter is configured and a render PNG is available
- **THEN** the system MAY produce a `VisionCritique` feedback to drive refinement
- **AND** when no vision adapter is configured, no `VisionCritique` feedback is ever produced

### Requirement: Preview-then-final rendering

For iteration speed, refinement renders SHALL use fast preview settings (reduced resolution, fewer
accumulation frames, denoised). A final high-quality render MUST be produced only once the user
accepts the scene. Render settings are an orchestrator-level concern; the LLM MUST NOT mutate render
settings in response to per-iteration user feedback.

#### Scenario: Iteration uses preview settings

- **WHEN** the refine loop renders a candidate scene during iteration
- **THEN** it uses preview render settings regardless of the scene's own quality settings

#### Scenario: Final render uses high quality only on acceptance

- **WHEN** the user accepts a scene
- **THEN** the system renders once at final high-quality settings

### Requirement: Bounded iteration and cost safety

The system SHALL enforce bounds on both the per-generation compile-fix loop (≤3 attempts) and the
overall refinement rounds (configurable ceiling). On exhaustion of either bound the system MUST stop
and ask the user rather than continue spending LLM calls. The user-facing outcome after any bounded
termination MUST be a clear status (last scene, last error, or last render), never an infinite loop.

#### Scenario: Refinement-round ceiling is honored

- **WHEN** the configured maximum number of refinement rounds is reached without acceptance
- **THEN** the system stops and reports the current state to the user

#### Scenario: Bounded termination always yields a usable status

- **WHEN** any bound (compile-fix or refinement-round) is reached
- **THEN** the system reports the last scene file, last error or render, and asks the user how to
  proceed

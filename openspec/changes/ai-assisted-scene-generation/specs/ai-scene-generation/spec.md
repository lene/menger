## ADDED Requirements

### Requirement: Natural-language scene generation

The system SHALL accept a natural-language description of a scene and produce a `.scala` file that
is valid against the `menger.dsl.*` surface and consumable by `menger --scene <file>.scala`. The
generated file MUST use only DSL constructs present in the tool-catalog (D3).

#### Scenario: Straightforward description produces a renderable scene

- **WHEN** the user submits "a chrome sphere at the origin with a single directional light"
- **THEN** the system emits a `.scala` file containing a `Sphere` with `Material.Chrome` and one
  `Directional` light, wrapped in a valid `Scene`
- **AND** the file passes the static guard and compiles via the runtime `SceneCompiler`

#### Scenario: Description with multiple objects, lights, and a feature block

- **WHEN** the user submits "a glass sphere and a gold cube on a checkered floor with warm
  three-point lighting and caustics"
- **THEN** the generated file contains two objects, three lights, a `Plane.checkered`, and
  `caustics = Some(Caustics(...))`
- **AND** every construct used is present in the tool-catalog

#### Scenario: Empty or meaningless description is rejected, not guessed

- **WHEN** the user submits a description with no identifiable scene content (e.g. empty string)
- **THEN** the system does NOT emit a scene file
- **AND** instead returns a clarification request

### Requirement: Clarification protocol for ambiguous intent

The system SHALL identify ambiguities or missing essentials in the description (unspecified scale,
missing material, contradictory lighting, undefined object count) and ask targeted clarifying
questions BEFORE generating, rather than silently inventing values. Generation MUST only proceed
once the description is sufficiently specified or the user explicitly accepts defaults.

#### Scenario: Missing material triggers a clarification

- **WHEN** the user submits "a sphere next to a cube" with no material for either
- **THEN** the system asks which material/preset to use for each object before generating

#### Scenario: Contradictory intent triggers a clarification

- **WHEN** the user submits "a bright sunny daytime scene, fully dark and moody"
- **THEN** the system asks the user to resolve the contradiction rather than picking one silently

#### Scenario: User accepts defaults to skip clarification

- **WHEN** the system offers to use sensible defaults and the user accepts
- **THEN** generation proceeds using catalog defaults without further questions

### Requirement: Tool-catalog constrains generation to the real DSL surface

The system SHALL consult a tool-catalog that enumerates every DSL object type, material preset,
light kind, camera/plane option, and feature block (caustics, fog, env-map, tone-mapping) with each
parameter's name, type, default, and range. The generator MUST NOT emit a type, preset, or parameter
that is absent from the catalog.

#### Scenario: Generator does not invent a non-existent preset

- **WHEN** the description implies a material not in the catalog (e.g. "jade")
- **THEN** the system maps it to the nearest catalog material or asks, and never emits a literal
  like `Material.Jade` that does not exist

#### Scenario: Out-of-range parameter is clamped or clarified

- **WHEN** the description implies an out-of-range value (e.g. sponge level 99)
- **THEN** the system either clamps to the catalog-stated range or asks the user, and never emits a
  value known to be invalid

### Requirement: Static guard rejects unsafe generated code

Before any generated `.scala` file is compiled or rendered, the system SHALL pass it through a
deterministic static guard that rejects any source whose imports or constructs fall outside an
allowlist. Permitted imports are restricted to `menger.dsl.*`, `scala.language.*`, `scala.math`, and
`examples.dsl.common.*`. References to `Runtime`, `sys.process`, `java.io`, `java.net`,
`System.exit`, or reflection MUST be rejected. A rejected file MUST be fed back to the generator as a
compile-style error; it MUST NOT reach `SceneCompiler`.

#### Scenario: Safe scene passes the guard

- **WHEN** the generated file imports only `menger.dsl.*` and `scala.language.implicitConversions`
- **THEN** the guard accepts it and compilation proceeds

#### Scenario: File with a forbidden import is rejected before compilation

- **WHEN** the generated file contains `import sys.process._`
- **THEN** the guard rejects it
- **AND** the rejection is returned to the generator as an error message
- **AND** `SceneCompiler` is never invoked on the file

#### Scenario: Runtime/exec reference is rejected

- **WHEN** the generated file references `Runtime.getRuntime` or `System.exit`
- **THEN** the guard rejects it before compilation

### Requirement: Catalog-parity fitness function

The system SHALL include an automated test that asserts the tool-catalog's set of object types and
material presets equals the set discoverable by reflection over `menger.dsl.*`. Divergence (a
catalog entry with no matching DSL type, or a DSL type missing from the catalog) MUST fail the test,
mirroring the existing `ScriptParitySuite` pattern.

#### Scenario: Catalog missing a DSL type fails CI

- **WHEN** a new object type is added to `menger.dsl.*` but not to the catalog
- **THEN** the catalog-parity test fails

#### Scenario: Catalog promises a non-existent type fails CI

- **WHEN** the catalog lists a material preset that does not exist in `menger.dsl.*`
- **THEN** the catalog-parity test fails

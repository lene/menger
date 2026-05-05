# Plan: 4D Polytope Test Strategy + Bug Fixes

## Context

Test coverage dropped sharply because three new 4D polytopes (`Pentachoron`, `Hexadecachoron`, `Icositetrachoron`) shipped with **zero tests**. Lack of tests masked geometry bugs:

1. **Icositetrachoron (24-cell)**: visibly mis-rendered. Pole-index sign mapping is inverted (`if si > 0 then 0 else 1` — but `axialVertices(0)` is `-s`, so a positive sign indexes the negative-pole vertex). Asymmetrically flips half the octahedral cells.
2. **Pentachoron (5-cell)**: irregular simplex. Comment says "regular simplex (unit edge length)" but `r5 = √(2/5)` produces corner-corner edges of length `2√2·s` and corner-apex edges of length `√13·s`. Correct value: `r5 = √(1/5)·s = s/√5`.
3. **All three**: face duplication. Each shared face is emitted once per incident cell. Counts: 20/64/192 (should be 10/32/96).
4. **Mesh4D API gap**: trait exposes only `faces`. Tesseract adds `edges` ad-hoc. No `cells` API anywhere.

The 600-cell and 120-cell are slated next; without a generic test contract, the same regression will recur.

User decisions (locked):
- Enforce regularity (Pentachoron must be fixed).
- Dedup faces; canonical counts (10/32/96).
- Add `edges` + `cells` to `Mesh4D` trait.
- Define a generic `Polytope4DContract` trait now; concrete 600/120-cell suites later.

## Strategy

**Two-tier testing:**

1. **Generic contract trait `Polytope4DContract`** (new test mixin) — applies the same battery of geometric tests to any `Mesh4D`, parameterised by expected (V, E, F, C) counts and per-polytope properties (`vertexNorm`, `edgeLength`, `faceShape`, `cellShape`).
2. **Polytope-specific suites** — extend the contract, add literal coordinate checks and polytope-unique invariants (e.g., 24-cell self-duality, 16-cell orthoplex axes).

**Test categories** (each contract polytope gets all of A–F; mesh suites get G):

| Category | Tests |
|---|---|
| **A. Vertex** | count; coords on common 3-sphere (norm = const); centroid at origin; scales linearly with `size`; no NaN/Inf; all distinct |
| **B. Edge** | count (unique); all equal length (regularity); each edge endpoints in vertex set |
| **C. Face** | count (unique); all triangular (or square for tesseract); equal area; no degenerate (`area > 0`); face vertices in vertex set |
| **D. Cell** | count; cell shape (tetrahedral / octahedral / cubic / dodecahedral): vertex count per cell, edge count per cell, regular-cell invariants. Dodecahedron contract (12 pentagonal faces, 30 edges, 20 vertices, all edges equal length, golden-ratio vertex coords) included now in preparation for 120-cell, even though no polytope currently uses it. |
| **E. Topology** | Euler-Poincaré: `V - E + F - C = 0`; every face shared by exactly 2 cells; every edge shared by ≥ 3 faces; closed-manifold via `MeshTopology.checkFace4D` (no boundary edges) |
| **F. Symmetry** | Vertex set invariant under coordinate permutation (`Sₙ` subgroup); under sign flips (`(Z/2)⁴` for cube/orthoplex/24-cell); centroid preserved |
| **G. Mesh projection** (separate `*MeshSuite`) | 3D mesh has expected vertex/triangle counts (faces × 3 for triangulated); all triangle normals normalised; no NaN/Inf in mesh data; closed-manifold after projection (`MeshTopology.checkTriangleMesh`); 4D rotations propagate to 3D output |

## Files to modify / create

### Production code (required for tests to pass)

- `menger-app/src/main/scala/menger/objects/higher_d/Mesh4D.scala`
  - Add `lazy val edges: Set[Edge4D]` (default impl: derive from faces, dedup as `Set`).
  - Add `lazy val cells: Seq[Cell4D]` (abstract; concrete polytopes provide).
  - Define `Edge4D` (unordered pair of `Vector[4]`) and `Cell4D` (vertex sequence + shape tag).

- `menger-app/src/main/scala/menger/objects/higher_d/Pentachoron.scala`
  - Fix `r5`: `scala.math.sqrt(0.2).toFloat * s` (= √(1/5)·s) — restores regularity.
  - Add `cells: Seq[Cell4D]` from existing cell-index data.
  - Dedup faces to canonical 10.

- `menger-app/src/main/scala/menger/objects/higher_d/Hexadecachoron.scala`
  - Add `cells` API.
  - Dedup faces to canonical 32.

- `menger-app/src/main/scala/menger/objects/higher_d/Icositetrachoron.scala`
  - Fix pole indexing: `if si > 0 then 1 else 0` (and same for `sj`).
  - Restructure to expose 24 octahedral cells explicitly.
  - Dedup faces to canonical 96.

- `menger-app/src/main/scala/menger/objects/higher_d/Tesseract.scala`
  - Add `cells: Seq[Cell4D]` (8 cubic cells).
  - Migrate ad-hoc `edges` → trait default.

### Test code (new)

- `menger-app/src/test/scala/menger/objects/higher_d/Polytope4DContract.scala` — abstract trait extending `AnyFlatSpec with Matchers`, exposes `polytope: Mesh4D` plus expected counts and properties; runs categories A–F.

- `menger-app/src/test/scala/menger/objects/higher_d/PentachoronSuite.scala`
  - Mixes `Polytope4DContract` with V=5, E=10, F=10, C=5, faceShape=triangle, cellShape=tetrahedron, common vertex norm, common edge length (after regularity fix).
  - Polytope-specific: literal vertex coords; apex on +w axis; vertex graph is K₅ (every pair connected).

- `menger-app/src/test/scala/menger/objects/higher_d/HexadecachoronSuite.scala`
  - Contract: V=8, E=24, F=32, C=16, tetrahedral cells, vertexNorm=s/√2, edgeLength=s.
  - Specific: vertices = signed unit-axis vectors; antipodal pairs not edge-connected (orthoplex); two vertices form an edge iff orthogonal in 4D.

- `menger-app/src/test/scala/menger/objects/higher_d/IcositetrachoronSuite.scala`
  - Contract: V=24, E=96, F=96, C=24, octahedral cells (6 verts, 12 edges, 8 triangle faces each), vertexNorm=s.
  - Specific: vertex set = 8 axial ∪ 16 half; **self-duality** (dual vertex set = scale·original up to permutation); rendering golden-image guard.

- `menger-app/src/test/scala/menger/objects/higher_d/TesseractSuite.scala` (extend, not rewrite)
  - Add: cell count = 8; V−E+F−C = 0; manifold check via `MeshTopology.checkFace4D`; vertex norm = s; (optional) face-area uniformity.

- `menger-app/src/test/scala/menger/objects/higher_d/PentachoronMeshSuite.scala`, `HexadecachoronMeshSuite.scala`, `IcositetrachoronMeshSuite.scala` — mirror `TesseractMeshSuite` pattern (post-projection topology, no NaN/Inf, rotations propagate).

- Reactivate / extend `TopologyDiagnosticSpec` pattern with one ignored per-stage entry per new polytope (per `debugging-rendering-bugs` skill).

### Future-prep (added now)

- `menger-app/src/test/scala/menger/objects/higher_d/CellShapeContract.scala` — defines reusable assertions per cell shape: `assertTetrahedron(cell)`, `assertOctahedron(cell)`, `assertCube(cell)`, **`assertDodecahedron(cell)`**. Each verifies vertex count, edge count, face count, edge-length uniformity, and face-shape (triangular / square / pentagonal). Used by category D in `Polytope4DContract`.
- **`DodecahedronContractSpec`** (new test file, no production code yet): exercises `assertDodecahedron` against a constructed reference dodecahedron fixture (golden-ratio vertex set, 20 vertices, 30 edges, 12 pentagonal faces). Validates the contract assertion itself before any 120-cell consumes it. Mirrors the `MeshTopologySpec` "validate the detector before trusting it" pattern from the `debugging-rendering-bugs` skill.

### Future (locked-in placeholder)

`Polytope4DContract` is the entry point for **600-cell** and **120-cell** when classes land. No suite files yet; implementer supplies a constants table:

```scala
class Hexacosichoron600CellSuite extends Polytope4DContract:
  val polytope = Hexacosichoron()
  val expected = PolytopeSpec(V=120, E=720, F=1200, C=600,
    faceShape=Triangle, cellShape=Tetrahedron, vertexNorm=...)

class Hecatonicosachoron120CellSuite extends Polytope4DContract:
  val polytope = Hecatonicosachoron()
  val expected = PolytopeSpec(V=600, E=1200, F=720, C=120,
    faceShape=Pentagon, cellShape=Dodecahedron, vertexNorm=...)
```

Because `assertDodecahedron` already exists and is fixture-validated, the 120-cell suite drops in with no extra cell-shape work.

## Critical files — reuse

- `menger-app/src/test/scala/menger/objects/higher_d/MeshTopology.scala` — `checkFace4D(faces)`, `checkTriangleMesh(mesh)` returning `isManifold`, `edgeUseHistogram`, `boundaryFaces`. Reuse for category E + G.
- `menger-app/src/test/scala/menger/objects/higher_d/MeshTopologySpec.scala` — fixture validation (unit cube manifold, cube-minus-triangle non-manifold). Trustworthy detector per `debugging-rendering-bugs` skill.
- `menger.common.Vector` — 4-vector ops (`dst`, `dst2`, `+`, `*`, `len`).
- `menger.common.Const.epsilon` — float comparison tolerance.
- `menger.objects.higher_d.Face4D.area` — already correct for triangles.
- `menger.objects.higher_d.Plane` — face plane classification (used by current `TesseractSuite`).

## Verification

1. `sbt mengerApp/test` — full suite green; all new contract tests pass.
2. `sbt "mengerApp/testOnly menger.objects.higher_d.PentachoronSuite"` etc. — run each new suite in isolation.
3. Render-correctness end-to-end:
   ```
   ./menger-app/target/universal/stage/bin/menger-app -o --headless \
     -s /tmp/icositetrachoron.png --objects type=icositetrachoron:level=1
   ```
   Visual inspection: 24-cell projection looks symmetric (per `debugging-rendering-bugs` skill: trust the eye, not just the number).
4. `MeshTopology.checkFace4D(Icositetrachoron().faces)` reports `isManifold=true`, `boundaryFaces=Set.empty` after fix.
5. Coverage report: `sbt mengerApp/jacoco` — `higher_d` package coverage restored.

## Order of execution (when plan approved)

1. **RED**: Write `Polytope4DContract` + concrete suites against current (broken) implementations. New tests must fail with specific signatures (regularity, count, manifold).
2. **API**: Extend `Mesh4D` with `edges`/`cells`. Tesseract migration.
3. **GREEN**: Fix Pentachoron `r5`; fix Icositetrachoron pole indexing; dedup faces in all three. Tests pass.
4. **Visual crosscheck**: render 24-cell headless; confirm symmetric projection.
5. **Document**: close affected `CODE_IMPROVEMENTS.md` entries; commit per-step.

## Out of scope

- 600-cell / 120-cell concrete classes (don't exist yet).
- Refactoring `Mesh4DProjection` / `Mesh4DGpuFlatten`.
- Performance benchmarks.
- Material/shader tests.

<!--
SYNC IMPACT REPORT
==================
Version change: N/A (new) → 1.0.0
Bump rationale: Initial constitution creation (MAJOR - new governance document)

Modified principles: None (initial creation)
Added sections:
  - Core Principles (6 principles derived from AGENTS.md and arc42)
  - Technology Constraints (from AGENTS.md BUILD REQUIREMENTS)
  - Development Workflow (from AGENTS.md DEVELOPMENT WORKFLOW)
  - Governance (new)
Removed sections: N/A (template placeholders replaced)

Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section present)
  - .specify/templates/spec-template.md: ✅ Compatible (no constitution-specific sections)
  - .specify/templates/tasks-template.md: ✅ Compatible (no constitution-specific sections)

Follow-up TODOs:
  - TODO(RATIFICATION_DATE): Using today's date as ratification; update if earlier
    adoption date is known
-->

# Menger Constitution

## Core Principles

### I. Functional Programming (NON-NEGOTIABLE)

All Scala code MUST follow functional programming principles:

- **Immutability**: Use `val` not `var`. Mutable state is forbidden except where
  LibGDX integration requires `@SuppressWarnings` annotation.
- **Error handling**: Use `Try`, `Either`, or `Option` instead of exceptions.
  The `throw` keyword is prohibited by Wartremover.
- **No nulls**: Null is forbidden in all Scala code. Use `Option` for optional values.
- **Pure functions**: Prefer functions without side effects. Side effects MUST be
  explicit and contained.

**Rationale**: Functional code is easier to test, reason about, and maintain. This
is enforced by Wartremover and Scalafix at compile time.

### II. Architecture Documentation First

The arc42 architecture documentation at `docs/arc42/README.md` is the **single source
of truth** for architectural decisions.

- **Before** making architectural decisions: Consult Section 9 (Decisions),
  Section 10 (Quality Requirements), Section 11 (Risks).
- **After** making changes: Update arc42 if affecting architecture, quality, or risks.
- Outdated documentation is worse than no documentation. Documentation MUST stay current.

**Rationale**: Centralized architecture documentation prevents knowledge silos and
enables informed decision-making.

### III. Code Quality Enforcement

All code MUST pass automated quality checks before merge:

- **Wartremover**: No `var`, `while`, `asInstanceOf`, `throw` in production code.
- **Scalafix**: OrganizeImports, DisableSyntax (noNulls, noReturns), no unused imports.
- **Line length**: Maximum 100 characters.
- **Imports**: One per line, organized per `.scalafix.conf`.

Quality checks are enforced via `sbt "scalafix --check"` and pre-push hooks.

**Rationale**: Automated enforcement removes subjectivity from code review and
maintains consistent quality.

### IV. Alpha Channel Convention (DOMAIN-CRITICAL)

This project uses a specific alpha channel convention that MUST NOT be confused:

- **alpha = 0.0** means **FULLY TRANSPARENT** (no opacity, no absorption)
- **alpha = 1.0** means **FULLY OPAQUE** (full opacity, maximum absorption)

This applies to: OptiX shaders (`sphere_combined.cu`), Beer-Lambert absorption,
Scala Color objects, and all tests.

**Rationale**: Incorrect alpha handling causes subtle rendering bugs that are
difficult to diagnose. This convention is critical for physically-based rendering.

### V. Dual Rendering Pipeline

The project maintains two parallel rendering paths:

- **LibGDX (OpenGL)**: Real-time preview, cross-platform, interactive exploration.
- **OptiX (ray tracing)**: High-quality output, physically-based rendering
  (refraction, caustics, shadows).

Geometry MUST be exportable to both formats. Each path has independent camera controllers.

**Rationale**: Different use cases require different rendering approaches.
Interactive exploration needs speed; final output needs quality.

### VI. Data Safety

- **Never delete data without explicit user permission**.
- Always confirm before destructive operations.
- Git operations: Never `git add -A`; add files explicitly. Never commit
  automatically; always show diff for user review first.

**Rationale**: Data loss is irreversible. User control over destructive operations
is non-negotiable.

## Technology Constraints

### Required Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Scala | 3.x | Never Scala 2 syntax |
| Java | 21+ | JNI requires compatible version |
| sbt | 1.11+ | Build tool |
| CUDA Toolkit | 12.0+ | GPU acceleration |
| NVIDIA OptiX SDK | 9.0+ | Ray tracing (must match driver) |
| CMake | 3.18+ | Native compilation |
| C++ | 17 | OptiX JNI bindings |

### OptiX SDK Version Matching (CRITICAL)

OptiX SDK version MUST match the installed NVIDIA driver:

- Driver 580.x+ requires OptiX SDK 9.0+
- Driver 535-575.x requires OptiX SDK 8.0

Mismatched versions cause CUDA error 718. Verify with:
```bash
strings /usr/lib/x86_64-linux-gnu/libnvoptix.so.* | grep "OptiX Version"
```

### Test Framework

- **Scala tests**: AnyFlatSpec (ScalaTest)
- **C++ tests**: Google Test
- **Current test count**: ~1,070 total (27 C++ + 1,043 Scala)

## Development Workflow

### Standard Development Cycle

1. **Make changes**: Follow code standards, update CHANGELOG.md (keepachangelog.com format)
2. **Compile**: `sbt compile`
3. **Test**: `sbt test`
4. **Quality check**: `sbt "scalafix --check"`
5. **Pre-push assessment**: Comprehensive code quality review, document in
   `CODE_IMPROVEMENTS.md`
6. **User review**: Show diff, user decides when to commit
7. **Push**: Run pre-push hook (up to 5 minutes)

### Pre-Push Quality Assessment

Before every `git push`, perform comprehensive assessment of:

- Clean code guidelines
- Clarity of intent
- Separation of concerns
- Functional programming adherence
- Code duplication
- Hardcoded constants
- Over-long functions and classes
- Architectural efficiency

Assess the **entire codebase**, not just new code. Document findings in
`CODE_IMPROVEMENTS.md`.

## Governance

### Amendment Process

1. Propose changes with rationale in a pull request.
2. Update this constitution with new version number.
3. If adding/removing principles, update dependent templates (plan, spec, tasks).
4. Amendments require explicit approval before merge.

### Version Policy

This constitution follows semantic versioning:

- **MAJOR**: Backward-incompatible changes (principle removal, redefinition)
- **MINOR**: New principles or materially expanded guidance
- **PATCH**: Clarifications, wording fixes, non-semantic refinements

### Compliance

- All pull requests MUST verify compliance with this constitution.
- Complexity beyond these guidelines MUST be justified in the PR description.
- For runtime development guidance, refer to `AGENTS.md`.

**Version**: 1.0.0 | **Ratified**: 2026-01-13 | **Last Amended**: 2026-01-13

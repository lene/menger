# Menger Architecture Documentation (arc42)

This directory contains the architecture documentation for the Menger fractal renderer, following the [arc42 template](https://arc42.org/).

## Document Structure

| Section | Description | Status |
|---------|-------------|--------|
| [1. Introduction and Goals](01-introduction-and-goals.md) | Business context, quality goals, stakeholders | Updated 2026-06-27 |
| [2. Constraints](02-constraints.md) | Technical and organizational constraints | Current |
| [3. Context and Scope](03-context-and-scope.md) | System boundaries and external interfaces | Current |
| [4. Solution Strategy](04-solution-strategy.md) | Fundamental technology decisions | Current |
| [5. Building Block View](05-building-block-view.md) | Static decomposition of the system | Current |
| [6. Runtime View](06-runtime-view.md) | Key runtime scenarios | Current |
| [7. Deployment View](07-deployment-view.md) | Infrastructure and deployment | Current |
| [8. Crosscutting Concepts](08-crosscutting-concepts.md) | Recurring patterns and concepts | Current |
| [9. Architectural Decisions](09-architectural-decisions.md) | Important architecture decisions | Updated 2026-06-27 (AD-29, AD-30 added; AD-24/25 duplicates fixed) |
| [10. Quality Requirements](10-quality-requirements.md) | Quality tree and scenarios | Updated 2026-06-27 (caustics markers, perf governance) |
| [11. Risks and Technical Debt](11-risks-and-technical-debt.md) | Known issues and risks | Updated 2026-06-27 (TR-5 JNI leak gate status) |
| [12. Glossary](12-glossary.md) | Important terms and definitions | Current |

## Quick Links

- **For developers:** Start with [Building Block View](05-building-block-view.md) and [Runtime View](06-runtime-view.md)
- **For operators:** See [Deployment View](07-deployment-view.md)
- **For new team members:** Read sections 1-4 for context
- **For physics/rendering:** See [Crosscutting Concepts](08-crosscutting-concepts.md)

## Conventions

- This documentation is the **single source of truth** for architecture
- Sprint planning documents contain detailed implementation decisions:
  - Active sprints: [docs/sprints/](../sprints/)
  - Completed sprints: [docs/archive/sprints/](../archive/sprints/)
- Troubleshooting guides remain in [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md) (referenced from section 11)

## Version

- **Last Updated:** 2026-06-27 (Sprint 30 coherence pass)
- **arc42 Template Version:** 8.2

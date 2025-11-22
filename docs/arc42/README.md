# Menger Architecture Documentation (arc42)

This directory contains the architecture documentation for the Menger fractal renderer, following the [arc42 template](https://arc42.org/).

## Document Structure

| Section | Description | Status |
|---------|-------------|--------|
| [1. Introduction and Goals](01-introduction-and-goals.md) | Business context, quality goals, stakeholders | Complete |
| [2. Constraints](02-constraints.md) | Technical and organizational constraints | Complete |
| [3. Context and Scope](03-context-and-scope.md) | System boundaries and external interfaces | Complete |
| [4. Solution Strategy](04-solution-strategy.md) | Fundamental technology decisions | Complete |
| [5. Building Block View](05-building-block-view.md) | Static decomposition of the system | Complete |
| [6. Runtime View](06-runtime-view.md) | Key runtime scenarios | Complete |
| [7. Deployment View](07-deployment-view.md) | Infrastructure and deployment | Complete |
| [8. Crosscutting Concepts](08-crosscutting-concepts.md) | Recurring patterns and concepts | Complete |
| [9. Architectural Decisions](09-architectural-decisions.md) | Important architecture decisions | Complete |
| [10. Quality Requirements](10-quality-requirements.md) | Quality tree and scenarios | Complete |
| [11. Risks and Technical Debt](11-risks-and-technical-debt.md) | Known issues and risks | Complete |
| [12. Glossary](12-glossary.md) | Important terms and definitions | Complete |

## Quick Links

- **For developers:** Start with [Building Block View](05-building-block-view.md) and [Runtime View](06-runtime-view.md)
- **For operators:** See [Deployment View](07-deployment-view.md)
- **For new team members:** Read sections 1-4 for context
- **For physics/rendering:** See [Crosscutting Concepts](08-crosscutting-concepts.md)

## Conventions

- This documentation is the **single source of truth** for architecture
- Sprint plans in `optix-jni/` contain detailed implementation decisions
- Troubleshooting guides remain in `docs/TROUBLESHOOTING.md` (referenced from section 11)

## Version

- **Last Updated:** 2025-11-22
- **arc42 Template Version:** 8.2

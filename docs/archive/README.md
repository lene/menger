# Documentation Archive

Completed planning documents preserved for historical reference.

## Structure

```
archive/
├── sprints/              # Completed sprint plans
│   ├── SPRINT_5.md       # Triangle Mesh + Cube
│   ├── SPRINT_6.md       # Full Geometry (IAS)
│   └── SPRINT_7.md       # Materials & Textures
├── refactoring/          # Completed refactoring docs
│   ├── OptiXEngineRefactor.md
│   ├── OPTIX_WRAPPER_REFACTORING.md
│   ├── REFACTORING_TWO_LAYER_ARCHITECTURE.md
│   └── REFACTORING_PLAN_COMMON_MODULE.md
├── investigations/       # Completed investigations
│   ├── window-resize-bug/
│   ├── GLASS_RENDERING_INVESTIGATION.md
│   ├── CODE_QUALITY_ASSESSMENT.md
│   ├── SHADOW_*.md
│   └── ...
├── demo_multiple_lights.cpp
├── demo_transparent_shadows.cpp
└── README.md             # This file
```

## Active Planning Documents

- [ROADMAP.md](../../ROADMAP.md) - Strategic feature planning (Sprints 8-13)
- [docs/sprints/](../sprints/) - Active sprint planning documents
  - [SPRINT.md](../sprints/SPRINT.md) - Current sprint details
  - [SPRINT8.md](../sprints/SPRINT8.md) through [SPRINT13.md](../sprints/SPRINT13.md) - Future sprint plans
- [TODO.md](../../TODO.md) - Quick notes and ideas
- [CODE_REVIEW.md](../../CODE_REVIEW.md) - Code quality findings
- [CODE_IMPROVEMENTS.md](../../CODE_IMPROVEMENTS.md) - Code quality assessments

## Archive Index

### Sprint Plans

| Sprint | Focus | File |
|--------|-------|------|
| 5 | Triangle Mesh + Cube | [sprints/SPRINT_5.md](sprints/SPRINT_5.md) |
| 6 | Full Geometry (IAS) | [sprints/SPRINT_6.md](sprints/SPRINT_6.md) |
| 7 | Materials & Textures | [sprints/SPRINT_7.md](sprints/SPRINT_7.md) |

### Refactoring

| Document | Description |
|----------|-------------|
| [OptiXEngineRefactor.md](refactoring/OptiXEngineRefactor.md) | 22-param constructor → config object |
| [OPTIX_WRAPPER_REFACTORING.md](refactoring/OPTIX_WRAPPER_REFACTORING.md) | OptiX wrapper decomposition |
| [REFACTORING_TWO_LAYER_ARCHITECTURE.md](refactoring/REFACTORING_TWO_LAYER_ARCHITECTURE.md) | Architecture refactoring |

### Investigations

| Document | Description |
|----------|-------------|
| [window-resize-bug/](investigations/window-resize-bug/) | Window resize bug investigation (deferred) |
| [GLASS_RENDERING_INVESTIGATION.md](investigations/GLASS_RENDERING_INVESTIGATION.md) | Glass rendering analysis |
| [SHADOW_*.md](investigations/) | Shadow implementation notes |
| [CODE_QUALITY_ASSESSMENT.md](investigations/CODE_QUALITY_ASSESSMENT.md) | Nov 2025 code review |

## Using Archive Documents

### When to Reference

1. **Historical Context** - Understanding past decisions
2. **Debugging Similar Issues** - Past investigations may help
3. **Feature Details** - How completed features were implemented

### Notes

- Documents preserved as-is, may be outdated
- Check git commit dates for currency
- For current information, consult active documentation first

---

*Last Updated: 2026-01-07*

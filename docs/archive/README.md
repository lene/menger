# Documentation Archive

Completed planning documents preserved for historical reference.

## Structure

```
archive/
├── sprints/              # Completed sprint plans
│   ├── SPRINT_5.md       # Triangle Mesh + Cube
│   ├── SPRINT_6.md       # Full Geometry (IAS)
│   ├── SPRINT_7.md       # Materials & Textures
│   ├── SPRINT_8.md       # 4D Projection + UX Improvements
│   ├── SPRINT_9.md       # TesseractSponge (4D Sponges)
│   ├── SPRINT10.md       # Scala DSL for Scene Description
│   ├── SPRINT11.md       # libGDX Wrapper, Thin-Film Physics, 4D Enhancements
│   ├── SPRINT12.md       # t-Parameter Animation System
│   └── SPRINT13.md       # Visual Quality & Material Enhancements
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
  - [SPRINT14.md](../sprints/SPRINT14.md) through [SPRINT19.md](../sprints/SPRINT19.md) - Future sprint plans
- [TODO.md](../../TODO.md) - Quick notes and ideas
- [CODE_IMPROVEMENTS.md](../../CODE_IMPROVEMENTS.md) - Code quality assessments

## Archive Index

### Sprint Plans

| Sprint | Focus | File |
|--------|-------|------|
| 13 | Visual Quality & Material Enhancements | [sprints/SPRINT13.md](sprints/SPRINT13.md) |
| 12 | t-Parameter Animation System | [sprints/SPRINT12.md](sprints/SPRINT12.md) |
| 11 | libGDX Wrapper, Thin-Film Physics, 4D Enhancements | [sprints/SPRINT11.md](sprints/SPRINT11.md) |
| 10 | Scala DSL for Scene Description | [sprints/SPRINT10.md](sprints/SPRINT10.md) |
| 9 | TesseractSponge (4D Sponges) | [sprints/SPRINT_9.md](sprints/SPRINT_9.md) |
| 8 | 4D Projection + UX Improvements | [sprints/SPRINT_8.md](sprints/SPRINT_8.md) |
| 7 | Materials & Textures | [sprints/SPRINT_7.md](sprints/SPRINT_7.md) |
| 6 | Full Geometry (IAS) | [sprints/SPRINT_6.md](sprints/SPRINT_6.md) |
| 5 | Triangle Mesh + Cube | [sprints/SPRINT_5.md](sprints/SPRINT_5.md) |

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

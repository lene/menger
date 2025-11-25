# Documentation Archive

This directory contains historical documentation, completed feature implementations, and investigation artifacts. These documents are preserved for reference but are not actively maintained.

## Index

### Completed Features (Sprint 1-3)

**Ray Statistics (Sprint 1.1)**
- `RAY_STATISTICS_FEATURE.md` - Implementation of ray statistics tracking and display

**Shadow Rays (Sprint 1.2)**
See `optix-jni/archive/` for shadow ray implementation docs:
- Shadow ray planning and implementation
- SBT fixes
- Transparent shadows
- Test coverage

**Common Module Refactoring**
- `REFACTORING_PLAN_COMMON_MODULE.md` - Common color module refactoring (completed)

### Investigation & Debugging Artifacts

**Window Resize Bug**
- `window-resize-bug/SPECIFICATION.md` - Bug specification
- `window-resize-bug/WINDOW_RESIZE_FIX_PLAN.md` - Proposed solutions
- `window-resize-bug/WINDOW_RESIZE_FIX_ATTEMPTS.md` - Debugging log (13 KB)
- `window-resize-bug/SCREENSHOTS.md` - Visual documentation
- `window-resize-bug/test_window_resize.sh` - Automated test script

**Glass Rendering Investigation**
- `GLASS_RENDERING_INVESTIGATION.md` (28 KB) - Comprehensive glass rendering investigation

**Build & PTX Issues**
- `PTX_LOADING_ISSUE.md` - Stale PTX file loading analysis and solutions
- `ISSUE_20_UPDATE.md` - Specific issue tracking (Oct 2025)

**Image Processing Tools**
- `IMAGEMAGICK_SPHERE_MEASUREMENT.md` - ImageMagick-based measurement tools
- `IMAGE_SIZE_REFACTORING.md` - Test image size optimization

### Code Quality Assessments

**November 2025 Assessments**
- `CODE_QUALITY_ASSESSMENT.md` (33 KB) - Comprehensive codebase review (Nov 19, 2025)
- `CODE_IMPROVEMENTS.md` (11 KB) - Specific improvement suggestions

**Feature Specifications**
- `PLANE_CLI_FEATURE.md` - Plane color customization specification

### Demo Programs

**C++ Standalone Demos** (not in build):
- `demo_multiple_lights.cpp` - Multiple light sources demo
- `demo_transparent_shadows.cpp` - Transparent shadow demo

## Active Documentation

For current, actively maintained documentation, see:
- **Main README**: `/README.md`
- **Architecture**: `/docs/arc42/` (single source of truth)
- **Troubleshooting**: `/docs/TROUBLESHOOTING.md`
- **Sprint Planning**: `/optix-jni/ENHANCEMENT_PLAN.md`
- **Installation**: `/docs/INSTALLATION_FROM_SCRATCH.md`

## Using Archive Documents

### When to Reference Archive

1. **Historical Context** - Understanding why decisions were made
2. **Debugging Similar Issues** - Past investigations may contain relevant insights
3. **Feature Implementation Details** - How completed features were implemented
4. **Code Archaeology** - Understanding evolution of the codebase

### Archive Maintenance

- Documents are preserved as-is, without updates
- Information may be outdated (check git commit dates)
- For current information, always consult active documentation first
- If archive content is critical for current work, consider updating active docs

## Archive Organization

```
docs/archive/
├── README.md (this file)
├── window-resize-bug/           # Consolidated bug investigation
│   ├── SPECIFICATION.md
│   ├── WINDOW_RESIZE_FIX_PLAN.md
│   ├── WINDOW_RESIZE_FIX_ATTEMPTS.md
│   ├── SCREENSHOTS.md
│   └── test_window_resize.sh
├── RAY_STATISTICS_FEATURE.md
├── REFACTORING_PLAN_COMMON_MODULE.md
├── GLASS_RENDERING_INVESTIGATION.md
├── PTX_LOADING_ISSUE.md
├── ISSUE_20_UPDATE.md
├── IMAGEMAGICK_SPHERE_MEASUREMENT.md
├── IMAGE_SIZE_REFACTORING.md
├── CODE_QUALITY_ASSESSMENT.md
├── CODE_IMPROVEMENTS.md
├── PLANE_CLI_FEATURE.md
├── demo_multiple_lights.cpp
└── demo_transparent_shadows.cpp
```

---

*Last Updated: 2025-11-25*

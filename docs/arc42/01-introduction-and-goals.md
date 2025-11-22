# 1. Introduction and Goals

## 1.1 Requirements Overview

**Menger** is a Scala 3 fractal renderer that generates and visualizes Menger sponges (3D) and tesseract sponges (4D) using surface subdivision algorithms.

### Core Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Render Menger sponges at configurable recursion levels (0-4+) | High |
| R2 | Support fractional levels with alpha blending | High |
| R3 | Interactive 3D/4D rotation and camera control | High |
| R4 | GPU-accelerated ray tracing via NVIDIA OptiX | High |
| R5 | Generate animation frame sequences | Medium |
| R6 | Support 4D tesseract sponge with 4D→3D projection | Medium |
| R7 | Configurable materials (glass, metal, matte) | Medium |

### Rendering Modes

1. **LibGDX Mode** - OpenGL rasterization for real-time preview
2. **OptiX Mode** - GPU ray tracing for high-quality output with refraction, shadows, caustics

## 1.2 Quality Goals

| Priority | Quality Goal | Description |
|----------|--------------|-------------|
| 1 | **Performance** | Efficient surface subdivision (O(12^n) vs O(20^n) volume) |
| 2 | **Visual Quality** | Physically-based rendering (Fresnel, Beer-Lambert, Snell's law) |
| 3 | **Extensibility** | Easy addition of new geometry types and materials |
| 4 | **Testability** | Comprehensive test coverage (~897 tests) |
| 5 | **Maintainability** | Functional Scala style, no mutable state, no exceptions |

## 1.3 Stakeholders

| Role | Expectations |
|------|--------------|
| **Developer** | Clear architecture, testable code, documented APIs |
| **User** | Easy CLI usage, high-quality renders, interactive exploration |
| **Researcher** | Accurate 4D projections, customizable parameters |

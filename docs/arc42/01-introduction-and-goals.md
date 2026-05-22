# 1. Introduction and Goals

## 1.1 Requirements Overview

**Menger** is a Scala 3 project that provides OptiX ray tracing capabilities, a rendering engine, and 3D/4D visualization tools. The current showcase application generates and visualizes Menger sponges (3D) and higher-dimensional fractal analogs (4D Menger, Sierpinski, hexadecachoron) using surface subdivision algorithms.

### Core Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Render Menger sponges at configurable recursion levels (0-4+) | High |
| R2 | Support fractional levels with alpha blending | High |
| R3 | Interactive 3D/4D rotation and camera control | High |
| R4 | GPU-accelerated ray tracing via NVIDIA OptiX | High |
| R5 | Generate animation frame sequences via `--animate` CLI flag | Medium |
| R6 | Support 4D fractal analogs (tesseract sponge, 4D Sierpinski, hexadecachoron) with 4D→3D projection | Medium |
| R7 | Configurable materials (glass, metal, matte) with image textures and PBR maps | Medium |
| R8 | Fog / depth cue (exponential distance attenuation) | Low |

### Rendering Modes

1. **LibGDX Mode** - OpenGL rasterization for real-time preview
2. **OptiX Mode** - GPU ray tracing for high-quality output with refraction, shadows, caustics, fog

## 1.2 Quality Goals

| Priority | Quality Goal | Description |
|----------|--------------|-------------|
| 1 | **Performance** | Efficient surface subdivision (O(12^n) vs O(20^n) volume) |
| 2 | **Visual Quality** | Physically-based rendering (Fresnel, Beer-Lambert, Snell's law) |
| 3 | **Extensibility** | Easy addition of new geometry types and materials |
| 4 | **Testability** | Comprehensive test coverage (2,823 tests across all modules) |
| 5 | **Maintainability** | Functional Scala style, no mutable state, no exceptions |

## 1.3 Stakeholders

| Role | Expectations |
|------|--------------|
| **Developer** | Clear architecture, testable code, documented APIs |
| **User** | Easy CLI usage, high-quality renders, interactive exploration |
| **Researcher** | Accurate 4D projections, customizable parameters |

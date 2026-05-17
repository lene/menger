# 12. Glossary

## Rendering Terms

| Term | Definition |
|------|------------|
| **Beer-Lambert Law** | Physical law describing light absorption in a medium: I = I₀ exp(-αd) |
| **BVH** | Bounding Volume Hierarchy - spatial acceleration structure for ray tracing |
| **Caustics** | Bright patterns formed by light focused through transparent objects |
| **Fresnel Equations** | Formulas for light reflection/transmission at material boundaries |
| **GAS** | Geometry Acceleration Structure - OptiX structure for geometry traversal |
| **IAS** | Instance Acceleration Structure - OptiX structure for multiple object instances |
| **IOR** | Index of Refraction - ratio of light speed in vacuum vs medium |
| **Normal Map** | Texture encoding per-texel surface normals to simulate fine surface detail without extra geometry |
| **PBR** | Physically Based Rendering - materials based on physical properties |
| **Procedural Texture** | GPU-computed texture pattern (e.g. checker, marble, wood) requiring no image file |
| **Roughness Map** | Texture encoding per-texel surface roughness for PBR materials |
| **PTX** | Parallel Thread Execution - NVIDIA intermediate shader format |
| **Ray Tracing** | Rendering by simulating light ray paths |
| **SBT** | Shader Binding Table - OptiX structure linking geometry to shaders |
| **Schlick Approximation** | Fast approximation of Fresnel equations |
| **Snell's Law** | Relationship between angles of incidence and refraction |
| **Total Internal Reflection** | Complete reflection when angle exceeds critical angle |

## Geometry Terms

| Term | Definition |
|------|------------|
| **Face** | Planar polygon surface (quad or triangle) |
| **Menger Sponge** | Fractal 3D shape with self-similar holes |
| **Surface Subdivision** | Recursive face splitting (12 faces per face) |
| **Tesseract** | 4D hypercube (8 cubic cells) |
| **TesseractSponge** | 4D analog of Menger sponge |
| **Volume Subdivision** | Recursive cube splitting (20 cubes per cube) |

## OptiX/CUDA Terms

| Term | Definition |
|------|------------|
| **Closest Hit** | Shader executed when ray finds nearest intersection |
| **Miss** | Shader executed when ray hits nothing |
| **optixTrace** | OptiX function to trace a ray through scene |
| **Params** | Launch parameters passed to GPU shaders |
| **Ray Generation** | Shader that initiates rays from camera |
| **Traversable** | OptiX handle to acceleration structure |

## Project-Specific Terms

| Term | Definition |
|------|------------|
| **FractionalLevel** | Non-integer sponge level using alpha blending |
| **LibGDX** | Cross-platform Java game framework (OpenGL); used for windowing and input only (AD-16) |
| **ObjectSpec** | Parsed representation of a `--object` CLI flag; contains typed value objects for rotation, geometry, textures, procedural config |
| **OptiX JNI** | Subproject containing native OptiX code |
| **ProceduralSpec** | Value object holding `proceduralType` (int) and `scale` (float) for procedural textures |
| **RotatedProjection** | Wrapper applying 4D rotation + projection |
| **SceneBuilder** | Strategy trait for wiring a specific geometry type into the OptiX scene; provides `applyInstanceTextures` helper |
| **SpongeBySurface** | Surface-subdivision sponge generator |
| **SpongeByVolume** | Volume-subdivision sponge generator |
| **TextureMaps** | Value object holding optional `normalMap` and `roughnessMap` file paths |

## Abbreviations

| Abbreviation | Meaning |
|--------------|---------|
| AA | Antialiasing |
| CLI | Command Line Interface |
| GPU | Graphics Processing Unit |
| JNI | Java Native Interface |
| JVM | Java Virtual Machine |
| RGB(A) | Red, Green, Blue, (Alpha) |
| SDK | Software Development Kit |
| UV | Texture coordinates (traditionally U and V axes) |

## Material Constants

| Material | IOR | Notes |
|----------|-----|-------|
| Vacuum/Air | 1.0 | Reference |
| Water | 1.33 | |
| Glass | 1.5 | Standard |
| Crystal | 2.0 | |
| Diamond | 2.42 | High dispersion |

## Alpha Convention

| Value | Meaning |
|-------|---------|
| α = 0.0 | Fully transparent (no absorption) |
| α = 1.0 | Fully opaque (maximum absorption) |

**Note:** This follows standard graphics convention. Never invert this.

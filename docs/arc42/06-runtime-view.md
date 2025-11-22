# 6. Runtime View

## 6.1 OptiX Rendering Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OptiX Ray Tracing Pipeline                        │
└─────────────────────────────────────────────────────────────────────┘

  User Request                                              Output
      │                                                        ▲
      ▼                                                        │
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐  │
│  render() │──►│ Build GAS │──►│ Setup SBT │──►│ Build     │  │
│  called   │   │ if dirty  │   │ if dirty  │   │ Params    │  │
└───────────┘   └───────────┘   └───────────┘   └─────┬─────┘  │
                                                      │        │
                                                      ▼        │
                                              ┌───────────────┐│
                                              │ optixLaunch() ││
                                              │ (GPU kernel)  ││
                                              └───────┬───────┘│
                                                      │        │
                      ┌───────────────────────────────┼────────┘
                      ▼                               │
            ┌─────────────────┐               ┌───────▼───────┐
            │  Ray Generation │               │  Copy Result  │
            │  (per pixel)    │               │  to Host      │
            └────────┬────────┘               └───────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Hit Sphere  │ │  Hit Mesh   │ │    Miss      │
│  (custom IS) │ │  (built-in) │ │  (plane/bg)  │
└──────┬───────┘ └──────┬───────┘ └──────────────┘
       │                │
       ▼                ▼
┌──────────────────────────────────┐
│  Closest Hit Shader              │
│  - Fresnel reflection/refraction │
│  - Beer-Lambert absorption       │
│  - Recursive ray tracing         │
│  - Shadow rays                   │
└──────────────────────────────────┘
```

## 6.2 Ray Tracing Flow (Single Ray)

```
1. Camera ray generation
   └─► Calculate ray origin + direction from pixel coordinates

2. Trace primary ray
   └─► optixTrace(handle, origin, direction, ...)

3. Intersection test
   ├─► Sphere: Custom __intersection__sphere (analytical)
   └─► Mesh: Built-in triangle intersection

4. Closest hit evaluation
   ├─► Opaque (alpha ≥ 0.996)
   │   └─► Diffuse shading + shadow test
   │
   ├─► Transparent (alpha ≤ 0.004)
   │   └─► Continue ray through object
   │
   └─► Semi-transparent (glass)
       ├─► Calculate Fresnel reflectance
       ├─► Trace reflection ray (recursive)
       ├─► Trace refraction ray (Snell's law)
       ├─► Apply Beer-Lambert absorption
       └─► Blend reflection + refraction

5. Miss (no hit)
   ├─► Check plane intersection
   │   ├─► Solid color or checkerboard
   │   └─► Shadow test
   └─► Return background color

6. Accumulate into pixel buffer
```

## 6.3 Scene Configuration Flow

```
CLI Arguments                 Scala Layer                 C++ Layer
     │                             │                           │
     ▼                             ▼                           ▼
┌─────────────┐           ┌─────────────────┐         ┌─────────────────┐
│ --optix     │──────────►│ OptiXResources  │────────►│ OptiXWrapper    │
│ --ior 1.5   │           │ .initialize()   │         │ .initialize()   │
│ --shadows   │           └────────┬────────┘         └────────┬────────┘
└─────────────┘                    │                           │
                                   ▼                           ▼
                          ┌─────────────────┐         ┌─────────────────┐
                          │ configureScene  │────────►│ setSphere()     │
                          │ (geometry,      │         │ setCamera()     │
                          │  camera, lights)│         │ setLights()     │
                          └────────┬────────┘         └────────┬────────┘
                                   │                           │
                                   ▼                           ▼
                          ┌─────────────────┐         ┌─────────────────┐
                          │ OptiXRenderer   │────────►│ render()        │
                          │ .render()       │  JNI    │ → optixLaunch   │
                          └─────────────────┘         └─────────────────┘
```

## 6.4 Geometry Generation (SpongeBySurface)

```
1. Start with 6 initial faces (cube sides)
   └─► Face(center, scale, normal)

2. For each level (0 to N):
   └─► For each face:
       ├─► Subdivide into 9 squares
       ├─► Remove center square (hole)
       ├─► Add 4 rotated faces (inner walls)
       └─► Result: 12 faces per input face

3. Face count: 6 × 12^level
   └─► Level 0: 6 faces
   └─► Level 1: 72 faces
   └─► Level 2: 864 faces
   └─► Level 3: 10,368 faces

4. Convert to triangles (2 per face)
   └─► Export vertex + index buffers
```

## 6.5 4D Rendering Flow

```
┌─────────────────┐
│ TesseractSponge │
│ (4D geometry)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ RotatedProjection   │
│ - Apply 4D rotation │
│   (XW, YW, ZW)      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 4D→3D Projection    │
│ - Perspective or    │
│   orthographic      │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3D ModelInstance    │
│ (LibGDX rendering)  │
└─────────────────────┘
```

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
│  - Fog (exponential attenuation) │
└──────────────────────────────────┘
```

## 6.2 Ray Tracing Flow (Single Ray)

```
1. Camera ray generation
   └─► Calculate ray origin + direction from pixel coordinates

2. Trace primary ray
   └─► optixTrace(handle, origin, direction, ...)

3. Intersection test
   ├─► Sphere/Cone/Cylinder: Custom IS programs (analytical)
   ├─► Menger4D/Sierpinski4D/Hexadecachoron4D: Custom IS + hit (GPU fractal eval)
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

5. Fog attenuation (if fog_density > 0)
   └─► `applyFogInPlace()`: exp(-density * t), blend toward fog color

6. Miss (no hit)
   ├─► Check plane intersection
   │   ├─► Solid color or checkerboard
   │   └─► Shadow test
   └─► Return background color

7. Accumulate into pixel buffer
```

## 6.3 Scene Configuration Flow

```
CLI Arguments                 Scala Layer                 C++ Layer
     │                             │                           │
     ▼                             ▼                           ▼
┌─────────────┐           ┌─────────────────┐         ┌─────────────────┐
│ --optix     │──────────►│ OptiXRender-    │────────►│ OptiXWrapper    │
│ --objects   │           │ Resources       │         │ .initialize()   │
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

Two paths depending on geometry type:

### CPU 4D Path (Tesseract, TesseractSponge, Polychora)

```
┌─────────────────┐
│ TesseractSponge │
│ (4D geometry)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Mesh4DProjection    │
│ - Apply 4D rotation │
│   (XW, YW, ZW)      │
│ - 4D→3D perspective │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Triangle mesh       │
│ uploaded to GPU     │
│ via OptiX BVH (GAS) │
└─────────────────────┘
```

### GPU 4D Path (Menger4D, Sierpinski4D, Hexadecachoron4D — Sprint 18+)

```
┌──────────────────────┐
│ 4D Fractal DSL node  │
│ (Menger4D, etc.)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Custom IS program    │
│ hit_menger4d.cu /    │
│ hit_sierpinski4d.cu  │
│ Evaluates fractal    │
│ SDF on GPU per ray   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Closest hit shader   │
│ (material, shading)  │
└──────────────────────┘
```

## 6.6 Animation Flow (`--animate`)

```
CLI --animate <spec>
       │
       ▼
┌─────────────────────────┐
│ CliAnimationEngine      │
│ - Parse frame spec      │
│   (start:end:step)      │
└──────────┬──────────────┘
           │
           ▼ (for each frame)
┌─────────────────────────┐
│ Update rotation/level   │
│ parameters              │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Rebuild geometry if     │
│ level changed           │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ render() → PNG output   │
│ frame_NNNN.png          │
└─────────────────────────┘
```

DSL animation uses `scene(t: Double)` — each frame calls `scene(t)`,
rebuilds the scene graph, and renders.

## 6.7 Stats JSON Export Flow (`--stats-json`)

```
render() [each frame]
       │
       ▼
renderWithStats(w, h)
       │
       ├─► rendererWrapper.renderSceneWithStats()
       │         │
       │         ▼ JNI
       │   optixLaunch() + RenderResult
       │   (frameMs, totalRays, primaryRays,
       │    reflectedRays, refractedRays,
       │    shadowRays, aaRays, msPerMray)
       │
       └─► lastRenderResult.set(Some(result))
               [AtomicReference — GL thread write]

dispose() [on exit]
       │
       ▼
execution.statsJsonPath.foreach(writeStatsJson)
       │
       ├─► lastRenderResult.get() → None?
       │       └─► logger.warn (no render completed)
       │
       └─► lastRenderResult.get() → Some(result)
               │
               ├─► extract scalar fields to locals
               ├─► Paths.get(path).toAbsolutePath
               ├─► Files.createDirectories(parent)
               ├─► Files.writeString(p, json)
               └─► Try / logger.error on IOException
```

Used by `scripts/benchmark.sh` to collect `frameMs` across 3 runs per scene
and compare against `scripts/perf-baseline.json` (15% regression threshold).

# Menger — Tutorials

**Version**: 0.9.0
**Last Updated**: September 2026

All tutorials open an interactive window. To render to a file without a window instead, add
`--headless --save-name <file>.png` (and prefix `xvfb-run -a` on a machine without a display).

← [DSL Reference](dsl-reference.md) | [User Guide Index](../USER_GUIDE.md)

---

## Examples and Tutorials

### Tutorial 1: Your First Render

Let's create a simple glass sphere with nice lighting.

**Step 1: Basic sphere**
```bash
sbt "run --objects 'type=sphere'"
```

**Step 2: Make it glass**
```bash
sbt "run --objects 'type=sphere:ior=1.5'"
```

**Step 3: Make it bigger**
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5'"
```

**Step 4: Add a floor and shadows** (shadows need a surface to fall on)
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' --plane y:-2 --shadows"
```

**Step 5: Improve lighting**
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' --plane y:-2 --shadows \
    --light directional:-1,1,-1:1.5 \
    --light point:2,3,2:0.8:ffffff"
```

**Step 6: Add antialiasing**
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' --plane y:-2 --shadows \
    --antialiasing \
    --light directional:-1,1,-1:1.5 \
    --light point:2,3,2:0.8:ffffff"
```

**Step 7: Save the image** (no window; the image is written once the render finishes)
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' --plane y:-2 --shadows \
    --antialiasing --headless --save-name my_first_sphere.png \
    --light directional:-1,1,-1:1.5 \
    --light point:2,3,2:0.8:ffffff"
```

Congratulations! You've created your first high-quality ray-traced image.

### Tutorial 2: Creating Glass Objects

Glass requires careful attention to IOR, lighting, and scene setup.

**Good Glass Setup:**
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' \
    --shadows \
    --antialiasing \
    --plane y:-2 --plane-color ffffff:d0d0d0 \
    --light directional:-1,1,-1:2.0 \
    --light directional:1,0.5,1:0.5:8080ff"
```

**Glass Menger Sponge:**
```bash
sbt "run --objects 'type=sponge-surface:level=2:ior=1.5' \
    --shadows \
    --antialiasing \
    --camera-pos -2,1.5,-2 \
    --plane y:-2 --plane-color ffffff:808080"
```

**Glass with Caustics** (caustics are the bright light patterns on the floor):
```bash
sbt "run --objects 'type=sphere:ior=1.5:size=1.5' --plane y:-2 \
    --caustics --caustics-photons 200000 --caustics-iterations 20 \
    --shadows --antialiasing \
    --light directional:-1,2,-1:2.0"
```

**Tips for Glass:**
- Use IOR between 1.3-1.6 for realistic glass
- Always enable shadows to see the transparency
- Antialiasing is crucial for clean edges
- A checkered ground plane helps visualize refraction
- Caustics add realism but are computationally expensive

### Tutorial 3: Animating a Rotating Sponge

Create a smooth 360° rotation animation.

**Step 1: Test a single frame**
```bash
sbt "run --objects 'type=sponge-surface:level=2'"
```

**Step 2: Add good lighting, a floor and a camera position**
```bash
sbt "run --objects 'type=sponge-surface:level=2' \
    --camera-pos -2,1.5,-2 \
    --plane y:-2 --shadows \
    --light directional:-1,1,-1"
```

**Step 3: Create animation (36 frames = 10° per frame)**
```bash
sbt "run --objects 'type=sponge-surface:level=2' \
    --camera-pos -2,1.5,-2 \
    --plane y:-2 --shadows \
    --light directional:-1,1,-1 \
    --save-name sponge_rot_%03d.png \
    --animate frames=36:rot-y=0-360"
```

(`--animate` renders every frame and then exits; don't combine it with `--timeout`.)

**Step 4: Add quality improvements**
```bash
sbt "run --objects 'type=sponge-surface:level=2' \
    --camera-pos -2,1.5,-2 \
    --shadows --antialiasing \
    --light directional:-1,1,-1:1.5 \
    --plane y:-2 --plane-color ffffff:808080 \
    --save-name sponge_hq_%03d.png \
    --animate frames=36:rot-y=0-360"
```

**Step 5: Create video**
```bash
ffmpeg -framerate 30 -i sponge_hq_%03d.png \
    -c:v libx264 -pix_fmt yuv420p sponge_rotation.mp4
```

**Pro tip:** For longer animations, render without a window (on a machine with no display,
prefix `xvfb-run -a`):
```bash
export __GL_THREADED_OPTIMIZATIONS=0
xvfb-run -a sbt "run --objects 'type=sponge-surface:level=2' --headless \
    --save-name sponge_%03d.png --animate frames=36:rot-y=0-360"
```

For animations beyond rotation and level sweeps (moving objects, changing materials, camera
paths) write an animated DSL scene and use `--frames` / `--video` — see
[Advanced Features](advanced.md#animated-scenes-t-parameter).

### Tutorial 4: 4D Visualization

Explore the fourth dimension with tesseract sponges.

**Step 1: Basic tesseract**
```bash
sbt "run --objects 'type=tesseract'"
```

**Step 2: Rotate in 4D space** (inside `--objects`, 4D planes are written `rot-xw`, `rot-yw`, `rot-zw`)
```bash
sbt "run --objects 'type=tesseract:rot-xw=30:rot-yw=45'"
```

You can also rotate interactively: hold **Shift** and use the arrow keys (XW/YW) or
Page Up/Down (ZW), or Shift+drag with the mouse.

**Step 3: Tesseract sponge level 1** (4D sponges require `level=`)
```bash
sbt "run --objects 'type=tesseract-sponge-volume:level=1'"
```

**Step 4: Animate 4D rotation** (inside `--animate`, planes are written `rot-x-w`)
```bash
sbt "run --objects 'type=tesseract-sponge-volume:level=1' \
    --save-name tesseract_%03d.png \
    --animate frames=36:rot-x-w=0-360"
```

**Step 5: Complex 4D motion** (two chained animation segments)
```bash
sbt "run --objects 'type=tesseract-sponge-volume:level=1' \
    --save-name 4d_complex_%03d.png \
    --animate frames=20:rot-x-w=0-90:rot-y-w=0-45 \
    --animate frames=20:rot-z-w=0-90"
```

**Adjust projection:**
```bash
sbt "run --objects 'type=tesseract:screen-w=2.0:eye-w=4.0'"
```

**Understanding 4D projection:**
- `screen-w`: Distance to the projection screen in the W dimension (analogous to Z in 3D→2D)
- `eye-w`: Eye position in the W dimension
- Larger `eye-w` = more "zoomed out" in 4D space (flatter projection)
- In the interactive window, **Shift + scroll** changes `eye-w` live
- Experiment with values to get intuition for 4D perspective

### Tutorial 5: Complex Scenes

Build a scene with multiple objects and sophisticated lighting.

**Material Showcase:**
```bash
sbt "run \
    --objects 'type=sphere:pos=-3,0,0:material=glass:size=0.8' \
    --objects 'type=sphere:pos=-1.5,0,0:material=water:size=0.8' \
    --objects 'type=sphere:pos=0,0,0:material=diamond:size=0.8' \
    --objects 'type=sphere:pos=1.5,0,0:material=gold:size=0.8' \
    --objects 'type=sphere:pos=3,0,0:material=chrome:size=0.8' \
    --shadows --antialiasing \
    --light directional:-1,1,-1:2.0 \
    --plane y:-1 --plane-color ffffff:c0c0c0 \
    --camera-pos 0,2,6"
```

**Sponge Gallery:**
```bash
sbt "run \
    --objects 'type=sponge-surface:level=1:pos=-2,0,0:material=glass' \
    --objects 'type=sponge-surface:level=1:pos=0,0,0:material=gold' \
    --objects 'type=sponge-surface:level=1:pos=2,0,0:material=copper' \
    --shadows --antialiasing \
    --light directional:-1,1,-1:1.5 \
    --light point:0,3,0:1.0 \
    --camera-pos 0,2,5"
```

**Mixed Geometry Scene:**
```bash
sbt "run \
    --objects 'type=sponge-surface:level=2:pos=0,0.5,0:material=diamond' \
    --objects 'type=sphere:pos=-3,0,0:size=0.5:color=#FF0000' \
    --objects 'type=sphere:pos=3,0,0:size=0.5:color=#0000FF' \
    --objects 'type=cube:pos=-1.5,0,-2:size=0.4:material=gold' \
    --objects 'type=cube:pos=1.5,0,-2:size=0.4:material=chrome' \
    --shadows --antialiasing \
    --light directional:-1,1,-1:2.0 \
    --light point:0,5,0:1.5:ffffd0 \
    --plane y:-1 --plane-color ffffff:808080"
```

---

← [DSL Reference](dsl-reference.md) | [User Guide Index](../USER_GUIDE.md) | → [Reference](reference.md)

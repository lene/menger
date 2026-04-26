# Menger — Reference

**Version**: 0.5.5
**Last Updated**: March 2026

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

---

## Reference

### Complete Option List

#### General Options
```
--timeout <seconds>          Auto-exit after specified time
--width <pixels>             Window width (default: 800)
--height <pixels>            Window height (default: 600)
--save-name <pattern>        Save frame pattern (e.g., "frame%03d.png")
```

#### LibGDX Geometry Options
```
--sponge-type <type>         Geometry type:
                             - square (2D square)
                             - cube (3D cube)
                             - square-sponge (Menger by surface)
                             - cube-sponge (Menger by volume)
                             - tesseract (4D hypercube)
                             - tesseract-sponge-volume (4D sponge, 48 tesseracts)
                             - tesseract-sponge-surface (4D sponge, 16 faces)
                             - composite[type1,type2,...] (overlay)
--level <float>              Recursion level (supports fractional)
--lines                      Wireframe mode
--color <rrggbb[aa]>         Hex color
--face-color <rrggbb[aa]>    Face color (overlay mode)
--line-color <rrggbb[aa]>    Line color (overlay mode)
--antialias-samples <int>    MSAA samples
```

#### 4D Projection Options
```
--projection-screen-w <float>    Projection screen distance in W
--projection-eye-w <float>       Eye position in W dimension
--rot-x-w <float>                4D rotation around XW plane
--rot-y-w <float>                4D rotation around YW plane
--rot-z-w <float>                4D rotation around ZW plane
```

#### OptiX Core Options
```
--optix                      Enable OptiX ray tracing
--object <type>              DEPRECATED: Use --objects instead
--radius <float>             DEPRECATED: Use --objects 'type=...:size=...' instead
--scale <float>              DEPRECATED: Use --objects 'type=...:size=...' instead
--center <x,y,z>             DEPRECATED: Use --objects 'type=...:pos=x,y,z' instead
--ior <float>                DEPRECATED: Use --objects 'type=...:ior=...' instead
--color <rrggbb[aa]>         DEPRECATED: Use --objects 'type=...:color=...' instead
```

#### OptiX Camera Options
```
--camera-pos <x,y,z>         Camera position (default: 0,0.5,3)
--camera-lookat <x,y,z>      Look-at target (default: 0,0,0)
--camera-up <x,y,z>          Up vector (default: 0,1,0)
```

#### OptiX Lighting Options
```
--light <spec>               Add light source (repeatable, max 8)
                             Types:
                               directional:x,y,z[:intensity[:color]]
                               point:x,y,z[:intensity[:color]]
                               area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color]]]
--shadows                    Enable shadow rays
```

#### OptiX Scene Options
```
--plane <spec>               Ground plane (default: +y:-2)
                             Format: [+-]?[xyz]:<value>
--plane-color <spec>         Plane color
                             Solid: #RRGGBB
                             Checkered: RRGGBB:RRGGBB
```

#### OptiX Quality Options
```
--antialiasing               Enable recursive adaptive AA
--aa-max-depth <int>         AA recursion depth (1-4, default: 2)
--aa-threshold <float>       AA threshold (0.0-1.0, default: 0.1)
--stats                      Display ray tracing statistics
```

#### OptiX Caustics Options
```
--caustics                   Enable caustics (PPM)
--caustics-photons <int>     Photons per iteration (default: 100000)
--caustics-iterations <int>  PPM iterations (default: 10)
--caustics-radius <float>    Initial gather radius (default: 0.1)
--caustics-alpha <float>     Radius reduction (0.0-1.0, default: 0.7)
```

#### OptiX Multi-Object Options
```
--objects "<spec>"           Add object (repeatable)
                             Format: param=value:param2=value2:...
                             Parameters:
                             - type: sphere|cube|sponge-volume|sponge-surface|sponge-recursive-ias
                             - pos: x,y,z
                             - size: float
                             - level: float (for sponges)
                             - color: #RRGGBB[AA]
                             - ior: float
                             - material: preset name
                             - texture: file path
                             - roughness: 0.0-1.0
                             - metallic: 0.0-1.0
                             - specular: 0.0-1.0
```

#### Animation Options
```
--animate <spec>             Animation specification (repeatable)
                             Format: frames=N:param=start-end[:param2=...]
                             Parameters: rot-x, rot-y, rot-z,
                             rot-x-w, rot-y-w, rot-z-w,
                             projection-screen-w, projection-eye-w, level
```

#### Scene Animation (t-Parameter) Options
```
--t <float>                  Evaluate animated scene at fixed t (freeze-frame)
--start-t <float>            Start of t range (default: 0)
--end-t <float>              End of t range (default: 1)
--frames <int>               Number of frames (requires --save-name with %)
```

### Keyboard Shortcuts

**LibGDX Mode:**

| Key | Action |
|-----|--------|
| ESC | Exit application |
| Space | Pause/resume animation |
| R | Reset camera to default position |
| S | Take screenshot |
| W | Toggle wireframe mode |
| F | Toggle fullscreen |

**LibGDX Mouse Controls:**

| Action | Effect |
|--------|--------|
| Left Click + Drag | Rotate around X/Y axes |
| Right Click + Drag | Rotate around Z axis |
| Scroll Wheel | Zoom in/out |

**OptiX Mode:**

| Key | Action |
|-----|--------|
| ESC | Reset 4D view to initial state |
| Ctrl + Q | Exit application |
| Shift + Left/Right arrows | Rotate 4D object in XW plane |
| Shift + Up/Down arrows | Rotate 4D object in YW plane |
| Shift + Page Up/Down | Rotate 4D object in ZW plane |

**OptiX Mouse Controls:**

| Action | Effect |
|--------|--------|
| Left Click + Drag | Rotate camera |
| Right Click + Drag | Pan camera |
| Scroll Wheel | Zoom in/out |
| Shift + Left Drag (horizontal) | Rotate 4D object in XW plane |
| Shift + Left Drag (vertical) | Rotate 4D object in YW plane |
| Shift + Right Drag (vertical) | Rotate 4D object in ZW plane |
| Shift + Scroll Wheel | Adjust 4D projection eye distance (eyeW) |

### File Formats

#### Supported Input Formats

**Textures:**
- PNG (`.png`) - Recommended for lossless quality
- JPEG (`.jpg`, `.jpeg`) - Supported but may show compression artifacts

#### Supported Output Formats

**Screenshots/Frames:**
- PNG (`.png`) - Default and recommended
- Specified via `--save-name` pattern (e.g., `frame%03d.png`)

**Creating Videos:**
Use ffmpeg to convert frame sequences:
```bash
# MP4 (H.264, widely compatible)
ffmpeg -framerate 30 -i frame%03d.png -c:v libx264 -pix_fmt yuv420p output.mp4

# WebM (VP9, web-friendly)
ffmpeg -framerate 30 -i frame%03d.png -c:v libvpx-vp9 -pix_fmt yuva420p output.webm

# GIF (for web, lower quality)
ffmpeg -framerate 15 -i frame%03d.png -vf "scale=640:-1" output.gif
```

---

## Appendix: Quick Reference Card

```bash
# QUICK START
sbt run                                    # Interactive mode
sbt "run --optix --objects 'type=sphere'"          # Ray-traced sphere

# COMMON OPERATIONS
sbt compile                                # Build project
sbt test                                   # Run tests
sbt clean                                  # Clean build artifacts

# BASIC RENDERS
sbt "run --level 2"                        # Level 2 sponge (LibGDX)
sbt "run --optix --objects 'type=sphere:ior=1.5'"    # Glass sphere

# QUALITY IMPROVEMENTS
--shadows                                  # Add shadows
--antialiasing                             # Smooth edges
--plane-color ffffff:808080                # Checkered floor

# ANIMATION
--save-name frame%03d.png                  # Save frames
--animate frames=36:rot-y=0-360            # 360° rotation
--timeout 1.0                              # Auto-exit per frame

# MATERIALS
--ior 1.5                                  # Glass
material=gold                              # Gold preset
material=glass:roughness=0.3               # Custom glass

# LIGHTING
--light directional:-1,1,-1                # Sun-like light
--light point:0,5,0:2.0                    # Bright overhead light
--light point:2,3,2::ffd700                # Gold-colored light

# TROUBLESHOOTING
rm -rf optix-jni/target/native && sbt compile    # Clean rebuild
export __GL_THREADED_OPTIMIZATIONS=0              # Fix SIGBUS crash
```

---

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

---

**Thank you for using Menger! Happy rendering!**

For questions, issues, or contributions:
- GitLab: https://gitlab.com/lilacashes/menger
- Issues: https://gitlab.com/lilacashes/menger/issues
- Documentation: https://gitlab.com/lilacashes/menger/-/tree/main/docs

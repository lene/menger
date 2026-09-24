# Menger — Reference

**Version**: 0.9.0
**Last Updated**: September 2026

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

---

## Reference

This page lists every command-line option and every `--objects` key. It mirrors
`menger-app --help`, which is always authoritative for the version you have installed.

Every run needs **something to render**: `--objects` or `--scene`. Without either, Menger exits
with `Error: SceneConfig must provide objectSpecs`.

### Complete Option List

#### General
```
-t, --timeout <seconds>      Close the interactive window after N seconds (0 = stay open)
-w, --width <pixels>         Window/image width (default: 800)
-h, --height <pixels>        Window/image height (default: 600)
-s, --save-name <file>       Save the rendered image; use a % pattern (frame%03d.png) for sequences
    --headless               Render without a window (requires --save-name)
-l, --log-level <level>      ERROR, WARN, INFO, DEBUG, TRACE
-p, --profile-min-ms <ms>    Log frames that take longer than N ms
-f, --fps-log-interval <ms>  FPS logging interval
    --stats / --nostats      Print ray tracing statistics (ms/frame, ms/Mray, ray counts)
    --stats-json <file>      Write last-frame statistics as JSON (implies --stats)
    --display <:N>           Render on this X11 display (re-executes with DISPLAY set)
    --render-lock-path <p>   Lock file enforcing one interactive session at a time
-c, --cross                  Show XYZ coordinate cross (toggle live with C)
    --cross-length <f>       Half-length of each axis arm (default: 2.0)
    --cross-thickness <f>    Axis cylinder radius (default: 0.03)
    --cross-material <name>  Material preset for the cross (default: chrome)
    --help / --version
```

> `-t` is `--timeout`; the *long* option `--t` is the animation freeze-frame (see below).

#### Objects and Scenes
```
-o, --objects <spec>         Object to render (repeatable): type=TYPE[:key=value...]
                             See "Object Keys" below
    --scene <name>           DSL scene: short name (glass-sphere), class name
                             (examples.dsl.GlassSphere) or path to a .scala file.
                             Mutually exclusive with --objects
-m, --max-instances <n>      Instance budget, 1-65536 (default 64; raised automatically for 4D edges)
```

#### Global Rotation (applied on top of each object's own rotation)
```
-r, --rot-x <deg>            Rotation around X
    --rot-y <deg>            Rotation around Y
    --rot-z <deg>            Rotation around Z
    --rot-x-w <deg>          4D rotation in the XW plane (4D objects only)
    --rot-y-w <deg>          4D rotation in the YW plane
    --rot-z-w <deg>          4D rotation in the ZW plane
    --rotation-4d XW,YW,ZW   Shorthand for the three above (mutually exclusive with them)
```

#### Camera
```
--camera-pos <x,y,z>         Camera position (default: 0,0.5,3)
--camera-lookat <x,y,z>      Look-at target (default: 0,0,0)
--camera-up <x,y,z>          Up vector (default: 0,1,0)
```

#### Lighting
```
--light <spec>               Add a light (repeatable, max 8):
                               directional:x,y,z[:intensity[:color]]   x,y,z points TO the light
                               point:x,y,z[:intensity[:color]]
                               area:px,py,pz:nx,ny,nz:radius[:samples[:intensity[:color[:shape]]]]
                                 (disk emitter → soft shadows)
                             Color: hex (ffffff) or R,G,B; skip intensity with "::"
--shadows                    Enable shadow rays
--transparent-shadows        Colored shadows through transparent objects (requires --shadows)
```

#### Scene
```
--plane <spec>               Ground plane: [+-]x|y|z:value, e.g. y:-2 (no plane unless given)
--plane-color <spec>         RRGGBB (solid) or RRGGBB:RRGGBB (checkered)
--plane-material <name>      Plane preset (matte, copper, plastic, film, glass-dispersive, water,
                             diamond-dispersive, metal, diamond, parchment, glass, chrome, gold).
                             Mutually exclusive with --plane-color
--env-map <file.hdr>         HDR equirectangular environment map (lighting + background)
--fog density=D:color=r,g,b  Depth-cue fog, e.g. density=0.05:color=0.8,0.8,0.9
--texture-dir <dir>          Base directory for texture=/texture-set=/…-map= files (default: .)
```

#### Quality
```
    --antialiasing           Recursive adaptive antialiasing
    --aa-max-depth <1-4>     AA recursion depth (default: 2)
    --aa-threshold <0-1>     AA edge threshold (default: 0.1)
    --max-ray-depth <1-5>    Maximum bounce depth (default: 5)
-d, --denoise                OptiX AI denoiser on the final accumulated frame
-n, --no-denoise             Force denoising off, even if a DSL scene enables it
    --nodenoise              Disable denoising (does not override DSL scenes)
    --accumulation-frames N  Average N frames to reduce noise (default: 1)
    --allow-uniform-render   Accept renders where ≥99% of pixels have the same colour
                             (otherwise reported as a failed render)
```

#### Caustics (Progressive Photon Mapping)
```
--caustics                   Enable caustics
--caustics-photons <n>       Photons per iteration (default: 100000)
--caustics-iterations <n>    Iterations (default: 10)
--caustics-radius <f>        Initial gather radius (default: derived from scene geometry)
--caustics-alpha <0-1>       Radius reduction factor (default: 0.7)
```

#### Animation — CLI parameter sweep
```
--animate <spec>             frames=N:param=start-end[:param2=...] (repeatable → chained segments)
                             Params: rot-x, rot-y, rot-z (all types); rot-x-w, rot-y-w, rot-z-w,
                             projection-screen-w, projection-eye-w (4D types); level (sponges).
                             Needs --objects and --save-name with %. Mutually exclusive with --timeout
```

#### Animation — DSL scenes (t-parameter)
```
    --t <f>                  Render an animated scene at a single t (freeze-frame)
    --start-t <f>            Start of t range (default: 0)
-e, --end-t <f>              End of t range (default: 1)
    --frames <n>             Number of frames (needs --save-name with %)
    --preview                Interactive scrubbing: Left/Right, Shift+Left/Right, Space, Home/End
```
All of these require `--scene` with an animated scene and are mutually exclusive with `--animate`.

#### Video Output
```
-v, --video <file>           Encode the --frames sequence: .mp4 (H.264/libx264) or .mkv (HEVC/NVENC)
    --video-quality <0-51>   Encoder QP; 0 = lossless, default 12
-k, --keep-frames            Keep the frame PNGs after encoding
```

#### Legacy flags (accepted, but currently without effect)
```
--sponge-type, --level (except as input to --animate validation), --lines, --color,
--face-color, --line-color, -a/--antialias-samples, --projection-screen-w, --projection-eye-w
```
Use the corresponding `--objects` keys (`level=`, `color=`, `eye-w=`, `screen-w=`) instead.
`--optix`, `--object`, `--radius`, `--scale`, `--center` and `--ior` were removed and are
rejected as unknown options.

### Object Types

| Group | Types |
|-------|-------|
| Analytic primitives | `sphere`, `cone`, `plane` |
| Meshes | `cube`, `tetrahedron`, `octahedron`, `dodecahedron`, `icosahedron`, `parametric` |
| Curves & L-systems | `curve`, `lsystem` |
| 3D Menger sponges | `sponge-surface` (O(12ⁿ) faces), `sponge-volume` (O(20ⁿ) cubes), `cube-sponge` (instanced cubes), `sponge-recursive-ias` (O(n·20) VRAM, integer levels 1-14) |
| 4D projected | `tesseract`, `pentachoron`, `16-cell`, `24-cell`, `120-cell`, `600-cell`, `tesseract-sponge-volume`, `tesseract-sponge-surface` |
| 4D IFS fractals | `menger4d`, `sierpinski4d`, `hexadecachoron4d` |

Deprecated aliases still accepted: `sponge` → `sponge-volume`, `sponge-2` → `sponge-surface`,
`tesseract-sponge` → `tesseract-sponge-volume`, `tesseract-sponge-2` → `tesseract-sponge-surface`.

### Object Keys

Unknown keys are rejected with an error. Colors are `#RRGGBB` or `#RRGGBBAA`
(alpha `00` = fully transparent, `FF` = fully opaque).

| Key | Applies to | Meaning |
|-----|-----------|---------|
| `type` | all | Object type (required) |
| `pos=x,y,z` | all | Position (default 0,0,0) |
| `size=S` | all | Scale (default 1.0) |
| `rot-x=`, `rot-y=`, `rot-z=` | all | 3D rotation in degrees |
| `level=L` | sponges, 4D sponges, IFS fractals | Recursion level; fractional values cross-fade. Required for 4D sponges |
| `color=#RRGGBB[AA]` | all | Base color / transparency |
| `material=PRESET` | all | glass, water, diamond, chrome, gold, copper, metal, plastic, matte, film, parchment |
| `ior=I` | all | Index of refraction (1.0 opaque, 1.33 water, 1.5 glass, 2.42 diamond) |
| `roughness=`, `metallic=`, `specular=` | all | Material overrides, 0.0-1.0 |
| `emission=E` | all | Self-illumination, 0.0-10.0 |
| `film-thickness=NM` | all | Thin-film interference thickness in nm (iridescence), e.g. 500 |
| `dispersion=ABBE` | transparent materials | Abbe number for wavelength-dependent refraction (0 = none; glass ≈ 33-60, lower = stronger rainbow) |
| `texture=FILE` | meshes, sphere, cone, plane | Image texture (PNG/JPEG), relative to `--texture-dir`. (Video textures are DSL-only — `VideoTexture`, see the [User Guide](user-guide.md)) |
| `normal-map=`, `roughness-map=`, `metallic-map=`, `ao-map=`, `height-map=` | textured objects | Individual PBR maps |
| `texture-set=DIR` | textured objects | Directory with a complete PBR set (color, normal, roughness, metallic, AO, height — detected automatically; DirectX normal maps are converted). ambientCG (`Name_Color_4K.jpg`) and Poly Haven (`1K/`, `2K/` subdirectories) layouts are recognised |
| `texture-set-res=RES` | with `texture-set` | Preferred resolution (e.g. `2K`); default: highest available |
| `uv-scale=S` | textured objects | Texture tiling factor |
| `procedural=NAME` | all | Procedural texture: value_noise, fbm, worley, gradient, wood, marble, layered_noise, xyz_rgb, heatmap, triplanar |
| `proc-scale=S` | with `procedural` | Procedural noise scale (default 1.0) |
| `eye-w=`, `screen-w=` | 4D types | 4D projection eye / screen distance (defaults 3.0 / 1.5; eye-w must exceed screen-w) |
| `rot-xw=`, `rot-yw=`, `rot-zw=` | 4D types | Rotation in the XW / YW / ZW plane (defaults 15 / 10 / 0 degrees) |
| `edge-radius=`, `edge-material=`, `edge-color=`, `edge-emission=` | 4D types | Render the 4D edges as cylinders |
| `dist-threshold=N` | `menger4d` | IFS distance threshold (integer, default 2) |
| `apex=x,y,z`, `base=x,y,z`, `radius=R` | `cone` | Cone geometry |
| `normal=x,y,z`, `distance=D`, `color2=#RRGGBB`, `checker-size=S` | `plane` object | Plane orientation, offset and checkerboard |
| `control-points=x,y,z,x,y,z,...`, `radius=R` | `curve` | B-spline control points (flat list, multiple of 3) and tube radius |
| `preset=NAME`, `angle=DEG`, `seed=N`, `dim=3\|4` | `lsystem` | Preset (tree, bush, fern3d, hilbert3d, kochisland, hilbert4d, tree4d), branch angle, random seed, 3D or 4D |

### Keyboard and Mouse (interactive window)

| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| Shift + Left/Right arrows, Shift + horizontal left-drag | Rotate 4D objects in the XW plane |
| Shift + Up/Down arrows, Shift + vertical left-drag | Rotate 4D objects in the YW plane |
| Shift + Page Up/Down, Shift + vertical right-drag | Rotate 4D objects in the ZW plane |
| Shift + scroll | Change 4D projection eye distance (`eye-w`) |
| C | Toggle coordinate cross |
| Esc | Reset 4D rotation and projection (does **not** quit) |
| Ctrl + Q | Quit |

In `--preview` mode: Left/Right step `t`, Shift+Left/Right larger steps, Space play/pause,
Home/End jump to the ends of the range.

### File Formats

**Input:** textures as PNG (recommended) or JPEG; video textures (DSL only); HDR (`.hdr`)
environment maps; DSL scenes as `.scala` files.

**Output:** PNG images via `--save-name`; MP4 (H.264) or MKV (HEVC) video via `--video`.
To encode frame sequences yourself:

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
sbt "run --objects type=sphere"                          # Interactive window
sbt "run --objects type=sphere --headless --save-name s.png"   # Straight to a file
sbt "run --scene glass-sphere"                           # Built-in DSL scene

# COMMON OPERATIONS
sbt compile                                # Build project
sbt test                                   # Run tests
./.git_hooks/pre-push                      # Full quality gate before pushing

# BASIC RENDERS
sbt "run --objects type=sponge-volume:level=2"     # Level 2 sponge
sbt "run --objects type=sphere:ior=1.5"            # Glass sphere
sbt "run --objects type=tesseract:rot-xw=30"       # Rotated tesseract

# QUALITY IMPROVEMENTS
--plane y:-2 --plane-color ffffff:808080   # Checkered floor
--shadows                                  # Add shadows
--antialiasing                             # Smooth edges
--accumulation-frames 8 --denoise          # Less noise

# ANIMATION
--save-name frame%03d.png --animate frames=36:rot-y=0-360   # 360° rotation
--scene examples.dsl.OrbitingSphere --frames 100 --save-name f%03d.png --video out.mp4

# MATERIALS (inside --objects)
ior=1.5                                    # Glass
material=gold                              # Gold preset
material=glass:roughness=0.3               # Frosted glass

# LIGHTING
--light directional:-1,1,-1                # Sun-like light
--light point:0,5,0:2.0                    # Bright overhead light
--light point:2,3,2::ffd700                # Gold-colored light

# TROUBLESHOOTING
sbt clean compile                          # Clean rebuild
export __GL_THREADED_OPTIMIZATIONS=0       # Fix NVIDIA crashes under xvfb-run
```

---

← [Tutorials](tutorials.md) | [User Guide Index](../USER_GUIDE.md)

---

For questions, issues, or contributions:
- GitHub: https://github.com/lene/menger
- Issues: https://github.com/lene/menger/issues
- Documentation: https://github.com/lene/menger/tree/main/docs

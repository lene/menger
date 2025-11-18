# Feature Request: CLI Support for Solid Color Plane

**Priority**: Medium
**Type**: Enhancement
**Component**: CLI / OptiX Renderer

## Feature Description

Add command-line interface support for the solid color plane mode that was recently implemented for shadow testing. Currently, solid plane mode is only accessible via Scala API (`setPlaneSolidColor(true)`), but not through CLI arguments.

## Current State

### What Exists
- ✅ Backend support: `setPlaneSolidColor(bool)` in OptiXWrapper
- ✅ JNI binding available
- ✅ Scala API: `renderer.setPlaneSolidColor(true)`
- ✅ Shader implementation: Plane renders as solid light gray (140, 140, 140)

### What's Missing
- ❌ CLI argument to enable solid plane mode
- ❌ CLI argument to set plane color (currently hardcoded to light gray)
- ❌ Documentation of new CLI options

## Proposed CLI Arguments

### Option 1: Simple Boolean Flag
```bash
./menger --plane-solid  # Uses default light gray (140, 140, 140)
./menger --plane-checkerboard  # Explicit checkerboard (default behavior)
```

### Option 2: With Color Specification (Recommended)
```bash
# Solid color with RGB values
./menger --plane-solid=140,140,140

# Solid color with hex
./menger --plane-solid=#8C8C8C

# Solid color with named preset
./menger --plane-solid=lightgray

# Checkerboard (default)
./menger --plane-checkerboard
```

## Implementation Plan

### 1. Extend OptiXWrapper API
Add color parameter to plane configuration:

```cpp
// In OptiXWrapper.h
void setPlaneSolidColor(bool enabled, const float* rgb = nullptr);

// In OptiXWrapper.cpp
void OptiXWrapper::setPlaneSolidColor(bool enabled, const float* rgb) {
    impl->plane_solid_color = enabled;
    if (rgb) {
        std::memcpy(impl->plane_color, rgb, 3 * sizeof(float));
    } else {
        // Default: light gray
        impl->plane_color[0] = 140.0f / 255.0f;
        impl->plane_color[1] = 140.0f / 255.0f;
        impl->plane_color[2] = 140.0f / 255.0f;
    }
}
```

### 2. Update Shader
Replace hardcoded `PLANE_SOLID_LIGHT_GRAY` constant with param:

```cuda
// In OptiXData.h
struct Params {
    // ...
    float plane_solid_color_rgb[3];  // RGB color for solid plane mode
    // ...
};

// In sphere_combined.cu
if (params.plane_solid_color) {
    r = static_cast<unsigned int>(params.plane_solid_color_rgb[0] * 255.0f);
    g = static_cast<unsigned int>(params.plane_solid_color_rgb[1] * 255.0f);
    b = static_cast<unsigned int>(params.plane_solid_color_rgb[2] * 255.0f);
}
```

### 3. Add CLI Parsing
In main CLI argument parser (likely in `Main.scala` or similar):

```scala
case class PlaneConfig(
  mode: PlaneMode = PlaneMode.Checkerboard,
  solidColor: Option[Color] = None
)

sealed trait PlaneMode
object PlaneMode {
  case object Checkerboard extends PlaneMode
  case object Solid extends PlaneMode
}

// Parse arguments
args match {
  case "--plane-solid" :: rest =>
    config = config.copy(plane = PlaneConfig(PlaneMode.Solid, Some(Color(140, 140, 140))))

  case "--plane-solid" :: color :: rest =>
    val rgb = parseColor(color)  // Parse "R,G,B" or "#RRGGBB"
    config = config.copy(plane = PlaneConfig(PlaneMode.Solid, Some(rgb)))

  case "--plane-checkerboard" :: rest =>
    config = config.copy(plane = PlaneConfig(PlaneMode.Checkerboard, None))
}
```

### 4. Update Documentation
- Add to `--help` output
- Document in README.md or CLI docs
- Add examples to CLAUDE.md

## Use Cases

### Shadow Testing
```bash
# Light gray plane for clear shadow visibility
./menger --plane-solid=140,140,140 --shadows

# White plane for maximum shadow contrast
./menger --plane-solid=255,255,255 --shadows
```

### Artistic/Aesthetic
```bash
# Dark plane for dramatic lighting
./menger --plane-solid=30,30,30

# Colored plane for specific aesthetic
./menger --plane-solid=100,150,200  # Light blue
```

### Performance Testing
```bash
# Solid color may have different performance than checkerboard
# (fewer conditional branches in shader)
./menger --plane-solid --benchmark
```

## Color Parsing Helper

```scala
object ColorParser {
  def parse(s: String): Option[Color] = s match {
    // RGB format: "R,G,B"
    case rgb if rgb.contains(',') =>
      rgb.split(',').map(_.trim.toInt) match {
        case Array(r, g, b) if validRange(r, g, b) => Some(Color(r, g, b))
        case _ => None
      }

    // Hex format: "#RRGGBB" or "RRGGBB"
    case hex if hex.startsWith("#") || hex.length == 6 =>
      val clean = hex.stripPrefix("#")
      Try {
        val r = Integer.parseInt(clean.substring(0, 2), 16)
        val g = Integer.parseInt(clean.substring(2, 4), 16)
        val b = Integer.parseInt(clean.substring(4, 6), 16)
        Color(r, g, b)
      }.toOption

    // Named colors
    case "white" => Some(Color(255, 255, 255))
    case "lightgray" | "lightgrey" => Some(Color(140, 140, 140))
    case "gray" | "grey" => Some(Color(128, 128, 128))
    case "darkgray" | "darkgrey" => Some(Color(80, 80, 80))
    case "black" => Some(Color(0, 0, 0))

    case _ => None
  }

  private def validRange(r: Int, g: Int, b: Int): Boolean =
    r >= 0 && r <= 255 && g >= 0 && g <= 255 && b >= 0 && b <= 255
}
```

## Testing

### Manual Testing
```bash
# Verify each format works
./menger --plane-solid
./menger --plane-solid=100,150,200
./menger --plane-solid=#64A0C8
./menger --plane-solid=white
./menger --plane-solid=lightgray

# Verify default behavior unchanged
./menger  # Should use checkerboard
./menger --plane-checkerboard  # Explicit checkerboard
```

### Automated Testing
Add unit tests for color parsing:
- Valid RGB formats
- Valid hex formats
- Named colors
- Invalid formats (should fail gracefully)
- Edge cases (0, 255, negative, > 255)

## Related Work

- Current implementation: `setPlaneSolidColor(bool)` in optix-jni/210c6b8
- Shadow testing: Uses solid plane for better shadow visibility
- Plane constants: `PLANE_SOLID_LIGHT_GRAY = 140` in OptiXData.h

## Success Criteria

- [ ] CLI argument `--plane-solid` enables solid plane mode
- [ ] CLI argument accepts color in RGB format: `--plane-solid=R,G,B`
- [ ] CLI argument accepts color in hex format: `--plane-solid=#RRGGBB`
- [ ] CLI argument accepts named colors: `--plane-solid=white`
- [ ] Default solid color is light gray (140, 140, 140)
- [ ] `--help` documents new options
- [ ] Backward compatible (no CLI args = checkerboard plane)
- [ ] Color validation provides helpful error messages
- [ ] Updated documentation in README/CLAUDE.md

## Priority Justification

**Medium** because:
- ✅ Feature already exists in backend (low implementation risk)
- ✅ Useful for shadow visualization and testing
- ✅ Relatively small scope (CLI parsing + plumbing)
- ❌ Not blocking any critical functionality
- ❌ Workaround exists (use Scala API directly)

## Estimated Effort

- Backend changes: 2-3 hours
- CLI parsing: 1-2 hours
- Testing: 1 hour
- Documentation: 30 minutes
- **Total**: ~5 hours

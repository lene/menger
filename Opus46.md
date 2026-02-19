# Top Complex Tasks for Opus 4.6:

## 1. Animation system (TODO.md lines 44-47) - ARCHITECTURALLY COMPLEX

  "Object transform animation (position, rotation, scale), Keyframe system with linear
interpolation, Frame sequence rendering, DSL animation
   syntax"

  This requires:

  - Designing the keyframe system architecture from scratch
  - DSL syntax design for animation definitions
  - Frame sequence rendering pipeline
  - Integration with existing scene system and DSL
  - Easing functions and interpolation (line 49)

  Why Opus 4.6: This is a significant feature requiring good architectural design, DSL syntax
design, and understanding of animation systems.

  ---

  # 2. Validate tesseract sponge generation (TODO.md line 15)

  "Validate tesseract sponge generation from surfaces by repeating it with cubes. same result?"

  This requires understanding 4D fractal geometry algorithms and implementing validation tests.

  ---

  My Recommendation:

  The Scala wrapper for libGDX (task #1) would benefit the most from Opus 4.6. It's a
large-scale architectural refactoring that requires:
  - Systematic analysis of the entire codebase
  - Sophisticated architectural decision-making
  - Understanding functional programming principles deeply
  - Careful planning to avoid breaking existing functionality

  This is the kind of "think deeply about the whole system" task where Opus 4.6's advanced reasoning
would shine.

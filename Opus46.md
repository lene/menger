# Top Complex Tasks for Opus 4.6:

## 1. Scala wrapper for libGDX (TODO.md line 27) - MOST COMPLEX

  This is a major architectural refactoring requiring:

  - Deep analysis of all current libGDX usage patterns across the entire codebase
  - Designing a clean abstraction layer that isolates var and null usage
  - Maintaining API compatibility while improving code quality
  - Architectural decisions about where boundaries should be drawn
  - This touches many files and requires sophisticated architectural thinking

  Why Opus 4.6: This needs strong architectural vision, understanding of functional programming
principles, and ability to reason about
  large-scale refactoring strategies.

  ---

## 2. Thin-film physics implementation (TODO.md line 14) - VERY COMPLEX

  "Implement proper thin-film physics with thickness parameter and interference effects (Film
material)"

  This requires:

  - Deep understanding of optical physics (thin-film interference, wave behavior)
  - Complex mathematical modeling of light wave interactions
  - Integration with the ray tracing and material system
  - Physically accurate rendering of iridescent effects

  Why Opus 4.6: This involves sophisticated physics and requires reasoning about how to translate
complex optical phenomena into practical
  rendering algorithms.

  ---

## 3. Animation system (TODO.md lines 44-47) - ARCHITECTURALLY COMPLEX

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

# 4. Debugging: Gold transparency & fractional sponge issues (TODO.md line 8)

  "Check output - seems buggy (e.g. gold is transparent, fractional sponge level weirdnesses)"

  This could be tricky debugging work if the issues are subtle or involve complex interactions
between materials, ray tracing, and fractal generation.
  Why Opus 4.6: Deep debugging of rendering issues may require sophisticated reasoning about ray
tracing behavior.

  ---

  # 5. Validate tesseract sponge generation (TODO.md line 15)

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

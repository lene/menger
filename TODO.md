- introduce overloaded JNI functions for imageSize and vector[3] and use them throughout
- review tuple return types: (Int, Int) for positions/coordinates, (Double, Double, Double) for RGB ratios - consider named types
- rename Boolean-returning visibility functions: backgroundVisibility -> isBackgroundVisible, planeVisibility -> isPlaneVisible (and others)
- ensure PTX are in the right place
  > Packaged app can't find optixjni library
  > The packaged application failure is expected because it's trying to load the
  > native library from the system path rather than the bundled one. This is a known
  > packaging issue but doesn't affect the correctness of the code.
- antialiasing/adaptive super-sampling
- --plane: default no plane, can be repeated for multiple planes
- caustics
- on-the-fly change of image dimension with resizing of menger main window
- fix optix cache corruption

## later
- composites
- Cubes
- mesh objects 
- sponges
- render coordinate cross
- DSL (domain specific language) aka SDL (scene description language)

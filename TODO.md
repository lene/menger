- comprehensive test suite V
- make test suite run faster V
- clean up the code V
- get rid of ENABLE_OPTIX_JNI V
- separate optix function call layer from application code V
- revise ifs where something < 0.01 is assumed to be 0 V
- make c++ tests less verbose V
- remove docstrings
- ensure PTX are in the right place
  > Packaged app can't find optixjni library
  > The packaged application failure is expected because it's trying to load the
  > native library from the system path rather than the bundled one. This is a known
  > packaging issue but doesn't affect the correctness of the code.
- ray casting and tracing statistics
- antialiasing/adaptive super-sampling
- --plane: default no plane, can be repeated for multiple planes
- caustics and shadows
- define light source(s)
- interactive camera positioning with mouse dragging in menger main window
- on-the-fly change of image dimension with resizing of menger main window

## later
- composites
- Cubes
- mesh objects 
- sponges
- render coordinate cross
- DSL (domain specific language) aka SDL (scene description language)

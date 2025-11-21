- ensure PTX are in the right place
  > Packaged app can't find optixjni library
  > The packaged application failure is expected because it's trying to load the
  > native library from the system path rather than the bundled one. This is a known
  > packaging issue but doesn't affect the correctness of the code.
- --plane: default no plane, can be repeated for multiple planes
- use a color class dereved from Vector instead of the libGDX class
- caustics
- on-the-fly change of image dimension with resizing of menger main window

## later
- composites
- Cubes
- mesh objects 
- sponges
- render coordinate cross
- DSL (domain specific language) aka SDL (scene description language)

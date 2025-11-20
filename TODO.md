- ensure PTX are in the right place
  > Packaged app can't find optixjni library
  > The packaged application failure is expected because it's trying to load the
  > native library from the system path rather than the bundled one. This is a known
  > packaging issue but doesn't affect the correctness of the code.
- antialiasing/adaptive super-sampling
- --plane: default no plane, can be repeated for multiple planes
- caustics
- on-the-fly change of image dimension with resizing of menger main window
- fix optix cache corruption:
     [OptiX][3][DISKCACHE]: Failed to insert into the cache. Query failed. database disk image is malformed (code: 11)
     [OptiX][3][DISKCACHE]: Database corruption detected. Deleting cache file: "/var/tmp/OptixCache_lepr/optix7cache.db".
- ensure dockeris up to date on home laptop, should be:
  $ docker --version
  Docker version 28.2.2  # Your runner is NEW
  $ docker version --format '{{.Client.APIVersion}}'
  1.50  # Your runner's API is NEW (well above 1.44)


## later
- composites
- Cubes
- mesh objects 
- sponges
- render coordinate cross
- DSL (domain specific language) aka SDL (scene description language)

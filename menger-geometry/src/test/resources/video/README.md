# Video Test Fixtures

These files are tiny deterministic SDR fixtures for native libav decode tests and
packaged-app integration tests. They contain no audio.

Regenerate the rectangular texture fixture:

```bash
ffmpeg -y \
  -f lavfi -i color=c=red:s=4x4:r=2:d=0.5 \
  -f lavfi -i color=c=green:s=4x4:r=2:d=0.5 \
  -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0,format=rgba' \
  -c:v qtrle -pix_fmt argb \
  menger-geometry/src/test/resources/video/two-frame-rgba.mov
```

Regenerate the equirectangular 360-degree fixture:

```bash
ffmpeg -y \
  -f lavfi -i color=c=red:s=4x2:r=2:d=0.5 \
  -f lavfi -i color=c=blue:s=4x2:r=2:d=0.5 \
  -filter_complex '[0:v][1:v]concat=n=2:v=1:a=0,format=rgba' \
  -c:v qtrle -pix_fmt argb \
  menger-geometry/src/test/resources/video/two-frame-equirect-rgba.mov
```

The equirectangular fixture must stay 2:1 because `EnvMapVideo` rejects ordinary
rectangular object-video dimensions.

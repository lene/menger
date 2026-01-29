# Setup Guide: Image Comparison Testing

## What Was Implemented

Visual regression testing has been added to both integration and manual test scripts. Tests now compare rendered images against reference images to catch unintended visual changes.

## Changes Made

### 1. Docker Image
- **File**: `optix-jni/Dockerfile`
- **Change**: Added `imagemagick` package to Layer 6
- **Action Required**: Rebuild and push the Docker image

### 2. Integration Tests
- **File**: `scripts/integration-tests.sh`
- **Changes**:
  - **Modified `run_test()` to run ALL tests in headless mode with image saving**
  - Added `--update-references` flag to regenerate golden images
  - Added image comparison using ImageMagick's `compare` command
  - Added diff image generation when tests fail
  - Added summary reporting for image comparison failures
- **New directories**:
  - `scripts/reference-images/` - Golden reference images (version controlled)
  - `scripts/test-diffs/` - Visual diff images (git ignored)
- **Important**: Tests now run in headless mode instead of timeout mode to enable full image comparison

### 3. Manual Tests
- **File**: `scripts/manual-test.sh`
- **Changes**: Same as integration tests
- **New flag**: `--update-references`

### 4. Git Configuration
- **File**: `.gitignore`
- **Changes**:
  - Added exception to commit reference images
  - Added ignore rules for test output directories

## Setup Steps

### Step 1: Rebuild Docker Image

The Docker image needs to be rebuilt with ImageMagick:

```bash
cd optix-jni

# Build the image (you'll need the OptiX SDK installer)
docker build -t registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.12.0 .

# Push to registry
docker push registry.gitlab.com/lilacashes/menger/optix-cuda:12.8-9.0-25-1.12.0
```

### Step 2: Generate Initial Reference Images

Generate the golden reference images locally:

```bash
# Build the application
sbt stage

# Generate reference images for integration tests
./scripts/integration-tests.sh \
  ./menger-app/target/universal/stage/bin/menger-app \
  --update-references

# Generate reference images for manual tests
./scripts/manual-test.sh --update-references
```

This will create PNG files in `scripts/reference-images/`.

### Step 3: Review and Commit Reference Images

**Important**: Carefully review the reference images before committing!

```bash
# View the generated images
ls -lh scripts/reference-images/

# Use an image viewer to inspect them
eog scripts/reference-images/*.png  # or any image viewer

# Once verified, commit them
git add scripts/reference-images/
git commit -m "test: Add initial reference images for visual regression testing"
```

### Step 4: Verify CI Integration

After committing and pushing:

1. CI will run automatically
2. Integration tests will now compare rendered images against references
3. Any visual differences will cause test failures with diff images in artifacts

## Usage After Setup

### Running Tests Locally

```bash
# Run with image comparison (normal mode)
./scripts/integration-tests.sh ./menger-app/target/universal/stage/bin/menger-app

# Update references after intentional rendering changes
./scripts/integration-tests.sh ./menger-app/target/universal/stage/bin/menger-app --update-references
```

### Updating References After Rendering Changes

When you make intentional rendering changes:

1. Run tests with `--update-references`
2. Review the changes carefully
3. Commit with a descriptive message:
   ```bash
   git add scripts/reference-images/
   git commit -m "test: Update references for [your feature/fix]"
   ```

### When Tests Fail

If image comparison fails:

1. Check `scripts/test-diffs/` for visual diff images
2. Determine if the difference is a bug or intentional change
3. If intentional, update references
4. If a bug, fix the code and rerun tests

## Threshold Configuration

Current threshold: **0.1% pixel difference** (0.001)

To adjust tolerance:
- Edit `IMAGE_DIFF_THRESHOLD` in `scripts/integration-tests.sh`
- Edit `IMAGE_DIFF_THRESHOLD` in `scripts/manual-test.sh`

## Troubleshooting

### "No reference image found"
Normal on first run or for new tests. Run with `--update-references` to create them.

### "ImageMagick 'compare' command not found"
Install ImageMagick:
```bash
sudo apt-get install imagemagick
```

### False positives (tiny differences)
Adjust `IMAGE_DIFF_THRESHOLD` if differences are imperceptible but tests fail.

### CI fails but local tests pass
Possible GPU driver differences. Check the diff images in CI artifacts.

## Architecture

### Comparison Method
- **Tool**: ImageMagick `compare` with AE (Absolute Error) metric
- **Metric**: Counts different pixels
- **Calculation**: `diff_percent = different_pixels / total_pixels`
- **Pass condition**: `diff_percent <= threshold`

### File Naming
Test names are sanitized for filenames:
- Spaces and special characters → underscores
- Multiple underscores → single underscore
- Example: "tesseract (4D)" → `tesseract__4D_.png`

### Directory Structure
```
scripts/
├── integration-tests.sh
├── manual-test.sh
├── reference-images/           # Golden images (committed)
│   ├── sphere.png
│   ├── cube.png
│   └── ...
├── test-diffs/                 # Diff images (ignored)
│   └── test_name_diff.png
└── test-output/                # Temp output (ignored)
    └── *.png
```

## Next Steps

1. ✅ Rebuild Docker image with ImageMagick
2. ✅ Generate initial reference images
3. ✅ Review and commit references
4. ✅ Push changes to trigger CI
5. ✅ Monitor CI for successful image comparison

# Integration Testing with Image Comparison

## Overview

The integration tests now include visual regression testing by comparing rendered images against reference images. This ensures that rendering changes are intentional and catches unexpected visual bugs.

## Directory Structure

```
scripts/
├── integration-tests.sh      # Automated integration tests
├── manual-test.sh            # Manual visual verification tests
├── reference-images/         # Golden reference images for comparison
├── test-diffs/              # Visual diff images (created when tests fail)
└── test-output/             # Temporary test output (manual-test.sh only)
```

## Running Tests

### Integration Tests (Automated)

```bash
# Run all tests with image comparison
./scripts/integration-tests.sh ./menger-app/target/universal/stage/bin/menger-app

# Update reference images after intentional rendering changes
./scripts/integration-tests.sh ./menger-app/target/universal/stage/bin/menger-app --update-references
```

### Manual Tests (Visual Verification)

```bash
# Run all static tests with image comparison
./scripts/manual-test.sh

# Update reference images
./scripts/manual-test.sh --update-references

# Skip to interactive tests
./scripts/manual-test.sh -i
```

## How It Works

1. **Test Execution**: Each test renders an image in headless mode
2. **Image Saving**: All tests save output to temporary files for comparison
3. **Comparison**: The rendered image is compared against a reference image using ImageMagick
4. **Threshold**: Images are considered matching if they differ by less than 0.1% of pixels
5. **Reporting**: Differences are reported with percentage and visual diff images are saved

**Note**: All tests now run in headless mode to enable image comparison. The `--timeout` parameter is no longer used during testing.

## Image Comparison Metrics

- **Metric**: Absolute Error (AE) - counts different pixels
- **Threshold**: 0.001 (0.1% of total pixels)
- **Tolerance**: Allows for small floating-point differences and driver variations

## When Tests Fail

If image comparison fails:

1. **Review Diff Images**: Check `scripts/test-diffs/` for visual differences
2. **Investigate**: Determine if the difference is intentional or a bug
3. **Update References**: If intentional, run with `--update-references`

Example diff image naming: `test_name_diff.png`

## Updating Reference Images

Reference images should be updated when:

- Intentional rendering changes are made (new features, bug fixes)
- Lighting, material, or shader changes are implemented
- Camera or projection algorithms are modified

**Important**: Always review the changes before updating references!

```bash
# Generate new reference images
./scripts/integration-tests.sh ./path/to/menger --update-references

# Commit the updated references
git add scripts/reference-images/
git commit -m "test: Update reference images for [reason]"
```

## CI/CD Integration

The integration tests run automatically in CI. The Docker image includes ImageMagick for image comparison.

**Note**: Reference images must be committed to the repository for CI to work correctly.

## Troubleshooting

### ImageMagick Not Found

```bash
sudo apt-get install imagemagick
```

### False Positives (Small Differences)

If you're getting failures for visually identical images:

1. Check the difference percentage in the output
2. Adjust `IMAGE_DIFF_THRESHOLD` in the test scripts if needed
3. Review diff images to ensure differences are truly insignificant

### GPU Driver Differences

Different GPU drivers may produce slightly different results. The 0.1% threshold should handle most variations, but if you see consistent failures across different hardware:

1. Review the actual visual difference
2. Consider using a dedicated CI runner with consistent hardware
3. Adjust threshold if differences are imperceptible

## Best Practices

1. **Always run tests before committing** rendering changes
2. **Review diff images** when tests fail - don't blindly update references
3. **Keep reference images small** - use default resolution unless testing specific sizes
4. **Document reference updates** in commit messages with the reason for changes
5. **Version control reference images** - they're part of the test suite

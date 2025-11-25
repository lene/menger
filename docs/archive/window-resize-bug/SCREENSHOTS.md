# Screenshot Analysis - Window Resize Bug

This document describes the screenshots captured during the window resize bug investigation.

## Screenshots Overview

All screenshots are stored in `docs/window-resize-bug/screenshots/`

### Initial Test Sequence (Failed FOV Scaling Attempt)

1. **2025-11-13-133743_800x600_scrot.png**
   - Initial window size: 800x600
   - Sphere appears circular and correct (baseline)

2. **2025-11-13-133903_1267x600_scrot.png**
   - Width increased to 1267x600
   - **Problem**: Sphere is horizontally stretched (elliptical)
   - Expected: Sphere should scale up but remain circular

3. **2025-11-13-133913_682x600_scrot.png**
   - Width decreased to 682x600
   - **Problems**:
     - Sphere is vertically stretched
     - Black border appears on the right side
   - Expected: Sphere should scale down but remain circular

4. **2025-11-13-133922_682x885_scrot.png**
   - Height increased to 682x885
   - **Problem**: Sphere is vertically stretched
   - Expected: Sphere should remain same size and circular

### Earlier Test Sequence

5. **2025-11-13-132149_1507x600_scrot.png**
   - Width increased significantly to 1507x600
   - Shows extreme horizontal stretching of sphere

6. **2025-11-13-132208_741x600_scrot.png**
   - Width reduced to 741x600
   - Shows vertical stretching with black border

7. **2025-11-13-132225_741x1007_scrot.png**
   - Height increased to 741x1007
   - Shows vertical stretching persists

## Key Observations

1. **Distortion Pattern**: The sphere becomes elliptical rather than maintaining its circular shape
2. **Black Borders**: Appear when width is reduced below initial size
3. **Consistent Failure**: All resize operations result in distorted spheres
4. **FOV Mismatch**: Evidence points to OptiX interpreting FOV as horizontal while we pass vertical FOV value

## Root Cause

The screenshots confirm that the OptiX renderer is interpreting the FOV parameter as horizontal field of view, while the Scala code is passing a vertical FOV value (45°). This mismatch causes the aspect ratio correction to be applied incorrectly, resulting in elliptical spheres during window resizing.
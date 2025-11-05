# Fix Plan for OptiX Volume Absorption

## Root Cause (CONFIRMED Nov 5, 2025)
The OptiX built-in sphere primitive does NOT support ray intersection from inside the sphere. When a refracted ray is traced from inside the sphere, it never hits the back surface, preventing the exit code path from executing.

## Definitive Evidence
Comprehensive testing with explicit hit_t value checks confirms:
- **Test 1**: Ray from sphere center (tmin=0) → hit_t=-1.0 (NO intersection)
- **Test 2**: Ray from just inside surface (tmin=0.001) → hit_t=-1.0 (NO intersection)
- **Test 3**: Same ray with tmin=0 → hit_t=-1.0 (tmin not the issue)
- **Conclusion**: This is a fundamental limitation of OptiX built-in sphere primitives

## Solution: Custom Intersection Program

See **[Glass_Implementation_Plan.md](Glass_Implementation_Plan.md)** for the complete implementation plan.

### Summary

Replace `OPTIX_BUILD_INPUT_TYPE_SPHERES` with custom intersection program that:
1. Detects BOTH entry and exit intersections (even from inside sphere)
2. Reports hit_kind (0=entry, 1=exit) to distinguish intersection type
3. Passes surface normal via intersection attributes
4. Enables proper Beer-Lambert absorption calculation

This is the production-tested, NVIDIA-recommended approach used in all OptiX SDK glass examples (e.g., optixWhitted).

### Implementation Phases

1. **Phase 1**: Code cleanup (remove test code) - 30 min
2. **Phase 2**: Implement custom intersection program - 2 hours
3. **Phase 3**: Update host-side pipeline (geometry type, program groups) - 1 hour
4. **Phase 4**: Fix ray offsetting and state management - 1 hour
5. **Phase 5**: Testing and validation - 1 hour

**Total**: 5-6 hours for complete implementation

See Glass_Implementation_Plan.md for detailed code and step-by-step instructions.
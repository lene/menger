# Scripts Archive

This directory contains obsolete test and verification scripts that are no longer used in the build or CI pipeline.

## Index

### OptiX Verification

**test-optix.cpp**
- Minimal OptiX header compilation test
- Compile-time verification that OptiX headers are accessible
- Replaced by:
  - `scripts/verify-optix.sh` (comprehensive verification)
  - OptiX JNI C++ unit tests (`optix-jni/src/main/native/tests/`)

**Purpose**: Header availability check only (no runtime execution)

**Compile**:
```bash
g++ -I$OPTIX_ROOT/include -c test-optix.cpp -o test-optix.o
```

## Active Scripts

For current, actively used scripts, see:
- `scripts/verify-optix.sh` - Comprehensive OptiX/CUDA/driver verification
- `scripts/setup-optix-local.sh` - Local OptiX SDK setup
- `scripts/nvidia-spot.sh` - AWS spot instance management

All other scripts in `/scripts/` are actively used for:
- AWS infrastructure (Terraform, EC2)
- OptiX SDK setup
- Environment verification

---

*Last Updated: 2025-11-25*

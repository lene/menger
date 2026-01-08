# AGENTS.md - Development Guidelines

## Build/Test Commands
```bash
sbt compile                    # Compile all modules (includes C++/CUDA)
sbt test --warn               # Run all tests (~1064 Scala + 27 C++ = ~1091 total)
sbt "testOnly ClassName"      # Run specific Scala test
sbt "project optixJni" nativeTest  # Run C++ Google Test suite
sbt "scalafix --check"        # Verify code quality
sbt run                       # Run application
```

## Code Style Requirements
- **Scala 3 only** - never Scala 2 syntax
- **Line length:** max 100 characters
- **Imports:** one per line, organized as: javax/scala/* (per .scalafix.conf)
- **No null values:** use Option/Try/Either everywhere
- **Functional style:** avoid mutable state, exceptions (prefer Try/Either)
- **No docstrings:** use descriptive function/parameter names instead
- **Wartremover enforced:** no var/while/asInstanceOf/throw in compile code
- **Test framework:** AnyFlatSpec for Scala tests

## Critical Conventions
- **Alpha channel:** 0.0 = fully transparent, 1.0 = fully opaque (OptiX shaders, Color objects)
- **Architecture docs:** consult docs/arc42/ before architectural changes
- **Manual edits only:** never use scripts/sed for bulk code changes
- **Never commit automatically:** never commit changes without the user reviewing them. Always show a diff for the user to review. 
- **Pre-push hook:** always run before commits
- **CUDA/OptiX:** SDK version must match driver (check with strings command)
- 

# Menger Release Checklist

Comprehensive step-by-step guide for releasing a new version of Menger.

**Estimated total time:** 15-20 minutes from commit to published release
- Pre-release preparation: 15-30 minutes (manual)
- Pre-push validation: 8-10 minutes (automated)
- Commit & push: 1-2 minutes (manual)
- Release pipeline: 12-15 minutes (automated)
- Post-release verification: 5-10 minutes (manual)

---

## Phase 1: Pre-Release Preparation (Manual - 15-30 minutes)

### 1. Version Management

Decide on version number following semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features, backwards compatible
- PATCH: Bug fixes only

**Update version in 3 locations (must match):**

- [ ] `menger-app/build.sbt` (line 2):
  ```scala
  version := "X.Y.Z"
  ```

- [ ] `.gitlab-ci.yml` (line 15):
  ```yaml
  DEPLOYABLE_VERSION: X.Y.Z
  ```

- [ ] `menger-app/src/main/scala/menger/MengerCLIOptions.scala` (line ~30):
  ```scala
  version("menger vX.Y.Z ...")
  ```

**Automated validation:** Pre-push hook verifies version consistency

### 2. Documentation Updates

#### CHANGELOG.md (Required)

- [ ] Update `[Unreleased]` section to `[X.Y.Z] - YYYY-MM-DD`
- [ ] Follow keepachangelog.com format:
  - **Added**: New features
  - **Changed**: Changes to existing functionality
  - **Deprecated**: Soon-to-be removed features
  - **Removed**: Removed features
  - **Fixed**: Bug fixes
  - **Security**: Security fixes

- [ ] Write clear, user-focused descriptions
- [ ] Document only bugs that existed in previous releases (not bugs introduced and fixed in same version)
- [ ] Add version comparison link at bottom:
  ```markdown
  [X.Y.Z]: https://gitlab.com/lilacashes/menger/-/compare/PREV...X.Y.Z
  ```

**Example entry:**
```markdown
## [0.4.3] - 2026-02-05

### Fixed
- Corrected directional light direction for accurate shadow rays
- Fixed PushToGithub CI job by adding GIT_DEPTH: 0 for complete history

### Added
- Reference images for 3D fractional sponge tests
```

#### ROADMAP.md

- [ ] Mark completed sprint as "✅ Complete"
- [ ] Update milestone status if version completes a milestone
- [ ] Remove obsolete items from TODO.md (move to ROADMAP backlog or delete)
- [ ] Archive completed sprint plan if needed:
  ```bash
  mv docs/sprints/SPRINT.md docs/archive/sprints/sprint-N.md
  ```

#### arc42 Documentation (If Applicable)

Update if architectural changes were made:

- [ ] `docs/arc42/09-architectural-decisions.md` - Document new architectural decisions
- [ ] `docs/arc42/10-quality-requirements.md` - Update if performance characteristics changed
- [ ] `docs/arc42/11-risks-and-technical-debt.md` - Document new technical debt or resolved risks

#### USER_GUIDE.md (If New Features)

- [ ] Update usage examples if new features added
- [ ] Add new command-line options to reference
- [ ] Update screenshots/examples if UI changed

### 3. Code Quality Assessment

Run comprehensive codebase assessment and document in CODE_IMPROVEMENTS.md:

- [ ] Check clean code guidelines:
  - Descriptive naming (functions, variables, classes)
  - Function length (max ~50 lines, prefer smaller)
  - Cyclomatic complexity (avoid deeply nested logic)

- [ ] Verify architectural quality:
  - Separation of concerns (rendering vs business logic)
  - Code duplication (DRY principle)
  - Hardcoded constants (extract to named constants)
  - Over-long functions and classes

- [ ] Review test coverage quality:
  - Are critical paths tested?
  - Are edge cases covered?
  - Are tests readable and maintainable?

- [ ] Document findings in CODE_IMPROVEMENTS.md:
  - Overall grade (A+ to F)
  - Strengths to maintain
  - Areas for improvement
  - Action items for future sprints

**IMPORTANT:** Always address critical issues before release. Do not defer critical bugs or security issues.

### 4. Dependency Updates (Required)

- [ ] Check for outdated dependencies:
  ```bash
  sbt dependencyUpdates
  ```

- [ ] Review for security vulnerabilities:
  ```bash
  sbt dependencyCheck
  ```

- [ ] Update dependencies if needed:
  - Security patches: **Update immediately**
  - Major versions: Test thoroughly, document breaking changes
  - Minor versions: Update if improvements justify risk

- [ ] After updates, run full pre-push hook:
  ```bash
  .git_hooks/pre-push
  ```

---

## Phase 2: Pre-Push Validation (Automated - 8-10 minutes)

The pre-push hook (`.git_hooks/pre-push`) runs automatically on `git push`. You can also run it manually:

```bash
.git_hooks/pre-push
```

**What it validates:**

### Environment Validation
- [x] `CUDA_HOME` set and valid (points to CUDA Toolkit)
- [x] `OPTIX_ROOT` set and valid (points to OptiX SDK)
- [x] GitLab CI config valid (requires `GITLAB_ACCESS_TOKEN` env var)

### Version Consistency
- [x] Scala version matches between `build.sbt` and `.gitlab-ci.yml`
- [x] Version number consistent across all 3 files
- [x] Git tag for version doesn't already exist

### Build & Tests (~8 minutes)
- [x] Full compilation: `sbt compile`
- [x] All tests pass: `xvfb-run sbt test` (~1,070 tests)
  - 27 C++ tests (Google Test)
  - 1,043 Scala tests (AnyFlatSpec)
- [x] Code quality: `sbt "scalafix --check"`
- [x] Package builds: `sbt "mengerApp / Universal / packageBin"`

### Test Coverage (Ratchet Mechanism)
- [x] Coverage ≥ 80% (absolute minimum)
- [x] Coverage drop ≤ 1% from baseline
- [x] If dropped, must stay above 80%

**Coverage baseline:** `.coverage_baseline`

### Memory Safety (~2-3 minutes)
- [x] GPU memory leaks: `compute-sanitizer --tool memcheck`
- [x] Host memory leaks: `valgrind --leak-check=full`

### Integration Tests (27 scenarios)
- [x] Basic rendering (spheres, cubes, sponges, tesseracts)
- [x] Materials (glass, chrome, gold, matte, etc.)
- [x] Multi-object scenes (IAS)
- [x] Textures
- [x] Shadows and lighting
- [x] 4D objects and rotations
- [x] Fractional sponge levels
- [x] Headless rendering
- [x] Error handling

**Test script:** `./scripts/integration-tests.sh`

**If pre-push hook fails:**
- Read error messages carefully
- Follow **TEST FAILURE PROTOCOL** from AGENTS.md (below)
- Fix issues and re-run
- **Never bypass tests for releases**

#### TEST FAILURE PROTOCOL

When tests fail during release preparation:

1. **Investigate root cause** - Don't just make tests pass
   - Run failing test in isolation
   - Check if test expectations are wrong OR implementation is wrong
   - Trace through the code to understand the failure

2. **Determine if test or code needs fixing:**
   - If implementation is wrong: Fix the code, verify test passes
   - If test expectations are wrong: Update test, document WHY in commit message

3. **Document test changes in commit message:**
   ```
   test: Update shadow direction test expectations

   The test was checking for inverted light direction. Updated to match
   the corrected implementation where directional lights point FROM the
   light source (standard convention).

   Previous test expected: direction = normalize(light_pos - hit_point)
   Corrected test expects: direction = -light.direction
   ```

4. **Never blindly rewrite tests to pass** - This hides bugs
   - Understand the failure completely before changing anything
   - If uncertain, ask for human review

---

## Phase 3: Commit & Push via Merge Request (Manual - 5-10 minutes)

**IMPORTANT:** The main branch is protected. All changes must go through merge requests.

### Create Feature Branch

- [ ] Create release branch from main:
  ```bash
  git checkout main
  git pull
  git checkout -b release/vX.Y.Z
  ```

### Review Changes

- [ ] Review all changes:
  ```bash
  git diff main
  ```

- [ ] Verify only intended changes present

### Stage Files

**Never use `git add -A` - add files explicitly:**

```bash
git add menger-app/build.sbt
git add .gitlab-ci.yml
git add menger-app/src/main/scala/menger/MengerCLIOptions.scala
git add CHANGELOG.md
git add ROADMAP.md
git add CODE_IMPROVEMENTS.md
git add docs/arc42/  # If updated
# Add other modified files explicitly
```

### Create Release Commit

- [ ] Write clear commit message:
  ```bash
  git commit -m "release: Version X.Y.Z

  Brief summary of major changes (1-2 sentences):
  - Fixed critical shadow direction bug affecting all scenes
  - Improved CI pipeline reliability with complete git history

  Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
  ```

### Push and Create Merge Request

- [ ] Push release branch:
  ```bash
  git push origin release/vX.Y.Z
  ```

**Pre-push hook runs automatically:**
- If it fails, fix issues and push again
- Never use `--no-verify` for releases

- [ ] Create merge request:
  ```bash
  glab mr create --title "Release vX.Y.Z" --description "Release version X.Y.Z

  ## Changes
  - Updated version to X.Y.Z across all files
  - Updated CHANGELOG.md with release notes
  - Updated ROADMAP.md marking sprint complete
  - Code quality assessment in CODE_IMPROVEMENTS.md

  ## Pre-Push Validation
  - ✅ All tests pass (1,070 tests)
  - ✅ Coverage: XX.X% (baseline: XX.X%)
  - ✅ No memory leaks detected
  - ✅ All integration tests pass (27 scenarios)

  Merge to trigger release pipeline." --target-branch main
  ```

### Wait for Pipeline and Merge

- [ ] Wait for pipeline to complete (all jobs must pass)
- [ ] Review the MR (code review if working with team)
- [ ] Merge the MR when pipeline is green

**After merge to main, the release pipeline triggers automatically**

---

## Phase 4: Release Pipeline (Fully Automated - 12-15 minutes)

GitLab CI automatically runs the full release pipeline when the MR merges to main.

### Monitor Pipeline Progress

- [ ] Navigate to: https://gitlab.com/lilacashes/menger/-/pipelines
- [ ] Find the latest pipeline (should be running)
- [ ] Watch jobs progress through stages

### Pipeline Stages

**Build Stage:**
- [x] Test:Full - Full test suite with GPU
- [x] Test:OptiXJni - OptiX JNI C++ tests
- [x] Test:ComputeSanitizer - GPU memory leak detection
- [x] Test:Valgrind - Host memory leak detection
- [x] Test:OptiXIntegration - 27 integration scenarios
- [x] CheckCoverage - Coverage ratchet validation
- [x] Scalafix - Code quality enforcement

**Release Stage:**
- [x] BuildDeployable - Package distribution `.zip`
- [x] **CreateRelease** - Creates GitLab release and git tag

**Tag Pipeline (Triggered by CreateRelease):**
- [x] Upload - Upload artifact to GitLab Package Registry
- [x] PushToGithub - Mirror to GitHub repository (with GIT_DEPTH: 0)
- [x] CreateGithubRelease - Create GitHub release with CHANGELOG link

### Verify All Jobs Pass

- [ ] All jobs show green checkmarks ✅
- [ ] No failed (red ❌) or canceled (gray ⊘) jobs

**If any job fails:**
1. Click on failed job to view logs
2. Identify root cause
3. Fix issue locally
4. Create fix commit and push to release branch
5. Pipeline re-runs automatically
6. Merge MR when green

**Common failures:**
- Test failures: Follow TEST FAILURE PROTOCOL (Phase 2)
- Coverage drop: Add tests or investigate removed code
- Memory leaks: Use compute-sanitizer/valgrind locally to debug
- Integration test failures: Compare rendered output to reference images

---

## Phase 5: Post-Release Verification (Manual - 5-10 minutes)

### Verify Releases Created

**GitLab Release:**
- [ ] Navigate to: https://gitlab.com/lilacashes/menger/-/releases
- [ ] Verify release `vX.Y.Z` exists
- [ ] Check release notes include CHANGELOG link
- [ ] Verify artifact is attached

**GitHub Release:**
- [ ] Navigate to: https://github.com/lene/menger/releases
- [ ] Verify release `vX.Y.Z` exists
- [ ] Check release notes match GitLab
- [ ] Verify tag points to correct commit

### Verify Git Tags

- [ ] Fetch tags:
  ```bash
  git fetch --all --tags
  ```

- [ ] Verify tag exists:
  ```bash
  git tag | grep X.Y.Z
  ```

- [ ] Verify tag points to correct commit:
  ```bash
  git show vX.Y.Z
  ```

### Test Packaged Distribution

Download and test the release artifact:

```bash
# Download from GitLab Package Registry or GitHub Releases
wget https://gitlab.com/lilacashes/menger/-/releases/vX.Y.Z/downloads/menger-X.Y.Z.zip

# Extract
unzip menger-X.Y.Z.zip
cd menger-app-X.Y.Z

# Verify version
./bin/menger-app --version
# Should output: menger vX.Y.Z ...

# Quick smoke test (headless render)
xvfb-run ./bin/menger-app --object sphere --timeout 0.1 --level 2 --headless --save-name /tmp/smoke-test.png

# Verify output
file /tmp/smoke-test.png
# Should be: PNG image data, 800 x 600 ...
```

### Documentation Verification

- [ ] CHANGELOG.md renders correctly on GitLab and GitHub
- [ ] GitHub release shows correct CHANGELOG link (not 404)
- [ ] README.md installation instructions still accurate
- [ ] Version number appears correctly in docs

### Cleanup & Next Steps

- [ ] Update `.coverage_baseline` if coverage improved significantly
- [ ] Archive sprint documentation (if not already done)
- [ ] Create next sprint planning document (if ready)
- [ ] Clean up release branch:
  ```bash
  git branch -d release/vX.Y.Z
  git push origin --delete release/vX.Y.Z
  ```
- [ ] Announce release (if applicable):
  - GitLab/GitHub release notes
  - Internal team notification
  - Community announcement (if public)

---

## Common Issues & Solutions

### Version Mismatch Errors

**Problem:** Pre-push hook reports version inconsistency

**Solution:**
- Check all 3 files (build.sbt, .gitlab-ci.yml, MengerCLIOptions.scala)
- Ensure exact version match (no extra spaces, consistent format)
- Commit and try again

### Tag Already Exists

**Problem:** `error: tag 'vX.Y.Z' already exists`

**Solution:**
- If pushing to GitHub after GitLab release: This is normal, PushToGithub job handles it
- If genuine duplicate: Increment version number and update all 3 files

### Pipeline Failures

**Test failures:**
- Follow TEST FAILURE PROTOCOL (Phase 2)
- Investigate root cause before changing tests
- Never blindly update tests to pass

**Coverage drops:**
- Identify which code was removed/changed
- Add tests for uncovered code paths
- Or justify coverage drop in commit message

**Memory leaks:**
- Run locally: `compute-sanitizer --tool memcheck ./standalone_test`
- Fix leaks before releasing
- Update suppressions if false positive

### Protected Branch Error

**Problem:** `remote: GitLab: You are not allowed to push code to protected branch main`

**Solution:**
- This is expected and correct - main branch is protected
- Always create a feature/release branch
- Push to that branch and create an MR
- Merge MR after pipeline passes

### Pre-Push Hook Skipped

**Problem:** Push succeeded but pre-push hook didn't run

**Solution:**
- Verify hook is installed: `ls -la .git/hooks/pre-push`
- Should be symlink to: `../../.git_hooks/pre-push`
- Re-run manually: `.git_hooks/pre-push`
- If tests fail, create fix commit

---

## Emergency Hotfix Process

For critical bugs that need immediate release:

1. **Create hotfix branch from main:**
   ```bash
   git checkout main
   git pull
   git checkout -b hotfix/vX.Y.Z-critical-bug-description
   ```

2. **Make minimal fix** (smallest possible change)

3. **Test thoroughly - run full pre-push hook:**
   ```bash
   .git_hooks/pre-push
   ```

4. **Update version** (bump PATCH number):
   - Example: 0.4.3 → 0.4.4
   - Update all 3 version files

5. **Update CHANGELOG.md:**
   ```markdown
   ## [0.4.4] - 2026-02-05

   ### Fixed
   - Critical: [Description of bug and fix]
   ```

6. **Commit and push hotfix branch:**
   ```bash
   git add [files]
   git commit -m "fix: Critical bug in [component]"
   git push origin hotfix/vX.Y.Z-critical-bug-description
   ```

7. **Create merge request and merge to main after CI passes:**
   ```bash
   glab mr create --title "Hotfix vX.Y.Z: Critical bug" --target-branch main
   ```

8. **Follow normal release verification** (Phase 5)

**Hotfix criteria:**
- Production is broken or severely degraded
- Security vulnerability
- Data loss risk

**For non-critical bugs:** Follow normal sprint cycle

---

## Release Retrospective

After each release, consider documenting:

- [ ] What went well?
- [ ] What could be improved?
- [ ] Were there any unexpected issues?
- [ ] Did the timeline estimate match reality?
- [ ] Should any checklist items be added/modified?

**Update this checklist** if you discover missing steps or improvements.

---

*Last updated: 2026-02-05*
*Maintained by: Development team with Claude Code assistance*

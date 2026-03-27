# Sprint Close & New Sprint Start

Interactive workflow to close a completed sprint, verify the release, and collaboratively plan the next sprint.

**Usage:** `/sprint-close` — run after the sprint MR has been merged in the GitLab web UI.

---

## Phase 1: Gather Context

Run these commands and read the output:

```bash
git fetch origin --tags
git log origin/main --oneline -5
head -5 docs/sprints/SPRINT.md
```

Then ask the user:
> "Which sprint was just closed and what version was released (e.g. Sprint 13 / v0.5.3)?"

Store SPRINT_NUM and VERSION. Derive:
- NEXT_SPRINT_NUM = SPRINT_NUM + 1

---

## Phase 2: Post-Merge CI Verification

### 2a. GitLab Pipeline Status

Get the main SHA and recent pipelines:

```bash
git rev-parse origin/main
```

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" \
  "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines?per_page=10" \
  | python3 -c "
import sys, json
ps = json.load(sys.stdin)
for p in ps:
    print(f\"{p['iid']:4d}  {p['sha'][:8]}  {p['ref']:<45s}  {p['status']:<10s}  {p['created_at'][:16]}\")"
```

Look for and verify these pipelines all show `success` on the main SHA:
- MR pipeline (`refs/merge-requests/N/head`)
- Tag pipeline (`refs/tags/vVERSION`)
- Main branch pipeline (`main`)

If any pipeline is `running` or `pending`, wait and retry. If `failed`, stop and report the failing job before proceeding.

### 2b. PushToGithub Job

Find the tag pipeline ID from the output above, then inspect its jobs:

```bash
curl -s --header "PRIVATE-TOKEN: $GITLAB_ACCESS_TOKEN" \
  "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/pipelines/PIPELINE_ID/jobs" \
  | python3 -c "
import sys, json
jobs = json.load(sys.stdin)
for j in jobs:
    print(f\"{j['name']:<30s}  {j['status']:<10s}  {j.get('failure_reason','') or ''}\")"
```

Report the status of each job. The `PushToGithub` and `CreateGithubRelease` jobs must be `success`.

### 2c. GitHub Mirror Verification

```bash
git ls-remote github refs/heads/main
git ls-remote github refs/tags/vVERSION
```

Verify:
- `refs/heads/main` SHA matches `origin/main`
- `refs/tags/vVERSION` exists and points to same SHA

Report ✅ or ❌ for each. If GitHub mirror is wrong, suggest re-running the PushToGithub job from the GitLab UI.

---

## Phase 3: Sprint Archiving

### 3a. Archive Completed Sprint Plan

```bash
ls docs/archive/sprints/
head -3 docs/sprints/SPRINT.md
```

Move the completed sprint:

```bash
git mv docs/sprints/SPRINTSPRINT_NUM.md docs/archive/sprints/SPRINTSPRINT_NUM.md
```

(Replace SPRINT_NUM with the actual number, e.g. `git mv docs/sprints/SPRINT13.md docs/archive/sprints/SPRINT13.md`)

Update `docs/sprints/SPRINT.md` to point to the next sprint number.

### 3b. Clean docs/plans/

```bash
ls docs/plans/
```

Read each file and check if its work is complete (implemented in the archived sprint). Ask the user to confirm before removing anything that isn't clearly obsolete. Remove confirmed completed plan files with `git rm`.

### 3c. Update ROADMAP.md

```bash
grep -n "Sprint SPRINT_NUM\|vVERSION" ROADMAP.md
```

Read the relevant section and:
- Mark the completed sprint milestone as `✅ Complete`
- Update next sprint status to `🔄 In Progress`

### 3d. Check arc42 Staleness

```bash
grep "Last Updated" docs/arc42/README.md
git log --after="ARC42_DATE" --oneline -- menger-app/src optix-jni/src
```

(Use the date from the arc42 README.) If significant source changes exist since the last arc42 update, list the affected files and ask the user which sections may need updating. Do not update arc42 automatically.

### 3e. Clean CODE_IMPROVEMENTS.md

Read `CODE_IMPROVEMENTS.md` in full. Identify issues marked as resolved, complete, or fixed (look for ✅, "Resolved:", "Fixed in", "Closed"). Present the list:

> "These CODE_IMPROVEMENTS.md issues appear resolved — remove them? (y/n/all)"

Remove confirmed ones. Do not touch deferred or in-progress items.

### 3f. Version Consistency Check

```bash
grep "version" menger-app/build.sbt
grep "DEPLOYABLE_VERSION" .gitlab-ci.yml
grep "menger v" menger-app/src/main/scala/menger/MengerCLIOptions.scala
grep -m1 "version" docs/guide/user-guide.md
```

All four must agree on VERSION. Report any mismatches as ❌.

### 3g. Commit Archiving Work

```bash
git add docs/archive/sprints/
git add docs/sprints/SPRINT.md
git add ROADMAP.md
git add CODE_IMPROVEMENTS.md
git add docs/plans/
```

Commit: `docs: archive sprint SPRINT_NUM, clean up for sprint NEXT_SPRINT_NUM`

---

## Phase 4: Interactive Sprint Planning

### 4a. Load and Summarise Next Sprint

Read `docs/sprints/SPRINTNEXT_SPRINT_NUM.md` and present a summary:

> "Sprint NEXT_SPRINT_NUM — **TITLE**
> Tasks: [numbered list with estimates]
> Total estimate: ~X hours"

### 4b. Collaborative Task Review

Walk through each task and ask:
1. Still relevant? (y/n/modified)
2. Priority change needed?
3. Estimate still accurate?

Then prompt:

> "Any items from CODE_IMPROVEMENTS.md to add to Sprint NEXT_SPRINT_NUM?"

```bash
grep -E "^### [HM][0-9]" CODE_IMPROVEMENTS.md
```

> "Any technical debt from arc42 to schedule?"

```bash
grep -A2 "^### TD-" docs/arc42/11-risks-and-technical-debt.md
```

Present the agreed task list and ask:
> "Confirm final Sprint NEXT_SPRINT_NUM scope? (y/n)"

### 4c. Update Sprint Plan if Changed

If the user approved any changes, update `docs/sprints/SPRINTNEXT_SPRINT_NUM.md` to reflect the agreed scope.

### 4d. Ensure Sprint Branch Exists

We should already be on `feature/sprint-NEXT_SPRINT_NUM`. If not:

```bash
git fetch origin main
git checkout -b feature/sprint-NEXT_SPRINT_NUM origin/main
git push -u origin feature/sprint-NEXT_SPRINT_NUM
```

### 4e. Commit Sprint Kickoff

```bash
git add docs/sprints/SPRINTNEXT_SPRINT_NUM.md
```

Commit: `docs: kick off sprint NEXT_SPRINT_NUM — TITLE`

```bash
git push
```

---

## Phase 5: Summary Report

Print this report, filling in ✅ or ❌ for each item:

```
╔══════════════════════════════════════════════════════╗
║         Sprint SPRINT_NUM Close Summary              ║
╠══════════════════════════════════════════════════════╣
║ Release vVERSION                                     ║
║   GitLab MR pipeline:       ✅/❌                    ║
║   GitLab tag pipeline:      ✅/❌                    ║
║   PushToGithub job:         ✅/❌                    ║
║   GitHub mirror (SHA+tag):  ✅/❌                    ║
╠══════════════════════════════════════════════════════╣
║ Housekeeping                                         ║
║   Sprint plan archived:     ✅/❌                    ║
║   docs/plans/ cleaned:      ✅/❌  (N files removed) ║
║   ROADMAP.md updated:       ✅/❌                    ║
║   CODE_IMPROVEMENTS:        ✅/❌  (N issues removed)║
║   Version consistency:      ✅/❌                    ║
║   arc42 staleness:          ✅/⚠️ (flagged/ok)       ║
╠══════════════════════════════════════════════════════╣
║ Sprint NEXT_SPRINT_NUM Ready                         ║
║   Branch: feature/sprint-NEXT_SPRINT_NUM             ║
║   Tasks agreed: N tasks, ~X hours                    ║
╚══════════════════════════════════════════════════════╝
```

List any remaining ❌ items with specific actions needed to resolve them.

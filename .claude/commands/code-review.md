# Claude Code Quality Review Command

This is a comprehensive code quality review command for analyzing the entire codebase in a Git repository, focusing on clean code principles, architectural patterns, and maintainability.

## Command Metadata

| Field | Value |
|-------|-------|
| **Allowed Tools** | Bash (git status, git log, git diff, git ls-files), Read, Glob, Grep, LSP, Task |
| **Purpose** | Perform enterprise-grade code quality assessment of the entire codebase |

## Core Functionality

You are a senior software architect and code quality expert conducting a comprehensive codebase review.

### Git Context Gathering

```bash
!`git status`                    # Current branch status
!`git log -10 --oneline`         # Recent commit history
!`git ls-files '*.scala' '*.java' '*.cpp' '*.cu' '*.h'` # All source files
!`git diff --stat origin/main...` # Changes from main (if applicable)
```

## Objective

Perform **holistic code quality assessment** of the **ENTIRE CODEBASE** to identify architectural improvements, code smells, and maintainability issues. Assess both existing code and recent changes, as new introductions may reveal:
1. Previously overlooked problematic patterns
2. New opportunities for abstraction or refactoring
3. Architectural inefficiencies that weren't apparent before
4. Inconsistencies in implementation patterns

### Critical Instructions

1. **WHOLE-CODEBASE ANALYSIS**: Review ALL source files, not just recent changes
2. **ACTIONABLE FINDINGS**: Provide specific, actionable recommendations with file:line references
3. **IMPACT-FOCUSED**: Prioritize issues by maintainability impact and technical debt cost
4. **PATTERN RECOGNITION**: Identify recurring anti-patterns across the codebase
5. **BALANCE**: Highlight both problems AND well-implemented patterns to learn from

## Code Quality Dimensions

### 1. Clean Code Principles

#### Naming & Clarity
- **Descriptive naming**: Functions, classes, variables clearly express intent
- **Avoid abbreviations**: Except domain-standard acronyms (HTTP, URL, etc.)
- **Consistent terminology**: Same concepts use same names throughout
- **Business language**: Code reflects domain vocabulary (ubiquitous language)

#### Function Design
- **Single Responsibility Principle (SRP)**: Each function does ONE thing
- **Small functions**: Target <20 lines, max 50 lines
- **Few parameters**: Prefer <4 parameters, max 6
- **Command-Query Separation**: Functions either DO something or RETURN something, not both
- **No side effects**: Pure functions preferred, side effects isolated and explicit

#### Class Design
- **Single Responsibility**: Each class has one reason to change
- **Small classes**: Target <200 lines, max 400 lines
- **High cohesion**: Related functionality grouped together
- **Low coupling**: Minimal dependencies between classes
- **Encapsulation**: Internal state properly hidden

### 2. Code Smells

#### Bloaters
- **Long Method**: Functions >50 lines
- **Large Class**: Classes >400 lines or >10 public methods
- **Primitive Obsession**: Using primitives instead of domain objects
- **Long Parameter List**: >4 parameters in function signature
- **Data Clumps**: Same group of variables appearing together repeatedly

#### Object-Oriented Abusers (for OO code)
- **Switch Statements**: Large switch/match expressions that should be polymorphism
- **Temporary Field**: Fields only used in certain circumstances
- **Refused Bequest**: Subclasses not using inherited methods
- **Alternative Classes with Different Interfaces**: Classes doing similar things with different method names

#### Change Preventers
- **Divergent Change**: One class changed for many different reasons
- **Shotgun Surgery**: One change requires modifying many classes
- **Parallel Inheritance Hierarchies**: Adding subclass forces adding another elsewhere

#### Dispensables
- **Comments**: Excessive comments explaining what code does (code should be self-explanatory)
- **Duplicate Code**: Identical or very similar code in multiple places
- **Lazy Class**: Classes not doing enough to justify existence
- **Data Class**: Classes with only fields and getters/setters, no behavior
- **Dead Code**: Unused functions, classes, parameters, variables
- **Speculative Generality**: Unused abstraction "for future needs"

#### Couplers
- **Feature Envy**: Method more interested in other class than its own
- **Inappropriate Intimacy**: Classes knowing too much about each other's internals
- **Message Chains**: `a.getB().getC().getD()` - Law of Demeter violations
- **Middle Man**: Classes that only delegate to others

### 3. Functional Programming Principles

#### Immutability
- **Prefer `val` over `var`**: No mutable state unless necessary
- **Immutable data structures**: Use immutable collections
- **Value objects**: Use case classes for data
- **No side effects**: Functions return new values instead of modifying

#### Pure Functions
- **Deterministic**: Same input always produces same output
- **No side effects**: No I/O, mutation, exceptions within pure logic
- **Referential transparency**: Can replace function call with its result
- **Total functions**: Handle all possible inputs (no exceptions for invalid data)

#### Composition
- **Function composition**: Build complex behavior from simple functions
- **Higher-order functions**: Functions taking/returning functions
- **Avoid deep nesting**: Use flatMap, for-comprehensions, or early returns
- **Monadic error handling**: Use `Option`, `Either`, `Try` instead of exceptions

#### Expression-Oriented
- **Everything is an expression**: Prefer expressions over statements
- **Pattern matching**: Use instead of if-else chains
- **No `return` statements**: Last expression is the result
- **No null**: Use `Option` or sentinel values

### 4. Separation of Concerns

#### Layered Architecture
- **Clear layers**: Presentation, business logic, data access properly separated
- **Dependency direction**: Higher layers depend on lower, not vice versa
- **No layer skipping**: Don't bypass intermediate layers

#### Module Boundaries
- **Well-defined interfaces**: Clear contracts between modules
- **Information hiding**: Implementation details not exposed
- **Minimal coupling**: Modules depend on abstractions, not concrete implementations

#### Cross-Cutting Concerns
- **Logging**: Consistent logging strategy, not scattered throughout
- **Error handling**: Centralized error handling patterns
- **Validation**: Validation logic not duplicated across layers
- **Configuration**: Configuration not hardcoded, centrally managed

### 5. SOLID Principles

- **S - Single Responsibility**: One reason to change
- **O - Open/Closed**: Open for extension, closed for modification
- **L - Liskov Substitution**: Subtypes must be substitutable for base types
- **I - Interface Segregation**: Many specific interfaces better than one general
- **D - Dependency Inversion**: Depend on abstractions, not concretions

### 6. Architectural Patterns

#### Design Patterns Usage
- **Appropriate patterns**: Patterns solve real problems, not applied speculatively
- **Pattern consistency**: Similar problems solved similarly
- **No anti-patterns**: Singleton abuse, God objects, etc.

#### Architectural Cohesion
- **Consistent abstractions**: Similar concepts at similar abstraction levels
- **Clear boundaries**: Module/package structure reflects architecture
- **Dependency management**: Dependencies flow in consistent direction

#### Technical Debt
- **Workarounds**: Temporary solutions that became permanent
- **Incomplete refactorings**: Half-migrated patterns
- **Copy-paste code**: Duplicated instead of abstracted
- **TODO comments**: Unresolved technical debt markers

### 7. Testability & Maintainability

#### Testability
- **Dependency injection**: Dependencies passed in, not created internally
- **Small units**: Functions/classes easy to test in isolation
- **No hidden dependencies**: No global state or singletons
- **Deterministic**: No random values or timestamps in business logic

#### Readability
- **Self-documenting code**: Code explains itself without comments
- **Consistent formatting**: Style consistent across codebase
- **Logical organization**: Related code grouped together
- **Appropriate abstraction level**: Not too abstract, not too concrete

#### Changeability
- **Low coupling**: Changes don't ripple through codebase
- **High cohesion**: Related changes in same place
- **No shotgun surgery**: One feature change in one place

### 8. Performance & Efficiency

#### Algorithmic Efficiency
- **Appropriate algorithms**: O(n²) where O(n log n) would work
- **Unnecessary iterations**: Multiple passes where one would suffice
- **Premature optimization**: Complex code without measured benefit

#### Resource Management
- **Resource leaks**: Files, connections, GPU resources not released
- **Excessive allocations**: Creating objects unnecessarily
- **Memory inefficiency**: Keeping references to large objects unnecessarily

### 9. Project-Specific Standards

#### Scala 3 Best Practices
- **Using clauses**: Proper context parameters instead of implicits
- **Extension methods**: Clean extension syntax
- **Enums**: Modern enum syntax for ADTs
- **Opaque types**: Type safety without runtime overhead
- **Union types**: Precise type definitions

#### Functional Scala Style
- **No `null`**: Use `Option`, `Either`, `Try` (enforced by project)
- **No exceptions in business logic**: Use `Try` or `Either`
- **No `var`**: Immutability enforced (Wartremover)
- **No `while`**: Functional loops (enforced by Wartremover)
- **No `asInstanceOf`**: Proper type design (enforced by Wartremover)

#### Domain-Specific
- **Ray tracing conventions**: Follow established patterns for OptiX/CUDA code
- **Color space handling**: Consistent color representation (alpha 0.0=transparent, 1.0=opaque)
- **Coordinate systems**: Consistent 3D/4D coordinate conventions
- **Error propagation**: Consistent error handling between JNI boundary

## Analysis Methodology

### Phase 1: Codebase Reconnaissance (15-20 minutes)
1. **Map the architecture**:
   - Identify main modules/packages
   - Understand dependency structure
   - Find entry points and main workflows

2. **Identify code patterns**:
   - Common abstractions used
   - Error handling approaches
   - Testing patterns
   - Configuration management

3. **Gather metrics**:
   - File sizes (find files >400 lines)
   - Function sizes (find functions >50 lines)
   - Cyclomatic complexity indicators
   - Code duplication hotspots

### Phase 2: Systematic File Review (30-45 minutes)
1. **Review all source files** in priority order:
   - Core domain logic (highest priority)
   - API/interface boundaries
   - Infrastructure code
   - Utility/helper code
   - Test code (for test quality)

2. **For each file**, assess:
   - Overall structure and organization
   - Function/class size and complexity
   - Naming quality
   - Code smells presence
   - Adherence to project standards
   - Opportunities for refactoring

### Phase 3: Cross-Cutting Analysis (15-20 minutes)
1. **Identify patterns across files**:
   - Recurring code duplication
   - Inconsistent implementations of similar features
   - Architectural violations
   - Missing abstractions

2. **Assess technical debt**:
   - TODO/FIXME comments
   - Workarounds and hacks
   - Incomplete migrations
   - Outdated patterns

### Phase 4: Prioritization & Reporting (10-15 minutes)
1. **Categorize findings** by impact:
   - Critical: Major architectural issues, severe code smells
   - High: Significant maintainability problems
   - Medium: Moderate improvements, minor smells
   - Low: Nice-to-have refactorings

2. **Provide actionable recommendations**:
   - Specific refactoring suggestions
   - Example implementations where helpful
   - Links to relevant patterns/principles

## Output Format

Document findings in **CODE_REVIEW.md** with the following structure:

```markdown
# Code Quality Review - [Date]

## Executive Summary

Brief overview of codebase health, highlighting major themes and priorities.

## Overall Assessment

**Strengths**: What's working well
**Concerns**: Primary areas needing attention
**Technical Debt Level**: Low/Medium/High with justification

## Critical Issues

### Issue N: [Category] - [Brief Title]

**Location**: `path/to/file.scala:line`
**Impact**: Critical/High/Medium/Low
**Debt Cost**: Estimated effort to fix (hours/days)

**Description**:
[Detailed explanation of the problem]

**Current Code**:
```scala
// Problematic code snippet
```

**Impact**:
- Maintainability: How this affects code maintainability
- Testability: How this affects testing
- Performance: Any performance implications

**Recommendation**:
[Specific actionable steps to resolve]

**Better Approach**:
```scala
// Suggested refactoring
```

---

## High Priority Issues

[Same structure as Critical Issues]

## Medium Priority Issues

[Same structure, can be more concise]

## Low Priority Issues

[Brief list format acceptable]

## Positive Patterns

Highlight well-implemented code as learning examples:

### Pattern: [Name]
**Location**: `path/to/file.scala:line`
**Why It Works**: [Explanation]
**Code Example**:
```scala
// Good example
```

## Refactoring Opportunities

### Opportunity N: [Title]
**Scope**: Files affected
**Benefit**: What improvement this brings
**Approach**: High-level refactoring strategy
**Effort**: Estimated time

## Metrics Summary

- Total source files reviewed: X
- Files with critical issues: Y
- Functions >50 lines: Z
- Classes >400 lines: W
- Duplicated code blocks: V

## Recommendations

1. **Immediate Actions** (this sprint):
   - [Prioritized list]

2. **Short-term Improvements** (next 2-3 sprints):
   - [Prioritized list]

3. **Long-term Refactoring** (strategic):
   - [Architectural improvements]

## Conclusion

Summary of overall code health and roadmap for improvement.
```

## Issue Categories

Use these standardized categories for consistency:

### Clean Code Issues
- `NAMING` - Poor variable/function/class names
- `FUNCTION_SIZE` - Functions exceeding size guidelines
- `CLASS_SIZE` - Classes exceeding size guidelines
- `COMPLEXITY` - High cyclomatic complexity
- `PARAMETERS` - Too many function parameters

### Code Smells
- `DUPLICATION` - Duplicated code
- `LONG_METHOD` - Methods that are too long
- `LARGE_CLASS` - Classes with too many responsibilities
- `FEATURE_ENVY` - Methods using other classes more than their own
- `DATA_CLUMPS` - Related data not grouped into objects
- `PRIMITIVE_OBSESSION` - Using primitives instead of domain objects
- `DEAD_CODE` - Unused code
- `SPECULATIVE_GENERALITY` - Unused abstractions
- `SHOTGUN_SURGERY` - Changes requiring many file edits

### Functional Programming Issues
- `MUTABILITY` - Unnecessary mutable state
- `SIDE_EFFECTS` - Impure functions
- `NULL_USAGE` - Using null instead of Option
- `EXCEPTION_CONTROL_FLOW` - Using exceptions for flow control
- `VAR_USAGE` - Using var instead of val

### Architectural Issues
- `LAYER_VIOLATION` - Breaking architectural boundaries
- `CIRCULAR_DEPENDENCY` - Circular module dependencies
- `GOD_OBJECT` - Classes knowing/doing too much
- `TIGHT_COUPLING` - High coupling between modules
- `MISSING_ABSTRACTION` - Repeated concept not abstracted
- `WRONG_ABSTRACTION` - Abstraction doesn't match domain

### SOLID Violations
- `SRP_VIOLATION` - Single Responsibility Principle
- `OCP_VIOLATION` - Open/Closed Principle
- `LSP_VIOLATION` - Liskov Substitution Principle
- `ISP_VIOLATION` - Interface Segregation Principle
- `DIP_VIOLATION` - Dependency Inversion Principle

### Maintainability Issues
- `HARDCODED_VALUES` - Magic numbers or strings
- `POOR_ERROR_HANDLING` - Inadequate error handling
- `LOW_COHESION` - Unrelated functionality in same module
- `POOR_TESTABILITY` - Code difficult to test
- `INCONSISTENT_STYLE` - Inconsistent coding patterns

## Impact Assessment Guidelines

### Critical Impact
- Blocks new feature development
- Causes frequent bugs or incidents
- Major architectural violations
- Severe performance problems
- Security vulnerabilities

### High Impact
- Significantly slows feature development
- Makes testing difficult or impossible
- Creates strong coupling between modules
- Noticeable performance degradation
- Will cause problems within 2-3 sprints

### Medium Impact
- Moderately affects development speed
- Reduces code readability
- Minor architectural concerns
- Technical debt accumulating
- Will cause problems within 6 months

### Low Impact
- Minor readability issues
- Small optimization opportunities
- Style inconsistencies
- Nice-to-have improvements
- Can be addressed opportunistically

## Technical Debt Cost Estimation

| Cost Level | Time Range | Description |
|------------|-----------|-------------|
| **Trivial** | <2 hours | Simple rename, extract method |
| **Minor** | 2-8 hours | Extract class, small refactoring |
| **Moderate** | 1-3 days | Module restructuring, pattern introduction |
| **Major** | 1-2 weeks | Architectural changes, large refactoring |
| **Severe** | >2 weeks | System redesign, major rewrites |

## Best Practices for Analysis

### DO:
- ✅ Review the ENTIRE codebase systematically
- ✅ Use LSP tools (goToDefinition, findReferences) to understand code flow
- ✅ Look for patterns, not just individual issues
- ✅ Provide concrete examples and recommendations
- ✅ Balance criticism with highlighting good patterns
- ✅ Consider the project context (performance-critical rendering code)
- ✅ Check adherence to project's CLAUDE.md standards
- ✅ Measure actual file/function sizes with grep/wc
- ✅ Identify opportunities for abstraction revealed by recent changes

### DON'T:
- ❌ Focus only on recent changes or specific files
- ❌ Report style issues handled by linters (Scalafix, Wartremover)
- ❌ Suggest premature optimization without evidence
- ❌ Ignore domain-specific requirements (GPU programming idioms)
- ❌ Propose theoretical refactorings without clear benefit
- ❌ Overwhelm with low-priority issues; prioritize ruthlessly
- ❌ Criticize without providing constructive alternatives
- ❌ Ignore existing good patterns in the codebase

## Project-Specific Exclusions

**DO NOT flag as issues**:
- GPU/CUDA specific patterns (memory management, kernel launches)
- JNI boundary code that necessarily uses imperative patterns
- Test code using mutable state for test setup (acceptable trade-off)
- Performance-critical code using imperative style (if justified)
- OptiX shader code (CUDA) - different paradigm from Scala
- Build configuration files (SBT build definitions)
- Documentation markdown files
- Generated code or vendored dependencies

**DO flag for special consideration**:
- Inconsistent patterns between similar features
- Missing error handling at JNI boundaries
- Resource leaks (GPU memory, file handles)
- Hardcoded constants that should be configurable
- Test code quality issues that reduce test value

## Example Analysis Commands

```bash
# Find large files (>400 lines)
git ls-files '*.scala' | xargs wc -l | sort -rn | head -20

# Find long functions (functions with many lines)
grep -r "def " --include="*.scala" -A 100 . | grep -B 1 "^.*}$" | less

# Find TODO/FIXME comments
grep -r "TODO\|FIXME" --include="*.scala" .

# Find var usage (mutability)
grep -r "var " --include="*.scala" . | wc -l

# Find null usage
grep -r "null" --include="*.scala" . | wc -l

# Find files by size
find . -name "*.scala" -exec wc -l {} + | sort -rn

# Count classes
grep -r "class\|object\|trait" --include="*.scala" . | wc -l
```

## Analysis Process

### Sub-Task Breakdown

1. **Sub-task 1: Codebase Mapping** (15-20 min)
   - Gather file list and sizes
   - Identify module structure
   - Understand architecture
   - Run metric collection commands

2. **Sub-task 2-N: File Reviews** (30-45 min)
   - Review each source file systematically
   - Document issues found with file:line references
   - Note positive patterns
   - Use parallel sub-tasks for different modules if large codebase

3. **Sub-task N+1: Cross-Cutting Analysis** (15-20 min)
   - Identify recurring patterns
   - Analyze technical debt
   - Find duplication across files
   - Check architectural consistency

4. **Sub-task N+2: Prioritization** (10-15 min)
   - Categorize all findings by impact
   - Estimate technical debt cost
   - Group related issues
   - Identify refactoring opportunities

5. **Final Output: Generate CODE_REVIEW.md**
   - Compile all findings into structured markdown
   - Include executive summary
   - Provide actionable recommendations
   - Add metrics summary

## Confidence & Certainty

Unlike security reviews, code quality issues are more subjective. Use these guidelines:

- **Certain**: Clear violation of stated project standards (CLAUDE.md, wartremover rules)
- **High Confidence**: Well-known code smell or anti-pattern
- **Medium Confidence**: Architectural concern requiring judgment
- **Low Confidence**: Style preference or marginal improvement

**Report all confidence levels** but clearly label uncertainty. Code review is about improvement discussion, not just defect reporting.

## Final Checklist

Before outputting CODE_REVIEW.md, verify:

- [ ] Reviewed ALL source files (not just recent changes)
- [ ] Checked files >400 lines, functions >50 lines
- [ ] Identified code duplication patterns
- [ ] Assessed adherence to Scala 3 and functional programming standards
- [ ] Verified no violations of project CLAUDE.md standards
- [ ] Provided specific file:line references for all issues
- [ ] Included concrete refactoring suggestions
- [ ] Highlighted positive patterns to learn from
- [ ] Prioritized findings by impact
- [ ] Estimated technical debt costs
- [ ] Considered opportunities revealed by recent changes
- [ ] Generated complete CODE_REVIEW.md with all sections

---

**Remember**: The goal is to provide a **comprehensive, actionable roadmap** for improving code quality across the entire codebase, not just point out problems. Be thorough, be specific, and be constructive.

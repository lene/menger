# Sprint 13: Visual Quality & Material Enhancements

**Sprint:** 13 - Visual Quality Improvements
**Status:** Not Started
**Estimate:** 10-14 hours
**Branch:** `feature/sprint-13`
**Dependencies:** Sprint 10 (Scala DSL) - optional

> **Note:** This content was originally planned as Sprint 12 but was deferred when Sprint 12 was reprioritized for the t-parameter animation system.

---

## Goal

Improve visual quality and material realism with better plane materials, transparent shadows, material validation, and visual enhancements for cubes and sponges.

## Success Criteria

- [ ] Plane supports textures and materials (not just solid colors)
- [ ] Shadows work correctly with transparent/glass objects
- [ ] Material physical correctness validation and documentation
- [ ] Mixed-metallic material examples (0 < metallic < 1)
- [ ] Rounded edges on cubes and sponges (optional stretch goal)
- [ ] Updated USER_GUIDE.md with material best practices
- [ ] All tests pass (~20-25 new tests)

---

## Background

### Current State

| Feature | Status | Notes |
|---------|--------|-------|
| Sphere materials | Complete | Glass, chrome, gold, etc. working |
| Mesh materials | Complete | Cubes and sponges support materials |
| Plane rendering | Complete | Solid colors and checkerboard only |
| Transparent objects | Complete | Glass spheres render correctly |
| Shadows | Partial | Opaque objects cast shadows, transparent objects don't |
| Material presets | Complete | 15+ presets (Glass, Chrome, Gold, etc.) |
| Metallic parameter | Complete | Binary only (0.0 or 1.0) in practice |

### Gaps

1. **Plane materials:** Planes only support solid colors, no textures or materials
2. **Transparent shadows:** Glass objects should cast colored shadows, not block light completely
3. **Material validation:** No validation that materials are physically plausible
4. **Mixed metallic:** No examples of partial metallic values (e.g., 0.5)
5. **Sharp edges:** Cubes and sponges have perfectly sharp edges (unrealistic)

---

## Tasks

### Task 13.1: Plane Materials and Textures

**Estimate:** 3 hours

Enable planes to use full material system (textures, colors, materials).

#### Current Limitation

Planes only support `color` parameter:
```bash
menger --optix --plane Y,-2.0,#808080  # Solid gray plane
```

#### Proposed Enhancement

Support textures and materials:
```bash
# Textured plane
menger --optix --plane Y,-2.0 --plane-texture wood.png

# Material plane (e.g., chrome mirror floor)
menger --optix --plane Y,-2.0 --plane-material chrome

# Custom plane material
menger --optix --plane Y,-2.0 --plane-color #FFFFFF --plane-metallic 1.0 --plane-roughness 0.1
```

#### Files to Modify

- `menger-app/src/main/scala/menger/MengerCLIOptions.scala` -- add plane material options
- `optix-jni/src/main/native/include/OptiXData.h` -- add material fields to Plane struct
- `optix-jni/src/main/native/shaders/miss_plane.cu` -- update plane shader to handle materials

#### Tests to Add

- Plane with texture renders correctly
- Plane with material preset creates expected surface
- Validation: cannot specify both texture and material

---

### Task 13.2: Transparent Shadows

**Estimate:** 3.5 hours

Enable glass objects to cast colored shadows (light passes through with attenuation).

#### Current Limitation

Glass objects completely block shadows (shadow ray returns occluded).

#### Proposed Enhancement

Shadow rays through glass objects should:
1. Continue tracing through the glass
2. Attenuate light by glass color and opacity
3. Allow multiple glass layers (stack depth limit)

#### Files to Modify

- `optix-jni/src/main/native/shaders/shadows.cu` -- add anyhit shader for transparent shadow handling

#### Configuration

```bash
menger --optix --transparent-shadows  # Enable (default: disabled)
```

#### Tests to Add

- Red glass sphere casts red-tinted shadow
- Multiple glass layers produce progressive attenuation

---

### Task 13.3: Material Physical Correctness Validation

**Estimate:** 2 hours

Validate and document material parameters against real-world physics.

#### Scope

1. Review all material presets for physical plausibility
2. Add validation warnings for unrealistic combinations
3. Document reference values in USER_GUIDE.md

#### Material Properties to Validate

| Material | IOR Range | Roughness | Metallic | Notes |
|----------|-----------|-----------|----------|-------|
| Glass | 1.45-1.90 | 0.0-0.1 | 0.0 | Never metallic |
| Water | 1.33 | 0.0-0.05 | 0.0 | Low roughness |
| Diamond | 2.42 | 0.0 | 0.0 | High IOR |
| Chrome | ~1.5 | 0.0-0.2 | 1.0 | Always metallic |
| Gold | ~0.47 | 0.0-0.3 | 1.0 | Complex IOR |
| Plastic | 1.4-1.6 | 0.3-0.7 | 0.0 | Never metallic |

#### Files to Modify

- DSL `Material` -- add `validate()` method returning `Seq[String]` warnings
- `USER_GUIDE.md` -- add "Material Reference" section

---

### Task 13.4: Mixed-Metallic Material Examples

**Estimate:** 1.5 hours

Create example scenes demonstrating partial metallic values (0 < metallic < 1).

#### Background

Current presets only use `metallic=0.0` or `metallic=1.0`. Partial values create interesting hybrid materials (painted metal, rusty metal, etc.).

#### Examples to Create

- DSL scene with spheres at metallic=0.0, 0.25, 0.5, 0.75, 1.0
- Documentation of partial metallic visual effects

---

### Task 13.5: Rounded Edges on Cubes/Sponges (Stretch Goal)

**Estimate:** 4 hours

Add optional edge rounding to cube and sponge geometry for more realistic appearance.

**Note:** This is a stretch goal. If time is limited, defer to a future sprint.

#### Approach

Use chamfered/beveled edge vertices:
- Add parameter: `edge-radius=0.05` (default: 0.0 for sharp edges)
- Subdivide edges into curved sections
- Recompute normals for smooth shading

---

### Task 13.6: Documentation and Examples

**Estimate:** 1 hour

Update USER_GUIDE.md with material best practices and new features.

#### Sections to Add

1. Material Reference (from Task 13.3)
2. Partial Metallic Values (from Task 13.4)
3. Plane Materials (from Task 13.1)
4. Transparent Shadows (from Task 13.2)
5. Common Material Mistakes (validation warnings)

---

## Summary

| Task | Description | Estimate | Priority |
|------|-------------|----------|----------|
| 13.1 | Plane materials/textures | 3h | High |
| 13.2 | Transparent shadows | 3.5h | High |
| 13.3 | Material validation | 2h | Medium |
| 13.4 | Mixed-metallic examples | 1.5h | Medium |
| 13.5 | Rounded edges (stretch) | 4h | Low |
| 13.6 | Documentation | 1h | High |
| **Total** | | **11-15h** | |

---

## Definition of Done

- [ ] All success criteria met
- [ ] All tests passing
- [ ] Code quality checks pass: `sbt "scalafix --check"`
- [ ] CHANGELOG.md updated
- [ ] USER_GUIDE.md updated with material reference
- [ ] Example scenes created and tested

---

## Notes

### Implementation Order

Recommended order:

1. **Task 13.1** (Plane materials) - Foundation for realistic floors
2. **Task 13.3** (Material validation) - Prevents mistakes early
3. **Task 13.4** (Mixed-metallic examples) - Simple, high visual impact
4. **Task 13.2** (Transparent shadows) - Complex but high impact
5. **Task 13.5** (Rounded edges) - Optional, time permitting
6. **Task 13.6** (Documentation) - Last

### Testing Strategy

- **Unit tests**: Material validation logic
- **Integration tests**: Plane materials, transparent shadows
- **Visual tests**: Manual verification of shadow colors, reflections
- **Performance tests**: Ensure transparent shadows don't degrade fps

### Visual Quality Philosophy

This sprint focuses on **realism and polish**. Every feature adds subtle but important visual improvements:
- Textured floors add environmental context
- Colored shadows add realism to glass objects
- Validated materials prevent "uncanny valley" renders
- Rounded edges soften harsh geometry

---

## References

- TODO.md: Transparent shadows, plane materials, mixed-metallic examples
- Material physics: PBRT book, Chapter 8 (Materials)
- OptiX shadow rays: [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix7/guide/index.html#ray_types)

package menger.tools

import java.io.IOException
import java.lang.reflect.Method
import java.lang.reflect.Modifier
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

import scala.compiletime.constValueTuple
import scala.deriving.Mirror
import scala.util.Try
import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging
import menger.dsl.AreaLightShape
import menger.dsl.AxisHelper
import menger.dsl.Bezier
import menger.dsl.Camera
import menger.dsl.CameraPath
import menger.dsl.Caustics
import menger.dsl.Color
import menger.dsl.DenoiseMode
import menger.dsl.FieldConstraints
import menger.dsl.Fog
import menger.dsl.IBL
import menger.dsl.Light
import menger.dsl.Material
import menger.dsl.Placement
import menger.dsl.Plane
import menger.dsl.ProceduralPresets
import menger.dsl.RenderSettings
import menger.dsl.Scene
import menger.dsl.SceneNode
import menger.dsl.SceneObject
import menger.dsl.SpongeType
import menger.dsl.TesseractSpongeType
import menger.dsl.Transform
import menger.dsl.TransparentShadows
import menger.dsl.Vec3
import menger.scene.RestrictedClasspath
import upickle.default.ReadWriter
import upickle.default.write

/** Generates a versioned JSON capability manifest describing menger's DSL surface (CAP-7).
  *
  * Every list in the emitted manifest is derived by reflecting over the real DSL types in
  * `menger.dsl` at generation time -- `SceneObject`'s sealed-trait subtypes, `Material`'s
  * declared preset vals, and so on. Nothing here is a hand-maintained duplicate of the DSL:
  * a new `SceneObject` subtype or `Material` preset is picked up on the next run with no
  * change to this file. `menger.dsl` itself is never modified -- only introspected.
  *
  * Usage: `sbt "mengerApp/runMain menger.tools.ManifestGenerator [output-path]"`.
  * Defaults to `target/dsl-manifest.json` when no path is given.
  */
object ManifestGenerator extends LazyLogging:

  // 1.1.0: field `description`s and top-level `conventions` from DslSemantics (F28).
  // 1.2.0: field `min`/`max` from DslSemantics (T1#3), backed by menger.dsl.ResourceLimits.
  // 1.3.0: field `warnAt` (F36), procedural preset descriptions (F37/F40) and 4D
  //        colouring/texture/projection conventions (F43, menger#21, menger#52).
  // 1.4.0: field `limitsBy` (per-`spongeType` level bounds, F59), 4D rotation units in degrees
  //        (F74), refractive-colour/caustics/duration/4D-size conventions (F66, F69, F84, F72).
  // 1.5.0: `val durationSeconds` (F84, `duration` deprecated) and `Projection4DSpec.wScale`
  //        (menger#65) in the conventions.
  // 1.6.0: procedural preset 11 xyz_rgb_local (F63) and the transparent-shadow convention (F67),
  //        optix-jni 0.4.6.
  // 1.7.0: a level limit per 3D sponge type, CubeSponge left out (deprecated) (F86, #76);
  //        edge-tube level limits, caustics visibility, the 4D projection-size formula and the
  //        preview playback keys in the notes (F94, F98, F101, F92).
  // 1.8.0: the window's playback modes and idle pause in the animation convention (F95, #71, #75).
  // 1.9.0: fractional refractive levels render both levels (render cost in the Sponge level
  //        note), and edge tubes of a fractional level blend up to 2.x (#69, #70).
  // 1.10.0: the scene-wide budget convention; max for RenderSettings.accumulation and
  //         ParametricSurface uSteps/vSteps (menger#80, #106).
  // 1.11.0: `contentRevision` (the DSL's identity, menger#83), `proceduralPresets`, `productCaps`,
  //         `compileSurface`, type `traits`, material `transparent`, and a `min`/`max` for every
  //         bounded field (FieldConstraints, menger#114, msa#28). From here `schemaVersion`
  //         describes the JSON shape only: additive changes are a minor bump, and
  //         ManifestShapeSuite fails when the shape changes without one.
  private[tools] val SchemaVersion = "1.11.0"

  // Toolchain version pins (Always rule: no sbt-buildinfo -- a hardcoded constant is enough).
  // Keep in sync with menger-app/build.sbt (scalaVersion), build.sbt (optixJniDependency),
  // and the architecture spine's Stack table (minimum driver version).
  private val ScalaVersionPin = "3.8.3"
  private val OptixJniVersionPin = "0.5.0"
  private val MinDriverVersion = "580.65"

  private val DefaultOutputPath = "target/dsl-manifest.json"

  private val ConstructorDefaultPattern = """^\$lessinit\$greater\$default\$(\d+)$""".r
  private val MethodDefaultPattern = """^(.+)\$default\$(\d+)$""".r

  /** Bounds of one value of a discriminating field (`limitsBy`). */
  case class BoundsManifest(
    min: Option[Double] = None,
    max: Option[Double] = None,
    warnAt: Option[Double] = None
  ) derives ReadWriter

  /** Field bounds that depend on another field's value: `field` names the discriminating
    * field (`spongeType`), `values` maps each of its values to the bounds that apply. The
    * field's own `min`/`max`/`warnAt` stay as the conservative default. */
  case class LimitsByManifest(field: String, values: Map[String, BoundsManifest])
      derives ReadWriter

  case class FieldManifest(
    name: String,
    `type`: String,
    default: Option[String],
    description: Option[String] = None,
    min: Option[Double] = None,
    max: Option[Double] = None,
    warnAt: Option[Double] = None,
    limitsBy: Option[LimitsByManifest] = None
  ) derives ReadWriter
  /** `traits`: what the agent needs to know about a type beyond its fields -- `holed` (an object
    * with holes through it never occludes), `positioned` (a light placed by `position`). The
    * agent kept these lists itself (review 2026-10-14, msa#28). */
  case class TypeManifest(name: String, fields: List[FieldManifest], traits: List[String] = Nil)
      derives ReadWriter

  /** One `proceduralType` preset, structured (menger#81: the agent parsed them out of prose). */
  case class PresetManifest(id: Int, name: String, effect: String, note: String) derives ReadWriter

  /** A cap on the product of fields of one type, which per-field bounds cannot express. */
  case class ProductCapManifest(`type`: String, fields: List[String], max: Double)
      derives ReadWriter

  /** What a scene can compile against in menger-app: these packages and root types (see
    * `RestrictedClasspath`), so the agent's advisory allowlist is derived, not hand-kept. */
  case class CompileSurfaceManifest(packages: List[String], types: List[String])
      derives ReadWriter

  /** An enum's admissible values. Without these the manifest names a field's *type*
    * (`spongeType : menger.dsl.TesseractSpongeType`) but never what may be written there, and
    * for the mandatory-argument enums that makes the flagship 4D objects unconstructable from
    * the manifest alone -- CAP-7's "absence is decidable" fails on the very capability the
    * MVP acceptance scene needs (review round 2). */
  case class EnumManifest(name: String, values: List[String]) derives ReadWriter
  case class MethodManifest(name: String, parameters: List[FieldManifest]) derives ReadWriter
  case class MaterialManifest(
    name: String,
    ior: Float,
    roughness: Float,
    metallic: Float,
    specular: Float,
    emission: Float,
    filmThickness: Float,
    dispersion: Float,
    // Light passes through it (alpha below 1, or refractive): what the agent's occlusion
    // checks need, kept by hand there until msa#28.
    transparent: Boolean
  ) derives ReadWriter

  case class DslManifest(
    schemaVersion: String,
    contentRevision: String,
    scalaVersion: String,
    optixJniVersion: String,
    minDriverVersion: String,
    conventions: List[String],
    objects: List[TypeManifest],
    enums: List[EnumManifest],
    sceneComposition: List[TypeManifest],
    materials: List[MaterialManifest],
    lights: List[TypeManifest],
    placement: List[MethodManifest],
    colorPresets: List[String],
    plane: TypeManifest,
    planeAxes: List[String],
    cameraPath: TypeManifest,
    cameraPathHelpers: List[MethodManifest],
    renderSettings: TypeManifest,
    renderSettingsPresets: List[String],
    proceduralPresets: List[PresetManifest],
    productCaps: List[ProductCapManifest],
    compileSurface: CompileSurfaceManifest
  ) derives ReadWriter

  def main(args: Array[String]): Unit =
    run(args) match
      case Right(()) => ()
      case Left(message) => ManifestGeneratorMain.reportFailureAndExit(message)

  /** Pure-ish entry point (the only I/O is the manifest write) -- kept separate from `main`
    * so tests can observe failures without triggering `sys.exit`. `buildManifest()` runs a
    * lot of reflection over `menger.dsl`; a `NonFatal` guard turns any failure there (a
    * missing class, a constructor-shape assumption that no longer holds, etc.) into the
    * same clear `Left(message)` shape `writeManifest`'s `IOException` handling already uses,
    * instead of an uncaught exception escaping `run`. */
  def run(args: Array[String]): Either[String, Unit] =
    val outputPath = resolveOutputPath(args)
    try
      writeManifest(buildManifest(), outputPath)
    catch
      case NonFatal(e) =>
        logger.error("Failed to build the DSL capability manifest via reflection", e)
        Left(s"Error: failed to build the DSL capability manifest: ${e.getMessage}")

  def resolveOutputPath(args: Array[String]): String =
    args.headOption.getOrElse(DefaultOutputPath)

  // ---------------------------------------------------------------------
  // Reflection helpers
  // ---------------------------------------------------------------------

  private def companionInstance(clazz: Class[?]): AnyRef =
    val companionClass = Class.forName(clazz.getName + "$")
    // Field.get's argument is ignored for a static field (per its Javadoc) -- passing the
    // class itself avoids a bare `null` literal (DisableSyntax.null) with no behavior change.
    companionClass.getField("MODULE$").get(companionClass)

  /** Field metadata for one case class, derived from its primary constructor.
    * Field names and generic types come from `java.lang.reflect` on the constructor
    * (Scala 3 emits real parameter names by default); default values are recovered
    * from the compiler-generated `<init>$default$N` methods on the companion object --
    * never a hand-maintained override (per the Always rule). A parameter with no such
    * method (e.g. `Sponge`'s `level`) is a genuinely required field: `default` is `None`.
    */
  private def fieldsOf(clazz: Class[?]): List[FieldManifest] =
    val ctors = clazz.getDeclaredConstructors
    require(ctors.length == 1, s"${clazz.getName} must have exactly one constructor, found ${ctors.length}")
    val ctor = ctors.head
    val companion = companionInstance(clazz)
    val defaults: Map[Int, Method] =
      companion.getClass.getDeclaredMethods.toList.flatMap { m =>
        m.getName match
          case ConstructorDefaultPattern(pos) => Some(pos.toInt -> m)
          case _ => None
      }.toMap
    ctor.getParameters.toList.zipWithIndex.map { case (p, i) =>
      // A default that *depends on earlier parameters* compiles to a `$default$N` method
      // taking those parameters, so invoking it with no arguments throws
      // IllegalArgumentException and (before review round 2) aborted the whole manifest. Such
      // a default cannot be rendered as a standalone literal anyway: report it as "no static
      // default" rather than failing generation.
      val default = defaults.get(i + 1)
        .filter(_.getParameterCount == 0)
        .flatMap(m => Try(String.valueOf(m.invoke(companion))).toOption)
      val (min, max) = DslSemantics.limitsOf(clazz.getSimpleName, p.getName)
      FieldManifest(
        p.getName,
        p.getParameterizedType.getTypeName,
        default,
        DslSemantics.descriptionOf(clazz.getSimpleName, p.getName),
        min,
        max,
        DslSemantics.warnAtOf(clazz.getSimpleName, p.getName),
        DslSemantics.limitsBySubtypeOf(clazz.getSimpleName, p.getName).map { bySubtype =>
          LimitsByManifest(
            bySubtype.field,
            bySubtype.values.map((value, b) => value -> BoundsManifest(b.min, b.max, b.warnAt))
          )
        }
      )
    }

  private def typeManifestOf(clazz: Class[?]): TypeManifest =
    TypeManifest(clazz.getSimpleName, fieldsOf(clazz))

  /** Case-class labels of a sealed trait's direct children, from the compiler-derived
    * `Mirror.SumOf` -- never a hand-maintained list. A new subtype added to the sealed
    * trait in `menger.dsl` appears here automatically on the next compile. */
  private inline def sealedSubtypeNames[T](using m: Mirror.SumOf[T]): List[String] =
    constValueTuple[m.MirroredElemLabels].toList.map(_.toString)

  /** Resolves a sealed-trait child's runtime class. A `case object` child compiles to
    * `menger.dsl.<Name>$`, not `menger.dsl.<Name>`, so the plain lookup throws
    * `ClassNotFoundException` and -- before review round 2 -- took the entire manifest down
    * the first time anyone added one. */
  private def dslClassOf(name: String): Class[?] =
    Try(Class.forName(s"menger.dsl.$name")).getOrElse(Class.forName(s"menger.dsl.$name$$"))

  /** Per-type failures are reported with the offending type named, rather than one bad DSL
    * type aborting generation with a message that never says which one (review round 2). */
  private def objectManifestsOf(names: List[String]): List[TypeManifest] =
    val (failures, manifests) = names
      .map(n => Try(typeManifestOf(dslClassOf(n))).toEither.left.map(e => s"$n: ${e.getMessage}"))
      .partitionMap(identity)
    require(failures.isEmpty, s"could not derive a manifest for: ${failures.mkString("; ")}")
    manifests

  /** Names of zero-arg methods on `obj`'s class whose return type is `returnType` -- the
    * reflective shape of a preset `val` (e.g. `Material.Glass`, `RenderSettings.Default`,
    * `Color.White`): each compiles to a synthetic zero-arg getter on the companion module. */
  private def zeroArgPresetNames(obj: AnyRef, returnType: Class[?]): List[String] =
    obj.getClass.getDeclaredMethods.toList
      // `getDeclaredMethods` includes non-public members, so a private helper val of the same
      // type was advertised to the agent as a preset and then failed the `getMethod` lookup
      // that reads it (review round 2).
      .filter(m => Modifier.isPublic(m.getModifiers))
      .filter(m => m.getParameterCount == 0 && returnType.isAssignableFrom(m.getReturnType))
      .map(_.getName)
      .filterNot(_.contains("$"))
      .distinct
      .sorted

  // The Float-valued fields MaterialManifest's fixed JSON shape names, plus `color`, which
  // isn't a Float and is deliberately left off the manifest here (materials are already
  // identified by name; color is Material-specific extra detail this manifest doesn't need).
  // This is the target shape only -- the *values* below are never hand-read off `m` (e.g.
  // `m.ior`); they come from `Product`, the same compiler-derived introspection every case
  // class carries (`productElementNames`/`productIterator`, index-aligned by the Scala
  // language spec). The `require` below is the guard against the gap that leaves: if
  // `menger.dsl.Material` ever gains, loses, or renames a field, its `productElementNames`
  // stops matching this set and generation fails loudly here instead of the new field
  // silently never appearing in the manifest.
  private val MaterialManifestFloatFields =
    Set("ior", "roughness", "metallic", "specular", "emission", "filmThickness", "dispersion")

  private def materialManifestOf(name: String): MaterialManifest =
    Material.getClass.getMethod(name).invoke(Material) match
      case m: Material =>
        val values: Map[String, Any] = m.productElementNames.zip(m.productIterator).toMap
        require(
          values.keySet == MaterialManifestFloatFields + "color",
          "menger.dsl.Material's fields changed (now " +
            s"${values.keySet.toList.sorted.mkString(", ")}) -- update MaterialManifest and " +
            "MaterialManifestFloatFields in ManifestGenerator to match"
        )
        def floatValue(field: String): Float = values(field) match
          case f: Float => f
          case other => sys.error(s"Expected Float for Material.$field, got ${other.getClass.getName}")
        MaterialManifest(
          name,
          ior = floatValue("ior"),
          roughness = floatValue("roughness"),
          metallic = floatValue("metallic"),
          specular = floatValue("specular"),
          emission = floatValue("emission"),
          filmThickness = floatValue("filmThickness"),
          dispersion = floatValue("dispersion"),
          // Alpha below 1 lets light through; an opaque material's IOR (Plastic's 1.5) only
          // shapes its reflection.
          transparent = m.color.a < 1f
        )
      case other =>
        sys.error(s"Expected menger.dsl.Material for preset '$name', got ${other.getClass.getName}")

  private val IgnoredObjectMethodNames =
    Set("getClass", "hashCode", "equals", "toString", "notify", "notifyAll", "wait")

  /** Public, non-synthetic methods of a DSL `object` (e.g. `Placement`, `Bezier`) -- the
    * reflective stand-in for vocabulary that is functions rather than case-class fields.
    * Per-parameter defaults are recovered the same way as `fieldsOf`, from the
    * compiler-generated `<methodName>$default$N` methods (e.g. `Placement.scatter`'s seed).
    *
    * A default method's own signature carries no owning-overload marker beyond its name,
    * position and return type -- so the lookup key includes the parameter's type name
    * alongside the method name and position (rather than just `(name, position)`), and a
    * default is only applied when that type matches the target parameter's type. A default
    * method's return type always equals the defaulted parameter's declared type by
    * construction, so this is exact for today's non-overloaded `Placement`/`Bezier`; it also
    * means a same-name, same-position default from a *differently typed* future overload
    * can no longer be silently misattributed to the wrong parameter. */
  private def methodsOf(obj: AnyRef): List[MethodManifest] =
    val allMethods = obj.getClass.getDeclaredMethods.toList
    val defaults: Map[(String, String, Int), Method] =
      allMethods.flatMap { m =>
        m.getName match
          case MethodDefaultPattern(methodName, pos) =>
            Some((methodName, m.getReturnType.getTypeName, pos.toInt) -> m)
          case _ => None
      }.toMap
    allMethods
      .filter(m => Modifier.isPublic(m.getModifiers))
      .filterNot(_.getName.contains("$"))
      .filterNot(m => IgnoredObjectMethodNames.contains(m.getName))
      .sortBy(_.getName)
      .map { m =>
        val params = m.getParameters.toList.zipWithIndex.map { case (p, i) =>
          val typeName = p.getParameterizedType.getTypeName
          val default = defaults.get((m.getName, typeName, i + 1)).map(d => String.valueOf(d.invoke(obj)))
          FieldManifest(p.getName, typeName, default)
        }
        MethodManifest(m.getName, params)
      }

  // ---------------------------------------------------------------------
  // Manifest assembly
  // ---------------------------------------------------------------------

  private def withTraits(t: TypeManifest, traits: List[(String, TypeManifest => Boolean)]) =
    t.copy(traits = traits.collect { case (name, applies) if applies(t) => name })

  private val objectTraits: List[(String, TypeManifest => Boolean)] =
    List("holed" -> (t => DslSemantics.holedTypes.contains(t.name)))

  private val lightTraits: List[(String, TypeManifest => Boolean)] =
    List("positioned" -> (t => t.fields.exists(_.name == "position")))

  /** What identifies the DSL a scene is written against (review 2026-10-14, menger#83): a hash
    * of the manifest without its version fields, so a pin bump (optix-jni, Scala) no longer
    * makes the agent refuse an otherwise identical validator image, as the whole-file hash did.
    * `schemaVersion` describes the JSON shape only. */
  private[tools] def contentRevisionOf(manifest: DslManifest): String =
    val content = write(manifest.copy(
      schemaVersion = "", contentRevision = "", scalaVersion = "", optixJniVersion = "",
      minDriverVersion = ""
    ))
    java.security.MessageDigest.getInstance("SHA-256")
      .digest(content.getBytes(java.nio.charset.StandardCharsets.UTF_8))
      .map(b => f"$b%02x").mkString

  private[tools] def buildManifest(): DslManifest =
    val manifest = buildManifestWithoutRevision()
    manifest.copy(contentRevision = contentRevisionOf(manifest))

  private def buildManifestWithoutRevision(): DslManifest =
    DslManifest(
      schemaVersion = SchemaVersion,
      contentRevision = "",
      scalaVersion = ScalaVersionPin,
      optixJniVersion = OptixJniVersionPin,
      minDriverVersion = MinDriverVersion,
      conventions = DslSemantics.conventions,
      objects = objectManifestsOf(sealedSubtypeNames[SceneObject]).map(withTraits(_, objectTraits)),
      enums = List(
        EnumManifest(
          "SpongeType",
          sealedSubtypeNames[SpongeType].filterNot(DslSemantics.deprecatedEnumValues)
        ),
        EnumManifest("TesseractSpongeType", sealedSubtypeNames[TesseractSpongeType]),
        EnumManifest("AreaLightShape", sealedSubtypeNames[AreaLightShape]),
        EnumManifest("DenoiseMode", sealedSubtypeNames[DenoiseMode]),
        EnumManifest("TransparentShadows", sealedSubtypeNames[TransparentShadows])
      ),
      // `dsl-surface.md`'s "Remaining surface": the scene-composition and environment types an
      // agent needs to assemble a scene at all. `Caustics` in particular is that document's
      // own worked example of the tacit knowledge this manifest exists to supply, and was
      // absent entirely (review round 2).
      sceneComposition = List(
        typeManifestOf(classOf[Camera]),
        typeManifestOf(classOf[Scene]),
        typeManifestOf(classOf[SceneNode]),
        typeManifestOf(classOf[Transform]),
        typeManifestOf(classOf[Caustics]),
        typeManifestOf(classOf[Fog]),
        typeManifestOf(classOf[IBL]),
        typeManifestOf(classOf[Vec3])
      ),
      materials = zeroArgPresetNames(Material, classOf[Material]).map(materialManifestOf),
      lights = objectManifestsOf(sealedSubtypeNames[Light]).map(withTraits(_, lightTraits)),
      placement = methodsOf(Placement),
      colorPresets = zeroArgPresetNames(Color, classOf[Color]),
      plane = typeManifestOf(classOf[Plane]),
      planeAxes = sealedSubtypeNames[AxisHelper],
      cameraPath = typeManifestOf(classOf[CameraPath]),
      cameraPathHelpers = methodsOf(Bezier),
      renderSettings = typeManifestOf(classOf[RenderSettings]),
      renderSettingsPresets = zeroArgPresetNames(RenderSettings, classOf[RenderSettings]),
      proceduralPresets = ProceduralPresets.All.map(p =>
        PresetManifest(p.id, p.name, p.effect.wire, p.note)),
      productCaps = FieldConstraints.products.map(c =>
        ProductCapManifest(c.typeName, c.fields, c.max)),
      compileSurface = CompileSurfaceManifest(
        RestrictedClasspath.ScenePackages.toList.sorted.map("menger." + _),
        List.empty // no root-package types any more; the field goes with manifest 2.0.0
      )
    )

  private def writeManifest(manifest: DslManifest, outputPath: String): Either[String, Unit] =
    try
      val path: Path = Paths.get(outputPath)
      Option(path.getParent).foreach(Files.createDirectories(_))
      // SA-AD-14: this artifact is mounted read-only into the agent domain, so a reader must
      // never observe a partial file. Write beside the target and rename into place -- a
      // rename is atomic within a filesystem, `Files.writeString` straight onto the target is
      // not (review round 2).
      val tmp = Files.createTempFile(
        Option(path.getParent).getOrElse(Paths.get(".")), ".dsl-manifest-", ".json.tmp"
      )
      Files.writeString(tmp, write(manifest, indent = 2))
      Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
      Right(())
    catch
      case e: IOException =>
        logger.error(s"Failed to write manifest to $outputPath", e)
        Left(s"Error: failed to write manifest to '$outputPath': ${e.getMessage}")
      case e: InvalidPathException =>
        logger.error(s"Invalid manifest output path: $outputPath", e)
        Left(s"Error: invalid output path '$outputPath': ${e.getMessage}")

/** Isolates the two process-level side effects `ArchitectureSpec` restricts to classes whose
  * name matches `.*Main.*` (writing to stderr, calling `sys.exit`) -- the same convention the
  * top-level `Main.scala` bootstrap follows. `ManifestGenerator.main` stays the sbt `runMain`
  * entry point required by this tool's acceptance criteria; only the exit side effect on
  * failure is delegated here. */
private object ManifestGeneratorMain extends LazyLogging:
  def reportFailureAndExit(message: String): Unit =
    logger.error(message)
    sys.exit(1)

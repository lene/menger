package menger.tools

import java.io.IOException
import java.lang.reflect.Method
import java.lang.reflect.Modifier
import java.nio.file.Files
import java.nio.file.InvalidPathException
import java.nio.file.Path
import java.nio.file.Paths

import scala.compiletime.constValueTuple
import scala.deriving.Mirror
import scala.util.control.NonFatal

import com.typesafe.scalalogging.LazyLogging
import menger.dsl.AxisHelper
import menger.dsl.Bezier
import menger.dsl.CameraPath
import menger.dsl.Color
import menger.dsl.Light
import menger.dsl.Material
import menger.dsl.Placement
import menger.dsl.Plane
import menger.dsl.RenderSettings
import menger.dsl.SceneObject
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

  private val SchemaVersion = "1.0.0"

  // Toolchain version pins (Always rule: no sbt-buildinfo -- a hardcoded constant is enough).
  // Keep in sync with menger-app/build.sbt (scalaVersion), build.sbt (optixJniDependency),
  // and the architecture spine's Stack table (minimum driver version).
  private val ScalaVersionPin = "3.8.3"
  private val OptixJniVersionPin = "0.3.3"
  private val MinDriverVersion = "580.65"

  private val DefaultOutputPath = "target/dsl-manifest.json"

  private val ConstructorDefaultPattern = """^\$lessinit\$greater\$default\$(\d+)$""".r
  private val MethodDefaultPattern = """^(.+)\$default\$(\d+)$""".r

  case class FieldManifest(name: String, `type`: String, default: Option[String]) derives ReadWriter
  case class TypeManifest(name: String, fields: List[FieldManifest]) derives ReadWriter
  case class MethodManifest(name: String, parameters: List[FieldManifest]) derives ReadWriter
  case class MaterialManifest(
    name: String,
    ior: Float,
    roughness: Float,
    metallic: Float,
    specular: Float,
    emission: Float,
    filmThickness: Float,
    dispersion: Float
  ) derives ReadWriter

  case class DslManifest(
    schemaVersion: String,
    scalaVersion: String,
    optixJniVersion: String,
    minDriverVersion: String,
    objects: List[TypeManifest],
    materials: List[MaterialManifest],
    lights: List[TypeManifest],
    placement: List[MethodManifest],
    colorPresets: List[String],
    plane: TypeManifest,
    planeAxes: List[String],
    cameraPath: TypeManifest,
    cameraPathHelpers: List[MethodManifest],
    renderSettings: TypeManifest,
    renderSettingsPresets: List[String]
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
      val default = defaults.get(i + 1).map(m => String.valueOf(m.invoke(companion)))
      FieldManifest(p.getName, p.getParameterizedType.getTypeName, default)
    }

  private def typeManifestOf(clazz: Class[?]): TypeManifest =
    TypeManifest(clazz.getSimpleName, fieldsOf(clazz))

  /** Case-class labels of a sealed trait's direct children, from the compiler-derived
    * `Mirror.SumOf` -- never a hand-maintained list. A new subtype added to the sealed
    * trait in `menger.dsl` appears here automatically on the next compile. */
  private inline def sealedSubtypeNames[T](using m: Mirror.SumOf[T]): List[String] =
    constValueTuple[m.MirroredElemLabels].toList.map(_.toString)

  private def objectManifestsOf(names: List[String]): List[TypeManifest] =
    names.map(n => typeManifestOf(Class.forName(s"menger.dsl.$n")))

  /** Names of zero-arg methods on `obj`'s class whose return type is `returnType` -- the
    * reflective shape of a preset `val` (e.g. `Material.Glass`, `RenderSettings.Default`,
    * `Color.White`): each compiles to a synthetic zero-arg getter on the companion module. */
  private def zeroArgPresetNames(obj: AnyRef, returnType: Class[?]): List[String] =
    obj.getClass.getDeclaredMethods.toList
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
          dispersion = floatValue("dispersion")
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

  private def buildManifest(): DslManifest =
    DslManifest(
      schemaVersion = SchemaVersion,
      scalaVersion = ScalaVersionPin,
      optixJniVersion = OptixJniVersionPin,
      minDriverVersion = MinDriverVersion,
      objects = objectManifestsOf(sealedSubtypeNames[SceneObject]),
      materials = zeroArgPresetNames(Material, classOf[Material]).map(materialManifestOf),
      lights = objectManifestsOf(sealedSubtypeNames[Light]),
      placement = methodsOf(Placement),
      colorPresets = zeroArgPresetNames(Color, classOf[Color]),
      plane = typeManifestOf(classOf[Plane]),
      planeAxes = sealedSubtypeNames[AxisHelper],
      cameraPath = typeManifestOf(classOf[CameraPath]),
      cameraPathHelpers = methodsOf(Bezier),
      renderSettings = typeManifestOf(classOf[RenderSettings]),
      renderSettingsPresets = zeroArgPresetNames(RenderSettings, classOf[RenderSettings])
    )

  private def writeManifest(manifest: DslManifest, outputPath: String): Either[String, Unit] =
    try
      val path: Path = Paths.get(outputPath)
      Option(path.getParent).foreach(Files.createDirectories(_))
      Files.writeString(path, write(manifest, indent = 2))
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

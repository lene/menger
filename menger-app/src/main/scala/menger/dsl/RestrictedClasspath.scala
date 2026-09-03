package menger.dsl

import java.io.File
import java.net.URLClassLoader

import scala.util.Try

import com.typesafe.scalalogging.LazyLogging

/** Scopes the compile-time classpath `SceneCompiler.compile` hands to the Scala 3 compiler
  * down to the DSL surface and its transitive needs (AD-4 rule 2), replacing the previous
  * unrestricted classpath -- every jar and directory reachable from the current JVM's
  * classloader chain, including test frameworks, LibGDX/LWJGL, the native OptiX JNI bridge,
  * and the sbt/coursier build toolchain.
  *
  * AD-4 rule 2's own text concedes this restriction "cannot exclude the JVM bootclasspath or
  * `scala-library.jar`" -- it narrows which *library jars* are visible to the compiler while
  * it typechecks a scene file; it does not achieve per-class isolation within `menger-app`'s
  * own compiled output (`menger.dsl`, `menger.objects`, `menger.engines`, `menger.tools`, ...
  * all still share one `classes` directory -- splitting that would need a source-set-level
  * restructuring of the build, out of this story's scope). AD-18's container sandbox is what
  * backstops the rest.
  *
  * The include list below was determined empirically, not guessed: compiling every one of
  * `examples/dsl`'s 33 example scenes (glass/caustics, procedural surfaces, curves, video
  * textures, 4D tesseracts/sponges/pentachora, materials, lighting) against candidate
  * restricted classpaths and observing what broke. Two results were not obvious in advance:
  *   - `menger-geometry`'s compiled classes (the native OptiX/JNI + 4D-caustics module) turned
  *     out *not* to be needed -- no `menger.dsl`/`menger.objects` source references
  *     `io.github.lene.optix` or `menger.geometry` directly (confirmed by source grep), and
  *     every example compiled successfully without that classes directory on the classpath.
  *   - `scala-logging_3` *is* needed, despite no scene file ever importing it directly:
  *     several DSL case classes (`SceneObject`'s subtypes, e.g. `ParametricSurface`) extend
  *     `LazyLogging`, and the Scala 3 compiler must fully resolve that supertype to typecheck
  *     any reference to those classes. Omitting it does not produce a normal compile error --
  *     it crashes the compiler with `AssertionError: class X has non-class parent` deep in
  *     base-class linearization (reproduced against `ParametricScenes.scala` and
  *     `CausticsMeshValidation.scala`). This is exactly the "Ask First" ambiguity the spec
  *     anticipated: resolved empirically here, since the failure mode (an opaque compiler
  *     crash, not "class not found") would otherwise be easy to misdiagnose as unrelated.
  *   - `slf4j-api` (scala-logging's own compile dependency) and `tasty-core_3` (needed by the
  *     compiler *itself* to read `.tasty` metadata -- already present on the compiler's own
  *     invocation classpath, not on the target file's `-classpath`) were tried and found
  *     genuinely unnecessary against all 33 examples -- left out to keep the restriction as
  *     tight as the empirical evidence supports.
  *
  * Excluded, all confirmed non-essential by the same sweep: test frameworks (ScalaTest,
  * ScalaMock, ScalaCheck, JUnit/Jupiter, ArchUnit), LibGDX/LWJGL and its native/audio jars,
  * `optix-jni` and `menger-geometry`'s native JNI bridge, `upickle`/`ujson`/`upack`/`geny`
  * (menger.tools' JSON manifest concern, not the DSL's), `scallop` (CLI parsing), `logback-
  * classic`/`logback-core` (the SLF4J *implementation* -- compile-time symbol resolution only
  * needs the API surface scala-logging itself pulls in; the runtime JVM that later loads and
  * executes the compiled scene already has logback on its own unrestricted classloader, via
  * `SceneLoader`'s `URLClassLoader(..., Thread.currentThread.getContextClassLoader)` parent),
  * the Scala 3 compiler's own jars (`scala3-compiler`, `scala3-interfaces`, `compiler-
  * interface`, `scala-asm` -- a scene file never imports `dotty.tools.*`), and any sbt/
  * coursier/zinc build-tool-only jars incidentally on the invoking JVM's classpath.
  */
object RestrictedClasspath extends LazyLogging:

  /** Matched against a classpath directory entry's absolute path -- only `menger-app`'s own
    * compiled *main* output directory (never `test-classes`, never `menger-geometry`'s). This
    * is the layout `sbt test`/`sbt run` (unforked, or forked via the plain `Test`/`Compile`
    * classpath -- confirmed empirically to be what `sbt testOnly` actually uses) presents;
    * see the jar patterns below for the *other* two layouts this project's own build produces.
    */
  private val ProjectClassesDir = """.*/menger-app/target/[^/]+/classes$""".r

  /** Matched against a classpath jar entry's file name, independent of version (so a
    * dependency bump doesn't silently fall out of the restriction) and independent of the
    * resolver's/packager's naming convention -- confirmed empirically to differ across this
    * project's own build outputs: Coursier's cache uses `<artifact>-<version>.jar`
    * (`scala-library-3.8.3.jar`, `menger-common_3-0.2.0.jar`); `sbt`'s own staged/hashed
    * `Test`-fork classpath additionally organization-prefixes menger-app's own jar as
    * `menger-app_3-<version>.jar` (still no dotted org prefix); `JavaAppPackaging`'s
    * `stage`/`Universal:packageZipTarball` output (what the Docker image in
    * `docker/scene-validator/` actually runs) prefixes *every* jar with its dotted
    * organization id, e.g. `io.github.lilacashes.menger-app-0.8.13.jar`,
    * `org.scala-lang.scala-library-3.8.3.jar`, `com.typesafe.scala-logging.scala-logging_3-
    * 3.9.6.jar` -- and drops menger-app's own `_3` cross-version suffix entirely. The optional
    * `(?:[\w.-]+\.)?` prefix and optional `(?:_3)?` suffix below tolerate all three; the
    * trailing `\d[\d.]*\.jar$` (digits/dots only, then `.jar` immediately) is what excludes
    * `-tests`/`-sources`/`-javadoc`/`-natives-*` classifier jars, which share the same
    * artifact-name-plus-version prefix but never end in bare digits-and-dots.
    */
  private val MengerAppJarPattern    = """(?:[\w.-]+\.)?menger-app(?:_3)?-\d[\d.]*\.jar""".r
  private val IncludedJarNamePatterns = List(
    "scala-library"  -> """(?:[\w.-]+\.)?scala-library-\d[\d.]*\.jar""".r,
    "scala3-library"  -> """(?:[\w.-]+\.)?scala3-library_3-\d[\d.]*\.jar""".r,
    "menger-common"  -> """(?:[\w.-]+\.)?menger-common_3-\d[\d.]*\.jar""".r,
    "scala-logging"  -> """(?:[\w.-]+\.)?scala-logging_3-\d[\d.]*\.jar""".r,
    "menger-app"     -> MengerAppJarPattern
  )

  private def isIncludedDir(path: String): Boolean = ProjectClassesDir.matches(path)
  private def isIncludedJar(name: String): Boolean = IncludedJarNamePatterns.exists(_._2.matches(name))

  /** Classifies a classpath entry from its *path string* alone -- deliberately not a
    * filesystem `isDirectory()` stat: a `.jar`-suffixed entry is a jar, anything else is a
    * directory-style entry. This keeps `build`/`preflight` pure functions of their `String`
    * input (testable against synthetic classpaths with no real files on disk) and avoids a
    * subtle production risk of its own -- a stat-based check would silently reclassify (and
    * therefore drop) an entry if the underlying path happened to be momentarily unreadable. */
  private def isIncluded(entryPath: String): Boolean =
    if entryPath.endsWith(".jar") then isIncludedJar(new File(entryPath).getName)
    else isIncludedDir(entryPath)

  /** The full, unrestricted classpath of the current JVM -- every URL reachable from the
    * current thread's classloader chain, plus the `java.class.path` system property (sbt's
    * layered classloaders mean neither alone is complete). This is `RestrictedClasspath`'s
    * own input, never its output. */
  @SuppressWarnings(Array("org.wartremover.warts.IsInstanceOf"))
  def fullClasspath: String =
    def urlsFrom(cl: ClassLoader): Seq[java.net.URL] = cl match
      case ucl: URLClassLoader => ucl.getURLs.toSeq ++ urlsFrom(ucl.getParent)
      // scalafix:off DisableSyntax.null
      case null                => Seq.empty
      // scalafix:on DisableSyntax.null
      case other               => urlsFrom(other.getParent)

    val loaderUrls = urlsFrom(Thread.currentThread.getContextClassLoader)

    val sysPropEntries = System.getProperty("java.class.path", "")
      .split(File.pathSeparator)
      .filter(_.nonEmpty)
      .toSeq

    val loaderPaths = loaderUrls.flatMap(u => Try(new File(u.toURI).getAbsolutePath).toOption)

    (loaderPaths ++ sysPropEntries).distinct.mkString(File.pathSeparator)

  /** Filters `classpath`'s own entries down to the DSL-surface allowlist -- restriction only
    * ever narrows, it never adds an entry that wasn't already present on the input. */
  def build(classpath: String): String =
    val entries = classpath
      .split(File.pathSeparator)
      .toSeq
      .filter(_.nonEmpty)
      .filter(isIncluded)
      .distinct

    if entries.isEmpty then
      logger.warn("Restricted classpath is empty -- no entry on the full classpath matched " +
        "the DSL-surface allowlist; this almost certainly means the allowlist is stale " +
        "(e.g. after a dependency version bump) rather than a genuinely empty classpath")

    entries.mkString(File.pathSeparator)

  /** Convenience overload: restricts the current JVM's own full classpath. This is what
    * `SceneCompiler.compile` uses in production; the `build(classpath: String)` overload
    * above is what tests exercise against a synthetic classpath. */
  def build(): String = build(fullClasspath)

  /** `None` when the restricted classpath is structurally complete (every allowlisted
    * artifact resolved to at least one entry); `Some(reason)` when it is not -- a stale
    * build, an unresolved dependency, a classpath that never reached this project's own
    * `menger-app` code at all. Used to tell AD-5's `refused` (a pipeline-level failure) apart
    * from a genuine scene defect: a scene that fails to compile against a *complete*
    * restricted classpath is the scene's own bug; a scene that fails to compile because the
    * classpath itself is missing an entry it should always have is `refused`, never
    * miscoded as `compile-errors` (AD-5's own stated rationale -- conflating the two makes an
    * agent retry forever against its own supposed bug).
    *
    * `menger-app` itself is satisfied by *either* its classes directory or its jar (the two
    * layouts described above); every other required artifact is jar-only. */
  def preflight(classpath: String): Option[String] =
    val entries = classpath.split(File.pathSeparator).toSeq.filter(_.nonEmpty)

    val missingLibs = IncludedJarNamePatterns
      .filterNot(_._1 == "menger-app")
      .collect { case (label, pattern) if !entries.exists(e => pattern.matches(new File(e).getName)) => label }

    val mengerAppPresent = entries.exists { e =>
      isIncludedDir(e) || mengerAppJarMatches(e)
    }

    if missingLibs.isEmpty && mengerAppPresent then None
    else
      val libPart = if missingLibs.isEmpty then Nil
        else List(s"missing required jar(s) for: ${missingLibs.mkString(", ")}")
      val appPart = if mengerAppPresent then Nil
        else List("missing menger-app's own compiled classes (neither the classes directory nor its jar is on the classpath)")
      Some((libPart ++ appPart).mkString("; "))

  private def mengerAppJarMatches(entryPath: String): Boolean =
    entryPath.endsWith(".jar") && MengerAppJarPattern.matches(new File(entryPath).getName)

  /** Convenience overload against the current JVM's own full classpath. */
  def preflight(): Option[String] = preflight(fullClasspath)

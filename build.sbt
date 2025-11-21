import sbt.Keys.libraryDependencies
import com.github.sbt.jni.build.{BuildTool, CMakeWithoutVersionBug}
import scala.sys.process._

// Define custom task key for native tests
lazy val nativeTest = taskKey[Unit]("Run native C++ tests")

// Common module - shared types between optixJni and root
lazy val mengerCommon = project
  .in(file("menger-common"))
  .settings(
    name := "menger-common",
    scalaVersion := "3.7.3",
    scalacOptions ++= Seq("-deprecation", "-explain", "-feature", "-Wunused:imports"),

    // ScalaTest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test
  )

lazy val optixJni = project
  .in(file("optix-jni"))
  .dependsOn(mengerCommon)
  .enablePlugins(JniNative)
  .settings(
    name := "optix-jni",
    scalaVersion := "3.7.3",

    // Configure native build
    nativeCompile / sourceDirectory := sourceDirectory.value / "main" / "native",
    nativeBuildTool := CMakeWithoutVersionBug.make(Seq(
      "-Wno-dev",  // Suppress developer warnings
      "--log-level=WARNING"  // Only show warnings and errors
    )),

    // Auto-clean CMake cache if it's from a different build location (e.g., Docker)
    // This prevents "CMake Error: The source ... does not match ... used to generate cache"
    Compile / compile := {
      val log = streams.value.log
      val cacheFile = target.value / "native" / "x86_64-linux" / "build" / "CMakeCache.txt"
      val nativeDir = target.value / "native"

      if (cacheFile.exists()) {
        val expectedPath = (nativeCompile / sourceDirectory).value.getAbsolutePath
        val cacheContent = IO.read(cacheFile)

        // Check if cache contains a different source path (e.g., from Docker build)
        if (!cacheContent.contains(expectedPath)) {
          log.warn(s"CMake cache from different build location detected (likely Docker container)")
          log.warn(s"Cleaning native build directory: ${nativeDir.getAbsolutePath}")
          IO.delete(nativeDir)
          log.info("Native build directory cleaned. CMake will regenerate cache on next build.")
        }
      }

      val compileResult = (Compile / compile).value

      // Copy PTX file to classes directory (sbt-jni only copies .so/.dll/.dylib)
      val ptxSource = target.value / "native" / "x86_64-linux" / "bin" / "sphere_combined.ptx"
      val ptxDest = (Compile / classDirectory).value / "native" / "x86_64-linux" / "sphere_combined.ptx"
      if (ptxSource.exists()) {
        IO.copyFile(ptxSource, ptxDest)
        log.debug(s"Copied PTX file: $ptxSource -> $ptxDest")
      }

      compileResult
    },

    // Use custom CMakeWithoutVersionBug to avoid spurious warning from sbt-jni version parsing bug
    // See project/CMakeWithoutVersionBug.scala for details

    // Native test task to run C++ Google Test suite
    nativeTest := {
      val log = streams.value.log
      val buildDir = target.value / "native" / "x86_64-linux" / "build"
      val testExe = buildDir / "optixcontext_test"

      // Ensure native code is compiled first
      nativeCompile.value

      if (testExe.exists()) {
        log.info("Running C++ unit tests (Google Test)...")

        val result = Process(
          Seq(testExe.getAbsolutePath),
          None,
          "LD_LIBRARY_PATH" -> "/usr/local/cuda/lib64"
        ).!

        if (result != 0) {
          throw new RuntimeException(s"Native tests failed with exit code $result")
        }
        log.info("C++ unit tests passed")
      } else {
        log.warn(s"C++ test executable not found at ${testExe.getAbsolutePath}")
        log.warn("Skipping native tests (BUILD_OPTIX_TESTS may be disabled)")
      }
    },

    // Make 'test' depend on 'nativeTest' so both Scala and C++ tests run
    Test / test := {
      nativeTest.value
      (Test / test).value
    },

    // Set library path for tests to find native library
    // Multiple paths to try: test-classes, classes, and native build output
    Test / javaOptions ++= Seq(
      s"-Djava.library.path=${(Test / classDirectory).value / "native" / "x86_64-linux"}:${(Compile / classDirectory).value / "native" / "x86_64-linux"}:${target.value / "native" / "x86_64-linux" / "bin"}",
      "-Dlogback.statusListenerClass=ch.qos.logback.core.status.NopStatusListener"
    ),
    Test / fork := true, // Required for javaOptions to take effect

    // Logging
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.19",

    // ScalaTest for future tests
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test
  )

lazy val root = {
  val base = project
    .in(file("."))
    .enablePlugins(JavaAppPackaging)
    .settings(
    name := "Menger",
    version := "0.3.7",
    maintainer := "lene.preuss@gmail.com",

    scalaVersion := "3.7.3",
    scalacOptions ++= Seq("-deprecation", "-explain", "-feature", "-Wunused:imports"),

    // Scalafix configuration
    inThisBuild(List(
      semanticdbVersion := scalafixSemanticdb.revision
    )),
    // Run Scalafix during compile only
    Compile / semanticdbEnabled := true,

    // Filter out plugin options from scaladoc to avoid warnings
    Compile / doc / scalacOptions := scalacOptions.value.filterNot(opt =>
      opt.startsWith("-Xplugin") || opt.startsWith("-P") || opt.contains("semanticdb")
    ),

    // Run WartRemover during compile only - explicit warts excluding LibGDX-incompatible ones
    Compile / wartremoverErrors ++= Seq(
      Wart.Var,           // Error on mutable variables (must use @SuppressWarnings)
      Wart.While,         // Prevent while loops
      Wart.AsInstanceOf,  // Prevent unsafe casting
      Wart.IsInstanceOf,  // Prevent runtime type checks
      Wart.Throw          // Prevent throwing exceptions
      // Excluding Wart.Null and Wart.Return for LibGDX compatibility
    ),

    // Logging
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.19",
    // JUnit
    libraryDependencies ++= Seq(
      "com.github.sbt.junit" % "jupiter-interface" % JupiterKeys.jupiterVersion.value % Test
    ),
    // ScalaTest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test,
    libraryDependencies += "org.scalamock" %% "scalamock" % "7.5.0" % Test,

    // Suppress SLF4J replay warnings during tests
    Test / javaOptions += "-Dlogback.statusListenerClass=ch.qos.logback.core.status.NopStatusListener",
    Test / fork := true,

    // Make root project's test depend on optixJni and mengerCommon tests
    Test / test := {
      (mengerCommon / Test / test).value
      (optixJni / Test / test).value
      (Test / test).value
    },

    // Scallop command line parser
    libraryDependencies += "org.rogach" %% "scallop" % "5.2.0",

    // libGDX
    libraryDependencies ++= Seq(
      "com.badlogicgames.gdx" % "gdx" % "1.13.5",
      "net.sf.proguard" % "proguard-base" % "6.2.2" % "provided",
      "com.badlogicgames.gdx" % "gdx-backend-lwjgl3" % "1.13.5",
      "com.badlogicgames.gdx" % "gdx-platform" % "1.13.5" classifier "natives-desktop",
    ),
    Test / scalacOptions += "-experimental"
  )

  // Always depend on OptiX JNI and common module, configure library path
  base
    .dependsOn(optixJni, mengerCommon)  // Compile and runtime dependency
    .settings(
      // Set library path for run to find native library and CUDA libraries
      run / javaOptions += s"-Djava.library.path=${(optixJni / target).value / "native" / "x86_64-linux" / "bin"}:/usr/local/cuda/lib64",
      run / fork := true // Required for javaOptions to take effect
    )
}

import com.github.sbt.jni.build.CMakeWithoutVersionBug

// Global settings applied to all projects
inThisBuild(List(
  semanticdbVersion := scalafixSemanticdb.revision,
  Test / parallelExecution := true
))

// Root project - aggregator only, no source code
lazy val root = project
  .in(file("."))
  .aggregate(mengerCommon, optixJni, mengerApp)
  .settings(
    name := "menger-root",
    publish / skip := true,
    // Delegate run to mengerApp so `sbt "run --args"` works from root
    Compile / run / aggregate := false,
    Compile / run := (mengerApp / Compile / run).evaluated,
    // Run all tests across all subprojects
    Test / test := {
      (mengerCommon / Test / test).value
      (optixJni / Test / test).value
      (mengerApp / Test / test).value
    }
  )

// Common module - shared types and utilities (no dependencies on other subprojects)
lazy val mengerCommon = project
  .in(file("menger-common"))

// OptiX JNI bindings - depends on common
lazy val optixJni = project
  .in(file("optix-jni"))
  .dependsOn(mengerCommon)
  .enablePlugins(JniNative)

// Main application - depends on both common and optix-jni
lazy val mengerApp = project
  .in(file("menger-app"))
  .dependsOn(optixJni, mengerCommon)
  .enablePlugins(JavaAppPackaging)
  .settings(
    // Set library path for run to find native library and CUDA libraries
    run / javaOptions += s"-Djava.library.path=${(optixJni / Compile / classDirectory).value / "native" / "x86_64-linux"}:${(optixJni / target).value / "native" / "x86_64-linux" / "bin"}:/usr/local/cuda/lib64",
    run / fork := true,
    // Use project root as working directory so file paths match packaged executable behavior
    run / baseDirectory := (ThisBuild / baseDirectory).value
  )

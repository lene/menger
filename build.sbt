import menger.build.CMakeWithoutVersionBug

// Global settings applied to all projects
inThisBuild(List(
  semanticdbEnabled := true,
  semanticdbVersion := scalafixSemanticdb.revision,
  Test / parallelExecution := true,
))

lazy val mengerCommonDependency = "io.github.lene" %% "menger-common" % "0.1.4"
lazy val optixJniDependency = "io.github.lene" % "optix-jni" % "0.1.10"

// Root project - aggregator only, no source code
lazy val root = project
  .in(file("."))
  .aggregate(mengerGeometry, mengerApp)
  .settings(
    name := "menger-root",
    publish / skip := true,
    // Delegate run to mengerApp so `sbt "run --args"` works from root
    Compile / run / aggregate := false,
    Compile / run := (mengerApp / Compile / run).evaluated,
    // Run all tests across all subprojects
    Test / test := {
      (mengerGeometry / Test / test).value
      (mengerApp / Test / test).value
    }
  )

// Menger-specific geometry layer — 4D fractals, caustics (not published)
lazy val mengerGeometry = project
  .in(file("menger-geometry"))
  .enablePlugins(JniNative)
  .settings(libraryDependencies ++= Seq(mengerCommonDependency, optixJniDependency))

// Main application - depends on menger-geometry and common
lazy val mengerApp = project
  .in(file("menger-app"))
  .dependsOn(mengerGeometry)
  .enablePlugins(JavaAppPackaging)
  .settings(
    libraryDependencies += mengerCommonDependency,
    // Set library path for run to find native library and CUDA libraries
    run / javaOptions += s"-Djava.library.path=${(mengerGeometry / target).value / "native" / "x86_64-linux" / "bin"}:/usr/local/cuda/lib64",
    run / fork := true,
    // Use project root as working directory so file paths match packaged executable behavior
    run / baseDirectory := (ThisBuild / baseDirectory).value
  )

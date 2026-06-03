import com.github.sbt.jni.build.CMakeWithoutVersionBug

// Global settings applied to all projects
inThisBuild(List(
  semanticdbEnabled := true,
  semanticdbVersion := scalafixSemanticdb.revision,
  Test / parallelExecution := true,
  resolvers += "GitLab Menger" at
    "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/packages/maven",
  credentials += Credentials(
    "GitLab Packages Registry",
    "gitlab.com",
    if (sys.env.contains("CI_JOB_TOKEN")) "gitlab-ci-token" else "Private-Token",
    sys.env.getOrElse(
      "CI_JOB_TOKEN",
      sys.env.getOrElse("GITLAB_PAT", sys.env.getOrElse("GITLAB_ACCESS_TOKEN", ""))
    )
  )
))

lazy val mengerCommonDependency = "io.github.lene" %% "menger-common" % "0.1.0"

// Root project - aggregator only, no source code
lazy val root = project
  .in(file("."))
  .aggregate(optixJni, mengerGeometry, mengerApp)
  .settings(
    name := "menger-root",
    publish / skip := true,
    // Delegate run to mengerApp so `sbt "run --args"` works from root
    Compile / run / aggregate := false,
    Compile / run := (mengerApp / Compile / run).evaluated,
    // Run all tests across all subprojects
    Test / test := {
      (optixJni / Test / test).value
      (mengerGeometry / Test / test).value
      (mengerApp / Test / test).value
    }
  )

// OptiX JNI bindings - depends on common
lazy val optixJni = project
  .in(file("optix-jni"))
  .enablePlugins(JniNative)
  .settings(libraryDependencies += mengerCommonDependency)

// Menger-specific geometry layer — 4D fractals, caustics (not published)
lazy val mengerGeometry = project
  .in(file("menger-geometry"))
  .dependsOn(optixJni)
  .enablePlugins(JniNative)
  .settings(libraryDependencies += mengerCommonDependency)

// Main application - depends on menger-geometry and common
lazy val mengerApp = project
  .in(file("menger-app"))
  .dependsOn(mengerGeometry)
  .enablePlugins(JavaAppPackaging)
  .settings(
    libraryDependencies += mengerCommonDependency,
    // Set library path for run to find native library and CUDA libraries
    run / javaOptions += s"-Djava.library.path=${(optixJni / Compile / classDirectory).value / "native" / "x86_64-linux"}:${(optixJni / target).value / "native" / "x86_64-linux" / "bin"}:${(mengerGeometry / target).value / "native" / "x86_64-linux" / "bin"}:/usr/local/cuda/lib64",
    run / fork := true,
    // Use project root as working directory so file paths match packaged executable behavior
    run / baseDirectory := (ThisBuild / baseDirectory).value
  )

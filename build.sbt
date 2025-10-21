import sbt.Keys.libraryDependencies

lazy val optixJni = project
  .in(file("optix-jni"))
  .enablePlugins(JniNative)
  .settings(
    name := "optix-jni",
    scalaVersion := "3.7.3",

    // Set library path for tests to find native library
    // Multiple paths to try: test-classes, classes, and native build output
    Test / javaOptions ++= Seq(
      s"-Djava.library.path=${(Test / classDirectory).value / "native" / "x86_64-linux"}:${(Compile / classDirectory).value / "native" / "x86_64-linux"}:${target.value / "native" / "x86_64-linux" / "bin"}"
    ),
    Test / fork := true, // Required for javaOptions to take effect

    // Logging
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.19",

    // ScalaTest for future tests
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test
  )

lazy val root = project
  .in(file("."))
  .enablePlugins(JavaAppPackaging)
  .dependsOn(optixJni)
  .settings(
    name := "Menger",
    version := "0.3.1",
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

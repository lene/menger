name := "menger-app"
version := "0.6.2"
scalaVersion := "3.8.2"

organization := "io.github.lilacashes"
maintainer := "lene.preuss@gmail.com"
description := "Menger Sponge ray tracer with OptiX acceleration"
homepage := Some(url("https://gitlab.com/lilacashes/menger"))
licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0"))
scmInfo := Some(ScmInfo(
  url("https://gitlab.com/lilacashes/menger"),
  "scm:git:git@gitlab.com:lilacashes/menger.git"
))
developers := List(
  Developer("lene", "Lene Preuss", "lene.preuss@gmail.com", url("https://gitlab.com/lilacashes"))
)

scalacOptions ++= Seq("-deprecation", "-explain", "-feature", "-Wunused:imports")

Compile / semanticdbEnabled := true

// Filter out plugin options from scaladoc to avoid warnings
Compile / doc / scalacOptions := scalacOptions.value.filterNot(opt =>
  opt.startsWith("-Xplugin") || opt.startsWith("-P") || opt.contains("semanticdb")
)

Compile / wartremoverErrors ++= Seq(
  Wart.Var,
  Wart.While,
  Wart.AsInstanceOf,
  Wart.IsInstanceOf,
  Wart.Throw
)

// Suppress SLF4J replay warnings during tests
Test / javaOptions += "-Dlogback.statusListenerClass=ch.qos.logback.core.status.NopStatusListener"
Test / fork := true
Test / scalacOptions += "-experimental"

// Coverage configuration - exclude untestable packages (JNI/GPU and LibGDX/OpenGL code)
// Also exclude LibGDX adapter handlers that require native library initialization for testing
coverageExcludedPackages := "menger\\.optix\\..*;menger\\.engines\\..*;" +
  "menger\\.input\\.LibGDXInputAdapter;" +
  "menger\\.input\\.OptiXCameraHandler;" +
  "menger\\.input\\.OptiXKeyHandler"

libraryDependencies ++= Seq(
  // Logging
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
  "ch.qos.logback" % "logback-classic" % "1.5.32",
  // JUnit
  "com.github.sbt.junit" % "jupiter-interface" % JupiterKeys.jupiterVersion.value % Test,
  // ScalaTest
  "org.scalatest" %% "scalatest" % "3.2.19" % Test,
  "org.scalamock" %% "scalamock" % "7.5.5" % Test,
  "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0" % Test,
  // Scallop command line parser
  "org.rogach" %% "scallop" % "6.0.0",
  // libGDX
  "com.badlogicgames.gdx" % "gdx" % "1.14.0",
  "com.badlogicgames.gdx" % "gdx-backend-lwjgl3" % "1.14.0",
  "com.badlogicgames.gdx" % "gdx-platform" % "1.14.0" classifier "natives-desktop",
  // Scala 3 compiler for runtime .scala scene file evaluation (Task 17.7)
  "org.scala-lang" %% "scala3-compiler" % scalaVersion.value,
  // ArchUnit — architecture rule enforcement
  "com.tngtech.archunit" % "archunit" % "1.4.2" % Test
)

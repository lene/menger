import com.github.sbt.jni.build.CMakeWithoutVersionBug

name := "menger-geometry"
version := "0.7.1"
scalaVersion := "3.8.3"

organization := "io.github.lene"
publish / skip := true   // internal module, not published separately

scalacOptions ++= Seq("-deprecation", "-explain", "-feature", "-Wunused:imports")

// Set library path for tests to find native libraries
Test / javaOptions ++= Seq(
  s"-Djava.library.path=${(Test / classDirectory).value / "native" / "x86_64-linux"}:${(Compile / classDirectory).value / "native" / "x86_64-linux"}:${target.value / "native" / "x86_64-linux" / "bin"}",
  "-Dlogback.statusListenerClass=ch.qos.logback.core.status.NopStatusListener"
)
Test / fork := true

nativeCompile / sourceDirectory := sourceDirectory.value / "main" / "native"
nativeBuildTool := CMakeWithoutVersionBug.make(Seq(
  "-Wno-dev",
  "--log-level=WARNING",
  "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc"
))

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
  "ch.qos.logback" % "logback-classic" % "1.5.32",
  "org.scalatest" %% "scalatest" % "3.2.20" % Test
)

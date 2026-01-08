name := "menger-common"
version := "0.4.1"
scalaVersion := "3.7.3"

organization := "io.github.lilacashes"
description := "Common types and utilities for Menger ray tracer"
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

Compile / wartremoverErrors ++= Seq(
  Wart.Var,
  Wart.While,
  Wart.AsInstanceOf,
  Wart.IsInstanceOf,
  Wart.Throw
)

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.2.19" % Test,
  "org.scalatestplus" %% "scalacheck-1-18" % "3.2.19.0" % Test
)

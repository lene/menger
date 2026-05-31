name := "menger-common"
version := "0.1.0"
scalaVersion := "3.8.3"

organization := "io.github.lene"
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

// Publication targets
// Primary: GitLab Package Registry
// CI uses CI_JOB_TOKEN; local use requires GITLAB_PAT env var
publishTo := {
  val base = "https://gitlab.com/api/v4/projects/lilacashes%2Fmenger/packages/maven"
  if (isSnapshot.value) Some("GitLab Snapshots" at base)
  else                  Some("GitLab Releases"  at base)
}

credentials += Credentials(
  "GitLab Packages Registry",
  "gitlab.com",
  if (sys.env.contains("CI_JOB_TOKEN")) "gitlab-ci-token" else "Private-Token",
  sys.env.getOrElse("CI_JOB_TOKEN", sys.env.getOrElse("GITLAB_PAT", ""))
)

sonatypeCredentialHost := "central.sonatype.com"
publishMavenStyle := true

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
  "org.scalatest" %% "scalatest" % "3.2.20" % Test,
  "org.scalatestplus" %% "scalacheck-1-19" % "3.2.20.0" % Test
)

package menger.dsl

import java.io.File

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RestrictedClasspathSuite extends AnyFlatSpec with Matchers:

  private val sep = File.pathSeparator

  private val syntheticFull = List(
    "/repo/menger-app/target/scala-3.8.3/classes",
    "/repo/menger-app/target/scala-3.8.3/test-classes",
    "/repo/menger-geometry/target/scala-3.8.3/classes",
    "/cache/org/scala-lang/scala-library/3.8.3/scala-library-3.8.3.jar",
    "/cache/org/scala-lang/scala3-library_3/3.8.3/scala3-library_3-3.8.3.jar",
    "/cache/io/github/lene/menger-common_3/0.2.0/menger-common_3-0.2.0.jar",
    "/cache/com/typesafe/scala-logging/scala-logging_3/3.9.6/scala-logging_3-3.9.6.jar",
    "/cache/ch/qos/logback/logback-classic/1.5.34/logback-classic-1.5.34.jar",
    "/cache/org/scalatest/scalatest_3/3.2.20/scalatest_3-3.2.20.jar",
    "/cache/com/badlogicgames/gdx/gdx/1.14.2/gdx-1.14.2.jar",
    "/cache/io/github/lene/optix-jni/0.3.3/optix-jni-0.3.3.jar",
    "/cache/org/scala-lang/scala3-compiler_3/3.8.3/scala3-compiler_3-3.8.3.jar",
    "/cache/com/lihaoyi/upickle_3/4.1.0/upickle_3-4.1.0.jar",
    "/cache/org/rogach/scallop_3/6.0.0/scallop_3-6.0.0.jar"
  ).mkString(sep)

  "RestrictedClasspath.build" should "include only menger-app's own main classes directory, never test-classes or menger-geometry's" in:
    val restricted = RestrictedClasspath.build(syntheticFull)
    restricted should include("/repo/menger-app/target/scala-3.8.3/classes")
    restricted should not include "/repo/menger-app/target/scala-3.8.3/test-classes"
    restricted should not include "/repo/menger-geometry/target/scala-3.8.3/classes"

  it should "include scala-library, scala3-library, menger-common, and scala-logging jars" in:
    val restricted = RestrictedClasspath.build(syntheticFull)
    restricted should include("scala-library-3.8.3.jar")
    restricted should include("scala3-library_3-3.8.3.jar")
    restricted should include("menger-common_3-0.2.0.jar")
    restricted should include("scala-logging_3-3.9.6.jar")

  it should "exclude test frameworks, LibGDX, optix-jni, the Scala compiler itself, upickle, and scallop" in:
    val restricted = RestrictedClasspath.build(syntheticFull)
    restricted should not include "logback-classic"
    restricted should not include "scalatest"
    restricted should not include "gdx"
    restricted should not include "optix-jni"
    restricted should not include "scala3-compiler"
    restricted should not include "upickle"
    restricted should not include "scallop"

  it should "return an empty string (with a warning, not a crash) when nothing on the input matches" in:
    RestrictedClasspath.build("/some/unrelated/dir:/some/unrelated.jar") shouldBe ""

  it should "never add an entry that wasn't already present on the input classpath" in:
    val restricted = RestrictedClasspath.build(syntheticFull)
    val inputEntries = syntheticFull.split(sep).toSet
    restricted.split(sep).filter(_.nonEmpty).foreach { e => inputEntries should contain(e) }

  "RestrictedClasspath.preflight" should "return None when every required entry is present" in:
    RestrictedClasspath.preflight(syntheticFull) shouldBe None

  it should "return Some(reason) when menger-app's own classes directory is missing" in:
    val withoutAppClasses = syntheticFull.split(sep).filterNot(_.endsWith("menger-app/target/scala-3.8.3/classes")).mkString(sep)
    RestrictedClasspath.preflight(withoutAppClasses) shouldBe defined

  it should "return Some(reason) when menger-common is missing (a stale/mismatched build)" in:
    val withoutCommon = syntheticFull.split(sep).filterNot(_.contains("menger-common")).mkString(sep)
    val reason = RestrictedClasspath.preflight(withoutCommon)
    reason shouldBe defined
    reason.get should include("menger-common")

  it should "return Some(reason) for a completely empty classpath" in:
    RestrictedClasspath.preflight("") shouldBe defined

  "RestrictedClasspath.build() and preflight() (no-arg, live JVM classpath)" should
    "produce a non-empty, structurally complete restricted classpath in this test's own JVM" in:
    RestrictedClasspath.preflight() shouldBe None
    val restricted = RestrictedClasspath.build()
    restricted should not be empty
    restricted should include("menger-common")
    restricted should include("scala-logging")
    restricted should not include "scalatest"
    restricted should not include "archunit"

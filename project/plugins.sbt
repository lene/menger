// Universal / packageBin
addSbtPlugin("com.github.sbt" % "sbt-native-packager" % "1.11.7")

// JNI native compilation
addSbtPlugin("com.github.sbt" % "sbt-jni" % "1.7.1")

// JUnit5/Jupiter
addSbtPlugin("com.github.sbt.junit" % "sbt-jupiter-interface" % "0.14.0")
// Test coverage (see https://www.baeldung.com/scala/sbt-scoverage-code-analysis)
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.4.4")
// Code quality and refactoring
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.14.6")
addSbtPlugin("org.wartremover" % "sbt-wartremover" % "3.5.6")
ThisBuild / libraryDependencySchemes += "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always
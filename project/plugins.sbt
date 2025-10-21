// Universal / packageBin
addSbtPlugin("com.github.sbt" % "sbt-native-packager" % "1.11.3")

// JNI native compilation
addSbtPlugin("com.github.sbt" % "sbt-jni" % "1.6.0")

// JUnit5/Jupiter
addSbtPlugin("com.github.sbt.junit" % "sbt-jupiter-interface" % "0.11.3")
// Test coverage (see https://www.baeldung.com/scala/sbt-scoverage-code-analysis)
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.3.1")
// Code quality and refactoring
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.14.3")
addSbtPlugin("org.wartremover" % "sbt-wartremover" % "3.4.1")
ThisBuild / libraryDependencySchemes += "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always
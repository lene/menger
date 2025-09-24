// Universal / packageBin
addSbtPlugin("com.github.sbt" % "sbt-native-packager" % "1.11.3")

// JUnit5/Jupiter
addSbtPlugin("com.github.sbt.junit" % "sbt-jupiter-interface" % "0.11.3")
// Test coverage (see https://www.baeldung.com/scala/sbt-scoverage-code-analysis)
addSbtPlugin("org.scoverage" % "sbt-scoverage" % "2.3.1")
// Code quality and refactoring
addSbtPlugin("ch.epfl.scala" % "sbt-scalafix" % "0.11.1")
ThisBuild / libraryDependencySchemes += "org.scala-lang.modules" %% "scala-xml" % VersionScheme.Always
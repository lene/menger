
lazy val root = project
  .in(file("."))
  .enablePlugins(JavaAppPackaging)
  .settings(
    name := "Menger",
    version := "0.2.8",
    maintainer := "lene.preuss@gmail.com",

    scalaVersion := "3.7.3",
    scalacOptions ++= Seq("-deprecation", "-explain", "-feature"),

    // Scalafix configuration
    ThisBuild / scalafixDependencies += "com.github.liancheng" %% "organize-imports" % "0.6.0",
    inThisBuild(List(
      semanticdbEnabled := true,
      semanticdbVersion := scalafixSemanticdb.revision
    )),

    // Logging
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.6",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.18",
    // JUnit
    libraryDependencies ++= Seq(
      "com.github.sbt.junit" % "jupiter-interface" % JupiterKeys.jupiterVersion.value % Test
    ),
    // ScalaTest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test,
    libraryDependencies += "org.scalamock" %% "scalamock" % "7.4.1" % Test,

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

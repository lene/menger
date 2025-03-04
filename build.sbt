
lazy val root = project
  .in(file("."))
  .enablePlugins(JavaAppPackaging)
  .settings(
    name := "Menger",
    version := "0.2.3",
    maintainer := "lene.preuss@gmail.com",

    scalaVersion := "3.6.4",
    scalacOptions ++= Seq("-deprecation", "-explain", "-feature"),

    // Logging
    libraryDependencies += "com.typesafe.scala-logging" %% "scala-logging" % "3.9.5",
    libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.5.17",
    // JUnit
    libraryDependencies ++= Seq(
      "net.aichler" % "jupiter-interface" % JupiterKeys.jupiterVersion.value % Test
    ),
    // ScalaTest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.19" % Test,
    libraryDependencies += "org.scalamock" %% "scalamock" % "6.1.1" % Test,

    // Scallop command line parser
    libraryDependencies += "org.rogach" %% "scallop" % "5.2.0",

    // libGDX
    libraryDependencies ++= Seq(
      "com.badlogicgames.gdx" % "gdx" % "1.13.1",
      "net.sf.proguard" % "proguard-base" % "6.2.2" % "provided",
      "com.badlogicgames.gdx" % "gdx-backend-lwjgl3" % "1.13.1",
      "com.badlogicgames.gdx" % "gdx-platform" % "1.13.1" classifier "natives-desktop",
    ),

  )

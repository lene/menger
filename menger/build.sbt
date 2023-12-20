
val scala3Version = "3.3.1"
val libgdxVersion = "1.12.1"

lazy val root = project
  .in(file("."))
  .settings(
    name := "Menger",
    version := "0.1.0-SNAPSHOT",
    maintainer := "lene.preuss@gmail.com",

    scalaVersion := scala3Version,

    // JUnit
    libraryDependencies ++= Seq(
      "net.aichler" % "jupiter-interface" % JupiterKeys.jupiterVersion.value % Test
    ),
    // ScalaTest
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.15" % "test",
    // libGDX
    libraryDependencies ++= Seq(
      "com.badlogicgames.gdx" % "gdx" % libgdxVersion,
      "net.sf.proguard" % "proguard-base" % "6.2.2" % "provided",
      "com.badlogicgames.gdx" % "gdx-backend-lwjgl3" % libgdxVersion,
      "com.badlogicgames.gdx" % "gdx-platform" % libgdxVersion classifier "natives-desktop",
    )
  )

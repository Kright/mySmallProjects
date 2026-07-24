ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.8.4"

lazy val root = (project in file("."))
  .enablePlugins(JmhPlugin)
  .settings(
    name := "scalaLists",
    scalacOptions ++= Seq(
      "-optimize",
      "-Xmax-inlines", "64"
    ),
    libraryDependencies += "com.github.sbt" % "junit-interface" % "0.13.3" % Test
  )

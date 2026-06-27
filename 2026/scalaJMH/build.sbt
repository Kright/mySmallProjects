ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.8.4"

lazy val root = (project in file("."))
  .enablePlugins(JmhPlugin)
  .settings(
    name := "scalaJMH",
    scalacOptions ++= Seq(
      "-optimize",
      "-Xmax-inlines", "64"
    ),
  )

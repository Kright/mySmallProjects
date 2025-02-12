ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.4.0"

lazy val root = (project in file("."))
  .settings(
    name := "scala_jmh",
    idePackagePrefix := Some("com.github.kright")
  )

enablePlugins(JmhPlugin)

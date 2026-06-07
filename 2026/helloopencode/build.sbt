ThisBuild / scalaVersion := "3.8.3"

lazy val root = (project in file("."))
  .settings(
    name := "helloopencode",
    libraryDependencies += "me.kright" %% "arrayview" % "0.3.2",
  )

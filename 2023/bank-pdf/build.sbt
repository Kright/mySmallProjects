ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.2.2"

lazy val root = (project in file("."))
  .settings(
    name := "bankpdf",
    libraryDependencies += "org.apache.pdfbox" % "pdfbox" % "2.0.27"
  )

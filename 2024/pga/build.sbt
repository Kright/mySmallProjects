ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.4.0"

lazy val root = (project in file("."))
  .settings(
    name := "pga"
  )

resolvers += "jitpack" at "https://jitpack.io"

libraryDependencies ++= Seq(
  "org.scalactic" %% "scalactic" % "3.2.18",
  "org.scalatest" %% "scalatest" % "3.2.18" % "test",
  "org.scalatestplus" %% "scalacheck-1-17" % "3.2.18.0" % "test",
  "org.scalacheck" %% "scalacheck" % "1.17.0" % "test",
  "symbolic" %% "symbolic" % "0.4.3", // local
//    "com.github.Kright.ScalaGameMath" % "symbolic_3" % "0.4.2",
)

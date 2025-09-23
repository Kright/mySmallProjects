plugins {
    kotlin("jvm") version "2.2.10"
    kotlin("kapt") version "2.2.10"
    id("org.jetbrains.kotlin.plugin.allopen") version "2.2.10"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))

    implementation("org.openjdk.jmh:jmh-core:1.37") // Use the latest JMH version
    kapt("org.openjdk.jmh:jmh-generator-annprocess:1.37") // Use the latest JMH version
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}

// Configure allOpen plugin for JMH
allOpen {
    annotation("org.openjdk.jmh.annotations.State")
    annotation("org.openjdk.jmh.annotations.Benchmark")
}
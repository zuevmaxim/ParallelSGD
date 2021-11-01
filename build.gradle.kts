plugins {
    kotlin("jvm") version "1.5.10"
}

group = "org.example"
version = "1.0-SNAPSHOT"

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    kotlinOptions { jvmTarget = "11" }
}

tasks.withType<Test> {
    maxHeapSize = "30g"
}

repositories {
    mavenCentral()
    maven("https://repo.kotlin.link")
}

dependencies {
    implementation(kotlin("stdlib"))

    implementation("org.jetbrains.kotlinx:atomicfu:0.16.3")

    implementation("org.jetbrains.lets-plot:lets-plot-common:2.1.0")
    implementation("org.jetbrains.lets-plot:lets-plot-image-export:2.1.0")
    implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:3.0.2")


    testImplementation("org.jetbrains.kotlin:kotlin-test")
}

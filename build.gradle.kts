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

    implementation("net.openhft:affinity:3.21ea82") // bind thread to core
    implementation("com.github.oshi:oshi-core:5.8.3") // retrieve NUMA configuration

    // plots
    implementation("org.jetbrains.lets-plot:lets-plot-common:2.2.0")
    implementation("org.jetbrains.lets-plot:lets-plot-image-export:2.2.0")
    implementation("org.jetbrains.lets-plot:lets-plot-kotlin-jvm:3.1.0")


    testImplementation("org.junit.jupiter:junit-jupiter:5.8.1")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.8.1")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

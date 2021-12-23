plugins {
    kotlin("jvm") version "1.5.10"
    id("me.champeau.jmh") version "0.6.6"
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
    mavenLocal()
}

dependencies {
    implementation(kotlin("stdlib"))

    implementation("org.jetbrains.kotlinx:atomicfu:0.16.3")

    implementation("net.openhft:affinity:3.21ea82") // bind thread to core
    implementation("com.github.oshi:oshi-core:5.8.3") // retrieve NUMA configuration

    implementation("org.jetbrains.kotlinx:smartbench:0.1-SNAPSHOT")

    testImplementation("org.junit.jupiter:junit-jupiter:5.8.1")
    testImplementation("org.junit.jupiter:junit-jupiter-api:5.8.1")
    testImplementation("org.junit.jupiter:junit-jupiter-engine:5.8.1")

    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.8.1")
    testRuntimeOnly("org.junit.vintage:junit-vintage-engine:5.8.1")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

jmh {
    includeTests.set(false)
    failOnError.set(true)
    profilers.add("perfasm")
}

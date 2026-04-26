package org.example.pipeline

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import org.example.convolution.kernels.Kernels3x3
import org.example.convolution.kernels.Kernels5x5
import javax.imageio.ImageIO
import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

private const val EPS     = 1f
private const val GPU_EPS = 2f   // GPU float32 vs JVM float32 + PNG-roundtrip может дать до ~1.5

// Задача 3 — тесты пайплайна

private fun tempImageFile(width: Int, height: Int, seed: Long = 52): Pair<File, GrayImage> {
    val rng = Random(seed)
    val img = GrayImage(width, height, FloatArray(width * height) { rng.nextFloat() * 255f })
    val file = File.createTempFile("pipeline_test_", ".png").also { it.deleteOnExit() }
    ImageIO.write(img.toBufferedImage(), "png", file)
    return file to img
}

private fun loadGray(file: File): GrayImage = ImageIO.read(file).toGrayImage()

/** Создаёт временную директорию и регистрирует её для удаления при завершении JVM */
private fun tempDir(prefix: String): File = createTempDirectory(prefix).toFile().also { it.deleteOnExit() }

private fun sortedByIndex(files: Array<File>): List<File> =
    files.sortedBy { it.nameWithoutExtension.substringAfterLast("_").toIntOrNull() ?: 0 }

/** Сравнение двух изображений попиксельно с заданной точностью */
private fun assertImagesEqual(
    expected: GrayImage, actual: GrayImage, eps: Float = EPS, label: String = ""
) {
    assertEquals(expected.width,  actual.width,  "width mismatch $label")
    assertEquals(expected.height, actual.height, "height mismatch $label")
    for (i in expected.data.indices) {
        val diff = abs(expected.data[i] - actual.data[i])
        assertTrue(diff <= eps,
            "pixel $i: expected=${expected.data[i]}, got=${actual.data[i]}, diff=$diff $label")
    }
}

private fun GrayImage.pngRoundTrip(): GrayImage = toBufferedImage().toGrayImage()

/**
 * Тесты пайплайна потоковой обработки изображений
 *
 * Основное свойство (из схемы тестирования задачи): пайплайн с любой конфигурацией
 * воркеров даёт тот же результат, что и последовательная обработка
 */
class PipelineTest {

    /** Пустой список входных путей — должен вернуть нулевую статистику без ошибок */
    @Test fun `empty input returns zero stats`() = runBlocking {
        val config = PipelineConfig(kernels = listOf(Kernels3x3.GAUSSIAN_BLUR))
        val stats  = runPipeline(emptyList(), null, config)
        assertEquals(0,   stats.totalImages)
        assertEquals(0L,  stats.totalMs)
        assertEquals(0.0, stats.throughput)
    }

    /** Одно изображение через пайплайн совпадает с последовательной свёрткой */
    @Test fun `single image matches sequential convolution`() = runBlocking {
        val (file, _) = tempImageFile(48, 48, seed = 1)
        val kernel    = Kernels3x3.GAUSSIAN_BLUR
        val expected  = convolveSequential(loadGray(file), kernel).pngRoundTrip()

        val outDir = tempDir("pipeline_single_")
        runPipeline(listOf(file.path), outDir.path, PipelineConfig(listOf(kernel), workerCount = 1))
        val actual = loadGray(outDir.listFiles()!!.first())

        assertImagesEqual(expected, actual, label = "single image pipeline")
    }

    /** N изображений через пайплайн: каждый результат совпадает с последовательной свёрткой, проверяется при 1, 2 и 4 воркерах */
    @Test fun `N images with various worker counts match sequential`() = runBlocking {
        val n      = 6
        val kernel = Kernels3x3.GAUSSIAN_BLUR
        val files  = (0 until n).map { i -> tempImageFile(40, 40, seed = i * 13L).first }

        val expected = files.map { f -> convolveSequential(loadGray(f), kernel).pngRoundTrip() }

        for (wc in listOf(1, 2, 4)) {
            val outDir = tempDir("pipeline_wc${wc}_")
            runPipeline(files.map { it.path }, outDir.path,
                PipelineConfig(kernels = listOf(kernel), workerCount = wc))

            val outFiles = sortedByIndex(outDir.listFiles()!!)
            assertEquals(n, outFiles.size, "workers=$wc: неверное количество результатов")

            for (i in outFiles.indices)
                assertImagesEqual(expected[i], loadGray(outFiles[i]), label = "workers=$wc image=$i")
        }
    }

    /** Результаты пайплайна не зависят от размера буферов */
    @Test fun `different buffer sizes produce identical results`() = runBlocking {
        val kernel = Kernels3x3.BOX_BLUR
        val files  = (0 until 5).map { i -> tempImageFile(32, 32, seed = i * 7L).first }

        val expected = files.map { f -> convolveSequential(loadGray(f), kernel).pngRoundTrip() }

        for (buf in listOf(1, 4, 16)) {
            val outDir = tempDir("pipeline_buf${buf}_")
            runPipeline(files.map { it.path }, outDir.path,
                PipelineConfig(listOf(kernel), workerCount = 2, inputBufferSize = buf, outputBufferSize = buf))
            val results = sortedByIndex(outDir.listFiles()!!)
            assertEquals(files.size, results.size, "buf=$buf: неверное количество выходных файлов")
            for (i in results.indices)
                assertImagesEqual(expected[i], loadGray(results[i]), label = "buf=$buf image=$i")
        }
    }

    /** Пайплайн с цепочкой нескольких фильтров совпадает с последовательным конвейером */
    @Test fun `pipeline with multi-kernel chain matches sequential pipeline`() = runBlocking {
        val kernels = listOf(Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.BOX_BLUR)
        val files   = (0 until 4).map { i -> tempImageFile(48, 48, seed = i * 11L).first }

        val expected = files.map { f ->
            convolveSequentialPipeline(loadGray(f), kernels).pngRoundTrip()
        }

        val outDir = tempDir("pipeline_multi_")
        runPipeline(files.map { it.path }, outDir.path,
            PipelineConfig(kernels = kernels, workerCount = 2))

        val results = sortedByIndex(outDir.listFiles()!!)
        assertEquals(files.size, results.size)
        for (i in results.indices)
            assertImagesEqual(expected[i], loadGray(results[i]), label = "multi-kernel image=$i")
    }

    /** Воркеры с Parallel-режимом свёртки дают тот же результат, что и Sequential-воркеры */
    @Test fun `parallel worker mode matches sequential worker mode`() = runBlocking {
        val kernel = Kernels5x5.GAUSSIAN_BLUR
        val files  = (0 until 4).map { i -> tempImageFile(60, 60, seed = i * 3L).first }

        val outSeq = tempDir("pipeline_seq_")
        runPipeline(files.map { it.path }, outSeq.path,
            PipelineConfig(listOf(kernel), workerCount = 2, workerMode = ConvolutionMode.Sequential))
        val seqResults = sortedByIndex(outSeq.listFiles()!!)

        val outPar = tempDir("pipeline_par_")
        runPipeline(files.map { it.path }, outPar.path,
            PipelineConfig(listOf(kernel), workerCount = 2,
                workerMode = ConvolutionMode.Parallel(ParallelMode.BY_ROW), workerThreads = 4))
        val parResults = sortedByIndex(outPar.listFiles()!!)

        assertEquals(seqResults.size, parResults.size)
        for (i in parResults.indices)
            assertImagesEqual(loadGray(seqResults[i]), loadGray(parResults[i]), label = "par-vs-seq image=$i")
    }

    /** Статистика: totalImages совпадает с размером батча */
    @Test fun `stats are consistent with batch size`() = runBlocking {
        val files  = (0 until 5).map { i -> tempImageFile(32, 32, seed = i.toLong()).first }
        val config = PipelineConfig(kernels = listOf(Kernels3x3.IDENTITY), workerCount = 2)
        val stats  = runPipeline(files.map { it.path }, null, config)

        assertEquals(5, stats.totalImages)
        assertTrue(stats.totalMs >= 0)
        assertTrue(stats.throughput > 0.0)
        assertTrue(stats.sumProcessMs >= 0)
    }

    /** Колбэк onProgress вызывается ровно totalImages раз, значения из диапазона [1, n] */
    @Test fun `onProgress callback is called for every image`() = runBlocking {
        val n      = 7
        val files  = (0 until n).map { i -> tempImageFile(24, 24, seed = i.toLong()).first }
        val config = PipelineConfig(kernels = listOf(Kernels3x3.IDENTITY), workerCount = 3)

        val progressValues = mutableListOf<Int>()
        runPipeline(files.map { it.path }, null, config) { done, _ -> progressValues += done }

        assertEquals(n, progressValues.size, "onProgress должен быть вызван $n раз")
        assertEquals(n, progressValues.toSet().size, "все значения должны быть уникальны")
        assertTrue(progressValues.all { it in 1..n })
    }

    // Задача 4b — гибридные воркеры (GPU + CPU)

    /** Гибридный конфиг (gpuWorkerCount > 0) даёт тот же результат, что и чистый CPU */
    @Test fun `hybrid config matches sequential result`() = runBlocking {
        val kernel = Kernels3x3.GAUSSIAN_BLUR
        val files  = (0 until 4).map { i -> tempImageFile(48, 48, seed = i * 5L).first }
        val expected = files.map { f -> convolveSequential(loadGray(f), kernel).pngRoundTrip() }

        val outDir = tempDir("pipeline_hybrid_")
        // gpuWorkerCount=1: если GPU доступен — один GPU-воркер, иначе тихий фолбэк на CPU
        runPipeline(files.map { it.path }, outDir.path,
            PipelineConfig(listOf(kernel), workerCount = 3, gpuWorkerCount = 1))

        val results = sortedByIndex(outDir.listFiles()!!)
        assertEquals(files.size, results.size)
        for (i in results.indices)
            assertImagesEqual(expected[i], loadGray(results[i]), eps = GPU_EPS, label = "hybrid image=$i")
    }

    /** gpuWorkerCount > workerCount — должно молча обрезаться до workerCount */
    @Test fun `gpuWorkerCount capped to workerCount`() = runBlocking {
        val kernel = Kernels3x3.IDENTITY
        val files  = (0 until 3).map { i -> tempImageFile(32, 32, seed = i.toLong()).first }
        val expected = files.map { f -> convolveSequential(loadGray(f), kernel).pngRoundTrip() }

        val outDir = tempDir("pipeline_capgpu_")
        // gpuWorkerCount (5) > workerCount (2) — корректно обрезается внутри runPipeline
        runPipeline(files.map { it.path }, outDir.path,
            PipelineConfig(listOf(kernel), workerCount = 2, gpuWorkerCount = 5))

        val results = sortedByIndex(outDir.listFiles()!!)
        assertEquals(files.size, results.size)
        for (i in results.indices)
            assertImagesEqual(expected[i], loadGray(results[i]), eps = GPU_EPS, label = "capped gpu image=$i")
    }

    /** При недоступном GPU (gpuWorkerCount > 0) воркеры переходят на workerMode без ошибок */
    @Test fun `gpuWorkerCount with unavailable GPU falls back to CPU silently`() = runBlocking {
        val kernel = Kernels3x3.BOX_BLUR
        val files  = (0 until 3).map { i -> tempImageFile(32, 32, seed = i * 2L).first }
        val expected = files.map { f -> convolveSequential(loadGray(f), kernel).pngRoundTrip() }

        val outDir = tempDir("pipeline_gpufallback_")
        // Если GPU недоступен — gpuWorkerCount игнорируется, всё идёт на CPU
        // Если GPU доступен — один GPU-воркер, тест проверяет корректность в обоих случаях
        runPipeline(files.map { it.path }, outDir.path,
            PipelineConfig(listOf(kernel), workerCount = 2,
                workerMode = ConvolutionMode.Sequential, gpuWorkerCount = 2))

        val results = sortedByIndex(outDir.listFiles()!!)
        assertEquals(files.size, results.size)
        for (i in results.indices)
            assertImagesEqual(expected[i], loadGray(results[i]), eps = GPU_EPS, label = "fallback image=$i")
    }

    /** collectImagePaths возвращает файл как список из одного пути */
    @Test fun `collectImagePaths returns single file as one-element list`() {
        val (file, _) = tempImageFile(10, 10)
        val paths = collectImagePaths(file.path)
        assertEquals(1, paths.size)
        assertEquals(file.path, paths[0])
    }

    /** collectImagePaths сканирует директорию и возвращает только изображения, игнорируя остальные файлы */
    @Test fun `collectImagePaths scans directory for images`() {
        val dir = tempDir("scan_test_")
        repeat(3) { i ->
            val (src, _) = tempImageFile(10, 10, seed = i.toLong())
            src.copyTo(File(dir, "img_$i.png"))
        }
        File(dir, "notes.txt").writeText("not an image")

        val paths = collectImagePaths(dir.path)
        assertEquals(3, paths.size, "должны найти 3 PNG и пропустить .txt")
        assertTrue(paths.all { it.endsWith(".png") })
    }
}

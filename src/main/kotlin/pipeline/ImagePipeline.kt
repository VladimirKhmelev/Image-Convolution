package org.example.pipeline

import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import org.example.convolution.*
import javax.imageio.ImageIO
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.system.measureTimeMillis

data class ImageTask(
    val index: Int,
    val sourcePath: String,
    val image: GrayImage
)

data class ImageResult(
    val index: Int,
    val sourcePath: String,
    val result: GrayImage,
    val processingMs: Long
)

data class PipelineConfig(
    val kernels: List<Array<FloatArray>>,
    val workerCount: Int = Runtime.getRuntime().availableProcessors(),
    val workerMode: ConvolutionMode = ConvolutionMode.Sequential,
    val workerThreads: Int = 1,
    val workerGridRows: Int = 0,
    val workerGridCols: Int = 0,
    val inputBufferSize: Int = workerCount * 2,
    val outputBufferSize: Int = workerCount * 2
)

data class PipelineStats(
    val totalImages: Int,
    val totalMs: Long,
    val sumReadMs: Long,
    val sumProcessMs: Long,
    val sumWriteMs: Long,
    val throughput: Double
)

suspend fun runPipeline(
    inputPaths: List<String>,
    outputDir: String?,
    config: PipelineConfig,
    onProgress: ((done: Int, total: Int) -> Unit)? = null
): PipelineStats = coroutineScope {
    if (inputPaths.isEmpty())
        return@coroutineScope PipelineStats(0, 0, 0, 0, 0, 0.0)

    val total = inputPaths.size

    val inputChannel = Channel<ImageTask>(capacity = config.inputBufferSize)
    val outputChannel = Channel<ImageResult>(capacity = config.outputBufferSize)

    val sumReadMs = AtomicLong(0)
    val sumProcessMs = AtomicLong(0)
    val sumWriteMs = AtomicLong(0)
    val doneCount = AtomicInteger(0)

    val totalMs = measureTimeMillis {

        val reader = launch(Dispatchers.IO) {
            try {
                for ((index, path) in inputPaths.withIndex()) {
                    lateinit var task: ImageTask
                    val ms = measureTimeMillis {
                        val buf = ImageIO.read(File(path))
                            ?: error("Не удалось прочитать изображение: $path")
                        task = ImageTask(index, path, buf.toGrayImage())
                    }
                    sumReadMs.addAndGet(ms)
                    inputChannel.send(task)
                }
            } finally {
                inputChannel.close()
            }
        }

        val workers = (1..config.workerCount).map {
            launch(Dispatchers.Default) {
                for (task in inputChannel) {
                    lateinit var result: GrayImage
                    val ms = measureTimeMillis {
                        result = when (val mode = config.workerMode) {
                            is ConvolutionMode.Sequential ->
                                convolveSequentialPipeline(task.image, config.kernels)
                            is ConvolutionMode.Parallel   ->
                                convolveParallelPipeline(
                                    task.image, config.kernels, mode.mode,
                                    config.workerThreads, config.workerGridRows, config.workerGridCols
                                )
                        }
                    }
                    sumProcessMs.addAndGet(ms)
                    outputChannel.send(ImageResult(task.index, task.sourcePath, result, ms))
                }
            }
        }

        launch {
            workers.joinAll()
            outputChannel.close()
        }

        val writer = launch(Dispatchers.IO) {
            for (res in outputChannel) {
                val ms = measureTimeMillis {
                    if (outputDir != null) {
                        val outFile = outputFilePath(res.sourcePath, outputDir, res.index)
                        outFile.parentFile?.mkdirs()
                        ImageIO.write(res.result.toBufferedImage(), "png", outFile)
                    }
                }
                sumWriteMs.addAndGet(ms)
                val done = doneCount.incrementAndGet()
                onProgress?.invoke(done, total)
            }
        }
        reader.join()
        writer.join()
    }

    PipelineStats(
        totalImages = total,
        totalMs = totalMs,
        sumReadMs = sumReadMs.get(),
        sumProcessMs = sumProcessMs.get(),
        sumWriteMs = sumWriteMs.get(),
        throughput = total * 1000.0 / totalMs.coerceAtLeast(1)
    )
}

fun outputFilePath(sourcePath: String, outputDir: String, index: Int): File {
    val base = File(sourcePath).nameWithoutExtension
    return File(outputDir, "${base}_${"%03d".format(index)}.png")
}

private val IMAGE_EXTENSIONS = setOf("png", "jpg", "jpeg", "bmp", "gif")

fun collectImagePaths(path: String): List<String> {
    val f = File(path)
    return if (f.isDirectory)
        f.listFiles { file -> file.extension.lowercase() in IMAGE_EXTENSIONS }
            ?.map { it.path }
            ?.sorted()
            ?: emptyList()
    else
        listOf(path)
}

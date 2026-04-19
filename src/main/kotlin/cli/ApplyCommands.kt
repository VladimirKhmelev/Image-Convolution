package org.example.cli

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import org.example.convolution.GpuContext
import org.example.pipeline.*
import javax.imageio.ImageIO
import java.io.File
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

fun runApply(cmd: CliCommand.Apply) {
    if (cmd.kernelNames.isEmpty()) {
        println("Укажите один или несколько фильтров для применения.")
        println("Доступные фильтры: ${Kernels.all.keys.joinToString()}")
        println("Для полного бенчмарка всех стратегий добавьте --benchmark.")
        return
    }

    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val threads = resolveThreads(cmd.maxThreads)
    val mode = parseStrategy(cmd.strategy)

    val (effectiveGridRows, effectiveGridCols) = if (cmd.tileSize != null) {
        tileToGrid(cmd.tileSize, image.width, image.height)
    } else {
        val autoR = maxOf(1, sqrt(threads.toDouble()).toInt())
        (cmd.gridRows ?: autoR) to (cmd.gridCols ?: ((threads + autoR - 1) / autoR))
    }

    val ks = cmd.kernelNames.map { name ->
        Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
    }
    val effectiveKernels = if (cmd.compose && ks.size > 1) listOf(ks.composed()) else ks

    val filterLabel = buildString {
        append(cmd.kernelNames.joinToString(" → "))
        if (cmd.compose && ks.size > 1) {
            val c = effectiveKernels[0]
            append(" (составной ${c.size}×${c[0].size})")
        }
    }

    val strategyLabel = when (mode) {
        is ConvolutionMode.Sequential -> mode.label
        is ConvolutionMode.GPU        -> "${mode.label}${GpuContext.deviceName()?.let { " [$it]" } ?: ""}"
        is ConvolutionMode.Parallel   -> when (mode.mode) {
            ParallelMode.BY_GRID -> "${mode.label} ${effectiveGridRows}×${effectiveGridCols}, $threads потоков"
            else                 -> "${mode.label}, $threads потоков"
        }
    }

    println("Изображение : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Стратегия   : $strategyLabel")
    println("Фильтры     : $filterLabel")
    if (cmd.tileSize != null)
        println("Тайл        : ${cmd.tileSize}px → сетка ${effectiveGridRows}×${effectiveGridCols}")

    val src = image.toGrayImage()
    val dst: GrayImage
    val ms = measureTimeMillis {
        dst = runBlocking {
            when (mode) {
                is ConvolutionMode.Sequential ->
                    convolveSequentialPipeline(src, effectiveKernels)
                is ConvolutionMode.GPU        ->
                    convolveGpuPipeline(src, effectiveKernels)
                is ConvolutionMode.Parallel   ->
                    convolveParallelPipeline(src, effectiveKernels, mode.mode, threads, effectiveGridRows, effectiveGridCols)
            }
        }
    }
    println("Время       : $ms мс")

    if (cmd.outputPath != null) {
        val outFile = File(cmd.outputPath).let {
            if ('.' in it.name) it else File("${cmd.outputPath}.png")
        }
        ImageIO.write(dst.toBufferedImage(), "png", outFile)
        println("Сохранено   : ${outFile.path}")
    } else {
        println("(Результат не сохранён — укажите --output <путь> для сохранения)")
    }
}

fun runPipelineApply(cmd: CliCommand.PipelineApply) {
    if (cmd.inputPaths.isEmpty()) { println("Нет входных изображений."); return }
    if (cmd.kernelNames.isEmpty()) {
        println("Укажите один или несколько фильтров.")
        println("Доступные: ${Kernels.all.keys.joinToString()}")
        return
    }

    val kernels = cmd.kernelNames.map { name ->
        Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
    }

    val available = Runtime.getRuntime().availableProcessors()
    val workers   = (cmd.workers ?: available).coerceAtLeast(1)
    val threads   = (cmd.workerThreads ?: 1).coerceAtLeast(1)
    val mode      = parseStrategy(cmd.workerStrategy)

    val config = PipelineConfig(
        kernels          = kernels,
        workerCount      = workers,
        workerMode       = mode,
        workerThreads    = threads,
        workerGridRows   = cmd.workerGridRows ?: 0,
        workerGridCols   = cmd.workerGridCols ?: 0,
        inputBufferSize  = cmd.inputBufferSize  ?: (workers * 2),
        outputBufferSize = cmd.outputBufferSize ?: (workers * 2)
    )

    println("Изображений  : ${cmd.inputPaths.size}")
    println("Воркеры      : $workers  |  Режим: ${mode.label}" +
            if (mode is ConvolutionMode.Parallel) "  |  Потоков/воркер: $threads" else "")
    println("Буферы       : вход=${config.inputBufferSize}  выход=${config.outputBufferSize}")
    println("Фильтры      : ${cmd.kernelNames.joinToString(" → ")}")
    if (cmd.outputDir != null) println("Вывод        : ${cmd.outputDir}")
    println()

    val stats = runBlocking {
        runPipeline(cmd.inputPaths, cmd.outputDir, config) { done, total ->
            print("\r  обработано: $done / $total")
        }
    }
    println()
    println("Всего        : ${stats.totalMs} мс  (${"%,.1f".format(stats.throughput)} изобр/с)")
    println("  чтение     : ${stats.sumReadMs} мс (суммарно)")
    println("  свёртка    : ${stats.sumProcessMs} мс (суммарно)")
    println("  запись     : ${stats.sumWriteMs} мс (суммарно)")
}

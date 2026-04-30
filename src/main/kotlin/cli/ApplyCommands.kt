package org.example.cli

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import javax.imageio.ImageIO
import java.io.File
import kotlin.system.measureTimeMillis

fun runApply(cmd: CliCommand.Apply) {
    if (cmd.kernelNames.isEmpty()) {
        println("Укажите один или несколько фильтров для применения.")
        println("Доступные фильтры: ${Kernels.all.keys.joinToString()}")
        println("Для бенчмарка всех стратегий добавьте --benchmark.")
        return
    }

    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val threads = resolveThreads(cmd.maxThreads)
    val mode    = parseStrategy(cmd.strategy)

    val (effectiveGridRows, effectiveGridCols) = if (cmd.tileSize != null) {
        tileToGrid(cmd.tileSize, image.width, image.height)
    } else {
        val (autoR, autoC) = autoGrid(threads)
        (cmd.gridRows ?: autoR) to (cmd.gridCols ?: autoC)
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
        is ConvolutionMode.Parallel -> when (mode.mode) {
            ParallelMode.BY_GRID -> "${mode.label} ${effectiveGridRows}×${effectiveGridCols}, $threads потоков"
            else -> "${mode.label}, $threads потоков"
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

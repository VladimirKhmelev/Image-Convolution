package org.example.cli

import org.example.convolution.ConvolutionMode
import org.example.convolution.ParallelMode

sealed class CliCommand {

    data class Apply(
        val imagePath: String,
        val kernelNames: List<String>,
        val strategy: String? = null,
        val maxThreads: Int? = null,
        val gridRows: Int? = null,
        val gridCols: Int? = null,
        val tileSize: Int? = null,
        val compose: Boolean = false,
        val outputPath: String? = null
    ) : CliCommand()

    data class Benchmark(
        val imagePath: String,
        val kernelNames: List<String>,
        val maxThreads: Int? = null,
        val gridRows: Int? = null,
        val gridCols: Int? = null,
        val tileSize: Int? = null,
        val compose: Boolean = false,
        val csvPath: String? = null,
        val kernelSizes: Set<Int> = emptySet()
    ) : CliCommand()
}

internal fun parseStrategy(s: String?): ConvolutionMode = when (s?.lowercase()?.trim()) {
    null, "seq", "sequential" -> ConvolutionMode.Sequential
    "pixels", "pixel" -> ConvolutionMode.Parallel(ParallelMode.BY_PIXEL)
    "rows",   "row"  -> ConvolutionMode.Parallel(ParallelMode.BY_ROW)
    "cols",   "col", "columns", "column" -> ConvolutionMode.Parallel(ParallelMode.BY_COLUMN)
    "grid" -> ConvolutionMode.Parallel(ParallelMode.BY_GRID)
    else -> error("Неизвестная стратегия: \"$s\". Допустимые: seq, pixels, rows, cols, grid")
}

internal fun resolveThreads(max: Int?): Int {
    val available = Runtime.getRuntime().availableProcessors()
    if (max == null) return available
    require(max >= 1) { "--threads должно быть >= 1, получено: $max" }
    return if (max > available) {
        println("Предупреждение: --threads=$max > доступных процессоров ($available), используем $available")
        available
    } else max
}

internal fun tileToGrid(tileSize: Int, imageW: Int, imageH: Int): Pair<Int, Int> {
    require(tileSize >= 1) { "--tile-size должен быть >= 1" }
    return (imageH + tileSize - 1) / tileSize to (imageW + tileSize - 1) / tileSize
}

fun parseArgs(args: Array<String>): CliCommand {
    val flagsWithValues = setOf(
        "threads", "grid-rows", "grid-cols", "tile-size", "output", "strategy", "csv", "kernel-size"
    )

    val valueIndices = mutableSetOf<Int>()
    for (i in args.indices) {
        if (args[i].startsWith("--") && !args[i].contains('=')) {
            val name = args[i].removePrefix("--")
            if (name in flagsWithValues && i + 1 < args.size) valueIndices.add(i + 1)
        }
    }
    val positional = args.filterIndexed { i, arg -> !arg.startsWith("--") && i !in valueIndices }

    fun parseInt(name: String): Int? {
        val eqArg = args.firstOrNull { it.startsWith("--$name=") }
        if (eqArg != null) {
            val raw = eqArg.removePrefix("--$name=")
            return raw.toIntOrNull() ?: error("Неверное значение --$name: \"$raw\", ожидается целое число")
        }
        val idx = args.indexOfFirst { it == "--$name" }
        if (idx >= 0) {
            val raw = args.getOrNull(idx + 1) ?: error("--$name требует значение")
            return raw.toIntOrNull() ?: error("Неверное значение --$name: \"$raw\", ожидается целое число")
        }
        return null
    }

    fun parseStr(name: String): String? {
        val eqArg = args.firstOrNull { it.startsWith("--$name=") }
        if (eqArg != null) return eqArg.removePrefix("--$name=")
        val idx = args.indexOfFirst { it == "--$name" }
        if (idx >= 0) return args.getOrNull(idx + 1) ?: error("--$name требует значение")
        return null
    }

    val imagePath = positional.getOrNull(0) ?: error("Укажите путь к изображению первым аргументом")
    val kernelNames = positional.drop(1)
    val benchmark = args.any { it == "--benchmark" }
    val compose  = args.any { it == "--compose" }

    return if (benchmark) {
        CliCommand.Benchmark(
            imagePath = imagePath,
            kernelNames = kernelNames,
            maxThreads = parseInt("threads"),
            gridRows = parseInt("grid-rows"),
            gridCols = parseInt("grid-cols"),
            tileSize = parseInt("tile-size"),
            compose = compose,
            csvPath = parseStr("csv"),
            kernelSizes = parseStr("kernel-size")
                ?.split(",")
                ?.map { it.trim().toIntOrNull() ?: error("Неверное значение --kernel-size: \"${it.trim()}\"") }
                ?.toSet()
                ?: emptySet()
        )
    } else {
        CliCommand.Apply(
            imagePath = imagePath,
            kernelNames = kernelNames,
            strategy  = parseStr("strategy"),
            maxThreads = parseInt("threads"),
            gridRows = parseInt("grid-rows"),
            gridCols = parseInt("grid-cols"),
            tileSize = parseInt("tile-size"),
            compose  = compose,
            outputPath  = parseStr("output")
        )
    }
}

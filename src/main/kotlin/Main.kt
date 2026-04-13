package org.example

import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import org.example.ui.AppContent

fun main(args: Array<String>) {
    if (args.isNotEmpty()) {
        // Если аргументы есть — работаем в консольном режиме, иначе запускаем GUI
        val flagsWithValues = setOf("threads", "grid-rows", "grid-cols", "tile-size", "output", "strategy", "csv")

        val valueIndices = mutableSetOf<Int>()
        for (i in args.indices) {
            if (args[i].startsWith("--") && !args[i].contains('=')) {
                val name = args[i].removePrefix("--")
                if (name in flagsWithValues && i + 1 < args.size) valueIndices.add(i + 1)
            }
        }
        val nonFlags = args.filterIndexed { i, arg -> !arg.startsWith("--") && i !in valueIndices }

        val imagePath   = nonFlags.getOrNull(0) ?: error("Укажите путь к изображению первым аргументом")
        val kernelNames = nonFlags.drop(1)

        fun parseIntFlag(name: String): Int? {
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

        fun parseStringFlag(name: String): String? {
            val eqArg = args.firstOrNull { it.startsWith("--$name=") }
            if (eqArg != null) return eqArg.removePrefix("--$name=")
            val idx = args.indexOfFirst { it == "--$name" }
            if (idx >= 0) return args.getOrNull(idx + 1) ?: error("--$name требует значение")
            return null
        }

        val maxThreads = parseIntFlag("threads")
        val gridRows   = parseIntFlag("grid-rows")
        val gridCols   = parseIntFlag("grid-cols")
        val tileSize   = parseIntFlag("tile-size")
        val compose    = args.any { it == "--compose" }
        val benchmark  = args.any { it == "--benchmark" }
        val outputPath = parseStringFlag("output")
        val strategy   = parseStringFlag("strategy")
        val csvPath    = parseStringFlag("csv")

        if (benchmark) {
            runBenchmark(
                imagePath   = imagePath,
                kernelNames = kernelNames,
                maxThreads  = maxThreads,
                gridRows    = gridRows,
                gridCols    = gridCols,
                tileSize    = tileSize,
                compose     = compose,
                csvPath     = csvPath
            )
        } else {
            runApply(
                imagePath   = imagePath,
                kernelNames = kernelNames,
                strategyStr = strategy,
                maxThreads  = maxThreads,
                gridRows    = gridRows,
                gridCols    = gridCols,
                tileSize    = tileSize,
                compose     = compose,
                outputPath  = outputPath
            )
        }
        return
    }

    // GUI режим
    application {
        Window(
            onCloseRequest = ::exitApplication,
            title = "Image Convolution",
            state = rememberWindowState(size = DpSize(1100.dp, 750.dp))
        ) {
            AppContent()
        }
    }
}

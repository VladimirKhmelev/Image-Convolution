package org.example.cli

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import javax.imageio.ImageIO
import java.io.File
import java.io.PrintWriter
import java.util.Locale
import kotlin.math.roundToInt
import kotlin.math.sqrt

private class Stats(rawTimes: List<Long>) {
    val n = rawTimes.size
    val mean = rawTimes.average()
    val std = sqrt(rawTimes.sumOf { t -> (t - mean) * (t - mean) } / n)
    val cv = if (mean > 0) std / mean * 100.0 else 0.0

    private val sorted = rawTimes.sorted()

    fun percentile(p: Double): Double {
        require(p in 0.0..100.0)
        if (n == 1) return sorted[0].toDouble()
        val idx  = p / 100.0 * (n - 1)
        val lo   = idx.toInt().coerceIn(0, n - 2)
        val frac = idx - lo
        return sorted[lo] * (1 - frac) + sorted[lo + 1] * frac
    }

    val p50 = percentile(50.0)
    val p95 = percentile(95.0)
    val p99 = percentile(99.0)

    val isNoisy get() = cv > 10.0
}

private data class BenchEntry(
    val label:    String,
    val mode:     ConvolutionMode,
    val gridRows: Int = 0,
    val gridCols: Int = 0
)

private data class CsvRow(
    val filter:   String,
    val strategy: String,
    val threads:  Int,
    val width:    Int,
    val height:   Int,
    val meanMs:   Double,
    val stdMs:    Double,
    val p50Ms:    Double,
    val p95Ms:    Double,
    val speedup:  Double
)

private fun runEntry(
    entry: BenchEntry,
    src: GrayImage,
    kernels: List<Array<FloatArray>>,
    threads: Int
) = runBlocking {
    when (val mode = entry.mode) {
        is ConvolutionMode.Sequential ->
            kernels.fold(src, ::convolveSequential)
        is ConvolutionMode.Parallel   ->
            kernels.fold(src) { acc, k ->
                convolveParallel(acc, k, mode.mode, threads, entry.gridRows, entry.gridCols)
            }
    }
}

private fun buildEntries(threads: Int, gridRows: Int?, gridCols: Int?): List<BenchEntry> = buildList {
    add(BenchEntry("Последовательный", ConvolutionMode.Sequential))
    add(BenchEntry("По пикселям", ConvolutionMode.Parallel(ParallelMode.BY_PIXEL)))
    add(BenchEntry("По строкам", ConvolutionMode.Parallel(ParallelMode.BY_ROW)))
    add(BenchEntry("По столбцам", ConvolutionMode.Parallel(ParallelMode.BY_COLUMN)))

    if (gridRows != null && gridCols != null) {
        add(BenchEntry("По сетке (${gridRows}×${gridCols})", ConvolutionMode.Parallel(ParallelMode.BY_GRID), gridRows, gridCols))
    } else {
        val (autoR, autoC) = autoGrid(threads)
        listOf(autoR to autoC, 1 to threads, threads to 1)
            .distinct()
            .forEach { (r, c) ->
                add(BenchEntry("По сетке (${r}×${c})", ConvolutionMode.Parallel(ParallelMode.BY_GRID), r, c))
            }
    }
}

fun runBenchmark(cmd: CliCommand.Benchmark) {
    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val threads = resolveThreads(cmd.maxThreads)

    val (effectiveGridRows, effectiveGridCols) = if (cmd.tileSize != null)
        tileToGrid(cmd.tileSize, image.width, image.height).let { (r, c) -> r as Int? to c as Int? }
    else cmd.gridRows to cmd.gridCols

    val available = Runtime.getRuntime().availableProcessors()
    println("Изображение : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Процессоров : $available  |  Потоков : $threads")
    if (cmd.tileSize != null)
        println("Тайл        : ${cmd.tileSize}px → сетка ${effectiveGridRows}×${effectiveGridCols}")
    println()

    val warmupIterations = 10
    val measuredIterations = 30
    val src = image.toGrayImage()

    val kernelConfigs: List<Pair<String, List<Array<FloatArray>>>> = when {
        cmd.kernelNames.isEmpty() -> {
            val allKernels = if (cmd.kernelSizes.isEmpty()) {
                Kernels.all.entries.toList()
            } else {
                Kernels.all.entries.filter { (_, k) -> k.size in cmd.kernelSizes }
            }
            require(allKernels.isNotEmpty()) {
                "Нет фильтров с размером ядра ${cmd.kernelSizes.sorted().joinToString()}. " +
                "Доступные размеры: ${Kernels.all.values.map { it.size }.toSortedSet().joinToString()}"
            }
            allKernels.map { (name, k) -> name to listOf(k) }
        }
        else -> {
            val ks = cmd.kernelNames.map { name ->
                Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
            }
            val label = cmd.kernelNames.joinToString(" → ")
            buildList {
                add(label to ks)
                if (cmd.compose && ks.size > 1) {
                    val composed = ks.composed()
                    add("$label (составной ${composed.size}×${composed[0].size})" to listOf(composed))
                }
            }
        }
    }

    val cLabel = 20
    val cVal = 10
    val cSd = 11
    val cPct = 9
    val cCv = 7
    val cSpd = 7
    val entries = buildEntries(threads, effectiveGridRows, effectiveGridCols)
    val csvRows = mutableListOf<CsvRow>()

    for ((configLabel, kernelList) in kernelConfigs) {
        println("═══ Фильтр: $configLabel")
        print("  прогрев: ")
        for (entry in entries) {
            repeat(warmupIterations) { runEntry(entry, src, kernelList, threads) }
            print("▪")
        }
        println()

        System.gc()
        Thread.sleep(300)

        // Замеряем всё в наносекундах
        val results = entries.map { entry ->
            val times = (1..measuredIterations).map {
                val t0 = System.nanoTime()
                runEntry(entry, src, kernelList, threads)
                System.nanoTime() - t0
            }
            entry to Stats(times)
        }

        // Определяем единицу по sequential
        val seqNs = results.first { it.first.mode is ConvolutionMode.Sequential }.second.mean
        val useUs = seqNs < 10_000_000.0  // < 10 мс → µс
        val div = if (useUs) 1_000.0 else 1_000_000.0
        val unit = if (useUs) "µс" else "мс"

        fun fmtVal(ns: Double) = "${"%,d".format((ns / div).roundToInt())} $unit"
        fun fmtStd(ns: Double) = "±${"%,d".format((ns / div).roundToInt())} $unit"

        val labelWidth = entries.maxOf { it.label.length }.coerceAtLeast(cLabel)
        println(
            "  " + "Стратегия".padEnd(labelWidth) + " " +
            "среднее".padStart(cVal)              + " " +
            "±std".padStart(cSd)                  + " " +
            "p50".padStart(cPct)                  + " " +
            "p95".padStart(cPct)                  + " " +
            "p99".padStart(cPct)                  + " " +
            "CV%".padStart(cCv)                   + " " +
            "ускор".padStart(cSpd)
        )
        val divider = "  " + "─".repeat(labelWidth) + " " +
            "─".repeat(cVal) + " " + "─".repeat(cSd) + " " +
            "─".repeat(cPct) + " " + "─".repeat(cPct) + " " +
            "─".repeat(cPct) + " " + "─".repeat(cCv) + " " + "─".repeat(cSpd)
        println(divider)

        var seqMeanNs: Double? = null
        for ((entry, s) in results) {
            if (entry.mode is ConvolutionMode.Sequential) seqMeanNs = s.mean
            val speedup = if (seqMeanNs != null && entry.mode !is ConvolutionMode.Sequential && s.mean > 0)
                "×${"%.2f".format(seqMeanNs / s.mean)}" else ""
            val cvStr = "${"%.1f".format(s.cv)}%${if (s.isNoisy) "!" else " "}"

            println(
                "  " + entry.label.padEnd(labelWidth) + " " +
                fmtVal(s.mean).padStart(cVal)          + " " +
                fmtStd(s.std).padStart(cSd)            + " " +
                fmtVal(s.p50).padStart(cPct)           + " " +
                fmtVal(s.p95).padStart(cPct)           + " " +
                fmtVal(s.p99).padStart(cPct)           + " " +
                cvStr.padStart(cCv)                    + " " +
                speedup.padStart(cSpd)
            )

            val msDiv = 1_000_000.0
            csvRows += CsvRow(
                filter = configLabel,
                strategy = entry.label,
                threads = threads,
                width = image.width,
                height = image.height,
                meanMs = s.mean / msDiv,
                stdMs = s.std  / msDiv,
                p50Ms = s.p50  / msDiv,
                p95Ms = s.p95  / msDiv,
                speedup = if (seqMeanNs != null && s.mean > 0) seqMeanNs / s.mean else 1.0
            )
        }
        println()
    }

    if (cmd.csvPath != null) {
        data class Col(val header: String, val values: List<String>) {
            val width = maxOf(header.length, values.maxOf { it.length })
            fun fmt(s: String) = s.padEnd(width)
        }
        val cols = listOf(
            Col("filter", csvRows.map { it.filter }),
            Col("strategy", csvRows.map { it.strategy }),
            Col("threads", csvRows.map { it.threads.toString() }),
            Col("width", csvRows.map { it.width.toString() }),
            Col("height",csvRows.map { it.height.toString() }),
            Col("mean_ms", csvRows.map { String.format(Locale.ROOT, "%.4f", it.meanMs) }),
            Col("std_ms", csvRows.map { String.format(Locale.ROOT, "%.4f", it.stdMs) }),
            Col("p50_ms", csvRows.map { String.format(Locale.ROOT, "%.4f", it.p50Ms) }),
            Col("p95_ms", csvRows.map { String.format(Locale.ROOT, "%.4f", it.p95Ms) }),
            Col("speedup", csvRows.map { String.format(Locale.ROOT, "%.4f", it.speedup) })
        )
        fun row(cells: List<String>) = cols.zip(cells).joinToString(", ") { (col, v) -> col.fmt(v) }

        val csvFile = File(cmd.csvPath)
        PrintWriter(csvFile.bufferedWriter()).use { pw ->
            pw.println(row(cols.map { it.header }))
            for (r in csvRows)
                pw.println(row(listOf(
                    r.filter, r.strategy, r.threads.toString(),
                    r.width.toString(), r.height.toString(),
                    String.format(Locale.ROOT, "%.4f", r.meanMs), String.format(Locale.ROOT, "%.4f", r.stdMs),
                    String.format(Locale.ROOT, "%.4f", r.p50Ms),  String.format(Locale.ROOT, "%.4f", r.p95Ms),
                    String.format(Locale.ROOT, "%.4f", r.speedup)
                )))
        }
        println("CSV → ${cmd.csvPath}")
    }

    println("─".repeat(80))
    println("n=$measuredIterations замеров / стратегию  |  прогрев=$warmupIterations  |  GC между фильтрами")
    println("CV > 10% (!) — нестабильные результаты, рекомендуется перезапуск")
}

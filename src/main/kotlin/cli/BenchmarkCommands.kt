package org.example.cli

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import org.example.pipeline.*
import javax.imageio.ImageIO
import java.io.File
import java.io.PrintWriter
import java.util.Locale
import kotlin.io.path.createTempDirectory
import kotlin.math.roundToInt
import kotlin.math.sqrt

private const val PIPELINE_MEASURED_ROUNDS = 5
private const val PIPELINE_WARMUP_IMAGES = 4

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
        is ConvolutionMode.Sequential -> convolveSequentialPipeline(src, kernels)
        is ConvolutionMode.Parallel -> convolveParallelPipeline(src, kernels, mode.mode, threads, entry.gridRows, entry.gridCols)
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

private fun runPipelineWorkerBenchmark(cmd: CliCommand.PipelineBenchmark) {
    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val kernels = if (cmd.kernelNames.isEmpty()) {
        listOf(Kernels.all.values.first())
    } else {
        cmd.kernelNames.map { name ->
            Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\"")
        }
    }
    val filterLabel = cmd.kernelNames.ifEmpty { listOf(Kernels.all.keys.first()) }.joinToString(" → ")
    val available = Runtime.getRuntime().availableProcessors()
    val fixedWorkers = (cmd.maxWorkers ?: 2).coerceAtLeast(1)
    val strategy = parseStrategy(cmd.workerStrategy ?: "rows")
    val strategyLabel = when (strategy) {
        is ConvolutionMode.Sequential -> "seq"
        is ConvolutionMode.Parallel   -> strategy.mode.name.lowercase()
    }

    val threadCounts = buildList {
        add(1)
        var t = 2
        while (t <= available) { add(t); t *= 2 }
        if (available !in this) add(available)
    }.distinct()

    val batchPaths = List(cmd.batchSize) { cmd.imagePath }
    val warmupBatch = List(PIPELINE_WARMUP_IMAGES) { cmd.imagePath }
    val outDir = if (cmd.noWrite) null else createTempDirectory("pipeline_wbench_").toFile()
        .also { dir -> Runtime.getRuntime().addShutdownHook(Thread { dir.deleteRecursively() }) }.path

    println("Изображение  : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Фильтры      : $filterLabel")
    println("Батч         : ${cmd.batchSize} изображений  |  Воркеров: $fixedWorkers")
    println("Стратегия    : $strategyLabel  |  Потоков: ${threadCounts.joinToString()}")
    println()

    data class WRow(
        val mode: String,
        val threads: Int,
        val throughput: Double,
        val totalMs: Long,
        val readMs: Long,
        val processMs: Long,
        val writeMs: Long
    )
    val rows = mutableListOf<WRow>()

    fun bench(workerMode: ConvolutionMode, threads: Int): WRow {
        val config = PipelineConfig(
            kernels = kernels,
            workerCount = fixedWorkers,
            workerMode = workerMode,
            workerThreads = threads
        )
        runBlocking { runPipeline(warmupBatch, outDir, config) }
        System.gc()
        Thread.sleep(200)
        val stats = (1..PIPELINE_MEASURED_ROUNDS).map { runBlocking { runPipeline(batchPaths, outDir, config) } }
        val modeLabel = when (workerMode) {
            is ConvolutionMode.Sequential -> "seq"
            is ConvolutionMode.Parallel -> "${workerMode.mode.name.lowercase()}×$threads"
        }
        return WRow(
            mode = modeLabel, threads = threads,
            throughput = stats.map { it.throughput }.average(),
            totalMs = stats.map { it.totalMs }.average().roundToInt().toLong(),
            readMs = stats.map { it.sumReadMs }.average().roundToInt().toLong(),
            processMs = stats.map { it.sumProcessMs }.average().roundToInt().toLong(),
            writeMs = stats.map { it.sumWriteMs }.average().roundToInt().toLong()
        )
    }

    val seqRow = bench(ConvolutionMode.Sequential, 1)
    rows += seqRow
    println("  seq:  ${"%6.1f".format(seqRow.throughput)} изобр/с")

    for (t in threadCounts) {
        val row = bench(strategy, t)
        rows += row
        val speedup = row.throughput / seqRow.throughput
        println("  ${strategyLabel}×$t: ${"%6.1f".format(row.throughput)} изобр/с  ×${"%.2f".format(speedup)}")
    }
    println()

    if (cmd.csvPath != null) {
        PrintWriter(File(cmd.csvPath).bufferedWriter()).use { pw ->
            pw.println("filter,worker_mode,workers,worker_threads,width,height,throughput_img_s,total_ms,sum_read_ms,sum_process_ms,sum_write_ms,speedup")
            val base = seqRow.throughput.coerceAtLeast(0.001)
            for (r in rows)
                pw.println("$filterLabel,${r.mode},$fixedWorkers,${r.threads},${image.width},${image.height}," +
                    "${String.format(Locale.ROOT, "%.2f", r.throughput)},${r.totalMs},${r.readMs},${r.processMs},${r.writeMs}," +
                    String.format(Locale.ROOT, "%.4f", r.throughput / base))
        }
        println("CSV → ${cmd.csvPath}")
    }
    println("n=$PIPELINE_MEASURED_ROUNDS прогонов / конфигурацию  |  прогрев=$PIPELINE_WARMUP_IMAGES изображения")
}

fun runPipelineBenchmark(cmd: CliCommand.PipelineBenchmark) {
    if (cmd.varyThreads) { runPipelineWorkerBenchmark(cmd); return }
    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val kernels = if (cmd.kernelNames.isEmpty()) {
        listOf(Kernels.all.values.first())
    } else {
        cmd.kernelNames.map { name ->
            Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\"")
        }
    }
    val filterLabel = cmd.kernelNames.ifEmpty { listOf(Kernels.all.keys.first()) }.joinToString(" → ")

    val available = Runtime.getRuntime().availableProcessors()
    val topWorkers = cmd.maxWorkers ?: available
    val batchPaths = List(cmd.batchSize) { cmd.imagePath }
    val tmpOutDir = createTempDirectory("pipeline_bench_").toFile()
        .also { dir -> Runtime.getRuntime().addShutdownHook(Thread { dir.deleteRecursively() }) }

    println("Изображение  : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Фильтры      : $filterLabel")
    println("Батч         : ${cmd.batchSize} изображений")
    println("Процессоров  : $available")
    println()

    val workerCounts = buildList {
        add(1)
        var w = 2
        while (w <= topWorkers) { add(w); w *= 2 }
        if (topWorkers !in this) add(topWorkers)
    }.distinct()

    val warmupBatch = List(PIPELINE_WARMUP_IMAGES) { cmd.imagePath }

    data class BenchConfig(val label: String, val workerCount: Int)

    val benchConfigs = workerCounts.map { wc -> BenchConfig("$wc воркеров", wc) }

    val cThroughput = 14
    val cTotal = 12
    val cRead = 11
    val cProcess = 12
    val cWrite = 11
    val cLabel = benchConfigs.maxOf { it.label.length }.coerceAtLeast(11)

    fun printDivider() = println(
        "  " + "─".repeat(cLabel) + " " + "─".repeat(cThroughput) + " " +
        "─".repeat(cTotal)        + " " + "─".repeat(cRead)        + " " +
        "─".repeat(cProcess)      + " " + "─".repeat(cWrite)       + " " + "─".repeat(7)
    )

    println(
        "  " + "конфигурация".padEnd(cLabel) + " " +
        "изобр/с".padStart(cThroughput) + " " +
        "всего мс".padStart(cTotal)  + " " +
        "чтение мс".padStart(cRead) + " " +
        "свёртка мс".padStart(cProcess) + " " +
        "запись мс".padStart(cWrite) + " " +
        "ускор".padStart(7)
    )
    printDivider()

    data class Row(val label: String, val workers: Int,
           val throughput: Double, val totalMs: Long,
           val readMs: Long, val processMs: Long, val writeMs: Long)
    val rows = mutableListOf<Row>()
    var baseThroughput: Double? = null

    for (bc in benchConfigs) {
        val config = PipelineConfig(kernels = kernels, workerCount = bc.workerCount)

        val outDir = if (cmd.noWrite) null else tmpOutDir.path
        runBlocking { runPipeline(warmupBatch, outDir, config) }
        System.gc(); Thread.sleep(200)

        val statsList = (1..PIPELINE_MEASURED_ROUNDS).map { runBlocking { runPipeline(batchPaths, outDir, config) } }

        val avgThroughput = statsList.map { it.throughput }.average()
        val avgTotal = statsList.map { it.totalMs }.average().roundToInt().toLong()
        val avgRead = statsList.map { it.sumReadMs }.average().roundToInt().toLong()
        val avgProcess = statsList.map { it.sumProcessMs }.average().roundToInt().toLong()
        val avgWrite = statsList.map { it.sumWriteMs }.average().roundToInt().toLong()

        if (baseThroughput == null) baseThroughput = avgThroughput
        val speedup = avgThroughput / baseThroughput

        rows += Row(bc.label, bc.workerCount, avgThroughput, avgTotal, avgRead, avgProcess, avgWrite)

        println(
            "  " + bc.label.padEnd(cLabel) + " " +
            "%.1f".format(avgThroughput).padStart(cThroughput) + " " +
            "$avgTotal".padStart(cTotal)  + " " +
            "$avgRead".padStart(cRead)  + " " +
            "$avgProcess".padStart(cProcess)  + " " +
            "$avgWrite".padStart(cWrite)  + " " +
            "×${"%.2f".format(speedup)}".padStart(7)
        )
    }
    println()

    if (cmd.csvPath != null) {
        PrintWriter(File(cmd.csvPath).bufferedWriter()).use { pw ->
            pw.println("filter,workers,batch_size,width,height,throughput_img_s,total_ms,sum_read_ms,sum_process_ms,sum_write_ms,speedup")
            val base = rows.firstOrNull()?.throughput ?: 1.0
            for (r in rows)
                pw.println("$filterLabel,${r.workers},${cmd.batchSize},${image.width},${image.height}," +
                    "${String.format(Locale.ROOT, "%.2f", r.throughput)},${r.totalMs},${r.readMs},${r.processMs},${r.writeMs}," +
                    String.format(Locale.ROOT, "%.4f", r.throughput / base))
        }
        println("CSV → ${cmd.csvPath}")
    }

    println("n=$PIPELINE_MEASURED_ROUNDS прогонов / конфигурацию  |  прогрев=$PIPELINE_WARMUP_IMAGES изображения")
}

package org.example.cli

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import org.example.convolution.GpuContext
import org.example.pipeline.*
import javax.imageio.ImageIO
import java.io.File
import java.io.PrintWriter
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

private class Stats(rawTimes: List<Long>) {
    val n    = rawTimes.size
    val mean = rawTimes.average()
    val std  = sqrt(rawTimes.sumOf { t -> (t - mean) * (t - mean) } / n)
    val cv   = if (mean > 0) std / mean * 100.0 else 0.0

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
    entry:   BenchEntry,
    src:     GrayImage,
    kernels: List<Array<FloatArray>>,
    threads: Int
) = runBlocking {
    when (val mode = entry.mode) {
        is ConvolutionMode.Sequential -> convolveSequentialPipeline(src, kernels)
        is ConvolutionMode.GPU        -> convolveGpuPipeline(src, kernels)
        is ConvolutionMode.Parallel   -> convolveParallelPipeline(src, kernels, mode.mode, threads, entry.gridRows, entry.gridCols)
    }
}

private fun buildEntries(threads: Int, gridRows: Int?, gridCols: Int?): List<BenchEntry> = buildList {
    add(BenchEntry("Последовательный", ConvolutionMode.Sequential))
    if (GpuContext.isAvailable())
        add(BenchEntry("GPU (OpenCL) [${GpuContext.deviceName()}]", ConvolutionMode.GPU))
    add(BenchEntry("По пикселям",      ConvolutionMode.Parallel(ParallelMode.BY_PIXEL)))
    add(BenchEntry("По строкам",       ConvolutionMode.Parallel(ParallelMode.BY_ROW)))
    add(BenchEntry("По столбцам",      ConvolutionMode.Parallel(ParallelMode.BY_COLUMN)))

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
    else
        cmd.gridRows to cmd.gridCols

    val available = Runtime.getRuntime().availableProcessors()
    println("Изображение : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Процессоров : $available  |  Потоков : $threads")
    if (cmd.tileSize != null)
        println("Тайл        : ${cmd.tileSize}px → сетка ${effectiveGridRows}×${effectiveGridCols}")
    println()

    val warmupIterations   = 10
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
    val cMs = 9
    val cSd = 9
    val cPct = 8
    val cCv = 7
    val cSpd = 7

    fun fmtMs(v: Double)  = "${"%,d".format(v.roundToInt())} мс"
    fun fmtSd(v: Double)  = "±${fmtMs(v)}"

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

        val labelWidth = entries.maxOf { it.label.length }.coerceAtLeast(cLabel)
        println(
            "  " + "Стратегия".padEnd(labelWidth) + " " +
            "среднее".padStart(cMs)               + " " +
            "±std".padStart(cSd)                  + " " +
            "p50".padStart(cPct)                  + " " +
            "p95".padStart(cPct)                  + " " +
            "p99".padStart(cPct)                  + " " +
            "CV%".padStart(cCv)                   + " " +
            "ускор".padStart(cSpd)
        )
        println(
            "  " + "(все значения в мс)".padEnd(labelWidth) + " " +
            "(n=$measuredIterations)".padStart(cMs + 1 + cSd) +
            " " + "интерполяция по методу C2".padStart(cPct * 3 + 2)
        )
        val divider = "  " + "─".repeat(labelWidth) + " " +
            "─".repeat(cMs) + " " + "─".repeat(cSd) + " " +
            "─".repeat(cPct) + " " + "─".repeat(cPct) + " " +
            "─".repeat(cPct) + " " +
            "─".repeat(cCv)  + " " + "─".repeat(cSpd)
        println(divider)

        var seqMean: Double? = null

        for (entry in entries) {
            val times = mutableListOf<Long>()
            repeat(measuredIterations) {
                times += measureTimeMillis { runEntry(entry, src, kernelList, threads) }
            }

            val s = Stats(times)
            if (entry.mode is ConvolutionMode.Sequential) seqMean = s.mean

            val speedup = if (seqMean != null && entry.mode !is ConvolutionMode.Sequential && s.mean > 0)
                "×${"%.2f".format(seqMean / s.mean)}" else ""
            val cvStr = "${"%.1f".format(s.cv)}%${if (s.isNoisy) "!" else " "}"

            println(
                "  " + entry.label.padEnd(labelWidth) + " " +
                fmtMs(s.mean).padStart(cMs)           + " " +
                fmtSd(s.std).padStart(cSd)            + " " +
                fmtMs(s.p50).padStart(cPct)          + " " +
                fmtMs(s.p95).padStart(cPct)          + " " +
                fmtMs(s.p99).padStart(cPct)          + " " +
                cvStr.padStart(cCv)                   + " " +
                speedup.padStart(cSpd)
            )

            csvRows += CsvRow(
                filter   = configLabel,
                strategy = entry.label,
                threads  = threads,
                width    = image.width,
                height   = image.height,
                meanMs   = s.mean,
                stdMs    = s.std,
                p50Ms    = s.p50,
                p95Ms    = s.p95,
                speedup  = if (seqMean != null && s.mean > 0) seqMean / s.mean else 1.0
            )
        }
        println()
    }

    if (cmd.csvPath != null) {
        val csvFile = File(cmd.csvPath)
        val writeHeader = !csvFile.exists() || csvFile.length() == 0L
        PrintWriter(csvFile.bufferedWriter()).use { pw ->
            if (writeHeader)
                pw.println("filter,strategy,threads,width,height,mean_ms,std_ms,p50_ms,p95_ms,speedup")
            for (r in csvRows)
                pw.println("${r.filter},${r.strategy},${r.threads},${r.width},${r.height}," +
                    "${"%.2f".format(r.meanMs)},${"%.2f".format(r.stdMs)}," +
                    "${"%.2f".format(r.p50Ms)},${"%.2f".format(r.p95Ms)}," +
                    "%.4f".format(r.speedup))
        }
        println("CSV → ${cmd.csvPath}")
    }

    println("─".repeat(72))
    println("n=$measuredIterations замеров / стратегию  |  прогрев=$warmupIterations  |  GC между фильтрами")
    println("CV > 10% (!) — нестабильные результаты, рекомендуется перезапуск")
    println("p99 при n=$measuredIterations — верхний ~1% выборки (≈ топ-1 значение)")
}

fun runPipelineBenchmark(cmd: CliCommand.PipelineBenchmark) {
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

    val available  = Runtime.getRuntime().availableProcessors()
    val topWorkers = cmd.maxWorkers ?: available
    val batchPaths = List(cmd.batchSize) { cmd.imagePath }

    println("Изображение  : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Фильтры      : $filterLabel")
    println("Батч         : ${cmd.batchSize} изображений")
    println("Процессоров  : $available")
    println()

    val workerCounts = buildList {
        add(1)
        var w = 2; while (w <= topWorkers) { add(w); w *= 2 }
        if (topWorkers !in this) add(topWorkers)
    }.distinct()

    val warmupBatch    = List(4) { cmd.imagePath }
    val measuredRounds = 5

    val cW          = 10
    val cThroughput = 14
    val cTotal      = 12
    val cRead       = 11
    val cProcess    = 12
    val cWrite      = 11

    println(
        "  " + "воркеры".padStart(cW)   + " " +
        "изобр/с".padStart(cThroughput) + " " +
        "всего мс".padStart(cTotal)     + " " +
        "чтение мс".padStart(cRead)     + " " +
        "свёртка мс".padStart(cProcess) + " " +
        "запись мс".padStart(cWrite)    + " " +
        "ускор".padStart(7)
    )
    println(
        "  " + "─".repeat(cW) + " " + "─".repeat(cThroughput) + " " +
        "─".repeat(cTotal)    + " " + "─".repeat(cRead)        + " " +
        "─".repeat(cProcess)  + " " + "─".repeat(cWrite)       + " " + "─".repeat(7)
    )

    data class Row(val workers: Int, val throughput: Double, val totalMs: Long,
                   val readMs: Long, val processMs: Long, val writeMs: Long)
    val rows = mutableListOf<Row>()
    var baseThroughput: Double? = null

    for (wc in workerCounts) {
        val config = PipelineConfig(kernels = kernels, workerCount = wc)

        runBlocking { runPipeline(warmupBatch, null, config) }
        System.gc(); Thread.sleep(200)

        val statsList = (1..measuredRounds).map { runBlocking { runPipeline(batchPaths, null, config) } }

        val avgThroughput = statsList.map { it.throughput   }.average()
        val avgTotal      = statsList.map { it.totalMs      }.average().roundToInt().toLong()
        val avgRead       = statsList.map { it.sumReadMs    }.average().roundToInt().toLong()
        val avgProcess    = statsList.map { it.sumProcessMs }.average().roundToInt().toLong()
        val avgWrite      = statsList.map { it.sumWriteMs   }.average().roundToInt().toLong()

        if (baseThroughput == null) baseThroughput = avgThroughput
        val speedup = avgThroughput / baseThroughput

        rows += Row(wc, avgThroughput, avgTotal, avgRead, avgProcess, avgWrite)

        println(
            "  " + "$wc".padStart(cW)                             + " " +
            "%.1f".format(avgThroughput).padStart(cThroughput)    + " " +
            "$avgTotal".padStart(cTotal)                           + " " +
            "$avgRead".padStart(cRead)                             + " " +
            "$avgProcess".padStart(cProcess)                       + " " +
            "$avgWrite".padStart(cWrite)                           + " " +
            "×${"%.2f".format(speedup)}".padStart(7)
        )
    }
    println()

    if (cmd.csvPath != null) {
        PrintWriter(File(cmd.csvPath).bufferedWriter()).use { pw ->
            pw.println("filter,workers,batch_size,throughput_img_s,total_ms,sum_read_ms,sum_process_ms,sum_write_ms,speedup")
            val base = rows.firstOrNull()?.throughput ?: 1.0
            for (r in rows)
                pw.println("$filterLabel,${r.workers},${cmd.batchSize}," +
                    "${"%.2f".format(r.throughput)},${r.totalMs},${r.readMs},${r.processMs},${r.writeMs}," +
                    "%.4f".format(r.throughput / base))
        }
        println("CSV → ${cmd.csvPath}")
    }

    println("n=$measuredRounds прогонов / конфигурацию  |  прогрев=4 изображения")
}

package org.example

import kotlinx.coroutines.runBlocking
import org.example.convolution.*
import javax.imageio.ImageIO
import java.io.File
import java.io.PrintWriter
import kotlin.math.roundToInt
import kotlin.math.sqrt
import kotlin.system.measureTimeMillis

/**
 * Быстрый однократный запуск обработки изображения с заданными фильтрами и стратегией.
 * Используется как из командной строки, так и для тестов.
 */
fun runApply(
    imagePath:   String,
    kernelNames: List<String> = emptyList(),
    strategyStr: String?      = null,
    maxThreads:  Int?         = null,
    gridRows:    Int?         = null,
    gridCols:    Int?         = null,
    tileSize:    Int?         = null,
    compose:     Boolean      = false,
    outputPath:  String?      = null
) {
    // Нужен хотя бы один фильтр
    if (kernelNames.isEmpty()) {
        println("Укажите один или несколько фильтров для применения.")
        println("Доступные фильтры: ${Kernels.all.keys.joinToString()}")
        println("Для полного бенчмарка всех стратегий добавьте --benchmark.")
        return
    }

    // Чтение изображения
    val file = File(imagePath)
    require(file.exists()) { "Файл не найден: $imagePath" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: $imagePath")

    val available = Runtime.getRuntime().availableProcessors()
    val threads = if (maxThreads != null) {
        require(maxThreads >= 1) { "--threads должно быть >= 1, получено: $maxThreads" }
        if (maxThreads > available) {
            println("Предупреждение: --threads=$maxThreads > доступных процессоров ($available), используем $available")
            available
        } else maxThreads
    } else available

    // Преобразование строки стратегии в тип ConvolutionMode
    val mode: ConvolutionMode = when (strategyStr?.lowercase()?.trim()) {
        null, "seq", "sequential"              -> ConvolutionMode.Sequential
        "pixels", "pixel"                      -> ConvolutionMode.Parallel(ParallelMode.BY_PIXEL)
        "rows", "row"                          -> ConvolutionMode.Parallel(ParallelMode.BY_ROW)
        "cols", "col", "columns", "column"     -> ConvolutionMode.Parallel(ParallelMode.BY_COLUMN)
        "grid"                                 -> ConvolutionMode.Parallel(ParallelMode.BY_GRID)
        else -> error(
            "Неизвестная стратегия: \"$strategyStr\". " +
            "Допустимые: seq, pixels, rows, cols, grid"
        )
    }

    // Расчёт параметров сетки для режима BY_GRID
    val effectiveGridRows: Int
    val effectiveGridCols: Int
    if (tileSize != null) {
        require(tileSize >= 1) { "--tile-size должен быть >= 1" }
        effectiveGridRows = (image.height + tileSize - 1) / tileSize
        effectiveGridCols = (image.width  + tileSize - 1) / tileSize
    } else {
        val autoR = maxOf(1, sqrt(threads.toDouble()).toInt())
        effectiveGridRows = gridRows ?: autoR
        effectiveGridCols = gridCols ?: ((threads + autoR - 1) / autoR)
    }

    // Получение ядер по именам и (опционально) их композиция
    val ks = kernelNames.map { name ->
        Kernels.all[name] ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
    }
    val effectiveKernels = if (compose && ks.size > 1) listOf(ks.composed()) else ks

    val filterLabel = buildString {
        append(kernelNames.joinToString(" → "))
        if (compose && ks.size > 1) {
            val c = effectiveKernels[0]
            append(" (составной ${c.size}×${c[0].size})")
        }
    }

    val strategyLabel = when (mode) {
        is ConvolutionMode.Sequential -> mode.label
        is ConvolutionMode.Parallel   -> when (mode.mode) {
            ParallelMode.BY_GRID -> "${mode.label} ${effectiveGridRows}×${effectiveGridCols}, $threads потоков"
            else                 -> "${mode.label}, $threads потоков"
        }
    }

    println("Изображение : $imagePath (${image.width}×${image.height})")
    println("Стратегия   : $strategyLabel")
    println("Фильтры     : $filterLabel")
    if (tileSize != null)
        println("Тайл        : ${tileSize}px → сетка ${effectiveGridRows}×${effectiveGridCols}")

    // Преобразование в оттенки серого и выполнение свёртки
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

    // Сохранение результата (если указан выходной путь)
    if (outputPath != null) {
        val outFile = File(outputPath).let {
            if ('.' in it.name) it else File("$outputPath.png")
        }
        ImageIO.write(dst.toBufferedImage(), "png", outFile)
        println("Сохранено   : ${outFile.path}")
    } else {
        println("(результат не сохранён — укажите --output <путь> для сохранения)")
    }
}

// Статистика для бенчмарка и графиков

private class Stats(rawTimes: List<Long>) {
    val n    = rawTimes.size                                                  // количество замеров
    val mean = rawTimes.average()                                             // среднее арифметическое
    val std  = sqrt(rawTimes.sumOf { t -> (t - mean) * (t - mean) } / n)  // стандартное отклонение (σ)
    val cv   = if (mean > 0) std / mean * 100.0 else 0.0                      // коэффициент вариации, %

    private val sorted = rawTimes.sorted()

    // Вычисление перцинтелей методом линейной интерполяции между соседними элементами
    fun percentile(p: Double): Double {
        require(p in 0.0..100.0)
        val idx = p / 100.0 * (n - 1)
        val lo  = idx.toInt().coerceIn(0, n - 2)
        val hi  = lo + 1
        val frac = idx - lo
        return sorted[lo] * (1 - frac) + sorted[hi] * frac
    }

    val p50 = percentile(50.0) // медиана
    val p75 = percentile(75.0)
    val p95 = percentile(95.0)
    val p99 = percentile(99.0)

    // CV > 10% — результаты нестабильны из-за фоновых процессов или недостаточного прогрева JIT
    val isNoisy get() = cv > 10.0
}

// Сравнение всех стратегий

// Описание одного бенчмарка - метка, режим и параметры сетки
private data class BenchEntry(
    val label: String,
    val mode: ConvolutionMode,
    val gridRows: Int = 0,
    val gridCols: Int = 0
)

// Запуск одного прогона
private fun runEntry(
    entry: BenchEntry,
    src: GrayImage,
    kernels: List<Array<FloatArray>>,
    threads: Int
) = runBlocking {
    when (val mode = entry.mode) {
        is ConvolutionMode.Sequential -> convolveSequentialPipeline(src, kernels)
        is ConvolutionMode.Parallel   -> convolveParallelPipeline(src, kernels, mode.mode, threads, entry.gridRows, entry.gridCols)
    }
}

// Генерация списка стратегий для тестирования
private fun buildEntries(threads: Int, gridRows: Int?, gridCols: Int?): List<BenchEntry> = buildList {
    add(BenchEntry("Последовательный",  ConvolutionMode.Sequential))
    add(BenchEntry("По пикселям",       ConvolutionMode.Parallel(ParallelMode.BY_PIXEL)))
    add(BenchEntry("По строкам",        ConvolutionMode.Parallel(ParallelMode.BY_ROW)))
    add(BenchEntry("По столбцам",       ConvolutionMode.Parallel(ParallelMode.BY_COLUMN)))

    if (gridRows != null && gridCols != null) {
        add(BenchEntry("По сетке (${gridRows}×${gridCols})", ConvolutionMode.Parallel(ParallelMode.BY_GRID), gridRows, gridCols))
    } else {
        // Автоматический подбор нескольких конфигураций сетки
        val autoR = maxOf(1, sqrt(threads.toDouble()).toInt())
        val autoC = (threads + autoR - 1) / autoR
        listOf(
            autoR   to autoC,   // близко к квадрату
            1       to threads,  // горизонтальные полосы
            threads to 1,        // вертикальные полосы
        ).distinct().forEach { (r, c) ->
            add(BenchEntry("По сетке (${r}×${c})", ConvolutionMode.Parallel(ParallelMode.BY_GRID), r, c))
        }
    }
}

// одна строка CSV-результата
private data class CsvRow(
    val filter:    String, // Описание фильтра
    val strategy:  String, // Название стратегии
    val threads:   Int,    // Кол-во потоков
    val width:     Int,    // Ширина изображения
    val height:    Int,    // Высота изображения
    val meanMs:    Double, // Среднее время
    val stdMs:     Double, // Стандартное отклонение
    val p50Ms:     Double, // Медиана
    val p95Ms:     Double, // 95-ый перцентиль
    val speedup:   Double   // 1.0 для sequential
)

// Запускает все стратегии на заданных фильтрах, выводит таблицу с детальной статистикой и сохраняет csv отчёт
fun runBenchmark(
    imagePath:   String,
    kernelNames: List<String> = emptyList(),
    maxThreads:  Int?         = null,
    gridRows:    Int?         = null,
    gridCols:    Int?         = null,
    tileSize:    Int?         = null,
    compose:     Boolean      = false,
    csvPath:     String?      = null,
    kernelSizes: Set<Int>     = emptySet()
) {
    val file = File(imagePath)
    require(file.exists()) { "Файл не найден: $imagePath" }

    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: $imagePath")

    val available = Runtime.getRuntime().availableProcessors()
    val threads = if (maxThreads != null) {
        require(maxThreads >= 1) { "--threads должно быть >= 1, получено: $maxThreads" }
        if (maxThreads > available) {
            println("Предупреждение: --threads=$maxThreads > доступных процессоров ($available), используем $available")
            available
        } else maxThreads
    } else available

    // Преобразование tileSize в размеры сетки
    val effectiveGridRows: Int?
    val effectiveGridCols: Int?
    if (tileSize != null) {
        require(tileSize >= 1) { "--tile-size должен быть >= 1" }
        effectiveGridRows = (image.height + tileSize - 1) / tileSize
        effectiveGridCols = (image.width  + tileSize - 1) / tileSize
    } else {
        effectiveGridRows = gridRows
        effectiveGridCols = gridCols
    }

    println("Изображение : $imagePath (${image.width}×${image.height})")
    println("Процессоров : $available  |  Потоков : $threads")
    if (tileSize != null)
        println("Тайл        : ${tileSize}px → сетка ${effectiveGridRows}×${effectiveGridCols}")
    println()

    // Настройки прогрева JIT и замеров
    val warmupIterations   = 10
    val measuredIterations = 30

    val src = image.toGrayImage()

    // Формирование списка конфигураций фильтров для тестирования
    //
    //  - Нет имён → каждый фильтр по отдельности (с фильтрацией по --kernel-size если задан)
    //  - Одно имя → один фильтр
    //  - Несколько → цепочка (pipeline), --compose добавляет вариант с составным ядром для сравнения
    //
    val kernelConfigs: List<Pair<String, List<Array<FloatArray>>>> = when {
        kernelNames.isEmpty() -> {
            val allKernels = if (kernelSizes.isEmpty()) {
                Kernels.all.entries.toList()
            } else {
                Kernels.all.entries.filter { (_, k) -> k.size in kernelSizes }
            }
            require(allKernels.isNotEmpty()) {
                "Нет фильтров с размером ядра ${kernelSizes.sorted().joinToString()}. " +
                "Доступные размеры: ${Kernels.all.values.map { it.size }.toSortedSet().joinToString()}"
            }
            allKernels.map { (name, k) -> name to listOf(k) }
        }
        else -> {
            val ks = kernelNames.map { name ->
                Kernels.all[name] ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
            }
            val label = kernelNames.joinToString(" → ")
            buildList {
                add(label to ks)
                if (compose && ks.size > 1) {
                    val composed = ks.composed()
                    add("$label (составной ${composed.size}×${composed[0].size})" to listOf(composed))
                }
            }
        }
    }

    // Форматирование
    val cLabel = 20
    val cMs    = 9
    val cSd    = 9
    val cPct   = 8
    val cCv    = 7
    val cSpd   = 7

    fun fmtMs(v: Double)  = "${"%,d".format(v.roundToInt())} мс"
    fun fmtSd(v: Double)  = "±${"%,d".format(v.roundToInt())} мс"
    fun fmtPct(v: Double) = "${"%,d".format(v.roundToInt())} мс"

    val entries = buildEntries(threads, effectiveGridRows, effectiveGridCols)
    val csvRows = mutableListOf<CsvRow>()

    // Основной цикл по конфигурациям фильтров
    for ((configLabel, kernelList) in kernelConfigs) {
        println("═══ Фильтр: $configLabel")

        // 1. Прогрев всех стратегий
        print("  прогрев: ")
        for (entry in entries) {
            repeat(warmupIterations) { runEntry(entry, src, kernelList, threads) }
            print("▪")
        }
        println()

        // 2. GC + пауза
        System.gc()
        Thread.sleep(300)

        // 3. Шапка таблицы
        val labelWidth = entries.maxOf { it.label.length }.coerceAtLeast(cLabel)
        println(
            "  " + "Стратегия".padEnd(labelWidth) + " " +
            "среднее".padStart(cMs)               + " " +
            "±std".padStart(cSd)                  + " " +
            "p50".padStart(cPct)                  + " " +
            "p75".padStart(cPct)                  + " " +
            "p95".padStart(cPct)                  + " " +
            "p99".padStart(cPct)                  + " " +
            "CV%".padStart(cCv)                   + " " +
            "ускор".padStart(cSpd)
        )
        println(
            "  " + "(все значения в мс)".padEnd(labelWidth) + " " +
            "(n=$measuredIterations)".padStart(cMs + 1 + cSd) +
            " " + "интерполяция по методу C2".padStart(cPct * 4 + 3)
        )
        val divider = "  " + "─".repeat(labelWidth) + " " +
            "─".repeat(cMs) + " " + "─".repeat(cSd) + " " +
            "─".repeat(cPct) + " " + "─".repeat(cPct) + " " +
            "─".repeat(cPct) + " " + "─".repeat(cPct) + " " +
            "─".repeat(cCv) + " " + "─".repeat(cSpd)
        println(divider)

        var seqMean: Double? = null

        // 4. Замеры
        for (entry in entries) {
            val times = mutableListOf<Long>()
            repeat(measuredIterations) {
                times += measureTimeMillis { runEntry(entry, src, kernelList, threads) }
            }

            val s = Stats(times)
            if (entry.mode is ConvolutionMode.Sequential) seqMean = s.mean

            val speedup = if (seqMean != null && entry.mode !is ConvolutionMode.Sequential && s.mean > 0)
                "×${"%.2f".format(seqMean / s.mean)}"
            else ""

            val cvStr = "${"%.1f".format(s.cv)}%${if (s.isNoisy) "!" else " "}"

            println(
                "  " + entry.label.padEnd(labelWidth)    + " " +
                fmtMs(s.mean).padStart(cMs)              + " " +
                fmtSd(s.std).padStart(cSd)               + " " +
                fmtPct(s.p50).padStart(cPct)             + " " +
                fmtPct(s.p75).padStart(cPct)             + " " +
                fmtPct(s.p95).padStart(cPct)             + " " +
                fmtPct(s.p99).padStart(cPct)             + " " +
                cvStr.padStart(cCv)                      + " " +
                speedup.padStart(cSpd)
            )

            val spd = if (seqMean != null && s.mean > 0) seqMean / s.mean else 1.0
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
                speedup  = spd
            )
        }
        println()
    }

    // Сохранение CSV
    if (csvPath != null) {
        val file = File(csvPath)
        val writeHeader = !file.exists() || file.length() == 0L
        PrintWriter(file.bufferedWriter().also { /* append = false, перезаписываем */ }).use { pw ->
            if (writeHeader)
                pw.println("filter,strategy,threads,width,height,mean_ms,std_ms,p50_ms,p95_ms,speedup")
            for (r in csvRows)
                pw.println("${r.filter},${r.strategy},${r.threads},${r.width},${r.height}," +
                    "${"%.2f".format(r.meanMs)},${"%.2f".format(r.stdMs)}," +
                    "${"%.2f".format(r.p50Ms)},${"%.2f".format(r.p95Ms)}," +
                        "%.4f".format(r.speedup)
                )
        }
        println("CSV → $csvPath")
    }

    println("─".repeat(72))
    println("n=$measuredIterations замеров / стратегию  |  прогрев=$warmupIterations  |  GC между фильтрами")
    println("CV > 10% (!) — нестабильные результаты, рекомендуется перезапуск")
    println("p99 при n=$measuredIterations — верхний ~1% выборки (≈ топ-1 значение)")
}

package org.example.convolution

import kotlinx.coroutines.*
import java.awt.image.BufferedImage
import kotlin.math.sqrt

/** Класс для хранения изображения в градациях серого */
class GrayImage(val width: Int, val height: Int, val data: FloatArray) {
    constructor(width: Int, height: Int) : this(width, height, FloatArray(width * height))

    operator fun get(y: Int, x: Int): Float = data[y * width + x]
    operator fun set(y: Int, x: Int, v: Float) { data[y * width + x] = v }
}

/**
 * Конвертация между BufferedImage и GrayImage
 * Используется стандартная формула Rec. 601: Y = 0.299*R + 0.587*G + 0.114*B
 */
fun BufferedImage.toGrayImage(): GrayImage {
    val w = width; val h = height
    val pixels = getRGB(0, 0, w, h, null, 0, w)
    val img = GrayImage(w, h)
    for (i in pixels.indices) {
        val rgb = pixels[i]
        val r = (rgb shr 16) and 0xFF
        val g = (rgb shr 8) and 0xFF
        val b = rgb and 0xFF
        img.data[i] = 0.299f * r + 0.587f * g + 0.114f * b
    }
    return img
}

/**  Преобразует GrayImage обратно в BufferedImage */
fun GrayImage.toBufferedImage(): BufferedImage {
    val out = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val pixels = IntArray(data.size) { i ->
        val v = data[i].coerceIn(0f, 255f).toInt()
        (v shl 16) or (v shl 8) or v
    }
    out.setRGB(0, 0, width, height, pixels, 0, width)
    return out
}

/**
 * Применяет ядро свёртки к одному пикселю (x, y) исходного изображения.
 * Обрабатывает границы: выход за пределы изображения игнорируется (не вносит вклад в сумму).
 */
private fun applyKernelAt(
    src: GrayImage,
    kernel: Array<FloatArray>,
    kH: Int, kW: Int, padY: Int, padX: Int,
    x: Int, y: Int
): Float {
    var sum = 0f
    for (ky in 0 until kH) {
        val sy = y + ky - padY
        if (sy !in 0 until src.height) continue
        val rowOffset = sy * src.width          // строка вычисляется один раз на ky для ускорения
        for (kx in 0 until kW) {
            val sx = x + kx - padX
            if (sx !in 0 until src.width) continue
            sum += src.data[rowOffset + sx] * kernel[ky][kx]
        }
    }
    return sum
}

// Задание 1 - последовательно
fun convolveSequential(src: GrayImage, kernel: Array<FloatArray>): GrayImage {
    val kH = kernel.size; val kW = kernel[0].size
    val padY = kH / 2;    val padX = kW / 2
    val dst = GrayImage(src.width, src.height)
    for (y in 0 until src.height)
        for (x in 0 until src.width)
            dst[y, x] = applyKernelAt(src, kernel, kH, kW, padY, padX, x, y)
    return dst
}

// Задание 2b - композиция фильтров
/**
 * Композиция двух ядер свёртки
 *
 * Если применить к изображению сначала фильтр с ядром k1, а затем фильтр с ядром k2,
 * то результат будет в точности таким же, как если бы применить один фильтр,
 * ядро которого равно свёртке k1 и k2: `result = k1 * k2`
 *
 * Позволяет объединить последовательность из нескольких фильтров
 * в один фильтр, что ускоряет обработку, особенно для больших изображений
 * @see - https://en.wikipedia.org/wiki/Convolution#Properties
 */
fun composeKernels(k1: Array<FloatArray>, k2: Array<FloatArray>): Array<FloatArray> {
    val h1 = k1.size; val w1 = k1[0].size
    val h2 = k2.size; val w2 = k2[0].size
    return Array(h1 + h2 - 1) { n ->
        FloatArray(w1 + w2 - 1) { m ->
            var sum = 0f
            for (k in 0 until h1) for (l in 0 until w1) {
                val nk = n - k; val ml = m - l
                if (nk in 0 until h2 && ml in 0 until w2)
                    sum += k1[k][l] * k2[nk][ml]
            }
            sum
        }
    }
}

/** Свёртка списка ядер в одно путём последовательной композиции */
fun List<Array<FloatArray>>.composed(): Array<FloatArray> = reduce(::composeKernels)

/** Применяет список ядер последовательно (одно за другим) в одном потоке */
fun convolveSequentialPipeline(
    src: GrayImage,
    kernels: List<Array<FloatArray>>
): GrayImage = kernels.fold(src, ::convolveSequential)

/** Применяет список ядер последовательно, но каждый шаг выполняется параллельно согласно заданному режиму */
suspend fun convolveParallelPipeline(
    src: GrayImage,
    kernels: List<Array<FloatArray>>,
    mode: ParallelMode,
    numThreads: Int = Runtime.getRuntime().availableProcessors(),
    gridRows: Int = 0,
    gridCols: Int = 0
): GrayImage = kernels.fold(src) { acc, k ->
    convolveParallel(acc, k, mode, numThreads, gridRows, gridCols)
}

// Задание 2 - параллельные режимы свёртки

enum class ParallelMode(val label: String) {
    BY_PIXEL ("По пикселям"),
    BY_ROW   ("По строкам"),
    BY_COLUMN("По столбцам"),
    BY_GRID  ("По сетке")
}

/** Выбор режима свёртки в пользовательском режиме */
sealed class ConvolutionMode(val label: String) {
    data object Sequential : ConvolutionMode("Последовательный")
    data class  Parallel(val mode: ParallelMode) : ConvolutionMode(mode.label)
    data object GPU : ConvolutionMode("GPU (OpenCL)")

    companion object {
        val all: List<ConvolutionMode> by lazy {
            listOf(Sequential, GPU) + ParallelMode.entries.map { Parallel(it) }
        }
    }
}

/**
 * Выполняет свёртку одного ядра с заданным изображением в параллельном режиме
 */
suspend fun convolveParallel(
    src: GrayImage,
    kernel: Array<FloatArray>,
    mode: ParallelMode,
    numThreads: Int = Runtime.getRuntime().availableProcessors(),
    gridRows: Int = 0,   // 0 = авто (√numThreads)
    gridCols: Int = 0    // 0 = авто
): GrayImage = withContext(Dispatchers.Default) {
    val h = src.height; val w = src.width
    val kH = kernel.size; val kW = kernel[0].size
    val padY = kH / 2;    val padX = kW / 2
    val dst = GrayImage(w, h)

    when (mode) {
        // Режим 1: разделение по отдельным пикселям (индексная арифметика)
        ParallelMode.BY_PIXEL -> {
            val total = h * w
            val chunk = (total + numThreads - 1) / numThreads
            (0 until numThreads).map { t ->
                async {
                    val start = t * chunk
                    val end   = minOf(start + chunk, total)
                    for (i in start until end) {
                        val py = i / w; val px = i % w
                        dst[py, px] = applyKernelAt(src, kernel, kH, kW, padY, padX, px, py)
                    }
                }
            }.awaitAll()
        }

        // Режим 2: разделение по строкам (каждая корутина получает непрерывный диапазон строк)
        ParallelMode.BY_ROW -> {
            val chunk = (h + numThreads - 1) / numThreads
            (0 until numThreads).map { t ->
                async {
                    for (y in (t * chunk) until minOf((t + 1) * chunk, h))
                        for (x in 0 until w)
                            dst[y, x] = applyKernelAt(src, kernel, kH, kW, padY, padX, x, y)
                }
            }.awaitAll()
        }

        // Режим 3: разделение по столбцам
        ParallelMode.BY_COLUMN -> {
            val chunk = (w + numThreads - 1) / numThreads
            (0 until numThreads).map { t ->
                async {
                    for (x in (t * chunk) until minOf((t + 1) * chunk, w))
                        for (y in 0 until h)
                            dst[y, x] = applyKernelAt(src, kernel, kH, kW, padY, padX, x, y)
                }
            }.awaitAll()
        }

        // Режим 4: разбиение на сетку прямоугольных блоков
        ParallelMode.BY_GRID -> {
            // Определяем количество строк и столбцов сетки
            val gridR = if (gridRows > 0) gridRows else maxOf(1, sqrt(numThreads.toDouble()).toInt())
            val gridC = if (gridCols > 0) gridCols else (numThreads + gridR - 1) / gridR
            val rowH  = (h + gridR - 1) / gridR
            val colW  = (w + gridC - 1) / gridC
            // Запускаем корутину для каждой ячейки сетки
            (0 until gridR).flatMap { gr ->
                (0 until gridC).map { gc ->
                    async {
                        val y0 = gr * rowH; val y1 = minOf(y0 + rowH, h)
                        val x0 = gc * colW; val x1 = minOf(x0 + colW, w)
                        for (y in y0 until y1)
                            for (x in x0 until x1)
                                dst[y, x] = applyKernelAt(src, kernel, kH, kW, padY, padX, x, y)
                    }
                }
            }.awaitAll()
        }
    }
    dst
}

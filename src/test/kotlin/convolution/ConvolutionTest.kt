package org.example.convolution

import kotlinx.coroutines.runBlocking
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*
import org.example.convolution.kernels.Kernels3x3
import org.example.convolution.kernels.Kernels5x5
import org.example.convolution.kernels.Kernels7x7
import org.example.convolution.kernels.Kernels9x9

// Погрешность при сравнении значений
private const val EPS = 2e-4f

private fun assertImagesEqual(
    expected: GrayImage,
    actual:   GrayImage,
    eps:      Float = EPS,
    label:    String = ""
) {
    assertEquals(expected.width,  actual.width,  "width mismatch $label")
    assertEquals(expected.height, actual.height, "height mismatch $label")
    for (i in expected.data.indices) {
        val diff = abs(expected.data[i] - actual.data[i])
        assertTrue(diff <= eps,
            "pixel $i (y=${i / expected.width}, x=${i % expected.width}): " +
            "expected=${expected.data[i]}, got=${actual.data[i]}, diff=$diff $label")
    }
}

/**
 * Сравнивает только внутреннюю область изображения, отступая [pad] пикселей от краёв
 */
private fun assertInteriorEqual(
    expected: GrayImage,
    actual:   GrayImage,
    pad:      Int,
    eps:      Float = EPS,
    label:    String = ""
) {
    assertEquals(expected.width,  actual.width)
    assertEquals(expected.height, actual.height)
    val w = expected.width
    val h = expected.height
    check(w > 2 * pad && h > 2 * pad) { "image too small for pad=$pad: ${w}×${h}" }
    for (y in pad until h - pad)
        for (x in pad until w - pad) {
            val diff = abs(expected[y, x] - actual[y, x])
            assertTrue(diff <= eps,
                "interior pixel ($y,$x): expected=${expected[y, x]}, " +
                "got=${actual[y, x]}, diff=$diff $label")
        }
}

/** Генерирует случайное изображение заданного размера с фиксированным seed */
private fun randomImageFast(width: Int, height: Int, seed: Long = 52): GrayImage {
    val rng = Random(seed)
    return GrayImage(width, height, FloatArray(width * height) { rng.nextFloat() * 255f })
}

private fun zeroImage(width: Int, height: Int) =
    GrayImage(width, height, FloatArray(width * height))

/**
 * Обрамляет ядро нулями по краям (добавляет по одной строке/столбцу сверху, снизу, слева, справа)
 * Результирующее ядро имеет размер (h+2)×(w+2)
 * Используется для проверки инвариантности свёртки относительно добавления нулевого обрамления
 */
private fun zeroPadKernel(k: Array<FloatArray>): Array<FloatArray> {
    val h = k.size + 2; val w = k[0].size + 2
    return Array(h) { y ->
        FloatArray(w) { x ->
            if (y in 1..k.size && x in 1..k[0].size) k[y - 1][x - 1] else 0f
        }
    }
}

/**
 * Тесты для autoGrid — функции разбиения потоков на прямоугольную сетку
 * Проверяют конкретные значения, инвариант покрытия и обработку недопустимых входов
 */
class AutoGridTest {

    /** autoGrid(1) должен вернуть сетку 1×1 */
    @Test fun `autoGrid(1) returns 1x1`() {
        assertEquals(1 to 1, autoGrid(1))
    }

    /** autoGrid(4) должен вернуть квадратную сетку 2×2 */
    @Test fun `autoGrid(4) returns 2x2`() {
        assertEquals(2 to 2, autoGrid(4))
    }

    /** Произведение r×c должно быть не меньше запрошенного числа потоков для любого n */
    @Test fun `autoGrid result covers all threads`() {
        for (n in 1..64) {
            val (r, c) = autoGrid(n)
            assertTrue(r * c >= n, "autoGrid($n) = ${r}x${c}, но ${r}x${c} < $n")
        }
    }

    /** autoGrid(0) должен бросить исключение — деление на ноль в BY_GRID */
    @Test fun `autoGrid(0) throws`() {
        assertFailsWith<IllegalArgumentException> { autoGrid(0) }
    }

    /** Отрицательное число потоков недопустимо */
    @Test fun `autoGrid negative throws`() {
        assertFailsWith<IllegalArgumentException> { autoGrid(-1) }
    }
}

// Задача 1 - последовательная свёртка

/**
 * Тесты для последовательной свёртки
 * Проверяют корректность на простых фильтрах, свойство композиции,
 * инвариантность относительно добавления нулевого обрамления,
 * а также работу на граничных размерах изображений
 */
class SequentialConvolutionTest {

    /** Тождественный фильтр не должен изменять изображение */
    @Test fun `identity filter returns same image`() {
        val img = randomImageFast(50, 50)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    /** Тождественный фильтр работает на любых размерах, включая 1×1, 1×N, N×1 */
    @Test fun `identity filter on various image sizes`() {
        for ((w, h) in listOf(1 to 1, 1 to 10, 10 to 1, 7 to 13, 100 to 100, 3 to 200)) {
            val img = randomImageFast(w, h, seed = w * 31L + h)
            assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY), label = "${w}×${h}")
        }
    }

    /** Нулевое ядро (все веса 0) должно давать полностью чёрное изображение */
    @Test fun `zero kernel produces all-black image`() {
        val img = randomImageFast(50, 50)
        val zeroKernel = Array(3) { FloatArray(3) }
        assertImagesEqual(zeroImage(50, 50), convolveSequential(img, zeroKernel))
    }

    /** Нулевое ядро размера 5×5 тоже даёт чёрное изображение */
    @Test fun `zero kernel of size 5x5 also produces all-black image`() {
        val img = randomImageFast(40, 40)
        assertImagesEqual(zeroImage(40, 40), convolveSequential(img, Array(5) { FloatArray(5) }))
    }

    /** Композиция двух ядер (Box Blur → Sharpen) эквивалентна составному ядру */
    @Test fun `composing two kernels equals sequential application (interior)`() {
        val img = randomImageFast(64, 64)
        val sequential = convolveSequential(convolveSequential(img, Kernels3x3.BOX_BLUR), Kernels3x3.SHARPEN)
        val composed   = convolveSequential(img, composeKernels(Kernels3x3.BOX_BLUR, Kernels3x3.SHARPEN))
        assertInteriorEqual(sequential, composed, pad = 2, eps = 1e-3f)
    }

    /** Композиция трёх ядер (Gaussian → Sharpen → Box) эквивалентна составному ядру */
    @Test fun `composing three kernels equals sequential application (interior)`() {
        val img = randomImageFast(64, 64)
        val ks = listOf(Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.SHARPEN, Kernels3x3.BOX_BLUR)
        val sequential = ks.fold(img, ::convolveSequential)
        val composed   = convolveSequential(img, ks.composed())
        assertInteriorEqual(sequential, composed, pad = 3, eps = 1e-2f)
    }

    /** Проверка, что расширение List<Array<FloatArray>>.composed() работает правильно */
    @Test fun `List composed() extension folds correctly`() {
        val img = randomImageFast(50, 50)
        val ks = listOf(Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.EDGE_DETECTION)
        val a = convolveSequential(convolveSequential(img, ks[0]), ks[1])
        val b = convolveSequential(img, ks.composed())
        assertInteriorEqual(a, b, pad = 2, eps = 1e-3f)
    }

    /** Добавление нулевой строки/столбца вокруг ядра не меняет результат свёртки */
    @Test fun `zero-padding a kernel does not change result`() {
        val img = randomImageFast(60, 60)
        for (kernel in listOf(Kernels3x3.BOX_BLUR, Kernels3x3.SHARPEN,
                              Kernels3x3.EDGE_DETECTION, Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.EMBOSS)) {
            val r1 = convolveSequential(img, kernel)
            val r2 = convolveSequential(img, zeroPadKernel(kernel))
            assertImagesEqual(r1, r2, label = "kernel size ${kernel.size}×${kernel[0].size}")
        }
    }

    /** Двукратное добавление нулевого обрамления также не меняет результат */
    @Test fun `double zero-padding a kernel does not change result`() {
        val img = randomImageFast(60, 60)
        val r1 = convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR)
        val r2 = convolveSequential(img, zeroPadKernel(zeroPadKernel(Kernels3x3.GAUSSIAN_BLUR)))
        assertImagesEqual(r1, r2)
    }

    /** Изображение 1×1 не должно вызывать исключений ни для одного предопределённого фильтра */
    @Test fun `1x1 image with identity`() {
        val img = GrayImage(1, 1, floatArrayOf(200f))
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    @Test fun `1x1 image with all predefined kernels does not throw`() {
        val img = GrayImage(1, 1, floatArrayOf(128f))
        for ((_, k) in Kernels.all) {
            convolveSequential(img, k)
        }
    }

    /** Изображение из одной строки (1×N) с тождественным фильтром */
    @Test fun `single-row image with identity`() {
        val img = randomImageFast(100, 1)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    /** Изображение из одного столбца (N×1) с тождественным фильтром */
    @Test fun `single-column image with identity`() {
        val img = randomImageFast(1, 100)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    /** Изображение размером ровно с ядром (3×3) — должно работать без ошибок */
    @Test fun `image exactly the same size as kernel`() {
        val img = randomImageFast(3, 3)
        val result = convolveSequential(img, Kernels3x3.IDENTITY)
        assertEquals(img[1, 1], result[1, 1])
    }

    /** Проверка инвариантности добавления нулей для случайных ядер размеров 1,3,5,7 */
    @Test fun `random odd-sized kernels zero-expansion invariance`() {
        val rng = Random(7)
        val img = randomImageFast(60, 60)
        for (kSize in listOf(1, 3, 5, 7)) {
            val kernel = Array(kSize) { FloatArray(kSize) { rng.nextFloat() * 2f - 1f } }
            val r1 = convolveSequential(img, kernel)
            val r2 = convolveSequential(img, zeroPadKernel(kernel))
            assertImagesEqual(r1, r2, eps = 1e-3f, label = "kSize=$kSize")
        }
    }
}

// Задача 2 - параллельная свёртка

/**
 * Тесты для параллельной свёртки (convolveParallel)
 * Проверяют, что все параллельные режимы (BY_PIXEL, BY_ROW, BY_COLUMN, BY_GRID)
 * дают тот же результат, что и последовательная версия, на разных размерах
 * изображений, ядрах и конфигурациях потоков
 */
class ParallelConvolutionTest {

    /**
     * Проверяет, что указанный параллельный режим для заданных
     * параметров выдаёт результат, идентичный последовательной свёртке
     */
    private fun checkMode(mode: ParallelMode, w: Int = 80, h: Int = 60,
                          kernel: Array<FloatArray> = Kernels3x3.GAUSSIAN_BLUR,
                          seed: Long = 52) = runBlocking {
        val img      = randomImageFast(w, h, seed)
        val expected = convolveSequential(img, kernel)
        for (threads in listOf(1, 2, 4, 8)) {
            val actual = convolveParallel(img, kernel, mode, numThreads = threads)
            assertImagesEqual(expected, actual, label = "$mode threads=$threads ${w}×${h}")
        }
    }

    @Test fun `BY_PIXEL matches sequential`()  = checkMode(ParallelMode.BY_PIXEL)
    @Test fun `BY_ROW matches sequential`()    = checkMode(ParallelMode.BY_ROW)
    @Test fun `BY_COLUMN matches sequential`() = checkMode(ParallelMode.BY_COLUMN)
    @Test fun `BY_GRID matches sequential`()   = checkMode(ParallelMode.BY_GRID)

    /** Все предопределённые фильтры работают корректно во всех параллельных режимах */
    @Test fun `all predefined kernels correct in all parallel modes`() = runBlocking {
        val img = randomImageFast(64, 48, seed = 123)
        for ((name, kernel) in Kernels.all) {
            val expected = convolveSequential(img, kernel)
            for (mode in ParallelMode.entries) {
                val actual = convolveParallel(img, kernel, mode, numThreads = 4)
                assertImagesEqual(expected, actual, label = "$name / $mode")
            }
        }
    }

    /** Явные формы сетки (различные соотношения строк/столбцов) должны работать правильно */
    @Test fun `BY_GRID explicit grid shapes match sequential`() = runBlocking {
        val img    = randomImageFast(100, 75)
        val kernel = Kernels3x3.EDGE_DETECTION
        val expected = convolveSequential(img, kernel)
        for ((gr, gc) in listOf(1 to 1, 2 to 3, 3 to 5, 7 to 4, 10 to 10)) {
            val actual = convolveParallel(img, kernel, ParallelMode.BY_GRID,
                numThreads = gr * gc, gridRows = gr, gridCols = gc)
            assertImagesEqual(expected, actual, label = "grid=${gr}×${gc}")
        }
    }

    /** Сетка, размер которой превышает размеры изображения, не должна вызывать ошибок */
    @Test fun `BY_GRID grid larger than image dimensions`() = runBlocking {
        val img    = randomImageFast(5, 5)
        val kernel = Kernels3x3.IDENTITY
        val expected = convolveSequential(img, kernel)
        val actual = convolveParallel(img, kernel, ParallelMode.BY_GRID,
            numThreads = 25, gridRows = 10, gridCols = 10)
        assertImagesEqual(expected, actual)
    }

    /** Параллельные режимы на маленьких и «неправильных» размерах изображений */
    @Test fun `parallel modes on small and odd-shaped images`() = runBlocking {
        for ((w, h) in listOf(1 to 1, 3 to 3, 5 to 7, 10 to 3, 2 to 100)) {
            val img      = randomImageFast(w, h, seed = w * 17L + h)
            val expected = convolveSequential(img, Kernels3x3.SHARPEN)
            for (mode in ParallelMode.entries) {
                val actual = convolveParallel(img, Kernels3x3.SHARPEN, mode, numThreads = 8)
                assertImagesEqual(expected, actual, label = "${w}×${h} / $mode")
            }
        }
    }

    /** Если потоков больше, чем строк (или пикселей), код не должен падать и давать правильный результат */
    @Test fun `more threads than rows does not crash or produce wrong result`() = runBlocking {
        val img      = randomImageFast(50, 3)   // only 3 rows
        val expected = convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR)
        for (mode in ParallelMode.entries) {
            val actual = convolveParallel(img, Kernels3x3.GAUSSIAN_BLUR, mode, numThreads = 16)
            assertImagesEqual(expected, actual, label = "$mode 16 threads on 3-row image")
        }
    }

    /** Случайные ядра разных размеров: параллельные режимы должны совпадать с последовательным */
    @Test fun `random odd-sized kernels parallel equals sequential`() = runBlocking {
        val rng = Random(52)
        val img = randomImageFast(64, 64)
        for (kSize in listOf(1, 3, 5, 7)) {
            val kernel = Array(kSize) { FloatArray(kSize) { rng.nextFloat() * 2f - 1f } }
            val expected = convolveSequential(img, kernel)
            for (mode in ParallelMode.entries) {
                val actual = convolveParallel(img, kernel, mode, numThreads = 4)
                assertImagesEqual(expected, actual, eps = 1e-3f, label = "${kSize}×${kSize} / $mode")
            }
        }
    }

    /** Параллельный конвейер должен давать тот же результат, что и последовательный конвейер */
    @Test fun `parallel pipeline matches sequential pipeline`() = runBlocking {
        val img = randomImageFast(64, 48)
        val kernels = listOf(Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.SHARPEN, Kernels3x3.BOX_BLUR)
        val expected = convolveSequentialPipeline(img, kernels)
        for (mode in ParallelMode.entries) {
            val actual = convolveParallelPipeline(img, kernels, mode, numThreads = 4)
            assertImagesEqual(expected, actual, label = "pipeline / $mode")
        }
    }
}

/**
 * Проверяются свойства, уникальные для крупных ядер:
 * граничные случаи по размеру, инвариантность нулевого обрамления и композиция.
 */
class LargeKernelTest {

    /** Изображение, равное по размеру ядру 5×5, 7×7, 9×9 — не должно падать */
    @Test fun `image exactly the same size as large kernel`() {
        for ((name, kernel) in listOf(
            "5×5 Gaussian" to Kernels5x5.GAUSSIAN_BLUR,
            "7×7 Gaussian" to Kernels7x7.GAUSSIAN_BLUR,
            "9×9 Gaussian" to Kernels9x9.GAUSSIAN_BLUR
        )) {
            val kH = kernel.size; val kW = kernel[0].size
            val img = randomImageFast(kW, kH, seed = kH * 31L)
            val result = convolveSequential(img, kernel)
            assertEquals(kW, result.width,  "width $name")
            assertEquals(kH, result.height, "height $name")
        }
    }

    /** Добавление нулевого обрамления вокруг ядер 5×5, 7×7, 9×9 не меняет результат */
    @Test fun `zero-padding invariance for large predefined kernels`() {
        val img = randomImageFast(60, 60)
        for ((name, kernel) in listOf(
            "5×5 Gaussian"  to Kernels5x5.GAUSSIAN_BLUR,
            "5×5 Sharpen"   to Kernels5x5.SHARPEN,
            "5×5 Edges"     to Kernels5x5.EDGE_DETECTION,
            "7×7 Gaussian"  to Kernels7x7.GAUSSIAN_BLUR,
            "7×7 Sharpen"   to Kernels7x7.SHARPEN,
            "9×9 Gaussian"  to Kernels9x9.GAUSSIAN_BLUR
        )) {
            val r1 = convolveSequential(img, kernel)
            val r2 = convolveSequential(img, zeroPadKernel(kernel))
            assertImagesEqual(r1, r2, eps = 1e-3f, label = name)
        }
    }

    /** Композиция двух 5×5 ядер должна давать ядро размером 9×9 */
    @Test fun `composing two 5x5 kernels yields 9x9 result size`() {
        val composed = composeKernels(Kernels5x5.GAUSSIAN_BLUR, Kernels5x5.SHARPEN)
        assertEquals(9, composed.size,    "composed height")
        assertEquals(9, composed[0].size, "composed width")
    }

    /** Композиция ядер 3×3 + 7×7 должна давать ядро размером 9×9 */
    @Test fun `composing 3x3 and 7x7 kernels yields 9x9 result size`() {
        val composed = composeKernels(Kernels3x3.GAUSSIAN_BLUR, Kernels7x7.GAUSSIAN_BLUR)
        assertEquals(9, composed.size,    "composed height")
        assertEquals(9, composed[0].size, "composed width")
    }

    /** Последовательное применение двух 5×5 ядер совпадает с применением их композиции на внутренних пикселях */
    @Test fun `composing two 5x5 kernels equals sequential pipeline (interior)`() {
        val img = randomImageFast(80, 80)
        val k1 = Kernels5x5.GAUSSIAN_BLUR
        val k2 = Kernels5x5.SHARPEN
        val sequential = convolveSequential(convolveSequential(img, k1), k2)
        val composed   = convolveSequential(img, composeKernels(k1, k2))
        assertInteriorEqual(sequential, composed, pad = 4, eps = 1e-2f)
    }

    /** Последовательное применение двух 7×7 ядер совпадает с применением их композиции */
    @Test fun `composing two 7x7 kernels equals sequential pipeline (interior)`() {
        val img = randomImageFast(100, 100)
        val k1 = Kernels7x7.GAUSSIAN_BLUR
        val k2 = Kernels7x7.SHARPEN
        val sequential = convolveSequential(convolveSequential(img, k1), k2)
        val composed   = convolveSequential(img, composeKernels(k1, k2))
        assertInteriorEqual(sequential, composed, pad = 6, eps = 1e-2f)
    }
}

package org.example.convolution

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.example.convolution.kernels.*

private const val GPU_EPS = 1e-2f

private fun assertGpuEqual(expected: GrayImage, actual: GrayImage, label: String = "") {
    assertEquals(expected.width,  actual.width,  "width mismatch $label")
    assertEquals(expected.height, actual.height, "height mismatch $label")
    for (i in expected.data.indices) {
        val diff = abs(expected.data[i] - actual.data[i])
        assertTrue(diff <= GPU_EPS,
            "pixel $i (y=${i / expected.width}, x=${i % expected.width}): " +
            "expected=${expected.data[i]}, got=${actual.data[i]}, diff=$diff $label")
    }
}

private fun rndImg(w: Int, h: Int, seed: Long = 52L): GrayImage {
    val rng = Random(seed)
    return GrayImage(w, h, FloatArray(w * h) { rng.nextFloat() * 255f })
}

/**
 * Все тесты автоматически пропускаются (assumeTrue), если OpenCL недоступен
 * Основное свойство: GPU и CPU должны давать одинаковый результат на любых данных
 */
class GpuConvolutionTest {

    private fun skipIfNoGpu() =
        assumeTrue(GpuContext.isAvailable(), "OpenCL недоступен — тест пропущен")

    /** isAvailable() не должен падать вне зависимости от наличия OpenCL на устройстве */
    @Test fun `GPU available or gracefully unavailable`() {
        val available = GpuContext.isAvailable()
        if (available) assertNotNull(GpuContext.deviceName())
    }

    /** Тождественный фильтр на GPU не должен изменять изображение */
    @Test fun `identity filter on GPU returns same image`() {
        skipIfNoGpu()
        val img = rndImg(64, 64)
        assertGpuEqual(img, convolveGpu(img, Kernels3x3.IDENTITY), "identity")
    }

    /** GPU-свёртка с Gaussian Blur совпадает с последовательной */
    @Test fun `GPU matches sequential for gaussian blur`() {
        skipIfNoGpu()
        val img = rndImg(80, 60)
        assertGpuEqual(
            convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR),
            convolveGpu(img, Kernels3x3.GAUSSIAN_BLUR),
            "gaussian 3x3"
        )
    }

    /** GPU-свёртка совпадает с последовательной для всех 24 предопределённых фильтров */
    @Test fun `GPU matches sequential for all predefined kernels`() {
        skipIfNoGpu()
        val img = rndImg(64, 48, seed = 7)
        for ((name, kernel) in Kernels.all) {
            assertGpuEqual(convolveSequential(img, kernel), convolveGpu(img, kernel), name)
        }
    }

    // Разные размеры изображений

    /** GPU корректно обрабатывает изображение размером 1×1 */
    @Test fun `GPU works on 1x1 image`() {
        skipIfNoGpu()
        val img = GrayImage(1, 1, floatArrayOf(128f))
        val result = convolveGpu(img, Kernels3x3.IDENTITY)
        assertEquals(1, result.width)
        assertEquals(1, result.height)
    }

    /** GPU корректно обрабатывает изображение из одной строки */
    @Test fun `GPU works on single-row image`() {
        skipIfNoGpu()
        val img = rndImg(100, 1)
        assertGpuEqual(
            convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR),
            convolveGpu(img, Kernels3x3.GAUSSIAN_BLUR),
            "1×100"
        )
    }

    /** GPU корректно обрабатывает изображение из одного столбца */
    @Test fun `GPU works on single-column image`() {
        skipIfNoGpu()
        val img = rndImg(1, 100)
        assertGpuEqual(
            convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR),
            convolveGpu(img, Kernels3x3.GAUSSIAN_BLUR),
            "100×1"
        )
    }

    /** GPU совпадает с последовательным на разнообразных размерах изображений */
    @Test fun `GPU matches sequential on various image sizes`() {
        skipIfNoGpu()
        val kernel = Kernels3x3.BOX_BLUR
        for ((w, h) in listOf(3 to 3, 7 to 13, 50 to 50, 120 to 80, 1 to 50, 50 to 1)) {
            val img = rndImg(w, h, seed = w * 31L + h)
            assertGpuEqual(convolveSequential(img, kernel), convolveGpu(img, kernel), "${w}×${h}")
        }
    }

    // Разные размеры ядра

    /** GPU совпадает с последовательным для фильтров 5×5 */
    @Test fun `GPU matches sequential for 5x5 kernels`() {
        skipIfNoGpu()
        val img = rndImg(80, 80, seed = 99)
        for ((name, kernel) in listOf(
            "5×5 Gaussian" to Kernels5x5.GAUSSIAN_BLUR,
            "5×5 Box Blur" to Kernels5x5.BOX_BLUR,
            "5×5 Sharpen"  to Kernels5x5.SHARPEN,
            "5×5 Edges"    to Kernels5x5.EDGE_DETECTION,
        )) {
            assertGpuEqual(convolveSequential(img, kernel), convolveGpu(img, kernel), name)
        }
    }

    /** GPU совпадает с последовательным для фильтров 7×7 и 9×9 */
    @Test fun `GPU matches sequential for 7x7 and 9x9 kernels`() {
        skipIfNoGpu()
        val img = rndImg(100, 100, seed = 13)
        for ((name, kernel) in listOf(
            "7×7 Gaussian" to Kernels7x7.GAUSSIAN_BLUR,
            "7×7 Sharpen"  to Kernels7x7.SHARPEN,
            "9×9 Gaussian" to Kernels9x9.GAUSSIAN_BLUR,
            "9×9 Box Blur" to Kernels9x9.BOX_BLUR,
        )) {
            assertGpuEqual(convolveSequential(img, kernel), convolveGpu(img, kernel), name)
        }
    }

    /** GPU совпадает с последовательным на случайных ядрах размеров от 1×1 до 9×9 */
    @Test fun `GPU matches sequential for random kernels of various sizes`() {
        skipIfNoGpu()
        val rng = Random(55)
        val img = rndImg(64, 64)
        for (kSize in listOf(1, 3, 5, 7, 9)) {
            val kernel = Array(kSize) { FloatArray(kSize) { rng.nextFloat() * 2f - 1f } }
            assertGpuEqual(
                convolveSequential(img, kernel),
                convolveGpu(img, kernel),
                "random ${kSize}×${kSize}"
            )
        }
    }

    // Pipeline

    /** GPU-конвейер из трёх фильтров совпадает с последовательным конвейером */
    @Test fun `GPU pipeline matches sequential pipeline`() {
        skipIfNoGpu()
        val img = rndImg(64, 48, seed = 21)
        val kernels = listOf(Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.SHARPEN, Kernels3x3.BOX_BLUR)
        assertGpuEqual(
            convolveSequentialPipeline(img, kernels),
            convolveGpuPipeline(img, kernels),
            "pipeline 3 kernels"
        )
    }

    /** GPU-конвейер из одного фильтра даёт тот же результат, что и прямой вызов convolveGpu */
    @Test fun `GPU single-kernel pipeline equals convolveGpu`() {
        skipIfNoGpu()
        val img = rndImg(50, 50)
        val kernel = Kernels3x3.GAUSSIAN_BLUR
        assertGpuEqual(
            convolveGpu(img, kernel),
            convolveGpuPipeline(img, listOf(kernel)),
            "single-kernel pipeline"
        )
    }

    // Стабильность контекста

    /** Повторные вызовы с одними данными возвращают одинаковый результат */
    @Test fun `multiple sequential GPU calls give consistent results`() {
        skipIfNoGpu()
        val img = rndImg(64, 64, seed = 77)
        val r1 = convolveGpu(img, Kernels3x3.GAUSSIAN_BLUR)
        val r2 = convolveGpu(img, Kernels3x3.GAUSSIAN_BLUR)
        assertGpuEqual(r1, r2, "repeated call")
    }

    /** Двадцать последовательных вызовов не приводят к утечкам или расхождению результата */
    @Test fun `GPU handles many back-to-back calls without leak`() {
        skipIfNoGpu()
        val img = rndImg(32, 32)
        val expected = convolveSequential(img, Kernels3x3.BOX_BLUR)
        repeat(20) { i ->
            assertGpuEqual(expected, convolveGpu(img, Kernels3x3.BOX_BLUR), "call #$i")
        }
    }
}

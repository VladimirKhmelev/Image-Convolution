package org.example.convolution

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*
import org.example.convolution.kernels.Kernels3x3

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

private fun randomImage(width: Int, height: Int, seed: Long = 52): GrayImage {
    val rng = Random(seed)
    return GrayImage(width, height, FloatArray(width * height) { rng.nextFloat() * 255f })
}

private fun zeroImage(width: Int, height: Int) =
    GrayImage(width, height, FloatArray(width * height))

private fun zeroPadKernel(k: Array<FloatArray>): Array<FloatArray> {
    val h = k.size + 2; val w = k[0].size + 2
    return Array(h) { y ->
        FloatArray(w) { x ->
            if (y in 1..k.size && x in 1..k[0].size) k[y - 1][x - 1] else 0f
        }
    }
}

class SequentialConvolutionTest {

    @Test fun `identity filter returns same image`() {
        val img = randomImage(50, 50)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    @Test fun `identity filter on various image sizes`() {
        for ((w, h) in listOf(1 to 1, 1 to 10, 10 to 1, 7 to 13, 100 to 100, 3 to 200)) {
            val img = randomImage(w, h, seed = w * 31L + h)
            assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY), label = "${w}×${h}")
        }
    }

    @Test fun `zero kernel produces all-black image`() {
        val img = randomImage(50, 50)
        assertImagesEqual(zeroImage(50, 50), convolveSequential(img, Array(3) { FloatArray(3) }))
    }

    @Test fun `zero kernel of size 5x5 also produces all-black image`() {
        val img = randomImage(40, 40)
        assertImagesEqual(zeroImage(40, 40), convolveSequential(img, Array(5) { FloatArray(5) }))
    }

    @Test fun `zero-padding a kernel does not change result`() {
        val img = randomImage(60, 60)
        for (kernel in listOf(Kernels3x3.BOX_BLUR, Kernels3x3.SHARPEN,
                              Kernels3x3.EDGE_DETECTION, Kernels3x3.GAUSSIAN_BLUR, Kernels3x3.EMBOSS)) {
            val r1 = convolveSequential(img, kernel)
            val r2 = convolveSequential(img, zeroPadKernel(kernel))
            assertImagesEqual(r1, r2, label = "kernel size ${kernel.size}×${kernel[0].size}")
        }
    }

    @Test fun `double zero-padding a kernel does not change result`() {
        val img = randomImage(60, 60)
        val r1 = convolveSequential(img, Kernels3x3.GAUSSIAN_BLUR)
        val r2 = convolveSequential(img, zeroPadKernel(zeroPadKernel(Kernels3x3.GAUSSIAN_BLUR)))
        assertImagesEqual(r1, r2)
    }

    @Test fun `1x1 image with identity`() {
        val img = GrayImage(1, 1, floatArrayOf(200f))
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    @Test fun `single-row image with identity`() {
        val img = randomImage(100, 1)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    @Test fun `single-column image with identity`() {
        val img = randomImage(1, 100)
        assertImagesEqual(img, convolveSequential(img, Kernels3x3.IDENTITY))
    }

    @Test fun `image exactly the same size as kernel`() {
        val img = randomImage(3, 3)
        val result = convolveSequential(img, Kernels3x3.IDENTITY)
        assertEquals(img[1, 1], result[1, 1])
    }

    @Test fun `random odd-sized kernels zero-expansion invariance`() {
        val rng = Random(7)
        val img = randomImage(60, 60)
        for (kSize in listOf(1, 3, 5, 7)) {
            val kernel = Array(kSize) { FloatArray(kSize) { rng.nextFloat() * 2f - 1f } }
            val r1 = convolveSequential(img, kernel)
            val r2 = convolveSequential(img, zeroPadKernel(kernel))
            assertImagesEqual(r1, r2, eps = 1e-3f, label = "kSize=$kSize")
        }
    }
}

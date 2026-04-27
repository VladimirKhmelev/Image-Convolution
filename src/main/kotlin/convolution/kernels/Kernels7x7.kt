package org.example.convolution.kernels

object Kernels7x7 {

    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
    )

    val BOX_BLUR = Array(7) { FloatArray(7) { 1f / 49f } }

    // Гауссовское размытие 7×7 (σ = 1.5)
    // Значения: exp(-(x²+y²) / (2·1.5²)), нормированы (сумма ≈ 1)
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf(0.0014f, 0.0043f, 0.0083f, 0.0104f, 0.0083f, 0.0043f, 0.0014f),
        floatArrayOf(0.0043f, 0.0131f, 0.0253f, 0.0316f, 0.0253f, 0.0131f, 0.0043f),
        floatArrayOf(0.0083f, 0.0253f, 0.0494f, 0.0617f, 0.0494f, 0.0253f, 0.0083f),
        floatArrayOf(0.0104f, 0.0316f, 0.0617f, 0.0771f, 0.0617f, 0.0316f, 0.0104f),
        floatArrayOf(0.0083f, 0.0253f, 0.0494f, 0.0617f, 0.0494f, 0.0253f, 0.0083f),
        floatArrayOf(0.0043f, 0.0131f, 0.0253f, 0.0316f, 0.0253f, 0.0131f, 0.0043f),
        floatArrayOf(0.0014f, 0.0043f, 0.0083f, 0.0104f, 0.0083f, 0.0043f, 0.0014f)
    )

    val SHARPEN = arrayOf(
        floatArrayOf( 0f,  0f,  0f, -1f,  0f,  0f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf(-1f, -1f, -1f, 25f, -1f, -1f, -1f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f,  0f,  0f, -1f,  0f,  0f,  0f)
    )

    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, 48f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f)
    )

    val EMBOSS = arrayOf(
        floatArrayOf(-6f, -5f, -4f, -3f, -2f, -1f, 0f),
        floatArrayOf(-5f, -4f, -3f, -2f, -1f,  0f, 1f),
        floatArrayOf(-4f, -3f, -2f, -1f,  0f,  1f, 2f),
        floatArrayOf(-3f, -2f, -1f,  1f,  1f,  2f, 3f),
        floatArrayOf(-2f, -1f,  0f,  1f,  2f,  3f, 4f),
        floatArrayOf(-1f,  0f,  1f,  2f,  3f,  4f, 5f),
        floatArrayOf( 0f,  1f,  2f,  3f,  4f,  5f, 6f)
    )
}

package org.example.convolution.kernels

object Kernels9x9 {

    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
    )

    val BOX_BLUR = Array(9) { FloatArray(9) { 1f / 81f } }

    // Гауссовское размытие 9×9 (σ = 2.0)
    // Значения: exp(-(x²+y²) / (2·2.0²)) = exp(-(x²+y²) / 8)
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf(0.0008f, 0.0018f, 0.0034f, 0.0050f, 0.0056f, 0.0050f, 0.0034f, 0.0018f, 0.0008f),
        floatArrayOf(0.0018f, 0.0044f, 0.0082f, 0.0119f, 0.0135f, 0.0119f, 0.0082f, 0.0044f, 0.0018f),
        floatArrayOf(0.0034f, 0.0082f, 0.0153f, 0.0223f, 0.0253f, 0.0223f, 0.0153f, 0.0082f, 0.0034f),
        floatArrayOf(0.0050f, 0.0119f, 0.0223f, 0.0325f, 0.0368f, 0.0325f, 0.0223f, 0.0119f, 0.0050f),
        floatArrayOf(0.0056f, 0.0135f, 0.0253f, 0.0368f, 0.0417f, 0.0368f, 0.0253f, 0.0135f, 0.0056f),
        floatArrayOf(0.0050f, 0.0119f, 0.0223f, 0.0325f, 0.0368f, 0.0325f, 0.0223f, 0.0119f, 0.0050f),
        floatArrayOf(0.0034f, 0.0082f, 0.0153f, 0.0223f, 0.0253f, 0.0223f, 0.0153f, 0.0082f, 0.0034f),
        floatArrayOf(0.0018f, 0.0044f, 0.0082f, 0.0119f, 0.0135f, 0.0119f, 0.0082f, 0.0044f, 0.0018f),
        floatArrayOf(0.0008f, 0.0018f, 0.0034f, 0.0050f, 0.0056f, 0.0050f, 0.0034f, 0.0018f, 0.0008f)
    )

    val SHARPEN = arrayOf(
        floatArrayOf( 0f,  0f,  0f,  0f, -1f,  0f,  0f,  0f,  0f),
        floatArrayOf( 0f,  0f,  0f, -1f, -1f, -1f,  0f,  0f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf(-1f, -1f, -1f, -1f, 41f, -1f, -1f, -1f, -1f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f,  0f,  0f, -1f, -1f, -1f,  0f,  0f,  0f),
        floatArrayOf( 0f,  0f,  0f,  0f, -1f,  0f,  0f,  0f,  0f)
    )

    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, 80f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f, -1f)
    )

    val EMBOSS = arrayOf(
        floatArrayOf(-8f, -7f, -6f, -5f, -4f, -3f, -2f, -1f, 0f),
        floatArrayOf(-7f, -6f, -5f, -4f, -3f, -2f, -1f,  0f, 1f),
        floatArrayOf(-6f, -5f, -4f, -3f, -2f, -1f,  0f,  1f, 2f),
        floatArrayOf(-5f, -4f, -3f, -2f, -1f,  0f,  1f,  2f, 3f),
        floatArrayOf(-4f, -3f, -2f, -1f,  1f,  1f,  2f,  3f, 4f),
        floatArrayOf(-3f, -2f, -1f,  0f,  1f,  2f,  3f,  4f, 5f),
        floatArrayOf(-2f, -1f,  0f,  1f,  2f,  3f,  4f,  5f, 6f),
        floatArrayOf(-1f,  0f,  1f,  2f,  3f,  4f,  5f,  6f, 7f),
        floatArrayOf( 0f,  1f,  2f,  3f,  4f,  5f,  6f,  7f, 8f)
    )
}

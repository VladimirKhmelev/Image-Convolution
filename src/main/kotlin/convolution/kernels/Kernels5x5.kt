package org.example.convolution.kernels

object Kernels5x5 {

    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 1f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f)
    )

    val BOX_BLUR = Array(5) { FloatArray(5) { 1f / 25f } }

    // Коэффициенты из строки треугольника Паскаля: 1 4 6 4 1
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf( 1f/256,  4f/256,  6f/256,  4f/256,  1f/256),
        floatArrayOf( 4f/256, 16f/256, 24f/256, 16f/256,  4f/256),
        floatArrayOf( 6f/256, 24f/256, 36f/256, 24f/256,  6f/256),
        floatArrayOf( 4f/256, 16f/256, 24f/256, 16f/256,  4f/256),
        floatArrayOf( 1f/256,  4f/256,  6f/256,  4f/256,  1f/256)
    )

    val SHARPEN = arrayOf(
        floatArrayOf( 0f,  0f, -1f,  0f,  0f),
        floatArrayOf( 0f, -1f, -1f, -1f,  0f),
        floatArrayOf(-1f, -1f, 13f, -1f, -1f),
        floatArrayOf( 0f, -1f, -1f, -1f,  0f),
        floatArrayOf( 0f,  0f, -1f,  0f,  0f)
    )

    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, 24f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f)
    )

    val EMBOSS = arrayOf(
        floatArrayOf(-4f, -3f, -2f, -1f, 0f),
        floatArrayOf(-3f, -2f, -1f,  0f, 1f),
        floatArrayOf(-2f, -1f,  1f,  1f, 2f),
        floatArrayOf(-1f,  0f,  1f,  2f, 3f),
        floatArrayOf( 0f,  1f,  2f,  3f, 4f)
    )
}

package org.example.convolution.kernels

object Kernels3x3 {

    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(0f, 1f, 0f),
        floatArrayOf(0f, 0f, 0f)
    )

    val BOX_BLUR = Array(3) { FloatArray(3) { 1f / 9f } }

    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf(1f / 16, 2f / 16, 1f / 16),
        floatArrayOf(2f / 16, 4f / 16, 2f / 16),
        floatArrayOf(1f / 16, 2f / 16, 1f / 16)
    )

    val SHARPEN = arrayOf(
        floatArrayOf( 0f, -1f,  0f),
        floatArrayOf(-1f,  5f, -1f),
        floatArrayOf( 0f, -1f,  0f)
    )

    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f),
        floatArrayOf(-1f,  8f, -1f),
        floatArrayOf(-1f, -1f, -1f)
    )

    val EMBOSS = arrayOf(
        floatArrayOf(-2f, -1f, 0f),
        floatArrayOf(-1f,  1f, 1f),
        floatArrayOf( 0f,  1f, 2f)
    )
}

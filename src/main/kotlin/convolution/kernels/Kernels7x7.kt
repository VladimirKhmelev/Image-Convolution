package org.example.convolution.kernels

/**
 * Ядра свёртки размером 7×7
 *
 * Захватывают радиус 3 вокруг пикселя: ощутимо более широкая область, чем 5×5
 * Вычислительная стоимость: 49 операций умножения на пиксель
 */
object Kernels7x7 {

    // Тождественное ядро — не изменяет изображение
    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 1f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f)
    )

    // Равномерное размытие: все 49 весов одинаковы, сумма = 1
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

    // Резкость 7×7 — ромбовидная «корона» отрицательных весов, центр = 25, сумма = 1
    val SHARPEN = arrayOf(
        floatArrayOf( 0f,  0f,  0f, -1f,  0f,  0f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf(-1f, -1f, -1f, 25f, -1f, -1f, -1f),
        floatArrayOf( 0f, -1f, -1f, -1f, -1f, -1f,  0f),
        floatArrayOf( 0f,  0f, -1f, -1f, -1f,  0f,  0f),
        floatArrayOf( 0f,  0f,  0f, -1f,  0f,  0f,  0f)
    )

    // Выделение краёв 7×7
    // Центр = 48, все 48 соседних = −1, сумма = 0
    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, 48f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f, -1f, -1f)
    )

    // Рельефный эффект 7×7 — расширенный диагональный градиент, сумма = 1
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

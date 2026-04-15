package org.example.convolution.kernels

/**
 * Ядра свёртки размером 9×9
 *
 * Захватывают радиус 4 — наибольшую окрестность в наборе
 * Вычислительная стоимость: 81 операция умножения на пиксель —
 * в 9 раз больше, чем у 3×3, что делает этот размер особенно
 * показательным при бенчмарках параллелизации
 */
object Kernels9x9 {

    // Тождественное ядро — не изменяет изображение
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

    // Равномерное размытие: все 49 весов одинаковы, сумма = 1
    val BOX_BLUR = Array(9) { FloatArray(9) { 1f / 81f } }

    // Гауссовское размытие 9×9 (σ = 2.0)
    // Значения: exp(-(x²+y²) / (2·2.0²)) = exp(-(x²+y²) / 8), нормированы (сумма ≈ 1)
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

    // Резкость 9×9 — ромбовидная «корона» отрицательных весов, центр = 41, сумма = 1
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

    // Выделение краёв 9×9
    // Центр = 80, все 80 соседних = −1, сумма = 0
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

    // Рельефный эффект 9×9 — расширенный диагональный градиент, сумма = 1
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

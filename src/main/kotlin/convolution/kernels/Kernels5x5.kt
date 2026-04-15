package org.example.convolution.kernels

/**
 * Ядра свёртки размером 5×5
 *
 * Захватывают радиус 2 вокруг пикселя: размытие плавнее, резкость сильнее,
 * края улавливаются на большем расстоянии
 * Вычислительная стоимость: 25 операций умножения на пиксель
 */
object Kernels5x5 {

    // Тождественное ядро — не изменяет изображение
    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 1f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f),
        floatArrayOf(0f, 0f, 0f, 0f, 0f)
    )

    // Равномерное размытие: все 25 весов одинаковы, сумма = 1
    val BOX_BLUR = Array(5) { FloatArray(5) { 1f / 25f } }

    // Гауссовское размытие 5×5 — биномиальное приближение
    // Коэффициенты из строки треугольника Паскаля: 1 4 6 4 1
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf( 1f/256,  4f/256,  6f/256,  4f/256,  1f/256),
        floatArrayOf( 4f/256, 16f/256, 24f/256, 16f/256,  4f/256),
        floatArrayOf( 6f/256, 24f/256, 36f/256, 24f/256,  6f/256),
        floatArrayOf( 4f/256, 16f/256, 24f/256, 16f/256,  4f/256),
        floatArrayOf( 1f/256,  4f/256,  6f/256,  4f/256,  1f/256)
    )

    // Резкость 5×5 — усиленный вариант фильтра резкости
    // Более широкая «корона» отрицательных весов, сумма = 1
    val SHARPEN = arrayOf(
        floatArrayOf( 0f,  0f, -1f,  0f,  0f),
        floatArrayOf( 0f, -1f, -1f, -1f,  0f),
        floatArrayOf(-1f, -1f, 13f, -1f, -1f),
        floatArrayOf( 0f, -1f, -1f, -1f,  0f),
        floatArrayOf( 0f,  0f, -1f,  0f,  0f)
    )

    // Выделение краёв 5×5
    // Центр = 24, все 24 соседних = −1, сумма = 0
    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, 24f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f),
        floatArrayOf(-1f, -1f, -1f, -1f, -1f)
    )

    // Рельефный эффект 5×5 — диагональный градиент, сумма = 1
    val EMBOSS = arrayOf(
        floatArrayOf(-4f, -3f, -2f, -1f, 0f),
        floatArrayOf(-3f, -2f, -1f,  0f, 1f),
        floatArrayOf(-2f, -1f,  1f,  1f, 2f),
        floatArrayOf(-1f,  0f,  1f,  2f, 3f),
        floatArrayOf( 0f,  1f,  2f,  3f, 4f)
    )
}

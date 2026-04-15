package org.example.convolution.kernels

/**
 * Ядра свёртки размером 3×3
 *
 * Базовый набор фильтров: минимальный радиус (1 пиксель), 9 операций умножения на пиксель
 * Служит точкой отсчёта при сравнении с более крупными ядрами в бенчмарках
 */
object Kernels3x3 {

    // Тождественное ядро — не изменяет изображение
    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(0f, 1f, 0f),
        floatArrayOf(0f, 0f, 0f)
    )

    // Равномерное размытие: все 9 весов одинаковы, сумма = 1
    val BOX_BLUR = Array(3) { FloatArray(3) { 1f / 9f } }

    // Гауссовское размытие — центр имеет наибольший вес, углы — наименьший
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf(1f / 16, 2f / 16, 1f / 16),
        floatArrayOf(2f / 16, 4f / 16, 2f / 16),
        floatArrayOf(1f / 16, 2f / 16, 1f / 16)
    )

    // Резкость: усиливает контрастность границ, сумма = 1
    val SHARPEN = arrayOf(
        floatArrayOf( 0f, -1f,  0f),
        floatArrayOf(-1f,  5f, -1f),
        floatArrayOf( 0f, -1f,  0f)
    )

    // Выделение границ: однородный фон → 0, переходы → яркий отклик
    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f),
        floatArrayOf(-1f,  8f, -1f),
        floatArrayOf(-1f, -1f, -1f)
    )

    // Рельефный эффект с выделением диагональных переходов, сумма = 1
    val EMBOSS = arrayOf(
        floatArrayOf(-2f, -1f, 0f),
        floatArrayOf(-1f,  1f, 1f),
        floatArrayOf( 0f,  1f, 2f)
    )
}

package org.example.convolution

object Kernels {

    // Тождественное ядро - не изменяет изображение
    val IDENTITY = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(0f, 1f, 0f),
        floatArrayOf(0f, 0f, 0f)
    )

    // Равномерное размытие
    val BOX_BLUR = Array(3) { FloatArray(3) { 1f / 9f } }

    // Гауссовское размытие - центр имеет наибольший вес, углы - наименьший
    val GAUSSIAN_BLUR = arrayOf(
        floatArrayOf(1f / 16, 2f / 16, 1f / 16),
        floatArrayOf(2f / 16, 4f / 16, 2f / 16),
        floatArrayOf(1f / 16, 2f / 16, 1f / 16)
    )

    // Резкость
    val SHARPEN = arrayOf(
        floatArrayOf( 0f, -1f,  0f),
        floatArrayOf(-1f,  5f, -1f),
        floatArrayOf( 0f, -1f,  0f)
    )

    // Выделение границ
    val EDGE_DETECTION = arrayOf(
        floatArrayOf(-1f, -1f, -1f),
        floatArrayOf(-1f,  8f, -1f),
        floatArrayOf(-1f, -1f, -1f)
    )

    // Создаёт эффект рельефа с выделением диагональных переходов
    val EMBOSS = arrayOf(
        floatArrayOf(-2f, -1f, 0f),
        floatArrayOf(-1f,  1f, 1f),
        floatArrayOf( 0f,  1f, 2f)
    )

    val all: Map<String, Array<FloatArray>> = linkedMapOf(
        "Идентити"      to IDENTITY,
        "Box Blur"      to BOX_BLUR,
        "Gaussian Blur" to GAUSSIAN_BLUR,
        "Резкость"      to SHARPEN,
        "Края"          to EDGE_DETECTION,
        "Эмбосс"        to EMBOSS
    )
}

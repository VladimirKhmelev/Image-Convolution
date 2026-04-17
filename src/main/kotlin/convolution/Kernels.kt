package org.example.convolution

import org.example.convolution.kernels.Kernels3x3
import org.example.convolution.kernels.Kernels5x5
import org.example.convolution.kernels.Kernels7x7
import org.example.convolution.kernels.Kernels9x9

/**
 * Все фильтры, доступные в CLI, GUI и бенчмарке
 */
object Kernels {

    val all: Map<String, Array<FloatArray>> = linkedMapOf(
        // 3x3
        "Идентити 3×3"      to Kernels3x3.IDENTITY,
        "Box Blur 3×3"      to Kernels3x3.BOX_BLUR,
        "Gaussian Blur 3×3" to Kernels3x3.GAUSSIAN_BLUR,
        "Резкость 3×3"      to Kernels3x3.SHARPEN,
        "Края 3×3"          to Kernels3x3.EDGE_DETECTION,
        "Эмбосс 3×3"        to Kernels3x3.EMBOSS,

        // 5x5
        "Идентити 5×5"      to Kernels5x5.IDENTITY,
        "Box Blur 5×5"      to Kernels5x5.BOX_BLUR,
        "Gaussian Blur 5×5" to Kernels5x5.GAUSSIAN_BLUR,
        "Резкость 5×5"      to Kernels5x5.SHARPEN,
        "Края 5×5"          to Kernels5x5.EDGE_DETECTION,
        "Эмбосс 5×5"        to Kernels5x5.EMBOSS,

        // 7x7
        "Идентити 7×7"      to Kernels7x7.IDENTITY,
        "Box Blur 7×7"      to Kernels7x7.BOX_BLUR,
        "Gaussian Blur 7×7" to Kernels7x7.GAUSSIAN_BLUR,
        "Резкость 7×7"      to Kernels7x7.SHARPEN,
        "Края 7×7"          to Kernels7x7.EDGE_DETECTION,
        "Эмбосс 7×7"        to Kernels7x7.EMBOSS,

        // 9x9
        "Идентити 9×9"      to Kernels9x9.IDENTITY,
        "Box Blur 9×9"      to Kernels9x9.BOX_BLUR,
        "Gaussian Blur 9×9" to Kernels9x9.GAUSSIAN_BLUR,
        "Резкость 9×9"      to Kernels9x9.SHARPEN,
        "Края 9×9"          to Kernels9x9.EDGE_DETECTION,
        "Эмбосс 9×9"        to Kernels9x9.EMBOSS
    )
}

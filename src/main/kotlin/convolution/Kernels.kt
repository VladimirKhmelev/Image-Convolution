package org.example.convolution

import org.example.convolution.kernels.Kernels3x3
import org.example.convolution.kernels.Kernels5x5
import org.example.convolution.kernels.Kernels7x7
import org.example.convolution.kernels.Kernels9x9

/**
 * Все фильтры, доступные в CLI, GUI и бенчмарке
 */
object Kernels {

    private val byNormalizedKey: Map<String, Array<FloatArray>> by lazy {
        all.entries.associate { (k, v) -> normalizeKey(k) to v }
    }

    private fun normalizeKey(name: String): String =
        name.trim().lowercase().replace(Regex("[xXхХ×✕⨯*✗]"), "×")

    fun find(name: String): Array<FloatArray>? =
        all[name] ?: byNormalizedKey[normalizeKey(name)]

    val all: Map<String, Array<FloatArray>> = linkedMapOf(
        // 3x3
        "Identity 3×3"      to Kernels3x3.IDENTITY,
        "Box Blur 3×3"      to Kernels3x3.BOX_BLUR,
        "Gaussian Blur 3×3" to Kernels3x3.GAUSSIAN_BLUR,
        "Sharpen 3×3"       to Kernels3x3.SHARPEN,
        "Edges 3×3"         to Kernels3x3.EDGE_DETECTION,
        "Emboss 3×3"        to Kernels3x3.EMBOSS,

        // 5x5
        "Identity 5×5"      to Kernels5x5.IDENTITY,
        "Box Blur 5×5"      to Kernels5x5.BOX_BLUR,
        "Gaussian Blur 5×5" to Kernels5x5.GAUSSIAN_BLUR,
        "Sharpen 5×5"       to Kernels5x5.SHARPEN,
        "Edges 5×5"         to Kernels5x5.EDGE_DETECTION,
        "Emboss 5×5"        to Kernels5x5.EMBOSS,

        // 7x7
        "Identity 7×7"      to Kernels7x7.IDENTITY,
        "Box Blur 7×7"      to Kernels7x7.BOX_BLUR,
        "Gaussian Blur 7×7" to Kernels7x7.GAUSSIAN_BLUR,
        "Sharpen 7×7"       to Kernels7x7.SHARPEN,
        "Edges 7×7"         to Kernels7x7.EDGE_DETECTION,
        "Emboss 7×7"        to Kernels7x7.EMBOSS,

        // 9x9
        "Identity 9×9"      to Kernels9x9.IDENTITY,
        "Box Blur 9×9"      to Kernels9x9.BOX_BLUR,
        "Gaussian Blur 9×9" to Kernels9x9.GAUSSIAN_BLUR,
        "Sharpen 9×9"       to Kernels9x9.SHARPEN,
        "Edges 9×9"         to Kernels9x9.EDGE_DETECTION,
        "Emboss 9×9"        to Kernels9x9.EMBOSS
    )
}

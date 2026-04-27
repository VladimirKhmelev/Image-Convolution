package org.example.convolution

import org.example.convolution.kernels.Kernels3x3

object Kernels {
    val all: Map<String, Array<FloatArray>> = linkedMapOf(
        "Identity 3×3"      to Kernels3x3.IDENTITY,
        "Box Blur 3×3"      to Kernels3x3.BOX_BLUR,
        "Gaussian Blur 3×3" to Kernels3x3.GAUSSIAN_BLUR,
        "Sharpen 3×3"       to Kernels3x3.SHARPEN,
        "Edges 3×3"         to Kernels3x3.EDGE_DETECTION,
        "Emboss 3×3"        to Kernels3x3.EMBOSS
    )

    fun find(name: String): Array<FloatArray>? =
        all[name] ?: all.entries.firstOrNull {
            it.key.trim().lowercase() == name.trim().lowercase()
        }?.value
}

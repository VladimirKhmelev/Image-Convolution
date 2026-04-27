package org.example.convolution

import java.awt.image.BufferedImage

class GrayImage(val width: Int, val height: Int, val data: FloatArray) {
    constructor(width: Int, height: Int) : this(width, height, FloatArray(width * height))

    operator fun get(y: Int, x: Int): Float = data[y * width + x]
    operator fun set(y: Int, x: Int, v: Float) { data[y * width + x] = v }
}

// Rec. 601: Y = 0.299*R + 0.587*G + 0.114*B
fun BufferedImage.toGrayImage(): GrayImage {
    val w = width
    val h = height
    val pixels = getRGB(0, 0, w, h, null, 0, w)
    val img = GrayImage(w, h)
    for (i in pixels.indices) {
        val rgb = pixels[i]
        val r = (rgb shr 16) and 0xFF
        val g = (rgb shr 8) and 0xFF
        val b = rgb and 0xFF
        img.data[i] = 0.299f * r + 0.587f * g + 0.114f * b
    }
    return img
}

fun GrayImage.toBufferedImage(): BufferedImage {
    val out = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val pixels = IntArray(data.size) { i ->
        val v = data[i].coerceIn(0f, 255f).toInt()
        (v shl 16) or (v shl 8) or v
    }
    out.setRGB(0, 0, width, height, pixels, 0, width)
    return out
}

private fun applyKernelAt(
    src: GrayImage,
    kernel: Array<FloatArray>,
    kH: Int, kW: Int, padY: Int, padX: Int,
    x: Int, y: Int
): Float {
    var sum = 0f
    for (ky in 0 until kH) {
        val sy = y + ky - padY
        if (sy !in 0 until src.height) continue
        val rowOffset = sy * src.width
        for (kx in 0 until kW) {
            val sx = x + kx - padX
            if (sx !in 0 until src.width) continue
            sum += src.data[rowOffset + sx] * kernel[ky][kx]
        }
    }
    return sum
}

fun convolveSequential(src: GrayImage, kernel: Array<FloatArray>): GrayImage {
    val kH = kernel.size
    val kW = kernel[0].size
    val padY = kH / 2
    val padX = kW / 2
    val dst = GrayImage(src.width, src.height)
    for (y in 0 until src.height)
        for (x in 0 until src.width)
            dst[y, x] = applyKernelAt(src, kernel, kH, kW, padY, padX, x, y)
    return dst
}

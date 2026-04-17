package org.example.ui

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.example.convolution.*
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter
import kotlin.system.measureTimeMillis

class AppViewModel {

    var originalImage  by mutableStateOf<BufferedImage?>(null)
        private set
    var processedImage by mutableStateOf<BufferedImage?>(null)
        private set

    val kernelPipeline = mutableStateListOf("Gaussian Blur 3×3")

    var useComposed  by mutableStateOf(false)
    var selectedMode by mutableStateOf<ConvolutionMode>(ConvolutionMode.Sequential)
    var numThreads   by mutableStateOf(Runtime.getRuntime().availableProcessors())
    var gridRows     by mutableStateOf(2)
    var gridCols     by mutableStateOf(4)

    var elapsedMs    by mutableStateOf<Long?>(null)
        private set
    var isProcessing by mutableStateOf(false)
        private set
    var errorMessage by mutableStateOf<String?>(null)
        private set

    fun loadImage() {
        val chooser = JFileChooser()
        chooser.fileFilter = FileNameExtensionFilter("Изображения", "png", "jpg", "jpeg", "bmp", "gif")
        if (chooser.showOpenDialog(null) != JFileChooser.APPROVE_OPTION) return
        try {
            originalImage  = ImageIO.read(chooser.selectedFile)
            processedImage = null
            elapsedMs      = null
            errorMessage   = null
        } catch (e: Exception) {
            errorMessage = "Ошибка загрузки: ${e.message}"
        }
    }

    fun saveImage() {
        val img = processedImage ?: return
        val chooser = JFileChooser()
        chooser.selectedFile = File("result.png")
        if (chooser.showSaveDialog(null) != JFileChooser.APPROVE_OPTION) return
        val file = chooser.selectedFile.let {
            if (it.name.endsWith(".png")) it else File("${it.path}.png")
        }
        try {
            ImageIO.write(img, "png", file)
        } catch (e: Exception) {
            errorMessage = "Ошибка сохранения: ${e.message}"
        }
    }

    // Свёртка

    suspend fun applyFilters() {
        val img = originalImage ?: return
        val rawKernels = kernelPipeline.mapNotNull { Kernels.all[it] }
        if (rawKernels.isEmpty()) return

        val effectiveKernels = if (useComposed && rawKernels.size > 1)
            listOf(rawKernels.composed()) else rawKernels

        val mode = selectedMode
        isProcessing = true
        errorMessage = null
        try {
            val src = withContext(Dispatchers.Default) { img.toGrayImage() }
            val dst: GrayImage
            val ms = measureTimeMillis {
                dst = when (mode) {
                    is ConvolutionMode.Sequential ->
                        withContext(Dispatchers.Default) {
                            convolveSequentialPipeline(src, effectiveKernels)
                        }
                    is ConvolutionMode.Parallel ->
                        convolveParallelPipeline(src, effectiveKernels, mode.mode, numThreads, gridRows, gridCols)
                }
            }
            processedImage = withContext(Dispatchers.Default) { dst.toBufferedImage() }
            elapsedMs = ms
        } catch (e: Exception) {
            errorMessage = "Ошибка: ${e.message}"
        } finally {
            isProcessing = false
        }
    }

    /** Размер составного ядра для отображения в UI, или null если не применимо. */
    fun composedKernelSize(): Pair<Int, Int>? {
        if (!useComposed || kernelPipeline.size < 2) return null
        val ks = kernelPipeline.mapNotNull { Kernels.all[it] }
        if (ks.size < 2) return null
        val c = ks.composed()
        return c.size to c[0].size
    }
}

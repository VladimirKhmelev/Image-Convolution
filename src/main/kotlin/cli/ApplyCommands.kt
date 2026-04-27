package org.example.cli

import org.example.convolution.*
import javax.imageio.ImageIO
import java.io.File
import kotlin.system.measureTimeMillis

fun runApply(cmd: ApplyCommand) {
    if (cmd.kernelNames.isEmpty()) {
        println("Укажите один или несколько фильтров.")
        println("Доступные: ${Kernels.all.keys.joinToString()}")
        return
    }

    val file = File(cmd.imagePath)
    require(file.exists()) { "Файл не найден: ${cmd.imagePath}" }
    val image = ImageIO.read(file) ?: error("Не удалось прочитать изображение: ${cmd.imagePath}")

    val kernels = cmd.kernelNames.map { name ->
        Kernels.find(name) ?: error("Неизвестный фильтр: \"$name\". Доступные: ${Kernels.all.keys.joinToString()}")
    }

    println("Изображение : ${cmd.imagePath} (${image.width}×${image.height})")
    println("Фильтры     : ${cmd.kernelNames.joinToString(" → ")}")

    val src = image.toGrayImage()
    val dst: GrayImage
    val ms = measureTimeMillis {
        dst = kernels.fold(src, ::convolveSequential)
    }
    println("Время       : $ms мс")

    if (cmd.outputPath != null) {
        val outFile = File(cmd.outputPath).let {
            if ('.' in it.name) it else File("${cmd.outputPath}.png")
        }
        ImageIO.write(dst.toBufferedImage(), "png", outFile)
        println("Сохранено   : ${outFile.path}")
    } else {
        println("(Результат не сохранён — укажите --output <путь>)")
    }
}

package org.example.cli

data class ApplyCommand(
    val imagePath:   String,
    val kernelNames: List<String>,
    val outputPath:  String?
)

fun parseArgs(args: Array<String>): ApplyCommand {
    val positional = mutableListOf<String>()
    var outputPath: String? = null
    var i = 0
    while (i < args.size) {
        when {
            args[i] == "--output" -> { outputPath = args.getOrNull(++i) ?: error("--output требует значение") }
            args[i].startsWith("--output=") -> outputPath = args[i].removePrefix("--output=")
            else -> positional.add(args[i])
        }
        i++
    }
    val imagePath   = positional.getOrNull(0) ?: error("Укажите путь к изображению первым аргументом")
    val kernelNames = positional.drop(1)
    return ApplyCommand(imagePath, kernelNames, outputPath)
}

package org.example

import org.example.cli.parseArgs
import org.example.cli.runApply

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Использование: <путь> <фильтр...> [--output <путь>]")
        return
    }
    runApply(parseArgs(args))
}

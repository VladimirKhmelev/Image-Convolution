package org.example

import org.example.cli.*

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Использование: <путь> <фильтр...> [--strategy <стратегия>] [--benchmark] ...")
        return
    }

    when (val cmd = parseArgs(args)) {
        is CliCommand.Apply -> runApply(cmd)
        is CliCommand.Benchmark -> runBenchmark(cmd)
    }
}

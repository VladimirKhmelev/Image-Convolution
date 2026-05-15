package org.example

import org.example.cli.*

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Использование: <путь> <фильтр...> [--strategy <стратегия>] [--benchmark] [--pipeline] ...")
        return
    }

    when (val cmd = parseArgs(args)) {
        is CliCommand.Apply -> runApply(cmd)
        is CliCommand.Benchmark -> runBenchmark(cmd)
        is CliCommand.PipelineApply -> runPipelineApply(cmd)
        is CliCommand.PipelineBenchmark -> runPipelineBenchmark(cmd)
    }
}

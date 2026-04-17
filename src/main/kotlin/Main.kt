package org.example

import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import org.example.cli.*
import org.example.ui.AppContent

fun main(args: Array<String>) {

    if (args.isEmpty()) {
        application {
            Window(
                onCloseRequest = ::exitApplication,
                title = "Image Convolution",
                state = rememberWindowState(size = DpSize(1100.dp, 750.dp))
            ) {
                AppContent()
            }
        }
        return
    }

    when (val cmd = parseArgs(args)) {
        is CliCommand.Apply             -> runApply(cmd)
        is CliCommand.Benchmark         -> runBenchmark(cmd)
        is CliCommand.PipelineApply     -> runPipelineApply(cmd)
        is CliCommand.PipelineBenchmark -> runPipelineBenchmark(cmd)
    }
}

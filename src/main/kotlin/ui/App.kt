package org.example.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import org.example.convolution.ConvolutionMode
import org.example.convolution.Kernels
import org.example.convolution.ParallelMode

@Composable
fun AppContent() {
    val vm = remember { AppViewModel() }
    val scope = rememberCoroutineScope()
    val maxThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(8)

    val originalBitmap  by remember(vm.originalImage)  { derivedStateOf { vm.originalImage?.toComposeImageBitmap() } }
    val processedBitmap by remember(vm.processedImage) { derivedStateOf { vm.processedImage?.toComposeImageBitmap() } }

    MaterialTheme {
        Row(modifier = Modifier.fillMaxSize()) {

            Surface(
                modifier = Modifier.width(264.dp).fillMaxHeight(),
                tonalElevation = 2.dp
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(rememberScrollState())
                        .padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {

                    SidebarSection("Фильтры")

                    vm.kernelPipeline.forEachIndexed { i, name ->
                        if (i > 0) {
                            Text(
                                "↓",
                                modifier = Modifier.padding(start = 20.dp),
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        PipelineItem(
                            index     = i,
                            name      = name,
                            canRemove = vm.kernelPipeline.size > 1,
                            onSelect  = { vm.kernelPipeline[i] = it },
                            onRemove  = { vm.kernelPipeline.removeAt(i) }
                        )
                    }

                    TextButton(
                        onClick = { vm.kernelPipeline.add(vm.kernelPipeline.last()) },
                        modifier = Modifier.fillMaxWidth()
                    ) { Text("+ Добавить фильтр") }

                    if (vm.kernelPipeline.size > 1) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Checkbox(checked = vm.useComposed, onCheckedChange = { vm.useComposed = it })
                            Text("Объединить в одно ядро", fontSize = 13.sp)
                        }
                        vm.composedKernelSize()?.let { (rows, cols) ->
                            Text(
                                "Составное ядро: ${rows}×${cols}",
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                modifier = Modifier.padding(start = 4.dp)
                            )
                        }
                    }

                    HorizontalDivider()

                    SidebarSection("Алгоритм")

                    LabeledDropdown(
                        selected = vm.selectedMode.label,
                        options  = ConvolutionMode.all.map { it.label },
                        onSelect = { label ->
                            vm.selectedMode = ConvolutionMode.all.first { it.label == label }
                        }
                    )

                    if (vm.selectedMode is ConvolutionMode.Parallel) {
                        Text("Потоки: ${vm.numThreads}", fontSize = 13.sp)
                        Slider(
                            value         = vm.numThreads.toFloat(),
                            onValueChange = { vm.numThreads = it.toInt().coerceAtLeast(1) },
                            valueRange    = 1f..maxThreads.toFloat(),
                            steps         = maxThreads - 2,
                            modifier      = Modifier.fillMaxWidth()
                        )
                    }

                    if (vm.selectedMode == ConvolutionMode.Parallel(ParallelMode.BY_GRID)) {
                        Text("Строк сетки: ${vm.gridRows}", fontSize = 13.sp)
                        Slider(
                            value         = vm.gridRows.toFloat(),
                            onValueChange = { vm.gridRows = it.toInt().coerceAtLeast(1) },
                            valueRange    = 1f..16f,
                            steps         = 14,
                            modifier      = Modifier.fillMaxWidth()
                        )
                        Text("Столбцов сетки: ${vm.gridCols}", fontSize = 13.sp)
                        Slider(
                            value         = vm.gridCols.toFloat(),
                            onValueChange = { vm.gridCols = it.toInt().coerceAtLeast(1) },
                            valueRange    = 1f..16f,
                            steps         = 14,
                            modifier      = Modifier.fillMaxWidth()
                        )
                    }

                    Spacer(Modifier.weight(1f))
                    HorizontalDivider()

                    Button(
                        onClick  = { scope.launch { vm.applyFilters() } },
                        enabled  = vm.originalImage != null && !vm.isProcessing,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        if (vm.isProcessing) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                        }
                        Text("Применить")
                    }

                    vm.elapsedMs?.let {
                        Text("Время: $it мс", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    if (vm.lastAppliedMode == vm.selectedMode && vm.applyCount > 1) {
                        Text(
                            "Повторный запуск — JIT уже прогрет, время может быть занижено",
                            fontSize = 11.sp,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                    vm.errorMessage?.let {
                        Text(it, fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                    }
                }
            }

            Row(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ImagePanel(
                    label       = "Оригинал (grayscale)",
                    bitmap      = originalBitmap,
                    modifier    = Modifier.weight(1f).fillMaxHeight(),
                    placeholder = "Загрузите изображение"
                ) {
                    Button(onClick = { vm.loadImage() }) { Text("Загрузить") }
                }

                ImagePanel(
                    label = buildString {
                        append("Результат")
                        vm.elapsedMs?.let { append(" — $it мс") }
                    },
                    bitmap      = processedBitmap,
                    modifier    = Modifier.weight(1f).fillMaxHeight(),
                    placeholder = "Здесь появится результат"
                ) {
                    if (vm.processedImage != null) {
                        Button(onClick = { vm.saveImage() }) { Text("Сохранить") }
                    }
                }
            }
        }
    }
}

// Вспомогательные composable

@Composable
private fun SidebarSection(title: String) {
    Text(
        title,
        style = MaterialTheme.typography.labelMedium,
        color = MaterialTheme.colorScheme.primary
    )
}

@Composable
private fun PipelineItem(
    index: Int,
    name: String,
    canRemove: Boolean,
    onSelect: (String) -> Unit,
    onRemove: () -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(
            "${index + 1}.", fontSize = 12.sp, modifier = Modifier.width(18.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Box(modifier = Modifier.weight(1f)) {
            OutlinedButton(
                onClick = { expanded = true },
                modifier = Modifier.fillMaxWidth(),
                contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
            ) {
                Text(name, fontSize = 13.sp)
            }
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                Kernels.all.keys.forEach { option ->
                    DropdownMenuItem(
                        text    = { Text(option, fontSize = 13.sp) },
                        onClick = { onSelect(option); expanded = false }
                    )
                }
            }
        }
        if (canRemove) {
            TextButton(
                onClick = onRemove,
                contentPadding = PaddingValues(0.dp),
                modifier = Modifier.width(32.dp)
            ) {
                Text("×", fontSize = 16.sp)
            }
        }
    }
}

@Composable
private fun ImagePanel(
    label: String,
    bitmap: ImageBitmap?,
    modifier: Modifier = Modifier,
    placeholder: String = "",
    button: @Composable ColumnScope.() -> Unit = {}
) {
    Column(modifier = modifier) {
        Text(label, fontSize = 13.sp, modifier = Modifier.padding(bottom = 4.dp))
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .background(Color(0xFF2B2B2B))
                .border(1.dp, Color(0xFF555555)),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap             = bitmap,
                    contentDescription = label,
                    modifier           = Modifier.fillMaxSize(),
                    contentScale       = ContentScale.Fit
                )
            } else {
                Text(placeholder, color = Color.Gray, fontSize = 13.sp)
            }
        }
        Spacer(Modifier.height(6.dp))
        button()
    }
}

@Composable
private fun LabeledDropdown(
    selected: String,
    options: List<String>,
    onSelect: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedButton(
            onClick = { expanded = true },
            modifier = Modifier.fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
        ) {
            Text(selected, fontSize = 13.sp)
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { option ->
                DropdownMenuItem(
                    text    = { Text(option, fontSize = 13.sp) },
                    onClick = { onSelect(option); expanded = false }
                )
            }
        }
    }
}

package org.example.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.example.convolution.*
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter
import kotlin.system.measureTimeMillis

@Composable
fun AppContent() {

    // Состояния UI
    var originalImage  by remember { mutableStateOf<BufferedImage?>(null) }
    var processedImage by remember { mutableStateOf<BufferedImage?>(null) }
    val kernelPipeline: SnapshotStateList<String> = remember { mutableStateListOf("Gaussian Blur 3×3") } // Список имён фильтров
    var useComposed    by remember { mutableStateOf(false) }                                 // Объединить ли в одно ядро
    var selectedMode   by remember { mutableStateOf<ConvolutionMode>(ConvolutionMode.Sequential) }
    var numThreads     by remember { mutableStateOf(Runtime.getRuntime().availableProcessors()) }
    var gridRows       by remember { mutableStateOf(2) }
    var gridCols       by remember { mutableStateOf(4) }
    var elapsedMs      by remember { mutableStateOf<Long?>(null) } // Время выполнения в мс
    var isProcessing   by remember { mutableStateOf(false) }
    var errorMessage   by remember { mutableStateOf<String?>(null) }

    val originalBitmap  by remember(originalImage)  { derivedStateOf { originalImage?.toComposeImageBitmap() } }
    val processedBitmap by remember(processedImage) { derivedStateOf { processedImage?.toComposeImageBitmap() } }

    val scope = rememberCoroutineScope()
    val maxThreads = Runtime.getRuntime().availableProcessors().coerceAtLeast(8)

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

                    kernelPipeline.forEachIndexed { i, name ->
                        if (i > 0) {
                            Text(
                                "↓",
                                modifier = Modifier.padding(start = 20.dp),
                                fontSize = 12.sp,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        PipelineItem(
                            index    = i,
                            name     = name,
                            canRemove = kernelPipeline.size > 1,
                            onSelect = { kernelPipeline[i] = it },
                            onRemove = { kernelPipeline.removeAt(i) }
                        )
                    }

                    TextButton(
                        onClick = { kernelPipeline.add(kernelPipeline.last()) },
                        modifier = Modifier.fillMaxWidth()
                    ) { Text("+ Добавить фильтр") }

                    // Объединение в одно ядро
                    if (kernelPipeline.size > 1) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Checkbox(checked = useComposed, onCheckedChange = { useComposed = it })
                            Text("Объединить в одно ядро", fontSize = 13.sp)
                        }
                        if (useComposed) {
                            val ks = kernelPipeline.mapNotNull { Kernels.all[it] }
                            if (ks.size > 1) {
                                val c = ks.composed() // Вычисление свертки
                                Text(
                                    "Составное ядро: ${c.size}×${c[0].size}",
                                    fontSize = 12.sp,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                    modifier = Modifier.padding(start = 4.dp)
                                )
                            }
                        }
                    }

                    HorizontalDivider()

                    SidebarSection("Алгоритм")

                    LabeledDropdown(
                        selected = selectedMode.label,
                        options  = ConvolutionMode.all.map { it.label },
                        onSelect = { label ->
                            selectedMode = ConvolutionMode.all.first { it.label == label }
                        }
                    )

                    // Настройка кол-во потоков
                    if (selectedMode is ConvolutionMode.Parallel) {
                        Text("Потоки: $numThreads", fontSize = 13.sp)
                        Slider(
                            value = numThreads.toFloat(),
                            onValueChange = { numThreads = it.toInt().coerceAtLeast(1) },
                            valueRange = 1f..maxThreads.toFloat(),
                            steps = maxThreads - 2,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    // Настройка сетки для BY_GRID
                    if (selectedMode == ConvolutionMode.Parallel(ParallelMode.BY_GRID)) {
                        Text("Строк сетки: $gridRows", fontSize = 13.sp)
                        Slider(
                            value = gridRows.toFloat(),
                            onValueChange = { gridRows = it.toInt().coerceAtLeast(1) },
                            valueRange = 1f..16f,
                            steps = 14,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Text("Столбцов сетки: $gridCols", fontSize = 13.sp)
                        Slider(
                            value = gridCols.toFloat(),
                            onValueChange = { gridCols = it.toInt().coerceAtLeast(1) },
                            valueRange = 1f..16f,
                            steps = 14,
                            modifier = Modifier.fillMaxWidth()
                        )
                    }

                    Spacer(Modifier.weight(1f))
                    HorizontalDivider()

                    Button(
                        onClick = {
                            val img = originalImage ?: return@Button
                            val mode = selectedMode
                            val rawKernels = kernelPipeline.mapNotNull { Kernels.all[it] }
                            if (rawKernels.isEmpty()) return@Button

                            // Составляем в одно ядро если выбрано объединение
                            val effectiveKernels = if (useComposed && rawKernels.size > 1)
                                listOf(rawKernels.composed()) else rawKernels
                            scope.launch {
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
                        },
                        enabled = originalImage != null && !isProcessing,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        if (isProcessing) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                            Spacer(Modifier.width(8.dp))
                        }
                        Text("Применить")
                    }

                    elapsedMs?.let {
                        Text("Время: $it мс", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    errorMessage?.let {
                        Text(it, fontSize = 12.sp, color = MaterialTheme.colorScheme.error)
                    }
                }
            }

            // Отображение картинок
            Row(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxHeight()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                ImagePanel(
                    label = "Оригинал (grayscale)",
                    bitmap = originalBitmap,
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    placeholder = "Загрузите изображение"
                ) {
                    Button(onClick = {
                        val chooser = JFileChooser()
                        chooser.fileFilter = FileNameExtensionFilter(
                            "Изображения", "png", "jpg", "jpeg", "bmp", "gif"
                        )
                        if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
                            try {
                                originalImage  = ImageIO.read(chooser.selectedFile)
                                processedImage = null
                                elapsedMs      = null
                                errorMessage   = null
                            } catch (e: Exception) {
                                errorMessage = "Ошибка загрузки: ${e.message}"
                            }
                        }
                    }) { Text("Загрузить") }
                }

                ImagePanel(
                    label = buildString {
                        append("Результат")
                        elapsedMs?.let { append(" — $it мс") }
                    },
                    bitmap = processedBitmap,
                    modifier = Modifier.weight(1f).fillMaxHeight(),
                    placeholder = "Здесь появится результат"
                ) {
                    if (processedImage != null) {
                        Button(onClick = {
                            val chooser = JFileChooser()
                            chooser.selectedFile = File("result.png")
                            if (chooser.showSaveDialog(null) == JFileChooser.APPROVE_OPTION) {
                                val file = chooser.selectedFile.let {
                                    if (it.name.endsWith(".png")) it else File("${it.path}.png")
                                }
                                try {
                                    ImageIO.write(processedImage!!, "png", file)
                                } catch (e: Exception) {
                                    errorMessage = "Ошибка сохранения: ${e.message}"
                                }
                            }
                        }) { Text("Сохранить") }
                    }
                }
            }
        }
    }
}

// Вспомогательные функции

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
        Text("${index + 1}.", fontSize = 12.sp, modifier = Modifier.width(18.dp),
            color = MaterialTheme.colorScheme.onSurfaceVariant)
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
                        text = { Text(option, fontSize = 13.sp) },
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
                    bitmap = bitmap,
                    contentDescription = label,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
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
                    text = { Text(option, fontSize = 13.sp) },
                    onClick = { onSelect(option); expanded = false }
                )
            }
        }
    }
}

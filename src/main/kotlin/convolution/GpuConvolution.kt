package org.example.convolution

import org.jocl.*
import org.jocl.CL.*

/**
 * Синглтон для выполнения свёртки через OpenCL
 *
 * При первом обращении ищет GPU среди доступных платформ, при отсутствии — берёт CPU
 * Контекст и скомпилированное ядро создаются один раз и переиспользуются
 */
object GpuContext {

    // Исходный код OpenCL ядра для операции свёртки
    private val KERNEL_SRC = """
        __kernel void convolve(
            __global const float* src,
            __global const float* kern,
            __global float* dst,
            const int width,
            const int height,
            const int kW,
            const int kH
        ) {
            int x = get_global_id(0);
            int y = get_global_id(1);
            if (x >= width || y >= height) return;                    // Проверка границ
            int padX = kW / 2;
            int padY = kH / 2;
            float sum = 0.0f;
            for (int ky = 0; ky < kH; ky++) {
                int sy = y + ky - padY;
                if (sy < 0 || sy >= height) continue;
                for (int kx = 0; kx < kW; kx++) {
                    int sx = x + kx - padX;
                    if (sx < 0 || sx >= width) continue;
                    sum += src[sy * width + sx] * kern[ky * kW + kx]; // Из 2D в 1D
                }
            }
            dst[y * width + x] = sum;
        }
    """.trimIndent()

    // Состояние OpenCL контекста (неинициализирован, готов, ошибка)
    private sealed class State {
        data object Uninitialized : State()
        data class Ready(
            val context:    cl_context,       // Контекст видеокарты
            val queue:      cl_command_queue, // Очередь команд
            val program:    cl_program,       // Скомпилированная программа
            val kernel:     cl_kernel,        // Функция ядра(convolve)
            val deviceName: String            // Название видеокарты
        ) : State()
        data class Failed(val reason: String) : State()
    }

    // @Volatile для видимости между потоками
    @Volatile private var state: State = State.Uninitialized

    // Может работать только в одном потоке
    @Synchronized
    private fun init() {
        if (state !is State.Uninitialized) return
        try {
            setExceptionsEnabled(true)

            // Получаем количество доступных OpenCL платформ
            val numPlatforms = IntArray(1)
            clGetPlatformIDs(0, null, numPlatforms)
            require(numPlatforms[0] > 0) { "OpenCL платформы не найдены" }

            // Получаем список всех платформ
            val platforms = arrayOfNulls<cl_platform_id>(numPlatforms[0])
            clGetPlatformIDs(platforms.size, platforms, null)

            var chosenPlatform: cl_platform_id? = null
            var chosenDevice:   cl_device_id?   = null

            // Ищем устройства - сначала GPU, потом CPU
            outer@ for (platform in platforms.filterNotNull()) {
                for (type in longArrayOf(CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU)) {
                    val numDevices = IntArray(1)
                    try {
                        clGetDeviceIDs(platform, type, 0, null, numDevices)
                    } catch (_: CLException) { continue } // Устройств данного типа нет
                    if (numDevices[0] > 0) {
                        val devices = arrayOfNulls<cl_device_id>(numDevices[0])
                        clGetDeviceIDs(platform, type, devices.size, devices, null)
                        chosenPlatform = platform
                        chosenDevice   = devices[0]
                        if (type == CL_DEVICE_TYPE_GPU) break@outer // Нашли GPU - выходим
                    }
                }
            }

            requireNotNull(chosenDevice) { "Нет доступных OpenCL устройств" }

            val nameBytes = ByteArray(256)
            val nameSize  = LongArray(1)
            clGetDeviceInfo(chosenDevice, CL_DEVICE_NAME,
                nameBytes.size.toLong(), Pointer.to(nameBytes), nameSize)
            val deviceName = String(nameBytes, 0, (nameSize[0] - 1).toInt().coerceIn(0, nameBytes.size))

            // Создаём контекст для выбранного устройства
            val ctxProps = cl_context_properties()
            ctxProps.addProperty(CL_CONTEXT_PLATFORM.toLong(), chosenPlatform)

            val context = clCreateContext(ctxProps, 1, arrayOf(chosenDevice), null, null, null)
            val queue   = clCreateCommandQueueWithProperties(context, chosenDevice, null, null)

            // Создаём программу из исходного кода
            val program = clCreateProgramWithSource(context, 1, arrayOf(KERNEL_SRC), null, null)
            try {
                clBuildProgram(program, 0, null, null, null, null)
            } catch (_: CLException) {
                val logBytes = ByteArray(65536)
                val logSize  = LongArray(1)
                clGetProgramBuildInfo(program, chosenDevice, CL_PROGRAM_BUILD_LOG,
                    logBytes.size.toLong(), Pointer.to(logBytes), logSize)
                error("Ошибка компиляции OpenCL ядра:\n${String(logBytes, 0, logSize[0].toInt())}")
            }
            val kernel = clCreateKernel(program, "convolve", null) // Создаём ядро

            state = State.Ready(context, queue, program, kernel, deviceName)

            Runtime.getRuntime().addShutdownHook(Thread {
                (state as? State.Ready)?.apply {
                    clReleaseKernel(kernel)
                    clReleaseProgram(program)
                    clReleaseCommandQueue(queue)
                    clReleaseContext(context)
                }
            })
        } catch (ex: Throwable) {
            state = State.Failed(ex.message ?: "неизвестная ошибка")
        }
    }

    /** Возвращает true если OpenCL устройство успешно инициализировано */
    fun isAvailable(): Boolean {
        init()
        return state is State.Ready
    }

    /** Возвращает название выбранного OpenCL устройства, или null если GPU недоступен */
    fun deviceName(): String? {
        init()
        return (state as? State.Ready)?.deviceName
    }

    /**
     * Применяет одно ядро свёртки к изображению на GPU
     * Выделяет буферы OpenCL, передаёт данные на устройство, запускает ядро и читает результат
     */
    @Synchronized
    fun convolve(src: GrayImage, filterKernel: Array<FloatArray>): GrayImage {
        init()
        val ready = state as? State.Ready
            ?: error("GPU недоступен: ${(state as? State.Failed)?.reason ?: "не инициализирован"}")

        val w  = src.width
        val h  = src.height
        val kH = filterKernel.size; val kW = filterKernel[0].size
        // Преобразуем 2D ядро в одномерный массив для OpenCL
        val kernelFlat = FloatArray(kH * kW) { i -> filterKernel[i / kW][i % kW] }

        val pixelBytes  = (w * h  * Sizeof.cl_float).toLong()
        val kernelBytes = (kH * kW * Sizeof.cl_float).toLong()

        // Создаём буферы OpenCL
        val srcBuf  = clCreateBuffer(ready.context, CL_MEM_READ_ONLY  or CL_MEM_COPY_HOST_PTR, pixelBytes,  Pointer.to(src.data),   null)
        val kernBuf = clCreateBuffer(ready.context, CL_MEM_READ_ONLY  or CL_MEM_COPY_HOST_PTR, kernelBytes, Pointer.to(kernelFlat), null)
        val dstBuf  = clCreateBuffer(ready.context, CL_MEM_WRITE_ONLY,                          pixelBytes,  null,                   null)

        try {
            clSetKernelArg(ready.kernel, 0, Sizeof.cl_mem.toLong(), Pointer.to(srcBuf))
            clSetKernelArg(ready.kernel, 1, Sizeof.cl_mem.toLong(), Pointer.to(kernBuf))
            clSetKernelArg(ready.kernel, 2, Sizeof.cl_mem.toLong(), Pointer.to(dstBuf))
            clSetKernelArg(ready.kernel, 3, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(w)))
            clSetKernelArg(ready.kernel, 4, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(h)))
            clSetKernelArg(ready.kernel, 5, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(kW)))
            clSetKernelArg(ready.kernel, 6, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(kH)))

            // Запускаем ядро на выполнение (2D-сетка потоков)
            clEnqueueNDRangeKernel(ready.queue, ready.kernel, 2, null,
                longArrayOf(w.toLong(), h.toLong()), null, 0, null, null)

            val result = FloatArray(w * h)
            clEnqueueReadBuffer(ready.queue, dstBuf, true, 0, pixelBytes,
                Pointer.to(result), 0, null, null)

            return GrayImage(w, h, result)
        } finally {
            clReleaseMemObject(srcBuf)
            clReleaseMemObject(kernBuf)
            clReleaseMemObject(dstBuf)
        }
    }
}

/** Применяет одно ядро свёртки к изображению через GPU */
fun convolveGpu(src: GrayImage, filterKernel: Array<FloatArray>): GrayImage =
    GpuContext.convolve(src, filterKernel)

/** Последовательно применяет список ядер к изображению через GPU */
fun convolveGpuPipeline(src: GrayImage, kernels: List<Array<FloatArray>>): GrayImage =
    kernels.fold(src, ::convolveGpu)

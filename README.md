# Image Convolution

Учебный проект по параллельной обработке изображений на Kotlin/JVM.
Реализует свёртку с 6 фильтрами, 4 стратегиями CPU-параллелизации и GPU-ускорением через OpenCL (JOCL).

## Быстрый старт

```bash
# GUI
./gradlew run

# CLI — применить фильтр
./gradlew run --args='photo.png "Gaussian Blur 3×3" --strategy rows --output result.png'

# CLI — цепочка фильтров
./gradlew run --args='photo.png "Gaussian Blur 3×3" "Sharpen 3×3" --strategy rows --threads 4 --output result.png'

# CLI — бенчмарк всех стратегий
./gradlew run --args='photo.png --benchmark'

# CLI — пайплайн (пакетная обработка директории)
./gradlew run --args='/path/to/images/ "Gaussian Blur 3×3" --pipeline --workers 4 --output results/'

# CLI — бенчмарк пайплайна
./gradlew run --args='photo.png "Gaussian Blur 3×3" --pipeline --benchmark --batch-size 32'
```

Все флаги CLI описаны в [DOCS.md](DOCS.md).
Структура пакетов и архитектура — там же, в разделе «Архитектура проекта».

## Тесты

```bash
./gradlew test
```

Отчёты: `build/reports/tests/test/index.html` и `build/reports/jacoco/test/html/index.html`

## Фильтры

24 фильтра четырёх размеров. Каждый размер содержит одинаковый набор типов:

| Размер | Оп/пиксель | Фильтры                                                   |
|--------|------------|-----------------------------------------------------------|
| 3×3    | 9          | Identity, Box Blur, Gaussian Blur, Sharpen, Edges, Emboss |
| 5×5    | 25         | то же самое                                               |
| 7×7    | 49         | то же самое                                               |
| 9×9    | 81         | то же самое                                               |

## Стратегии параллелизации (одно изображение)

| Ключ     | Описание                                    |
|----------|---------------------------------------------|
| `seq`    | Последовательный                            |
| `gpu`    | GPU-свёртка через OpenCL (JOCL)             |
| `pixels` | По пикселям                                 |
| `rows`   | По строкам                                  |
| `cols`   | По столбцам                                 |
| `grid`   | По сетке (2D, настраиваемая)                |

## Пайплайн (пакетная обработка)

Режим `--pipeline` обрабатывает массив изображений потоком: ридер → N воркеров → врайтер.

Подробнее — [DOCS.md](DOCS.md).

## Технологии

- **Язык:** Kotlin 2.1.0
- **JVM:** 21+
- **Сборка:** Gradle 8.x
- **GPU:** JOCL (Java Bindings for OpenCL)
- **Тестирование:** JUnit 5, JaCoCo

## Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

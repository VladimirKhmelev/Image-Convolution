# Image Convolution

Учебный проект по параллельной обработке изображений на Kotlin/JVM.
Реализует свёртку с 6 фильтрами и 4 стратегиями параллелизации.

## Быстрый старт

```bash
# GUI
./gradlew run

# CLI — применить фильтр
./gradlew run --args='photo.png "Gaussian Blur" --strategy rows --output result.png'

# CLI — с потоками и цепочкой фильтров
./gradlew run --args='photo.png "Gaussian Blur" "Резкость" --strategy rows --threads 4 --output result.png'

# CLI — скомпоновать цепочку в одно ядро
./gradlew run --args='photo.png "Gaussian Blur" "Резкость" --compose --strategy rows --threads 4 --output result.png'

# CLI — бенчмарк всех стратегий
./gradlew run --args='photo.png --benchmark'
```

Все флаги CLI описаны в [DOCS.md](DOCS.md).

## Тесты

```bash
./gradlew test
```

Отчёты: `build/reports/tests/test/index.html` и `build/reports/jacoco/test/html/index.html`

## Фильтры

24 фильтра четырёх размеров. Каждый размер содержит одинаковый набор типов:

| Размер | Оп/пиксель | Фильтры                                                   |
|--------|------------|-----------------------------------------------------------|
| 3×3    | 9          | Идентити, Box Blur, Gaussian Blur, Резкость, Края, Эмбосс |
| 5×5    | 25         | то же самое                                               |
| 7×7    | 49         | то же самое                                               |
| 9×9    | 81         | то же самое                                               |

## Стратегии параллелизации

| Ключ      | Описание                        |
|-----------|---------------------------------|
| `seq`     | Последовательный                |
| `pixels`  | По пикселям                     |
| `rows`    | По строкам                      |
| `cols`    | По столбцам                     |
| `grid`    | По сетке (2D, настраиваемая)    |

Подробнее — [DOCS.md](DOCS.md).

## Требования

- JVM 21+
- Kotlin 2.1.0
- Gradle 8.x

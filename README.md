# Image Convolution

Учебный проект по параллельной обработке изображений на Kotlin/JVM.
Реализует свёртку с 6 фильтрами и 4 стратегиями параллелизации.

## Быстрый старт

```bash
# GUI
./gradlew run

# CLI — применить фильтр
./gradlew run --args='photo.png "Gaussian Blur" --strategy rows --output result.png'

# CLI — бенчмарк всех стратегий
./gradlew run --args='photo.png --benchmark'
```

## Тесты

```bash
./gradlew test
```

Отчёты: `build/reports/tests/test/index.html` и `build/reports/jacoco/test/html/index.html`

## Фильтры

- Идентити
- Box Blur
- Gaussian Blur
- Резкость
- Края
- Эмбосс

## Стратегии параллелизации

| Ключ      | Описание                        |
|-----------|---------------------------------|
| `seq`     | Последовательный                |
| `pixels`  | По пикселям                     |
| `rows`    | По строкам                      |
| `cols`    | По столбцам                     |
| `grid`    | По сетке (2D, настраиваемая)    |

Подробнее — [DOCS.md](DOCS.md).

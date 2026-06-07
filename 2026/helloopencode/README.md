# HelloOpenCode

Проект на **Scala 3** с **рейтрейсингом** (ray tracing).

## Стек

- **Язык:** Scala 3.8.3
- **Сборка:** sbt 1.12.11
- **Пакет:** `me.kright.raytracer`

## Запуск

```sh
sbt run
```

## Профилирование

См. [PROFILING.md](PROFILING.md) — инструкция по запуску async-profiler (CPU, allocation).

## Бенчмаркинг

Программа принимает опциональный аргумент — имя запуска:

```sh
sbt "run baseline"       # эталон
sbt "run threadLocal"    # ThreadLocalRandom
sbt "run iterative"      # цикл вместо рекурсии
```

Каждый запуск дописывает строку в `benchmarks.tsv`:

```
baseline   12451305717   2026-05-26T15:39:57.765493251
```

Формат: `<runName>\t<elapsedNanos>\t<timestamp>`.

Время замеряется через `System.nanoTime()` вокруг вызова `accumulate()` (только рендер, без сохранения PNG).

## Style Guide

- **Каждый класс** — в своём собственном файле.
- **Иммутабельность** — предпочитать `val`, `case class`, неизменяемые коллекции. Мутабельность (`var`, `ArrayBuffer` и т.п.) допустима только если есть веская причина и это обосновано в коде.

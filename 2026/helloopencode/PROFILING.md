# Profiling

Профилирование выполняется с помощью [async-profiler](https://github.com/async-profiler/async-profiler) (доступен в Corretto JDK).

## Сбор classpath

```sh
sbt "export runtime:fullClasspath" > .classpath
```

## CPU-профиль (flamegraph HTML)

```sh
java -agentpath:$HOME/.jdks/corretto-25.0.3/lib/libasyncProfiler.so=start,event=cpu,file=/tmp/profile.html \
  -cp "$(cat .classpath)" me.kright.raytracer.main
```

## Allocation-профиль (collapsed stacks)

```sh
java -agentpath:$HOME/.jdks/corretto-25.0.3/lib/libasyncProfiler.so=start,event=alloc,file=/tmp/alloc.txt,collapsed \
  -cp "$(cat .classpath)" me.kright.raytracer.main
```

## Анализ collapsed stacks

```sh
# top-фреймы по числу сэмплов
sort -t' ' -k2 -rn /tmp/profile-cpu-collapsed.txt | head -40

# группировка по ключевым категориям
cat /tmp/alloc.txt | cut -d';' -f15- | sort | uniq -c | sort -rn | head -20
```

## Альтернатива: asprof (attach к процессу)

```sh
# запустить рендер в фоне
java -cp "$(cat .classpath)" me.kright.raytracer.main &

# приаттачить профайлер на 10 секунд
asprof -d 10 -e cpu -f /tmp/profile.html $!

# или start/stop
asprof start -e cpu -f /tmp/profile.html $!
sleep 8
asprof stop $!
```

## Результаты анализа

Подробный список выявленных узких мест и предложений по оптимизации находится в файле [OPTIMIZATIONS.md](OPTIMIZATIONS.md).

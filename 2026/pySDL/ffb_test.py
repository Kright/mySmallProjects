import sys
import time
import os
import ctypes
import csv

import matplotlib.pyplot as plt

# Принудительное использование системной библиотеки SDL2
system_sdl_paths = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib"
]
for path in system_sdl_paths:
    if os.path.exists(os.path.join(path, "libSDL2-2.0.so.0")):
        os.environ["PYSDL2_DLL_PATH"] = path
        break

from sdl2 import *

def main():
    if SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC | SDL_INIT_EVENTS) != 0:
        print(f"Ошибка инициализации SDL2: {SDL_GetError().decode()}")
        return -1

    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
    SDL_JoystickUpdate()

    num_joysticks = SDL_NumJoysticks()
    if num_joysticks == 0:
        print("Джойстики не найдены.")
        SDL_Quit()
        return

    print("\n--- Доступные устройства ---")
    devices = []
    for i in range(num_joysticks):
        joy = SDL_JoystickOpen(i)
        if joy:
            name = SDL_JoystickName(joy).decode()
            haptic = SDL_HapticOpenFromJoystick(joy)
            has_haptic = haptic is not None
            if haptic:
                SDL_HapticClose(haptic)
            SDL_JoystickClose(joy)
            
            devices.append((i, name, has_haptic))
            print(f"{i}: {name} [Haptic: {'Yes' if has_haptic else 'No'}]")

    if not any(d[2] for d in devices):
        print("Устройств с поддержкой Force Feedback не найдено.")
        SDL_Quit()
        return

    # Если устройств несколько, берем первое с Haptic или просим выбрать (в данном случае возьмем первое подходящее)
    target_idx = -1
    for idx, name, has_haptic in devices:
        if has_haptic:
            target_idx = idx
            print(f"\nВыбрано устройство: {name}")
            break

    joy = SDL_JoystickOpen(target_idx)
    haptic = SDL_HapticOpenFromJoystick(joy)
    
    if SDL_HapticQuery(haptic) & SDL_HAPTIC_CONSTANT:
        print("Эффект CONSTANT поддерживается.")
    else:
        print("Эффект CONSTANT НЕ поддерживается. Тест может не сработать.")

    # Настройка эффекта
    effect = SDL_HapticEffect()
    effect.type = SDL_HAPTIC_CONSTANT
    effect.constant.direction.type = SDL_HAPTIC_CARTESIAN
    effect.constant.direction.dir[0] = 1 # Направление
    effect.constant.level = 15000 # Сила (0-32767)
    effect.constant.length = 10 # 10 мс = 0.01 сек
    effect.constant.attack_length = 0
    effect.constant.fade_length = 0

    effect_id = SDL_HapticNewEffect(haptic, ctypes.byref(effect))
    SDL_HapticRunEffect(haptic, effect_id, 1)
    time.sleep(1)
    SDL_HapticDestroyEffect(haptic, effect_id)


    effect_length = 200
    effect_level = 5000

    effect = SDL_HapticEffect()
    effect.type = SDL_HAPTIC_CONSTANT
    effect.constant.direction.type = SDL_HAPTIC_CARTESIAN
    effect.constant.direction.dir[0] = 1 # Направление
    effect.constant.level = effect_level # Сила (0-32767)
    effect.constant.length = effect_length # 10 мс = 0.01 сек
    effect.constant.attack_length = 0
    effect.constant.fade_length = 0

    effect_id = SDL_HapticNewEffect(haptic, ctypes.byref(effect))
    if effect_id < 0:
        print(f"Ошибка создания эффекта: {SDL_GetError().decode()}")
        return

    print("\nПодготовка к тесту...")
    print("Пожалуйста, отпустите руль и держите его ровно.")
    time.sleep(2)
    print("Поехали!")

    data_time = []
    data_pos = []
    
    # Замеряем начальное положение
    SDL_JoystickUpdate()
    start_pos = SDL_JoystickGetAxis(joy, 0)
    
    start_time = time.perf_counter()
    SDL_HapticRunEffect(haptic, effect_id, 1)
    
    # Цикл сбора данных в течение 0.5 секунд
    test_duration = 0.5 + effect_length * 0.001
    detected_time = None
    threshold = 1 # Порог изменения оси для фиксации движения

    while True:
        curr_time = time.perf_counter() - start_time
        if curr_time > test_duration:
            break
            
        SDL_JoystickUpdate()
        pos = SDL_JoystickGetAxis(joy, 0)
        
        data_time.append(curr_time * 1000) # в мс
        data_pos.append(pos)
        
        if detected_time is None and abs(pos - start_pos) > threshold:
            detected_time = curr_time
            print(f"Движение обнаружено через {detected_time*1000:.2f} мс")

    if detected_time is None:
        print("Движение не было зафиксировано. Попробуйте увеличить силу эффекта или проверьте устройство.")
    
    # Очистка
    SDL_HapticDestroyEffect(haptic, effect_id)
    SDL_HapticClose(haptic)
    SDL_JoystickClose(joy)
    SDL_Quit()

    # Сохранение в CSV
    csv_file = 'ffb_data.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_ms', 'Axis_Pos'])
        for t, p in zip(data_time, data_pos):
            writer.writerow([f"{t:.4f}", p])
    print(f"Данные сохранены в {csv_file}")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(data_time, data_pos, label='Положение оси 0')
    if detected_time:
        plt.axvline(x=detected_time*1000, color='r', linestyle='--', label=f'Отклик ({detected_time*1000:.2f} мс)')
    
    # Отметка начала действия силы (0-10 мс)
    plt.axvspan(0, effect_length, color='gray', alpha=0.3, label=f'Импульс силы ({effect_length}мс)')
    
    plt.title('График отклика руля на усилие (FFB Latency Test)')
    plt.xlabel('Время (мс)')
    plt.ylabel('Значение оси')
    plt.legend()
    plt.grid(True)
    
    output_file = 'ffb_response_plot.png'
    plt.savefig(output_file)
    print(f"График сохранен в {output_file}")

if __name__ == "__main__":
    main()

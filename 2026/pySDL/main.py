import sys
import time
import os

# Принудительное использование системной библиотеки SDL2, если она доступна.
# Библиотека в pysdl2-dll (2.32.x) на Linux может иметь проблемы с обнаружением Moza устройств.
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

def clear_screen():
    # Очистка экрана для разных ОС
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # Инициализация SDL2 с максимальным набором флагов
    if SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK | SDL_INIT_GAMECONTROLLER | SDL_INIT_HAPTIC | SDL_INIT_EVENTS) != 0:
        print(f"Ошибка инициализации SDL2: {SDL_GetError().decode()}")
        return -1

    # Вывод версии SDL2 для отладки
    ver = SDL_version()
    SDL_GetVersion(ver)
    print(f"SDL2 Version: {ver.major}.{ver.minor}.{ver.patch}")
    
    # Установка подсказок для SDL (критично для некоторых HID устройств на Linux)
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, b"1")
    SDL_SetHint(SDL_HINT_LINUX_JOYSTICK_DEADZONES, b"1")
    # Этот хинт заставляет SDL видеть устройства даже без кнопок (как педали Moza)
    SDL_SetHint(b"SDL_JOYSTICK_LINUX_USE_ALL_JOYDEVS", b"1")
    
    # Попробуем обновить список устройств
    SDL_JoystickUpdate()

    # Информация о мониторах
    num_displays = SDL_GetNumVideoDisplays()
    displays_info = []
    for i in range(num_displays):
        display_name = SDL_GetDisplayName(i).decode()
        mode = SDL_DisplayMode()
        SDL_GetCurrentDisplayMode(i, mode)
        displays_info.append(f"Display {i}: {display_name}, Refresh Rate: {mode.refresh_rate}Hz")

    # Инициализация джойстиков
    num_joysticks = SDL_NumJoysticks()
    joysticks = []
    for i in range(num_joysticks):
        joy = SDL_JoystickOpen(i)
        if joy:
            haptic = SDL_HapticOpenFromJoystick(joy)
            haptic_support = "Yes" if haptic else "No"
            if haptic:
                SDL_HapticClose(haptic)
            
            joysticks.append({
                "handle": joy,
                "index": i,
                "instance_id": SDL_JoystickInstanceID(joy),
                "name": SDL_JoystickName(joy).decode(),
                "axes": SDL_JoystickNumAxes(joy),
                "buttons": SDL_JoystickNumButtons(joy),
                "haptic": haptic_support,
                "is_controller": SDL_IsGameController(i),
                "update_count": 0,
                "last_ups": 0,
                "max_ups": 0
            })

    last_time = time.time()
    try:
        while True:
            # Обработка событий SDL
            event = SDL_Event()
            while SDL_PollEvent(event):
                if event.type == SDL_QUIT:
                    raise KeyboardInterrupt
                elif event.type in (SDL_JOYAXISMOTION, SDL_JOYBALLMOTION, SDL_JOYHATMOTION, 
                                  SDL_JOYBUTTONDOWN, SDL_JOYBUTTONUP):
                    for j in joysticks:
                        if j["instance_id"] == event.jdevice.which:
                            j["update_count"] += 1
                            break

            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed >= 1.0:
                # Проверка на появление новых джойстиков
                if SDL_NumJoysticks() != len(joysticks):
                    print("\n[!] Изменение количества устройств. Рекомендуется перезапуск.")
                
                for j in joysticks:
                    j["last_ups"] = j["update_count"] / elapsed
                    if j["last_ups"] > j["max_ups"]:
                        j["max_ups"] = j["last_ups"]
                    j["update_count"] = 0
                last_time = current_time
                clear_screen()

                # Вывод данных
                for info in displays_info:
                    print(info)
                
                print("\n--- Информация о джойстиках ---")
                if not joysticks:
                    print("Джойстики не найдены. (Попробуйте запустить с правами доступа к /dev/input/ или проверьте PYSDL2_DLL_PATH)")
                
                for j in joysticks:
                    joy_handle = j["handle"]
                    SDL_JoystickUpdate()
                    
                    print(f"Имя: {j['name']}")
                    type_str = "GameController" if j['is_controller'] else "Raw Joystick"
                    print(f"Тип: {type_str}")
                    print(f"Осей: {j['axes']}, Кнопок: {j['buttons']}, Haptic: {j['haptic']}, UPS: {j['last_ups']:.1f}, maxUPS: {j['max_ups']:.1f}")
                    
                    axes_states = [f"{i}:{SDL_JoystickGetAxis(joy_handle, i)}" for i in range(j["axes"])]
                    print(f"Оси: {', '.join(axes_states)}")
                    
                    button_states = [f"{i}:{'on' if SDL_JoystickGetButton(joy_handle, i) else 'off'}" for i in range(j["buttons"])]
                    print(f"Кнопки: {', '.join(button_states)}")
                    print("-" * 20)

                print("\n(Нажмите Ctrl+C для выхода)")
            
            time.sleep(0.01) # Чаще опрашиваем события, но спим чуть-чуть

    except KeyboardInterrupt:
        print("\nВыход...")
    finally:
        for j in joysticks:
            SDL_JoystickClose(j["handle"])
        SDL_Quit()

if __name__ == "__main__":
    main()


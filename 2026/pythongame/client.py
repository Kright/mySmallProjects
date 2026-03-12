import arcade
import json
import socket
import threading
import queue
import math
import time

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Multiplayer Platformer"

class GameClient(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.players_data = []
        self.platforms_data = []
        self.grenades_data = []
        self.explosions_data = []
        self.played_explosions = set()
        self.explosion_sound = arcade.load_sound(":resources:sounds/explosion2.wav")
        self.jump_sound = arcade.load_sound(":resources:sounds/jump1.wav")
        self.walk_sound = arcade.load_sound(":resources:sounds/hit4.wav")
        self.played_jumps = {} # {player_id: last_jump_trigger}
        self.walk_timer = 0
        self.inputs = {"left": False, "right": False, "jump": False}
        self.mouse_click = None
        self.state_queue = queue.Queue()
        self.anim_time = 0
        
        # Облака (просто статические для красоты)
        self.clouds = [
            (100, 500, 60), (250, 550, 40), (450, 520, 80), (650, 560, 50), (750, 480, 70)
        ]
        
        # Устанавливаем цвет фона (небо)
        self.background_color = arcade.color.SKY_BLUE
        
        # Сетевой поток
        self.running = True
        self.net_thread = threading.Thread(target=self.network_loop, daemon=True)
        self.net_thread.start()

    def network_loop(self):
        while self.running:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Устанавливаем таймаут на попытку подключения
                s.settimeout(2.0)
                print(f"Попытка подключения к серверу...")
                s.connect(('127.0.0.1', 8888))
                s.setblocking(False)
                print("Успешное подключение к серверу!")
                
                buffer = ""
                while self.running:
                    # Отправка ввода
                    try:
                        data_to_send = {"inputs": self.inputs}
                        if self.mouse_click:
                            data_to_send["mouse_click"] = self.mouse_click
                            self.mouse_click = None
                        msg = (json.dumps(data_to_send) + "\n").encode()
                        s.sendall(msg)
                    except:
                        print("Ошибка отправки данных, переподключение...")
                        break

                    # Получение состояния
                    try:
                        data = s.recv(4096).decode()
                        if not data:
                            print("Сервер закрыл соединение, переподключение...")
                            break
                        buffer += data
                        if '\n' in buffer:
                            lines = buffer.split('\n')
                            for line in lines[:-1]:
                                if line:
                                    try:
                                        state = json.loads(line)
                                        self.state_queue.put(state)
                                    except:
                                        pass
                            buffer = lines[-1]
                    except BlockingIOError:
                        pass
                    except Exception as e:
                        print(f"Ошибка получения данных: {e}, переподключение...")
                        break
                    
                    import time
                    time.sleep(1/60) # 60 FPS для сетевого потока вполне достаточно
                
                s.close()
            except (ConnectionRefusedError, socket.timeout):
                import time
                time.sleep(1) # Ждем секунду перед повторной попыткой
            except Exception as e:
                print(f"Ошибка сети: {e}")
                import time
                time.sleep(1)
        
        self.running = False

    def on_draw(self):
        self.clear()
        
        # Отрисовка облаков
        for x, y, size in self.clouds:
            arcade.draw_circle_filled(x, y, size, arcade.color.WHITE)
            arcade.draw_circle_filled(x + size * 0.5, y - size * 0.2, size * 0.8, arcade.color.WHITE)
            arcade.draw_circle_filled(x - size * 0.5, y - size * 0.2, size * 0.8, arcade.color.WHITE)
        
        # Отрисовка платформ
        for plat in self.platforms_data:
            # Используем цвет от сервера, если он есть, иначе серый
            color = plat.get("color", arcade.color.GRAY)
            arcade.draw_rect_filled(
                arcade.XYWH(plat["x"], plat["y"], plat["width"], plat["height"]),
                color
            )
            
        # Отрисовка игроков
        for player in self.players_data:
            # Используем цвет от сервера, если он есть, иначе синий
            color = player.get("color", arcade.color.BLUE)
            px, py = player["x"], player["y"]
            vx = player.get("vx", 0)
            on_ground = player.get("on_ground", False)
            
            # Размеры персонажа (30x50)
            # Голова: 16x16
            # Тело: 16x20
            # Руки и ноги: 7x20 (условно)
            
            # Анимация
            swing = 0
            if on_ground and abs(vx) > 0.5:
                # Раскачивание зависит от скорости и времени
                swing = math.sin(self.anim_time * 15) * 15
            elif not on_ground:
                # В прыжке ноги/руки замирают в позе
                swing = 20
                
            # Рисуем ноги (двигаются в противофазе)
            # Левая нога
            arcade.draw_rect_filled(
                arcade.XYWH(px - 4, py - 15, 8, 20),
                color,
                tilt_angle=-swing
            )
            # Правая нога
            arcade.draw_rect_filled(
                arcade.XYWH(px + 4, py - 15, 8, 20),
                color,
                tilt_angle=swing
            )
            
            # Рисуем тело
            arcade.draw_rect_filled(
                arcade.XYWH(px, py + 5, 16, 20),
                color
            )
            
            # Рисуем руки
            # Левая рука
            arcade.draw_rect_filled(
                arcade.XYWH(px - 12, py + 5, 8, 20),
                color,
                tilt_angle=swing
            )
            # Правая рука
            arcade.draw_rect_filled(
                arcade.XYWH(px + 12, py + 5, 8, 20),
                color,
                tilt_angle=-swing
            )
            
            # Рисуем голову
            arcade.draw_rect_filled(
                arcade.XYWH(px, py + 23, 16, 16),
                color
            )
            # Глаза (черные точки)
            arcade.draw_rect_filled(
                arcade.XYWH(px - 4, py + 25, 3, 3),
                arcade.color.BLACK
            )
            arcade.draw_rect_filled(
                arcade.XYWH(px + 4, py + 25, 3, 3),
                arcade.color.BLACK
            )
            
        # Отрисовка гранат
        for grenade in self.grenades_data:
            arcade.draw_circle_filled(grenade["x"], grenade["y"], 5, arcade.color.BLACK)
            arcade.draw_circle_filled(grenade["x"], grenade["y"], 3, arcade.color.DARK_GRAY)

        # Отрисовка взрывов
        for explosion in self.explosions_data:
            # Рисуем вспышку из нескольких кругов для эффекта
            arcade.draw_circle_filled(explosion["x"], explosion["y"], 40, arcade.color.ORANGE)
            arcade.draw_circle_filled(explosion["x"], explosion["y"], 20, arcade.color.YELLOW)
            arcade.draw_circle_filled(explosion["x"], explosion["y"], 10, arcade.color.WHITE)

    def on_update(self, delta_time):
        self.anim_time += delta_time
        # Получаем последнее состояние из очереди
        while not self.state_queue.empty():
            state = self.state_queue.get()
            self.players_data = state.get("players", [])
            self.platforms_data = state.get("platforms", [])
            self.grenades_data = state.get("grenades", [])
            self.explosions_data = state.get("explosions", [])
            
            # Проигрываем звук для новых взрывов
            for explosion in self.explosions_data:
                expl_id = explosion.get("id")
                if expl_id and expl_id not in self.played_explosions:
                    arcade.play_sound(self.explosion_sound)
                    self.played_explosions.add(expl_id)
            
            # Очищаем старые ID, которых больше нет в списке взрывов на сервере
            current_ids = {e.get("id") for e in self.explosions_data if e.get("id")}
            self.played_explosions = {eid for eid in self.played_explosions if eid in current_ids}

            # Проигрываем звук для новых прыжков
            for player in self.players_data:
                pid = player.get("id")
                jump_trigger = player.get("jump_trigger", 0)
                if pid not in self.played_jumps:
                    self.played_jumps[pid] = jump_trigger
                elif jump_trigger > self.played_jumps[pid]:
                    arcade.play_sound(self.jump_sound)
                    self.played_jumps[pid] = jump_trigger
            
            # Проигрываем звук ходьбы (если хоть кто-то идет по земле)
            anyone_walking = False
            for player in self.players_data:
                if player.get("on_ground") and abs(player.get("vx", 0)) > 1.0:
                    anyone_walking = True
                    break
            
            if anyone_walking:
                self.walk_timer -= delta_time
                if self.walk_timer <= 0:
                    arcade.play_sound(self.walk_sound, volume=0.5)
                    self.walk_timer = 0.2 # Интервал между шагами
            else:
                self.walk_timer = 0 # Сброс при остановке

    def on_key_press(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.A:
            self.inputs["left"] = True
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.inputs["right"] = True
        elif key == arcade.key.UP or key == arcade.key.W or key == arcade.key.SPACE:
            self.inputs["jump"] = True

    def on_key_release(self, key, modifiers):
        if key == arcade.key.LEFT or key == arcade.key.A:
            self.inputs["left"] = False
        elif key == arcade.key.RIGHT or key == arcade.key.D:
            self.inputs["right"] = False
        elif key == arcade.key.UP or key == arcade.key.W or key == arcade.key.SPACE:
            self.inputs["jump"] = False

    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.mouse_click = (x, y)

def main():
    client = GameClient()
    arcade.run()

if __name__ == "__main__":
    main()

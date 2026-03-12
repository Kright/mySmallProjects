import asyncio
import json
import time
import random

# Константы игры
TICK_RATE = 60
TICK_TIME = 1 / TICK_RATE
GRAVITY = -0.5
JUMP_FORCE = 12
PLAYER_SPEED = 5
FRICTION = 0.8

# Константы гранат
GRENADE_RADIUS = 5
GRENADE_LIFETIME = 3.0
GRENADE_SPEED = 18
GRENADE_BOUNCE = 0.6 # Коэффициент упругости
GRENADE_FRICTION = 0.95 # Трение (замедление при качении)
EXPLOSION_RADIUS = 150
EXPLOSION_FORCE = 25
EXPLOSION_LIFETIME = 0.5 # Время жизни визуального эффекта (сек)

# Константы карты (простые платформы)
PLATFORMS = [
    {"x": 400, "y": 50, "width": 800, "height": 100, "color": (100, 100, 100)},  # Пол
    {"x": 200, "y": 250, "width": 200, "height": 20, "color": (150, 75, 0)},
    {"x": 600, "y": 400, "width": 200, "height": 20, "color": (0, 150, 0)},
    {"x": 300, "y": 480, "width": 150, "height": 20, "color": (0, 0, 200)},
    {"x": 100, "y": 380, "width": 100, "height": 20, "color": (200, 0, 0)},
    {"x": 500, "y": 280, "width": 120, "height": 20, "color": (150, 0, 150)},
    {"x": 750, "y": 200, "width": 100, "height": 20, "color": (0, 150, 150)},
    {"x": 50, "y": 150, "width": 100, "height": 20, "color": (150, 150, 0)},
    {"x": 650, "y": 120, "width": 150, "height": 20, "color": (255, 165, 0)},
    {"x": 400, "y": 350, "width": 100, "height": 20, "color": (0, 255, 0)},
]

class Grenade:
    def __init__(self, x, y, vx, vy, owner_id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = GRENADE_RADIUS
        self.timer = GRENADE_LIFETIME
        self.is_exploded = False
        self.owner_id = owner_id
        self.ignore_owner = True # Игнорируем владельца пока не вылетим из него

    def update(self, players, grenades):
        self.timer -= TICK_TIME
        if self.timer <= 0:
            self.explode(players, grenades)
            self.is_exploded = True
            return

        # Гравитация
        self.vy += GRAVITY
        
        # Движение
        self.x += self.vx
        self.y += self.vy

        # Проверка нахождения на поверхности для трения
        on_surface = False

        # Отскок от краев
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * GRENADE_BOUNCE
        elif self.x + self.radius > 800:
            self.x = 800 - self.radius
            self.vx = -self.vx * GRENADE_BOUNCE

        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * GRENADE_BOUNCE
            on_surface = True
        elif self.y + self.radius > 600:
            self.y = 600 - self.radius
            self.vy = -self.vy * GRENADE_BOUNCE

        # Отскок от платформ
        for plat in PLATFORMS:
            if self.check_collision_rect(plat):
                # Определяем с какой стороны удар (упрощенно)
                dx = self.x - plat["x"]
                dy = self.y - plat["y"]
                
                # Если по вертикали ближе к краю чем по горизонтали (с учетом размеров)
                if abs(dx) / (plat["width"]/2) < abs(dy) / (plat["height"]/2):
                    if dy > 0: # Сверху
                        self.y = plat["y"] + plat["height"]/2 + self.radius
                        self.vy = -self.vy * GRENADE_BOUNCE
                        on_surface = True
                    else: # Снизу
                        self.y = plat["y"] - plat["height"]/2 - self.radius
                        self.vy = -self.vy * GRENADE_BOUNCE
                else:
                    if dx > 0: # Справа
                        self.x = plat["x"] + plat["width"]/2 + self.radius
                        self.vx = -self.vx * GRENADE_BOUNCE
                    else: # Слева
                        self.x = plat["x"] - plat["width"]/2 - self.radius
                        self.vx = -self.vx * GRENADE_BOUNCE

        # Применяем трение если на поверхности
        if on_surface:
            self.vx *= GRENADE_FRICTION
            # Если скорость совсем маленькая, останавливаем, чтобы не дергалось
            if abs(self.vx) < 0.1:
                self.vx = 0

        # Отскок от игроков
        for player in players:
            if self.ignore_owner and player.id == self.owner_id:
                # Проверяем, вылетела ли граната из игрока
                player_rect = {"x": player.x, "y": player.y, "width": player.width, "height": player.height}
                if not self.check_collision_rect(player_rect):
                    self.ignore_owner = False
                continue

            player_rect = {"x": player.x, "y": player.y, "width": player.width, "height": player.height}
            if self.check_collision_rect(player_rect):
                dx = self.x - player.x
                dy = self.y - player.y
                if abs(dx) / (player.width/2) < abs(dy) / (player.height/2):
                    if dy > 0:
                        self.y = player.y + player.height/2 + self.radius
                        self.vy = -self.vy * GRENADE_BOUNCE
                    else:
                        self.y = player.y - player.height/2 - self.radius
                        self.vy = -self.vy * GRENADE_BOUNCE
                else:
                    if dx > 0:
                        self.x = player.x + player.width/2 + self.radius
                        self.vx = -self.vx * GRENADE_BOUNCE
                    else:
                        self.x = player.x - player.width/2 - self.radius
                        self.vx = -self.vx * GRENADE_BOUNCE

    def check_collision_rect(self, rect):
        return (self.x - self.radius < rect["x"] + rect["width"]/2 and
                self.x + self.radius > rect["x"] - rect["width"]/2 and
                self.y - self.radius < rect["y"] + rect["height"]/2 and
                self.y + self.radius > rect["y"] - rect["height"]/2)

    def explode(self, players, grenades):
        import math
        # Отбрасывание игроков
        for player in players:
            dx = player.x - self.x
            dy = player.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < EXPLOSION_RADIUS:
                # Сила затухает с расстоянием
                force = EXPLOSION_FORCE * (1 - distance / EXPLOSION_RADIUS)
                angle = math.atan2(dy, dx)
                player.vx += math.cos(angle) * force
                player.vy += math.sin(angle) * force
                player.on_ground = False # Отрываем от земли при взрыве
        
        # Отбрасывание других гранат
        for grenade in grenades:
            if grenade == self: continue # Себя не отбрасываем
            dx = grenade.x - self.x
            dy = grenade.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < EXPLOSION_RADIUS:
                # Сила затухает с расстоянием
                force = EXPLOSION_FORCE * (1 - distance / EXPLOSION_RADIUS)
                angle = math.atan2(dy, dx)
                grenade.vx += math.cos(angle) * force
                grenade.vy += math.sin(angle) * force

    def get_state(self):
        return {"x": self.x, "y": self.y}

class Player:
    def __init__(self, id):
        self.id = id
        self.x = 400
        self.y = 150
        self.vx = 0
        self.vy = 0
        self.width = 30
        self.height = 50
        self.on_ground = False
        self.inputs = {"left": False, "right": False, "jump": False}
        self.mouse_click = None # (x, y) куда кликнули
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.jump_trigger = 0 # Увеличивается при каждом прыжке

    def update(self):
        # Применяем ввод
        if self.inputs["left"]:
            self.vx = -PLAYER_SPEED
        elif self.inputs["right"]:
            self.vx = PLAYER_SPEED
        else:
            self.vx *= FRICTION

        # Применяем гравитацию
        self.vy += GRAVITY

        # Обновляем позицию
        self.x += self.vx
        self.y += self.vy

        # Коллизии с платформами
        self.on_ground = False
        for plat in PLATFORMS:
            if self.check_collision(plat):
                # Простая обработка коллизий (только сверху для платформ)
                if self.vy < 0 and self.y - self.height/2 >= plat["y"] + plat["height"]/2 - abs(self.vy):
                    self.y = plat["y"] + plat["height"]/2 + self.height/2
                    self.vy = 0
                    self.on_ground = True

        # Прыжок
        if self.inputs["jump"] and self.on_ground:
            self.vy = JUMP_FORCE
            self.on_ground = False
            self.jump_trigger += 1

        # Ограничения экрана (условно)
        if self.x < 0: self.x = 0
        if self.x > 800: self.x = 800
        if self.y < 0: self.y = 150; self.vy = 0 # Спавн если упал

    def check_collision(self, plat):
        return (self.x - self.width/2 < plat["x"] + plat["width"]/2 and
                self.x + self.width/2 > plat["x"] - plat["width"]/2 and
                self.y - self.height/2 < plat["y"] + plat["height"]/2 and
                self.y + self.height/2 > plat["y"] - plat["height"]/2)

    def get_state(self):
        return {
            "id": self.id, 
            "x": self.x, 
            "y": self.y, 
            "color": self.color, 
            "on_ground": self.on_ground,
            "vx": self.vx,
            "jump_trigger": self.jump_trigger
        }

class GameServer:
    def __init__(self):
        self.players = {}
        self.clients = {}
        self.grenades = []
        self.explosions = [] # [{"x": x, "y": y, "timer": time, "id": id}]
        self.explosion_counter = 0

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        player_id = f"{addr[0]}:{addr[1]}"
        print(f"Новое подключение: {player_id}")
        
        self.players[player_id] = Player(player_id)
        self.clients[player_id] = writer

        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                
                try:
                    lines = data.decode().split('\n')
                    for line in lines:
                        if not line.strip():
                            continue
                        message = json.loads(line)
                        if "inputs" in message:
                            self.players[player_id].inputs = message["inputs"]
                        if "mouse_click" in message:
                            self.players[player_id].mouse_click = message["mouse_click"]
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            print(f"Ошибка с клиентом {player_id}: {e}")
        finally:
            print(f"Клиент отключился: {player_id}")
            del self.players[player_id]
            del self.clients[player_id]
            writer.close()
            await writer.wait_closed()

    async def game_loop(self):
        while True:
            start_time = time.time()
            
            # Обновляем всех игроков
            players_list = list(self.players.values())
            for player in players_list:
                player.update()
                
                # Проверяем бросок гранаты
                if player.mouse_click:
                    mx, my = player.mouse_click
                    import math
                    dx = mx - player.x
                    dy = my - player.y
                    angle = math.atan2(dy, dx)
                    vx = math.cos(angle) * GRENADE_SPEED
                    vy = math.sin(angle) * GRENADE_SPEED
                    self.grenades.append(Grenade(player.x, player.y, vx, vy, player.id))
                    player.mouse_click = None

            # Обновляем гранаты
            for grenade in self.grenades:
                grenade.update(players_list, self.grenades)
                if grenade.is_exploded:
                    # Добавляем данные для вспышки взрыва
                    self.explosion_counter += 1
                    self.explosions.append({
                        "x": grenade.x,
                        "y": grenade.y,
                        "timer": EXPLOSION_LIFETIME,
                        "id": self.explosion_counter
                    })
            
            # Удаляем взорванные
            self.grenades = [g for g in self.grenades if not g.is_exploded]

            # Обновляем эффекты взрывов
            for explosion in self.explosions:
                explosion["timer"] -= TICK_TIME
            self.explosions = [e for e in self.explosions if e["timer"] > 0]

            # Состояние мира
            game_state = {
                "players": [p.get_state() for p in self.players.values()],
                "platforms": PLATFORMS,
                "grenades": [g.get_state() for g in self.grenades],
                "explosions": self.explosions
            }
            payload = json.dumps(game_state).encode() + b'\n'

            # Рассылаем всем
            for writer in list(self.clients.values()):
                try:
                    writer.write(payload)
                    await writer.drain()
                except:
                    pass

            # Спим до следующего тика
            elapsed = time.time() - start_time
            await asyncio.sleep(max(0, TICK_TIME - elapsed))

    async def main(self):
        server = await asyncio.start_server(self.handle_client, '0.0.0.0', 8888)
        print("Сервер запущен на порту 8888")
        
        async with server:
            await asyncio.gather(server.serve_forever(), self.game_loop())

if __name__ == "__main__":
    try:
        asyncio.run(GameServer().main())
    except KeyboardInterrupt:
        pass

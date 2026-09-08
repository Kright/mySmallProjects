#!/usr/bin/env python3
"""
fuse_track.py -- восстанавливает трек по gnss-nav.csv + imu.csv (EKF + RTS-сглаживатель).

Идея: гироскоп даёт приращение курса, доплеровская скорость GNSS -- пройденный путь,
а позиции GNSS привязывают это счисление к карте. Позиции входят в фильтр с робастным
весом: если невязка слишком велика (скачок решения, смена RTK-кадра), измерение не
отбрасывается, но его дисперсия раздувается, и фильтр "подтягивается" к новому кадру
плавно, а не прыжком. Обратный проход (сглаживатель Рауха-Тунга-Штрибеля) использует
и будущие измерения, поэтому переход симметричный и без запаздывания.

Акселерометр в этих логах для скорости бесполезен (корреляция с dv/dt ~ 0), его
ось Y видит лишь центростремительное ускорение, поэтому он в фильтре не используется.

Состояние: [e, n, psi, v, b, h, g]
  e, n  -- восток/север от центра, м (плоская земля, как в gpx2obj)
  psi   -- курс, рад, по часовой от севера
  v     -- скорость вдоль курса, м/с
  b     -- смещение гироскопа, рад/с
  h     -- высота, м
  g     -- продольный уклон dh/ds (высота меняется только с пройденным путём:
           dh = v * g * dt, поэтому скачок высоты GNSS на стоянке не проходит)

Выход: <трек>-fused.obj (точки) и <трек>-fused.gpx (для gpx2obj --mode road).
Только стандартная библиотека.
"""

import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpx2obj  # noqa: E402  (проекция, оси OBJ, центр координат)

GYRO_SCALE = -1.0        # psi_dot = GYRO_SCALE * gyroZ + b  (gyroZ в рад/с, + = против часовой)
GATE_POS = 9.21          # chi^2(2) 99%: порог робастного взвешивания позиции
GATE_1D = 6.63           # chi^2(1) 99%: для скорости / курса / высоты
NS = 7                   # размер вектора состояния


# --------------------------------------------------------------------------- #
# Маленькая линейная алгебра (плотные матрицы как списки списков)
# --------------------------------------------------------------------------- #

def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def mat_t(a):
    return [list(col) for col in zip(*a)]


def mat_mul(a, b):
    bt = mat_t(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def mat_add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_sub(a, b):
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def mat_vec(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def mat_inv(a):
    """Гаусс-Жордан с выбором ведущего элемента."""
    n = len(a)
    m = [row[:] + eye(n)[i] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-15:
            raise ZeroDivisionError('singular matrix')
        m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        m[c] = [x / pv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0.0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def symmetrize(p):
    return [[0.5 * (p[i][j] + p[j][i]) for j in range(len(p))] for i in range(len(p))]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# Чтение логов
# --------------------------------------------------------------------------- #

def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def fnum(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def load_nav(path, proj, all_samples=False):
    """Эпохи GNSS. Приёмник выдаёт 10 Гц, но новая позиция появляется ~1 Гц,
    остальное -- интерполяция. Берём только сэмплы, где сменились hAcc/vAcc/speed/sats."""
    rows = read_csv(path)
    out = []
    prev_key = None
    for r in rows:
        if r.get('fixOk', 'true') != 'true':
            continue
        # в режиме "GNSS + Dead Reckoning Combined" приёмник (с некалиброванным ESF)
        # держит позицию и выдаёт скорость 0 -- это не измерение, а заморозка
        if r.get('fixType', '3D-Fix') != '3D-Fix':
            continue
        key = (r['hAccuracy'], r['vAccuracy'], r['speed'], r['numSats'])
        if not all_samples and key == prev_key:
            continue
        prev_key = key
        lat, lon = fnum(r['lat']), fnum(r['lon'])
        if lat is None or lon is None:
            continue
        e, n, h = proj.to_enu(lat, lon, fnum(r['height'], 0.0))
        out.append({
            't': int(r['timestamp']) / 1000.0,
            'e': e, 'n': n, 'h': h,
            'hacc': max(fnum(r['hAccuracy'], 5.0), 0.1),
            'vacc': max(fnum(r['vAccuracy'], 5.0), 0.1),
            'speed': fnum(r['speed'], 0.0),
            'sacc': max(fnum(r['speedAccuracy'], 0.5), 0.05),
            'hdg': fnum(r['heading']),
            'hdgacc': fnum(r['headingAccuracy'], 180.0),
            'rtk': r.get('rtkType', ''),
        })
    return out


def load_imu(path):
    rows = read_csv(path)
    return [{'t': int(r['timestamp']) / 1000.0, 'gz': fnum(r['gyroZ'], 0.0)} for r in rows]


# --------------------------------------------------------------------------- #
# EKF + RTS
# --------------------------------------------------------------------------- #

class Fuser:
    def __init__(self, args):
        self.a = args
        self.n_down = {'pos': 0, 'speed': 0, 'hdg': 0, 'h': 0}
        self.n_upd = {'pos': 0, 'speed': 0, 'hdg': 0, 'h': 0}

    # --- модель движения
    def predict(self, x, P, dt, gz):
        e, n, psi, v, b, h, g = x
        s, c = math.sin(psi), math.cos(psi)
        xn = [e + v * s * dt,
              n + v * c * dt,
              psi + (GYRO_SCALE * gz + b) * dt,
              v, b,
              h + v * g * dt,
              g]
        F = eye(NS)
        F[0][2] = v * c * dt
        F[0][3] = s * dt
        F[1][2] = -v * s * dt
        F[1][3] = c * dt
        F[2][4] = dt
        F[5][3] = g * dt
        F[5][6] = v * dt
        a = self.a
        Q = zeros(NS, NS)
        Q[0][0] = Q[1][1] = (a.q_pos * dt) ** 2
        Q[2][2] = (a.q_gyro * dt) ** 2
        Q[3][3] = (a.q_acc * dt) ** 2
        Q[4][4] = (a.q_bias * dt) ** 2
        Q[5][5] = (a.q_h * dt) ** 2
        Q[6][6] = (a.q_grade * dt) ** 2
        Pn = mat_add(mat_mul(mat_mul(F, P), mat_t(F)), Q)
        return xn, symmetrize(Pn), F

    # --- робастное обновление: R раздувается, если невязка не проходит порог
    def update(self, x, P, H, z, R, gate, kind, angular=()):
        y = [zi - sum(hr[j] * x[j] for j in range(NS)) for zi, hr in zip(z, H)]
        for i in angular:
            y[i] = wrap(y[i])
        Ht = mat_t(H)
        PHt = mat_mul(P, Ht)
        S = mat_add(mat_mul(H, PHt), R)
        Si = mat_inv(S)
        d2 = sum(y[i] * Si[i][j] * y[j] for i in range(len(y)) for j in range(len(y)))
        self.n_upd[kind] += 1
        if d2 > gate:
            # взвешивание по Хуберу: вес ~ 1/d, т.е. R умножаем на d/sqrt(gate)
            f = math.sqrt(d2 / gate)
            R = [[r * f for r in row] for row in R]
            S = mat_add(mat_mul(H, PHt), R)
            Si = mat_inv(S)
            self.n_down[kind] += 1
        K = mat_mul(PHt, Si)
        Ky = mat_vec(K, y)
        xn = [xi + ki for xi, ki in zip(x, Ky)]
        xn[2] = wrap(xn[2])
        IKH = mat_sub(eye(NS), mat_mul(K, H))
        # форма Джозефа -- устойчивее численно
        Pn = mat_add(mat_mul(mat_mul(IKH, P), mat_t(IKH)), mat_mul(mat_mul(K, R), mat_t(K)))
        return xn, symmetrize(Pn)

    def nav_update(self, x, P, m):
        a = self.a
        # позиция
        H = [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]
        r = (m['hacc'] * a.k_pos) ** 2
        x, P = self.update(x, P, H, [m['e'], m['n']], [[r, 0], [0, r]], GATE_POS, 'pos')
        # скорость (доплер)
        H = [[0, 0, 0, 1, 0, 0, 0]]
        x, P = self.update(x, P, H, [m['speed']], [[(m['sacc'] * a.k_speed) ** 2]], GATE_1D, 'speed')
        # курс: только на ходу и когда приёмник в нём уверен
        if m['hdg'] is not None and m['speed'] >= a.hdg_min_speed and m['hdgacc'] <= a.hdg_max_acc:
            H = [[0, 0, 1, 0, 0, 0, 0]]
            r = (math.radians(max(m['hdgacc'], 0.5)) * a.k_hdg) ** 2
            x, P = self.update(x, P, H, [math.radians(m['hdg'])], [[r]], GATE_1D, 'hdg', angular=(0,))
        # высота
        H = [[0, 0, 0, 0, 0, 1, 0]]
        x, P = self.update(x, P, H, [m['h']], [[(m['vacc'] * a.k_h) ** 2]], GATE_1D, 'h')
        return x, P

    def run(self, nav, imu):
        a = self.a
        # события: (t, 'imu', gz) и (t, 'nav', measurement); начинаем с первой эпохи GNSS
        t0 = nav[0]['t']
        events = [(m['t'], 1, m) for m in nav] + [(s['t'], 0, s) for s in imu if s['t'] >= t0]
        events.sort(key=lambda ev: (ev[0], ev[1]))
        m0 = nav[0]
        x = [m0['e'], m0['n'], math.radians(m0['hdg'] or 0.0), m0['speed'], 0.0, m0['h'], 0.0]
        P = zeros(NS, NS)
        P[0][0] = P[1][1] = m0['hacc'] ** 2
        P[2][2] = math.radians(30.0) ** 2 if m0['speed'] > 1 else math.pi ** 2
        P[3][3] = 1.0
        P[4][4] = 0.02 ** 2
        P[5][5] = m0['vacc'] ** 2
        P[6][6] = 0.15 ** 2
        gz = 0.0
        t_prev = t0
        hist = []   # (t, x_pred, P_pred, F, x_upd, P_upd, is_nav)
        for t, kind, payload in events:
            dt = t - t_prev
            if dt > 0:
                xp, Pp, F = self.predict(x, P, dt, gz)
            else:
                xp, Pp, F = x, P, eye(NS)
            t_prev = t
            if kind == 0:
                gz = payload['gz']
                x, P = xp, Pp
                hist.append((t, xp, Pp, F, x, P, False))
            else:
                x, P = self.nav_update(xp, Pp, payload)
                hist.append((t, xp, Pp, F, x, P, True))
        # --- RTS назад
        N = len(hist)
        xs = [None] * N
        xs[N - 1] = hist[N - 1][4]
        for k in range(N - 2, -1, -1):
            _, _, _, _, xu, Pu, _ = hist[k]
            _, xp1, Pp1, F1, _, _, _ = hist[k + 1]
            try:
                C = mat_mul(mat_mul(Pu, mat_t(F1)), mat_inv(Pp1))
            except ZeroDivisionError:
                xs[k] = xu
                continue
            d = [xs[k + 1][i] - xp1[i] for i in range(NS)]
            d[2] = wrap(d[2])
            Cd = mat_vec(C, d)
            xk = [xu[i] + Cd[i] for i in range(NS)]
            xk[2] = wrap(xk[2])
            xs[k] = xk
        return [(hist[k][0], xs[k], hist[k][6]) for k in range(N)]


# --------------------------------------------------------------------------- #
# Вывод
# --------------------------------------------------------------------------- #

def iso_time(t):
    import datetime as dt
    d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.') + '%03dZ' % (d.microsecond // 1000)


def write_obj(path, track, name, args):
    with open(path, 'w') as f:
        f.write('# generated by fuse_track.py (EKF + RTS, gnss-nav.csv + imu.csv)\n')
        f.write('# source: %s\n' % name)
        f.write('# origin: lat=%.12f lon=%.12f\n' % (args.lat0, args.lon0))
        f.write('# axes: %s\n' % ('X=east Y=north Z=up' if args.z_up else 'X=east Y=up Z=south'))
        f.write('# points: %d\n' % len(track))
        f.write('o fused\n')
        for _, x, _ in track:
            X, Y, Z = gpx2obj.enu_to_obj(x[0], x[1], x[5], args.z_up)
            f.write('v %s %s %s\n' % (gpx2obj.fmt(X), gpx2obj.fmt(Y), gpx2obj.fmt(Z)))
        for i in range(1, len(track) + 1):
            f.write('p %d\n' % i)


def write_gpx(path, track, name, proj):
    with open(path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="fuse_track.py" '
                'xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write('  <trk>\n    <name>%s-fused</name>\n    <trkseg>\n' % name)
        for t, x, _ in track:
            lat = proj.lat0 + x[1] / proj.m_lat
            lon = proj.lon0 + x[0] / proj.m_lon
            f.write('      <trkpt lat="%.8f" lon="%.8f">\n' % (lat, lon))
            f.write('        <ele>%.3f</ele>\n        <time>%s</time>\n      </trkpt>\n'
                    % (x[5], iso_time(t)))
        f.write('    </trkseg>\n  </trk>\n</gpx>\n')


def stats(track, nav):
    """Максимальный шаг между соседними точками и расхождение с сырыми GNSS-эпохами."""
    steps = [math.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1]) for a, b in zip(track, track[1:])]
    import bisect
    ts = [p[0] for p in track]
    dev = []
    for m in nav:
        k = min(bisect.bisect_left(ts, m['t']), len(ts) - 1)
        x = track[k][1]
        dev.append(math.hypot(x[0] - m['e'], x[1] - m['n']))
    dev.sort()
    return {'max_step': max(steps), 'med_dev': dev[len(dev) // 2],
            'p95_dev': dev[int(0.95 * len(dev))], 'max_dev': dev[-1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('track', nargs='+', help='папки треков (с gnss-nav.csv и imu.csv)')
    ap.add_argument('--lat0', type=float, default=gpx2obj.LAT0)
    ap.add_argument('--lon0', type=float, default=gpx2obj.LON0)
    ap.add_argument('--z-up', action='store_true', help='оси X=восток Y=север Z=вверх')
    ap.add_argument('-o', '--outdir', default=None, help='куда писать (default: рядом с папкой)')
    ap.add_argument('--all-nav', action='store_true',
                    help='использовать все 10 Гц сэмплы GNSS, а не только новые эпохи')
    ap.add_argument('--every', type=int, default=1,
                    help='выводить каждую N-ю точку (default 1 = ~10 Гц)')
    g = ap.add_argument_group('шумы процесса (1 сигма)')
    g.add_argument('--q-pos', type=float, default=0.3, help='м/с, ошибка модели движения (снос)')
    g.add_argument('--q-gyro', type=float, default=0.03, help='рад/с, шум гироскопа')
    g.add_argument('--q-acc', type=float, default=0.8, help='м/с^2, случайное ускорение')
    g.add_argument('--q-bias', type=float, default=2e-4, help='рад/с/с, дрейф смещения гироскопа')
    g.add_argument('--q-h', type=float, default=0.05, help='м/с, ошибка модели высоты')
    g.add_argument('--q-grade', type=float, default=0.005, help='1/с, случайное изменение уклона')
    g = ap.add_argument_group('масштаб заявленных точностей GNSS')
    g.add_argument('--k-pos', type=float, default=1.0, help='множитель hAccuracy')
    g.add_argument('--k-speed', type=float, default=1.0, help='множитель speedAccuracy')
    g.add_argument('--k-hdg', type=float, default=1.0, help='множитель headingAccuracy')
    g.add_argument('--k-h', type=float, default=1.0, help='множитель vAccuracy')
    g.add_argument('--hdg-min-speed', type=float, default=2.0, help='м/с, курс GNSS берём только быстрее')
    g.add_argument('--hdg-max-acc', type=float, default=15.0, help='град, курс GNSS берём только точнее')
    args = ap.parse_args(argv)

    proj = gpx2obj.FlatProjection(args.lat0, args.lon0)
    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    for folder in args.track:
        folder = folder.rstrip('/')
        name = os.path.basename(folder)
        nav = load_nav(os.path.join(folder, 'gnss-nav.csv'), proj, args.all_nav)
        imu = load_imu(os.path.join(folder, 'imu.csv'))
        if len(nav) < 2:
            print('%s: нет эпох GNSS' % name, file=sys.stderr)
            continue
        fz = Fuser(args)
        track = fz.run(nav, imu)
        st = stats(track, nav)          # статистика по полному треку, до прореживания
        if args.every > 1:
            track = track[::args.every]
        outdir = args.outdir or os.path.dirname(folder) or '.'
        obj = os.path.join(outdir, name + '-fused.obj')
        gpx = os.path.join(outdir, name + '-fused.gpx')
        write_obj(obj, track, name, args)
        write_gpx(gpx, track, name, proj)
        print('%s -> %s, %s' % (name, os.path.basename(obj), os.path.basename(gpx)))
        print('   nav epochs %d, imu %d, points %d; pos updates downweighted %d/%d, hdg %d/%d, h %d/%d'
              % (len(nav), len(imu), len(track), fz.n_down['pos'], fz.n_upd['pos'],
                 fz.n_down['hdg'], fz.n_upd['hdg'], fz.n_down['h'], fz.n_upd['h']))
        print('   max step %.2f m; deviation from raw GNSS: median %.2f, p95 %.2f, max %.2f m'
              % (st['max_step'], st['med_dev'], st['p95_dev'], st['max_dev']))


if __name__ == '__main__':
    main()

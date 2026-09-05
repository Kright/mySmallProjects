#!/usr/bin/env python3
"""
Склейка записей видеорегистратора в отдельные поездки без перекодирования.

Вход: папки с файлами вида REC_YYYYMMDD_HHMMSS_F.MP4 (непрерывная запись)
и EVT_YYYYMMDD_HHMMSS_F.MP4 (запись по событию: ~10 с до и ~10 с после
срабатывания G-сенсора; первая половина дублирует REC, вторая половина
закрывает дыру, пока REC не возобновится).

Что делает скрипт:
  1. Читает у каждого файла moov-атом напрямую (без ffprobe): длительность,
     ключевые кадры видео и все сэмплы дорожки субтитров (телеметрия
     g-sensor + GNRMC).
  2. Раскладывает файлы на общей шкале времени: по имени файла, уточняет
     по GPS-времени из субтитров, а затем по совпадению отсчётов g-сенсора
     между пересекающимися файлами (точность 0.1 с).
  3. Разбивает на поездки по паузам (--gap).
  4. Для каждой поездки собирает список кусков (файл, in, out) так, чтобы
     время шло непрерывно, а каждый кусок начинался с ключевого кадра.
  5. Вызывает ffmpeg (concat demuxer, -c copy): видео, звук и субтитры
     копируются как есть. Дополнительно пишет .srt и .gpx рядом с mp4.
     С ключом --encode видео пережимается libx265 (crf 22, medium по
     умолчанию; ~4-6x меньше без видимой потери), звук/субтитры копируются.

Требования: python3 (стандартная библиотека), ffmpeg в PATH.

Примеры:
  python3 dashcam_merge.py cont_rec evt_rec -o merged
  python3 dashcam_merge.py cont_rec evt_rec --dry-run
  python3 dashcam_merge.py cont_rec evt_rec -o merged_x265 --encode --crf 22
"""

import argparse
import calendar
import datetime as dt
import json
import os
import re
import statistics
import struct
import subprocess
import sys
import tempfile

EPS = 1e-3
NAME_RE = re.compile(r"^([A-Za-z]+)_(\d{8})_(\d{6})(?:_([A-Za-z0-9]+))?\.(mp4|mov)$", re.I)
REC_PREFIXES = {"REC", "NOR", "NORMAL", "MOV", "MOVIE"}
GNRMC_RE = re.compile(
    r"G[NP]RMC,(\d{2})(\d{2})(\d{2}(?:\.\d+)?),([AV]),"
    r"(\d*\.?\d*),([NS]?),(\d*\.?\d*),([EW]?),([\d.]*),([\d.]*),(\d{6})?"
)
GSENSOR_RE = re.compile(r"gsensori?,[^;]*?(-?\d+),(-?\d+),(-?\d+)")
TAG_RE = re.compile(r"<[^>]+>|\{\\an\d\}")


# ----------------------------------------------------------------------------
# MP4 parsing (moov only + subtitle samples)
# ----------------------------------------------------------------------------

class Mp4Error(Exception):
    pass


def iter_boxes(fh, start, end):
    pos = start
    while pos + 8 <= end:
        fh.seek(pos)
        hdr = fh.read(8)
        if len(hdr) < 8:
            return
        size, typ = struct.unpack(">I4s", hdr)
        hlen = 8
        if size == 1:
            size = struct.unpack(">Q", fh.read(8))[0]
            hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen:
            return
        yield typ, pos + hlen, pos + size
        pos += size


def find_box(fh, start, end, typ):
    for t, s, e in iter_boxes(fh, start, end):
        if t == typ:
            return s, e
    return None


def read_table(fh, start, end, entry_fmt):
    """Read a 'full box' table: version/flags, count, entries."""
    fh.seek(start)
    data = fh.read(end - start)
    count = struct.unpack(">I", data[4:8])[0]
    esz = struct.calcsize(entry_fmt)
    return [struct.unpack(entry_fmt, data[8 + i * esz: 8 + (i + 1) * esz]) for i in range(count)]


def parse_track(fh, s, e):
    mdia = find_box(fh, s, e, b"mdia")
    if not mdia:
        return None
    mdhd = find_box(fh, mdia[0], mdia[1], b"mdhd")
    hdlr = find_box(fh, mdia[0], mdia[1], b"hdlr")
    minf = find_box(fh, mdia[0], mdia[1], b"minf")
    if not (mdhd and hdlr and minf):
        return None
    fh.seek(mdhd[0])
    v = fh.read(1)[0]
    fh.seek(mdhd[0] + (12 + 8 if v else 12))
    timescale = struct.unpack(">I", fh.read(4))[0]
    duration = struct.unpack(">Q" if v else ">I", fh.read(8 if v else 4))[0]
    fh.seek(hdlr[0] + 8)
    handler = fh.read(4)
    stbl = find_box(fh, minf[0], minf[1], b"stbl")
    if not stbl:
        return None
    boxes = {t: (bs, be) for t, bs, be in iter_boxes(fh, stbl[0], stbl[1])}
    stsd = boxes.get(b"stsd")
    fh.seek(stsd[0] + 12)
    fmt = fh.read(4) if stsd else b"????"
    tr = {
        "handler": handler.decode("latin1"),
        "format": fmt.decode("latin1"),
        "timescale": timescale,
        "duration": duration / timescale if timescale else 0.0,
        "boxes": boxes,
    }
    return tr


def sample_times(fh, boxes):
    """Decoding time (in track timescale) of every sample, plus durations."""
    stts = read_table(fh, *boxes[b"stts"], ">II")
    times, durs = [], []
    t = 0
    for count, delta in stts:
        for _ in range(count):
            times.append(t)
            durs.append(delta)
            t += delta
    return times, durs


def sample_offsets_sizes(fh, boxes):
    stsc = read_table(fh, *boxes[b"stsc"], ">III")
    if b"stco" in boxes:
        chunks = [c[0] for c in read_table(fh, *boxes[b"stco"], ">I")]
    else:
        chunks = [c[0] for c in read_table(fh, *boxes[b"co64"], ">Q")]
    fh.seek(boxes[b"stsz"][0])
    _, sample_size, count = struct.unpack(">III", fh.read(12))
    if sample_size:
        sizes = [sample_size] * count
    else:
        sizes = [x[0] for x in read_table(fh, boxes[b"stsz"][0] + 4, boxes[b"stsz"][1], ">I")]
    offsets = []
    si = 0
    for i, (first, per_chunk, _) in enumerate(stsc):
        last = stsc[i + 1][0] - 1 if i + 1 < len(stsc) else len(chunks)
        for ci in range(first - 1, last):
            off = chunks[ci]
            for _ in range(per_chunk):
                if si >= count:
                    break
                offsets.append(off)
                off += sizes[si]
                si += 1
    return offsets, sizes


def probe_native(path):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        moov = None
        for t, s, e in iter_boxes(fh, 0, size):
            if t == b"moov":
                moov = (s, e)
                break
        if not moov:
            raise Mp4Error("no moov atom (файл не дописан?)")
        tracks = []
        for t, s, e in iter_boxes(fh, *moov):
            if t == b"trak":
                tr = parse_track(fh, s, e)
                if tr:
                    tracks.append(tr)
        video = next((t for t in tracks if t["handler"] == "vide"), None)
        if not video:
            raise Mp4Error("no video track")
        vtimes, _ = sample_times(fh, video["boxes"])
        ts = video["timescale"]
        if b"stss" in video["boxes"]:
            kf_idx = [x[0] - 1 for x in read_table(fh, *video["boxes"][b"stss"], ">I")]
        else:
            kf_idx = range(len(vtimes))
        keyframes = [vtimes[i] / ts for i in kf_idx if i < len(vtimes)]
        # video duration as ffmpeg sees it: last sample dts + its duration
        vdur = video["duration"]

        subs = []
        text = next((t for t in tracks if t["handler"] in ("sbtl", "text", "subt")
                     or t["format"] in ("tx3g", "text")), None)
        if text:
            tts = text["timescale"]
            stimes, sdurs = sample_times(fh, text["boxes"])
            offs, sizes = sample_offsets_sizes(fh, text["boxes"])
            for i in range(min(len(stimes), len(offs))):
                fh.seek(offs[i])
                raw = fh.read(sizes[i])
                if len(raw) < 2:
                    continue
                n = struct.unpack(">H", raw[:2])[0]
                txt = raw[2:2 + n].decode("utf-8", "replace")
                subs.append((stimes[i] / tts, sdurs[i] / tts, txt))
    return {"duration": vdur, "keyframes": keyframes, "subs": subs}


def probe_ffmpeg(path):
    """Fallback: same info through ffprobe/ffmpeg (much slower: reads whole file)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration:packet=pts_time,flags", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    keyframes = [float(p["pts_time"]) for p in j.get("packets", []) if "K" in p.get("flags", "")]
    duration = float(j["streams"][0].get("duration") or 0)
    srt = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "0:s:0?", "-f", "srt", "-"],
        capture_output=True, text=True).stdout
    subs = []
    for blk in srt.strip().split("\n\n"):
        lines = blk.split("\n")
        if len(lines) < 3:
            continue
        m = re.match(r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not m:
            continue
        a = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        b = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
        subs.append((a, b - a, "\n".join(lines[2:])))
    return {"duration": duration, "keyframes": sorted(keyframes), "subs": subs}


# ----------------------------------------------------------------------------
# Clip model
# ----------------------------------------------------------------------------

class Clip:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(path)
        m = NAME_RE.match(self.name)
        if not m:
            raise ValueError("unexpected name")
        prefix, d, t, cam, _ = m.groups()
        self.kind = "rec" if prefix.upper() in REC_PREFIXES else "evt"
        self.cam = (cam or "").upper()
        self.fn_time = calendar.timegm(dt.datetime.strptime(d + t, "%Y%m%d%H%M%S").timetuple())
        self.duration = 0.0
        self.keyframes = []
        self.subs = []
        self.gps_time = None   # epoch (UTC) of local time 0, from GNRMC
        self.start = None      # final position on the camera clock (epoch seconds)
        self.align = "name"

    @property
    def end(self):
        return self.start + self.duration

    def load(self, slow=False):
        info = None
        if not slow:
            try:
                info = probe_native(self.path)
            except (Mp4Error, KeyError, struct.error, IndexError) as e:
                print(f"  [warn] {self.name}: native parse failed ({e}), falling back to ffprobe")
        if info is None:
            info = probe_ffmpeg(self.path)
        self.duration = info["duration"]
        self.keyframes = info["keyframes"] or [0.0]
        self.subs = info["subs"]
        # fingerprint: decisecond index -> g-sensor triple
        self.gs = {}
        for t, _, txt in self.subs:
            m = GSENSOR_RE.search(txt)
            if m:
                self.gs[round(t * 10)] = m.groups()
        # GPS time reference: median over all valid fixes (single GNRMC lines
        # are attached to the subtitle stream with up to 1 s of jitter)
        refs = []
        for t, _, txt in self.subs:
            g = parse_gnrmc(txt)
            if g:
                refs.append(g[0] - t)
        if refs:
            self.gps_time = statistics.median(refs)


def parse_gnrmc(txt):
    m = GNRMC_RE.search(txt)
    if not m or m[4] != "A" or not m[5] or not m[7] or not m[11]:
        return None
    lat = float(m[5]); lat = int(lat / 100) + (lat % 100) / 60
    lon = float(m[7]); lon = int(lon / 100) + (lon % 100) / 60
    if m[6] == "S":
        lat = -lat
    if m[8] == "W":
        lon = -lon
    day = dt.datetime.strptime(m[11], "%d%m%y")
    epoch = calendar.timegm(day.timetuple()) + int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    speed_kn = float(m[9]) if m[9] else 0.0
    return epoch, lat, lon, speed_kn * 0.514444


# ----------------------------------------------------------------------------
# Timeline alignment
# ----------------------------------------------------------------------------

def align_clips(clips, evt_lead_default=9.0, verbose=False):
    """Assign clip.start (camera-clock epoch seconds) to every clip."""
    # 1. clock offset gps-vs-filename per kind (median: robust to clock jumps)
    offs = {k: [c.gps_time - c.fn_time for c in clips if c.kind == k and c.gps_time is not None]
            for k in ("rec", "evt")}
    off_rec = statistics.median(offs["rec"]) if offs["rec"] else None
    off_evt = statistics.median(offs["evt"]) if offs["evt"] else None
    if off_rec is None and off_evt is None:
        off_rec = 0.0
        off_evt = -evt_lead_default
    elif off_rec is None:
        off_rec = off_evt + evt_lead_default
    elif off_evt is None:
        off_evt = off_rec - evt_lead_default
    lead = off_rec - off_evt
    if verbose:
        print(f"  clock: gps-name offset rec={off_rec:.1f}s evt={off_evt:.1f}s (evt lead {lead:.1f}s)")

    # name-based estimate; GPS-based where available
    for c in clips:
        c.est_name = c.fn_time - (0.0 if c.kind == "rec" else lead)
        if c.gps_time is not None:
            c.est = c.gps_time - off_rec
            c.corr = c.est - c.est_name
            c.align = "gps"
        else:
            c.est = None
    # Files without GPS: the camera clock may have been wrong until the first fix
    # (it is corrected mid-recording), so borrow the name->gps correction of the
    # next GPS-aligned file in name order (it was named with the same old clock).
    by_name = sorted(clips, key=lambda x: x.fn_time)
    next_corr = [None] * len(by_name)
    last = None
    for i in range(len(by_name) - 1, -1, -1):
        if by_name[i].est is not None:
            last = by_name[i].corr
        next_corr[i] = last
    prev = None
    for i, c in enumerate(by_name):
        if c.est is not None:
            prev = c.corr
            continue
        corr = next_corr[i] if next_corr[i] is not None else (prev if prev is not None else 0.0)
        c.est = c.est_name + corr
        c.align = "name" if abs(corr) < 5 else f"name+{corr:.0f}s(clock jump)"

    # 2. refine by g-sensor fingerprint against already placed overlapping clips
    placed = []
    for c in sorted(clips, key=lambda x: x.est):
        best = None
        for p in placed[-8:]:
            if p.end < c.est - 3 or p.start > c.est + c.duration + 3 or not c.gs or not p.gs:
                continue
            d0 = round((c.est - p.start) * 10)
            scores = []
            for d in range(d0 - 30, d0 + 31):
                m = sum(1 for i, v in c.gs.items() if p.gs.get(i + d) == v)
                scores.append((m, d))
            scores.sort(reverse=True)
            m1, d1 = scores[0]
            m2 = scores[1][0] if len(scores) > 1 else 0
            if m1 >= 10 and m1 >= 2 * m2 and (best is None or m1 > best[0]):
                best = (m1, p.start + d1 / 10, p.name)
        if best:
            c.start = best[1]
            c.align = f"gsensor({best[0]}, vs {best[2]}, shift {c.start - c.est:+.1f}s)"
        else:
            c.start = c.est
        placed.append(c)


def split_trips(clips, gap):
    trips, cur, cur_end = [], [], None
    for c in sorted(clips, key=lambda x: (x.start, x.kind != "rec")):
        if cur and c.start > cur_end + gap:
            trips.append(cur)
            cur, cur_end = [], None
        cur.append(c)
        cur_end = c.end if cur_end is None else max(cur_end, c.end)
    if cur:
        trips.append(cur)
    return trips


def plan_pieces(clips):
    """Greedy cover: stay on the current clip until it ends, then take the
    covering clip that reaches furthest (REC preferred on ties)."""
    pieces, gaps = [], []
    t = min(c.start for c in clips)
    trip_end = max(c.end for c in clips)
    while t < trip_end - EPS:
        cands = [c for c in clips if c.start <= t + EPS and c.end > t + EPS]
        if not cands:
            nxt = min(c.start for c in clips if c.start > t)
            gaps.append((t, nxt))
            t = nxt
            continue
        cur = max(cands, key=lambda c: (round(c.end, 1), c.kind == "rec"))
        pieces.append({"clip": cur, "in": t, "out": cur.end})
        t = cur.end

    # snap piece starts to keyframes, moving the seam so time stays continuous
    i = 1
    while i < len(pieces):
        pc = pieces[i]
        c = pc["clip"]
        loc = pc["in"] - c.start
        le = max((k for k in c.keyframes if k <= loc + EPS), default=0.0)
        gt = min((k for k in c.keyframes if k > loc + EPS), default=None)
        prev = pieces[i - 1]
        choice = le
        if gt is not None and (gt - loc) < (loc - le) and prev["clip"].end >= c.start + gt - EPS:
            choice = gt
        new_in = c.start + choice
        pc["in"] = new_in
        pc["in_local"] = choice
        prev["out"] = min(prev["clip"].end, new_in)
        if prev["out"] <= prev["in"] + EPS:
            # previous piece got swallowed: drop it and re-seam with the one before
            pieces.pop(i - 1)
            if i - 2 >= 0:
                pieces[i - 2]["out"] = min(pieces[i - 2]["clip"].end, new_in)
            i -= 1
        i += 1
    for pc in pieces:
        pc.setdefault("in_local", pc["in"] - pc["clip"].start)
        pc["out_local"] = min(pc["clip"].duration, pc["out"] - pc["clip"].start)
    pieces = [p for p in pieces if p["out_local"] - p["in_local"] > EPS]
    return pieces, gaps


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def utc(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.timezone.utc)


def fmt_ts(sec):
    ms = int(round(sec * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path, pieces):
    n, off = 0, 0.0
    with open(path, "w", encoding="utf-8") as f:
        for pc in pieces:
            for t, d, txt in pc["clip"].subs:
                if t < pc["in_local"] - EPS or t >= pc["out_local"] - EPS:
                    continue
                n += 1
                a = off + (t - pc["in_local"])
                f.write(f"{n}\n{fmt_ts(a)} --> {fmt_ts(a + d)}\n{TAG_RE.sub('', txt).strip()}\n\n")
            off += pc["out_local"] - pc["in_local"]


def write_gpx(path, pieces, name):
    pts, last = [], None
    for pc in pieces:
        for t, _, txt in pc["clip"].subs:
            if t < pc["in_local"] - EPS or t >= pc["out_local"] - EPS:
                continue
            g = parse_gnrmc(txt)
            if g and g[0] != last:
                pts.append(g)
                last = g[0]
    if not pts:
        return 0
    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gpx version="1.1" creator="dashcam_merge.py" xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write(f"<trk><name>{name}</name><trkseg>\n")
        for epoch, lat, lon, spd in pts:
            ts = dt.datetime.fromtimestamp(epoch, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            f.write(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"><time>{ts}</time>'
                    f"<extensions><speed>{spd:.2f}</speed></extensions></trkpt>\n")
        f.write("</trkseg></trk>\n</gpx>\n")
    return len(pts)


def concat_list(pieces):
    lines = []
    for pc in pieces:
        p = pc["clip"].path.replace("'", "'\\''")
        lines.append(f"file '{p}'")
        if pc["in_local"] > EPS:
            # round UP to microseconds so the seek lands on this keyframe, not the previous one
            lines.append(f"inpoint {(int(pc['in_local'] * 1e6) + 1) / 1e6:.6f}")
        lines.append(f"outpoint {pc['out_local']:.6f}")
    return "\n".join(lines) + "\n"


def run_ffmpeg(list_path, out_path, faststart, encode=None):
    """encode=None: stream copy. encode=(crf, preset): video via libx265, audio/subs copied."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y",
           "-f", "concat", "-safe", "0", "-i", list_path, "-map", "0"]
    if encode:
        crf, preset = encode
        cmd += ["-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-tag:v", "hvc1",
                "-x265-params", "keyint=120:log-level=error", "-c:a", "copy", "-c:s", "copy"]
    else:
        cmd += ["-c", "copy"]
    if faststart:
        cmd += ["-movflags", "+faststart"]
    cmd.append(out_path)
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="папки (или файлы) с записями")
    ap.add_argument("-o", "--out", default="merged", help="папка для результата (default: merged)")
    ap.add_argument("--gap", type=float, default=300, help="пауза в секундах, разделяющая поездки (default 300)")
    ap.add_argument("--evt-lead", type=float, default=9.0,
                    help="на сколько секунд EVT-файл начинается раньше времени в имени, если нет GPS (default 9)")
    ap.add_argument("--prefix", default="TRIP", help="префикс имён выходных файлов (default TRIP)")
    ap.add_argument("--no-srt", action="store_true", help="не писать .srt рядом с mp4")
    ap.add_argument("--no-gpx", action="store_true", help="не писать .gpx рядом с mp4")
    ap.add_argument("--faststart", action="store_true", help="перенести moov в начало (второй проход по файлу)")
    ap.add_argument("--encode", action="store_true",
                    help="пережать видео через libx265 (звук и субтитры копируются); без ключа видео копируется как есть")
    ap.add_argument("--crf", type=float, default=22, help="качество x265 при --encode (default 22, больше = меньше файл)")
    ap.add_argument("--preset", default="medium", help="preset x265 при --encode (default medium)")
    ap.add_argument("--slow", action="store_true", help="разбирать файлы через ffprobe/ffmpeg вместо встроенного парсера")
    ap.add_argument("--dry-run", "-n", action="store_true", help="только показать план, ничего не склеивать")
    ap.add_argument("--overwrite", action="store_true", help="перезаписывать уже существующие результаты")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    paths = []
    for d in args.dirs:
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d))]
        else:
            paths.append(d)
    clips = []
    for p in paths:
        if not NAME_RE.match(os.path.basename(p)):
            continue
        clips.append(Clip(p))
    if not clips:
        sys.exit("файлов вида REC_YYYYMMDD_HHMMSS_F.MP4 не найдено")

    print(f"Читаю метаданные {len(clips)} файлов...")
    ok = []
    for i, c in enumerate(clips, 1):
        try:
            c.load(slow=args.slow)
            ok.append(c)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {c.name}: {e}")
        if i % 10 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}", end="\r", flush=True)
    print()
    clips = ok

    os.makedirs(args.out, exist_ok=True)
    cams = sorted({c.cam for c in clips})
    total_rc = 0
    for cam in cams:
        cc = [c for c in clips if c.cam == cam]
        print(f"=== камера '{cam or '-'}': {len(cc)} файлов "
              f"(rec {sum(c.kind == 'rec' for c in cc)}, evt {sum(c.kind == 'evt' for c in cc)})")
        align_clips(cc, args.evt_lead, args.verbose)
        if args.verbose:
            for c in sorted(cc, key=lambda x: x.start):
                print(f"  {c.name:32s} {c.kind} start={utc(c.start).strftime('%H:%M:%S.%f')[:-5]}"
                      f" dur={c.duration:5.1f} subs={len(c.subs):4d} kf={len(c.keyframes):3d} align={c.align}")
        trips = split_trips(cc, args.gap)
        for ti, trip in enumerate(trips, 1):
            pieces, gaps = plan_pieces(trip)
            t0 = utc(min(c.start for c in trip))
            base = f"{args.prefix}_{t0.strftime('%Y%m%d_%H%M%S')}" + (f"_{cam}" if cam else "")
            out_mp4 = os.path.join(args.out, base + ".mp4")
            total = sum(p["out_local"] - p["in_local"] for p in pieces)
            print(f"\n--- поездка {ti}/{len(trips)}: {base}  файлов={len(trip)} кусков={len(pieces)} "
                  f"длительность={total / 60:.1f} мин  дыр={len(gaps)}")
            for g in gaps:
                print(f"    [gap] нет записи {utc(g[0]).strftime('%H:%M:%S')} -> "
                      f"{utc(g[1]).strftime('%H:%M:%S')} ({g[1] - g[0]:.1f}s)")
            if args.verbose or args.dry_run:
                off = 0.0
                for p in pieces:
                    ln = p["out_local"] - p["in_local"]
                    print(f"    {fmt_ts(off)}  {p['clip'].name:32s} {p['in_local']:7.3f} -> {p['out_local']:7.3f} ({ln:5.1f}s)")
                    off += ln
            if args.dry_run:
                continue
            if os.path.exists(out_mp4) and not args.overwrite:
                print(f"    уже есть, пропускаю: {out_mp4} (--overwrite чтобы пересобрать)")
                continue
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=args.out) as lf:
                lf.write(concat_list(pieces))
                list_path = lf.name
            try:
                rc = run_ffmpeg(list_path, out_mp4, args.faststart,
                                (args.crf, args.preset) if args.encode else None)
            finally:
                if rc == 0 and not args.verbose:
                    os.unlink(list_path)
                else:
                    print(f"    список кусков сохранён: {list_path}")
            if rc != 0:
                print(f"    [ERROR] ffmpeg завершился с кодом {rc}: {out_mp4}")
                total_rc = 1
                continue
            if not args.no_srt:
                write_srt(os.path.join(args.out, base + ".srt"), pieces)
            if not args.no_gpx:
                n = write_gpx(os.path.join(args.out, base + ".gpx"), pieces, base)
                if n == 0:
                    print("    GPS-точек нет, .gpx не записан")
            print(f"    готово: {out_mp4} ({os.path.getsize(out_mp4) / 2**30:.2f} GiB)")
    sys.exit(total_rc)


if __name__ == "__main__":
    main()

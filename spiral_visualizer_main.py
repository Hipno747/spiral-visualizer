#!/usr/bin/env python3
"""
SPIRAL — terminal audio visualizer (prompts to choose a song; copies it into input_audio next to the exe/script;
removes only the file(s) it copied when the program ends)
Requirements: pip install numpy librosa pygame
Controls: SPACE Play/Pause | R Reset | Q Quit
"""
import os, sys, time, math, shutil, platform, atexit
from pathlib import Path

WINDOWS = platform.system() == "Windows"
if WINDOWS:
    try: import msvcrt
    except Exception: msvcrt = None
else:
    try: import termios, tty, select
    except Exception: termios = tty = select = None

try:
    import numpy as np, librosa, pygame
except Exception:
    print("Missing packages. Install: pip install numpy librosa pygame"); sys.exit(1)

# Try tkinter for file dialog; fall back to CLI prompt
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# -------- config ----------
SUPPORTED_AUDIO_EXT = (".mp3", ".wav", ".ogg", ".flac")
def ansi_dim(code): return f"\x1b[2;38;5;{code}m"
RESET = "\x1b[0m"
COLOR_SPEED = 0.45
PURPLE_COLS = [129, 135, 141, 147, 141]
GREEN_COLS  = [22, 28, 34, 40, 46]
PURPLE_PALETTE = [ansi_dim(c) for c in PURPLE_COLS]
GREEN_PALETTE  = [ansi_dim(c) for c in GREEN_COLS]
SPECTRUM_CHARS = set(["░", "▒", "▓", "█"])

# -------- helpers to find / prepare input_audio ----------
def exe_parent_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()

def choose_audio_and_copy(input_sub="input_audio"):
    base = exe_parent_dir()
    folder = base / input_sub
    folder_created = False
    if not folder.exists():
        try:
            folder.mkdir(parents=True, exist_ok=True)
            folder_created = True
        except Exception:
            folder_created = False
    chosen = None
    if TK_AVAILABLE:
        try:
            root = tk.Tk(); root.withdraw()
            filetypes = [("Audio files", "*.mp3 *.wav *.ogg *.flac"), ("All files", "*.*")]
            p = filedialog.askopenfilename(title="Select an audio file to copy into input_audio", filetypes=filetypes)
            root.update(); root.destroy()
            if p: chosen = Path(p)
        except Exception:
            chosen = None
    if chosen is None:
        print(f"Please enter path to an audio file (or press Enter to cancel). Supported: {SUPPORTED_AUDIO_EXT}")
        entry = input("Audio file path: ").strip()
        if entry:
            chosen = Path(entry)
    if not chosen:
        return None, [], folder_created
    if not chosen.exists() or not chosen.is_file() or chosen.suffix.lower() not in SUPPORTED_AUDIO_EXT:
        print("Chosen file is not valid or not a supported audio file."); return None, [], folder_created
    dest = folder / chosen.name
    if dest.exists():
        stem = chosen.stem; suffix = chosen.suffix; t = int(time.time() * 1000)
        dest = folder / f"{stem}_{t}{suffix}"
    try:
        shutil.copy2(str(chosen), str(dest))
    except Exception as e:
        print("Failed to copy file:", e); return None, [], folder_created
    return dest, [dest], folder_created

# --------- Display, Input, Visualizer, Audio (kept from previous working version) ----------
class NoFlickerDisplay:
    def __init__(self):
        tw, th = shutil.get_terminal_size(); self.term_width, self.term_height = tw, th
        base_w = max(36, min(80, tw - 6)); base_h = max(12, min(28, th - 10))
        self.width = min(tw - 4, int(base_w * 1.7)); self.height = min(th - 8, int(base_h * 1.4))
        self.width = max(40, self.width); self.height = max(12, self.height)
        self.canvas = [" " * self.width for _ in range(self.height)]
        self.hide, self.show, self.clear = "\33[?25l", "\33[?25h", "\33[2J\33[H"
        print(self.hide + self.clear, end="", flush=True)
        purple_line = PURPLE_PALETTE[0] + "=" * self.term_width + RESET
        green_title = GREEN_PALETTE[2] + "SPIRAL".center(self.term_width) + RESET
        print(purple_line); print(green_title); print(purple_line)
        print(GREEN_PALETTE[1] + "Status: Ready | Controls: SPACE=Play R=Reset Q=Quit" + RESET + "\n", flush=True)

    def goto(self, x, y): return f"\33[{y + 6};{x + 2}H"

    def _colorize_row_with_mask(self, row, mask_row, base_idx):
        out = []
        n_p = len(PURPLE_PALETTE); n_g = len(GREEN_PALETTE)
        for x, ch in enumerate(row):
            if ch == " ":
                out.append(" ")
            elif mask_row and mask_row[x] == "1":
                idx = (base_idx + (x // 4)) % n_g
                out.append(GREEN_PALETTE[idx] + ch + RESET)
            else:
                idx = (base_idx + (x // 6)) % n_p
                out.append(PURPLE_PALETTE[idx] + ch + RESET)
        return "".join(out)

    def update_display(self, new_rows, mask_rows=None):
        out_parts = []; gw = self.goto; cbuf = self.canvas
        base = int(time.time() * COLOR_SPEED)
        for y, row in enumerate(new_rows):
            if y >= self.height: break
            if row != cbuf[y]:
                mask_row = mask_rows[y] if mask_rows is not None else None
                colored = self._colorize_row_with_mask(row, mask_row, base + (y // 3))
                out_parts.append(gw(0, y) + colored)
                cbuf[y] = row
        if out_parts:
            print("".join(out_parts), end="", flush=True)

    def update_status(self, status, progress=""):
        tw = self.term_width
        print("\33[4;1H" + " " * (tw - 1), end="")
        print("\33[4;1H" + GREEN_PALETTE[2] + status[: tw - 1] + RESET, end="", flush=True)
        if progress:
            print("\33[5;1H" + " " * (tw - 1), end="")
            print("\33[5;1H" + GREEN_PALETTE[1] + progress[: tw - 1] + RESET, end="", flush=True)

    def cleanup(self):
        print(self.show + self.clear, end="", flush=True)

class StableInput:
    def __init__(self):
        self.old = None; self.stdin_is_tty = sys.stdin.isatty()
        if not WINDOWS and self.stdin_is_tty and termios:
            try:
                self.old = termios.tcgetattr(sys.stdin.fileno()); tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.old = None

    def get_key(self):
        try:
            if WINDOWS and msvcrt:
                if msvcrt.kbhit(): return msvcrt.getwch().lower()
                return None
            if not self.stdin_is_tty or select is None:
                return None
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r: return sys.stdin.read(1).lower()
            return None
        except Exception:
            return None

    def cleanup(self):
        if not WINDOWS and self.old and termios:
            try: termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)
            except Exception: pass

class ASCIIVisualizer:
    GRAD = [" ", ".", ":", "-", "=", "+", "*", "█", "▓", "█"]
    SPECTRUM = ["░", "▒", "▓", "█"]
    SHAPES = ["mandala", "spiral", "checker", "rings", "waves"]

    def __init__(self, w, h):
        self.w, self.h = w, h; self.cx, self.cy = w // 2, h // 2
        self.frame = 0; self.shape_idx = 0; self.prev_intensity = 0.0; self.cool = 0
        self.prev_levels = None; self.smooth_alpha = 0.55

    def blank_rows_and_mask(self):
        rows = [list(" " * self.w) for _ in range(self.h)]
        mask = [list("0" * self.w) for _ in range(self.h)]
        return rows, mask

    def set_char(self, rows, x, y, ch):
        if 0 <= x < self.w and 0 <= y < self.h:
            rows[y][x] = ch

    def set_spectrum_char(self, rows, mask, x, y, ch):
        if 0 <= x < self.w and 0 <= y < self.h:
            rows[y][x] = ch
            mask[y][x] = "1"

    def _mandala_at(self, rows, cx, cy, scale, levels):
        maxr = scale; grad_len = len(self.GRAD) - 1
        bass = float(np.mean(levels[: max(1, len(levels) // 6)])) if len(levels) else 0.2
        spokes = max(6, int(6 + bass * 18)); rings = max(3, int(3 + bass * 8))
        hi = levels[len(levels) // 2] if len(levels) > 1 else 0.2
        for yy in range(cy - maxr, cy + maxr + 1):
            ydiff = yy - cy
            for xx in range(cx - maxr, cx + maxr + 1):
                xdiff = xx - cx; r = math.hypot(xdiff, ydiff)
                if r > maxr: continue
                angle = math.atan2(ydiff, xdiff)
                val = 0.5 * (math.cos(r * (rings * 0.6)) + 0.5 * math.cos(angle * (spokes / 2)))
                val = val * (0.7 + 0.6 * hi)
                idx = int((val + 1.0) / 2.0 * grad_len)
                self.set_char(rows, xx, yy, self.GRAD[max(0, min(grad_len, idx))])

    def _spiral_at(self, rows, cx, cy, scale, levels):
        maxr = scale; grad_len = len(self.GRAD) - 1
        mid = float(np.mean(levels[len(levels) // 4: len(levels) // 2])) if len(levels) > 4 else 0.2
        turns = 2 + int(4 * mid); freq = 3 + int(6 * mid)
        for yy in range(cy - maxr, cy + maxr + 1):
            ydiff = yy - cy
            for xx in range(cx - maxr, cx + maxr + 1):
                xdiff = xx - cx; r = math.hypot(xdiff, ydiff)
                if r > maxr: continue
                theta = math.atan2(ydiff, xdiff)
                s = math.sin(theta * turns + r * 0.6 * freq)
                v = (s + math.cos(r * 0.8)) * (0.6 + 0.6 * mid)
                idx = int((v + 1.0) / 2.0 * grad_len)
                self.set_char(rows, xx, yy, self.GRAD[max(0, min(grad_len, idx))])

    def _checker_at(self, rows, cx, cy, scale, levels):
        bass = float(levels[0]) if len(levels) > 0 else 0.0
        cell = max(2, int(max(2, scale // 6) + int(bass * 6))); alt = len(levels) > 2 and levels[2] > 0.4
        half = int(scale)
        for yy in range(cy - half, cy + half + 1):
            for xx in range(cx - half, cx + half + 1):
                xdiff = xx - cx; r = math.hypot(xdiff, yy - cy)
                if r > half: continue
                sx = ((xx + half) // cell) & 1; sy = ((yy + half) // cell) & 1; c = sx ^ sy
                ch = "#" if c else "."
                if alt and ((xx + yy) % (cell * 2) == 0): ch = "*"
                self.set_char(rows, xx, yy, ch)

    def _rings_at(self, rows, cx, cy, scale, levels):
        maxr = scale; grad_len = len(self.GRAD) - 1
        low = float(np.mean(levels[: max(1, len(levels) // 6)])) if len(levels) else 0.2
        rings = 4 + int(8 * low)
        for yy in range(cy - maxr, cy + maxr + 1):
            ydiff = yy - cy
            for xx in range(cx - maxr, cx + maxr + 1):
                xdiff = xx - cx; r = math.hypot(xdiff, ydiff)
                if r > maxr: continue
                v = math.sin(r * (rings * 0.9) - self.frame * 0.08)
                idx = int((v + 1.0) / 2.0 * grad_len)
                self.set_char(rows, xx, yy, self.GRAD[max(0, min(grad_len, idx))])

    def _waves_at(self, rows, cx, cy, scale, levels):
        mid = float(np.mean(levels[len(levels) // 3: len(levels) // 3 * 2])) if len(levels) > 3 else 0.3
        freq = 0.08 + mid * 0.25; phase = self.frame * 0.12; grad_len = len(self.GRAD) - 1; half = scale
        for yy in range(cy - half, cy + half + 1):
            ydiff = yy - cy
            for xx in range(cx - half, cx + half + 1):
                xdiff = xx - cx; r = math.hypot(xdiff, ydiff)
                if r > half: continue
                v = math.sin(xdiff * freq + math.cos(ydiff * freq * 0.8) + phase)
                idx = int((v + 1.0) / 2.0 * grad_len)
                self.set_char(rows, xx, yy, self.GRAD[max(0, min(grad_len, idx))])

    def draw_pattern(self, rows, mask, levels):
        name = self.SHAPES[self.shape_idx]
        base_scale = min(int(self.h * 0.42), int(self.w * 0.18)); base_scale = max(6, base_scale)
        gap = int(base_scale * 2.8); offsets = [0]
        if self.w >= gap * 3: offsets = [-gap, 0, gap]
        elif self.w >= gap * 2: offsets = [-gap // 2, gap // 2]
        for off in offsets:
            cx = self.cx + off; cy = self.cy
            if name == "mandala": self._mandala_at(rows, cx, cy, base_scale, levels)
            elif name == "spiral": self._spiral_at(rows, cx, cy, base_scale, levels)
            elif name == "checker": self._checker_at(rows, cx, cy, base_scale, levels)
            elif name == "rings": self._rings_at(rows, cx, cy, base_scale, levels)
            else: self._waves_at(rows, cx, cy, base_scale, levels)

    def add_spectrum(self, rows, mask, levels, bar_slots=None):
        if bar_slots is None: bar_slots = min(self.w - 6, len(levels))
        if bar_slots <= 0: return
        L = np.asarray(levels, dtype=float)
        if self.prev_levels is None or self.prev_levels.shape != L.shape: self.prev_levels = L.copy()
        a = self.smooth_alpha; smooth = a * L + (1 - a) * self.prev_levels; self.prev_levels = smooth
        src = np.linspace(0.0, smooth.size - 1.0, smooth.size); tgt = np.linspace(0.0, smooth.size - 1.0, bar_slots)
        mapped = np.interp(tgt, src, smooth)
        maxh = self.h - 3; chars = self.SPECTRUM; clen = len(chars)
        base_x = 3
        for i, v in enumerate(mapped):
            bar_h = int(v * maxh); x = base_x + i
            for j in range(bar_h):
                y = self.h - 2 - j
                ch = chars[min(j * clen // max(1, maxh), clen - 1)]
                self.set_spectrum_char(rows, mask, x, y, ch)

    def generate(self, levels):
        rows, mask = self.blank_rows_and_mask()
        self.draw_pattern(rows, mask, levels)
        self.add_spectrum(rows, mask, levels, bar_slots=min(self.w - 6, len(levels)))
        intensity = float(np.mean(levels)) if len(levels) else 0.0; beat = False
        if self.cool <= 0 and self.prev_intensity > 0 and intensity > max(0.12, self.prev_intensity * 1.9):
            beat = True; self.cool = 12
        self.prev_intensity = self.prev_intensity * 0.82 + intensity * 0.18
        if self.cool > 0: self.cool -= 1
        if beat: self.shape_idx = (self.shape_idx + 1) % len(self.SHAPES)
        self.frame += 1
        rows_str = ["".join(r)[: self.w] for r in rows]
        mask_str = ["".join(m)[: self.w] for m in mask]
        return rows_str, mask_str

class ReliableAudio:
    LOW_BAND_ATTENUATION = 1.0
    def __init__(self, path):
        self.path = path; self.sr = 22050; self.data = None; self.duration = 0.0; self.t = 0.0; self.playing = False; self.start = 0.0
        self._band_cache = {}
        try:
            pygame.mixer.pre_init(frequency=self.sr, size=-16, channels=2, buffer=1024)
            pygame.mixer.init()
        except Exception:
            pass
        self._load()

    def _load(self):
        try:
            self.data, sr = librosa.load(self.path, sr=self.sr, mono=True); self.sr = sr
            self.duration = len(self.data) / float(self.sr)
            try: pygame.mixer.music.load(self.path)
            except Exception: pass
        except Exception as e:
            print("Audio load error:", e); sys.exit(1)

    def _get_band_indices(self, bands, win):
        key = (bands, win, int(self.sr))
        if key in self._band_cache: return self._band_cache[key]
        freqs = np.fft.rfftfreq(win, 1.0 / self.sr); fmin = 20.0; nyq = self.sr / 2.0
        edges = np.logspace(math.log10(fmin), math.log10(max(fmin + 1.0, nyq)), num=bands + 1)
        starts = np.searchsorted(freqs, edges[:-1], side="left"); ends = np.searchsorted(freqs, edges[1:], side="left")
        slices = [(int(s), int(e)) for s, e in zip(starts, ends)]; self._band_cache[key] = slices; return slices

    def get_frequency_data(self, bands=64, fmin=20.0):
        if self.data is None or not self.playing: return np.zeros(bands)
        pos = int(self.t * self.sr); win = 4096 if len(self.data) > 16384 else 2048
        if pos + win >= len(self.data): return np.zeros(bands)
        window = self.data[pos: pos + win] * np.hanning(win); mag = np.abs(np.fft.rfft(window))
        slices = self._get_band_indices(bands, win); levels = np.zeros(bands, dtype=float)
        for i, (s, e) in enumerate(slices):
            if e > s: levels[i] = mag[s:e].mean()
            else:
                center = (s + e) // 2
                levels[i] = mag[center] if center < mag.size else 0.0
        low_count = min(3, levels.size)
        if low_count > 0 and self.LOW_BAND_ATTENUATION != 1.0: levels[:low_count] *= self.LOW_BAND_ATTENUATION
        maxv = levels.max()
        if maxv > 0: levels /= (maxv + 1e-12)
        return np.power(levels, 0.9)

    def play(self):
        if not self.playing and self.t < self.duration:
            try: pygame.mixer.music.play(loops=0, start=self.t)
            except Exception:
                try: pygame.mixer.music.play()
                except Exception: pass
            self.playing = True; self.start = time.time() - self.t

    def pause(self):
        try: pygame.mixer.music.pause()
        except Exception: pass
        self.playing = False

    def toggle(self): self.pause() if self.playing else self.play()

    def reset(self):
        try: pygame.mixer.music.stop()
        except Exception: pass
        self.t = 0.0; self.playing = False

    def update_time(self):
        if self.playing:
            self.t = time.time() - self.start
            if self.t >= self.duration: self.playing = False

# -------- main ----------
def main():
    created_files = []
    created_folder = False
    if len(sys.argv) == 2:
        path_arg = Path(sys.argv[1])
        if not path_arg.exists():
            print("File not found:", path_arg); return
        chosen = path_arg
    else:
        dest, created_files, created_folder = choose_audio_and_copy("input_audio")
        if dest is None:
            print("No audio selected. Exiting."); return
        chosen = dest
    print("Using audio:", chosen)
    disp = NoFlickerDisplay(); inp = StableInput(); audio = ReliableAudio(str(chosen)); viz = ASCIIVisualizer(disp.width, disp.height)

    def _cleanup_copied():
        for f in created_files:
            try:
                if f.exists(): f.unlink()
            except Exception:
                pass
        try:
            folder = exe_parent_dir() / "input_audio"
            if created_folder and folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
        except Exception:
            pass
    atexit.register(_cleanup_copied)

    fps = 28.0; delay = 1.0 / fps; frame = 0
    try:
        while True:
            loop_start = time.time(); audio.update_time()
            bands = min(128, max(24, disp.width - 6)); levels = audio.get_frequency_data(bands=bands)
            rows, mask = viz.generate(levels.tolist() if isinstance(levels, np.ndarray) else list(levels))
            disp.update_display(rows, mask)
            if frame % 4 == 0:
                status = f"Status: {'PLAYING' if audio.playing else 'PAUSED'} | Shape: {ASCIIVisualizer.SHAPES[viz.shape_idx].upper()}"
                if audio.duration > 0:
                    p = min(1.0, audio.t / audio.duration); bl = 38; filled = int(bl * p)
                    prog = "█" * filled + "░" * (bl - filled); disp.update_status(status, f"[{prog}]")
                else:
                    disp.update_status(status, "")
            k = inp.get_key()
            if k:
                k = k.lower()
                if k in ("q", "\x03", "\x04"): break
                if k == " ": audio.toggle()
                elif k == "r": audio.reset()
            frame += 1; elapsed = time.time() - loop_start; to_sleep = delay - elapsed
            if to_sleep > 0: time.sleep(to_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        disp.cleanup(); inp.cleanup()
        try: pygame.mixer.quit()
        except Exception: pass
        _cleanup_copied()
        print("Thanks for using the visualizer")

if __name__ == "__main__": main()

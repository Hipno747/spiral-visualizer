#!/usr/bin/env python3
"""
SPIRAL — compact, faster terminal audio visualizer with side speakers driven by volume
Requires: pip install numpy librosa pygame
Controls: SPACE Play/Pause | R Reset | Q Quit
"""
import sys, time, math, random, shutil, platform, atexit, re
from pathlib import Path

WINDOWS = platform.system() == "Windows"
if WINDOWS:
    try: import msvcrt
    except: msvcrt = None
else:
    try: import termios, tty, select
    except: termios = tty = select = None

try:
    import numpy as np, librosa, pygame
except Exception:
    print("Missing packages. Install: pip install numpy librosa pygame"); sys.exit(1)

try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# --- resource-aware exe parent helper (replacement requested)
def exe_parent_dir():
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    try:
        return Path(__file__).resolve().parent
    except:
        return Path.cwd()

# --- tiny helpers & palette
def ansi_dim(c): return f"\x1b[2;38;5;{c}m"
RESET = "\x1b[0m"
# Centralized color palettes (256-color codes). Each palette is a list of ANSI color strings.
PALETTES = {
    "red":    [ansi_dim(c) for c in (196, 160, 124, 88, 52)],
    "orange": [ansi_dim(c) for c in (208, 202, 166, 130, 94)],
    "yellow": [ansi_dim(c) for c in (226, 220, 214, 184, 148)],
    "green":  [ansi_dim(c) for c in (22, 28, 34, 40, 46)],
    "blue":   [ansi_dim(c) for c in (19, 27, 33, 39, 45)],
    "indigo": [ansi_dim(c) for c in (54, 55, 56, 57, 63)],
    "violet": [ansi_dim(c) for c in (99, 105, 111, 127, 135)],
    "purple": [ansi_dim(c) for c in (129, 135, 141, 147, 141)],
}
COLOR_SPEED = 0.45
SUPPORTED = (".mp3", ".wav", ".ogg", ".flac")

# Load active palette names from colors.txt (first line -> header, second line -> accent)
def load_color_selection(fname="colors.txt"):
    # Preference order for colors.txt (so users can ship an editable file next to the exe):
    # 1) external file next to the executable (when frozen) or cwd
    # 2) bundled resource in PyInstaller's _MEIPASS (if present)
    # 3) project/source directory
    default_header, default_accent = "purple", "green"
    def parse_text(txt):
        toks = [t.strip() for t in re.split('[,\n]+', txt) if t.strip() and not t.strip().startswith('#')]
        if not toks:
            return None
        header = toks[0]
        accent = toks[1] if len(toks) > 1 else default_accent
        if header not in PALETTES: header = default_header
        if accent not in PALETTES: accent = default_accent
        return header, accent
    try:
        # 1) Check external next to the packaged EXE (works for both --onefile and --onedir):
        if getattr(sys, 'frozen', False):
            try:
                argv0_dir = Path(sys.argv[0]).resolve().parent
                p_ext = argv0_dir / fname
                if p_ext.exists():
                    parsed = parse_text(p_ext.read_text(encoding='utf-8'))
                    if parsed: return parsed
            except Exception:
                pass
            # fallback: also check sys.executable parent
            try:
                exe_dir = Path(sys.executable).resolve().parent
                p_ext2 = exe_dir / fname
                if p_ext2.exists():
                    parsed = parse_text(p_ext2.read_text(encoding='utf-8'))
                    if parsed: return parsed
            except Exception:
                pass
            # also check one level up from the executable directory (useful when exe is inside a subfolder)
            try:
                exe_parent = Path(sys.executable).resolve().parent.parent
                p_ext3 = exe_parent / fname
                if p_ext3.exists():
                    parsed = parse_text(p_ext3.read_text(encoding='utf-8'))
                    if parsed: return parsed
            except Exception:
                pass
        # 1b) also check current working directory (helpful for double-click runs)
        p_cwd = Path.cwd() / fname
        if p_cwd.exists():
            parsed = parse_text(p_cwd.read_text(encoding='utf-8'))
            if parsed: return parsed
        # 2) If frozen with PyInstaller, check the bundled temp folder (_MEIPASS)
        meipass = getattr(sys, '_MEIPASS', None)
        if meipass:
            p_mei = Path(meipass) / fname
            if p_mei.exists():
                parsed = parse_text(p_mei.read_text(encoding='utf-8'))
                if parsed: return parsed
        # 3) Fallback: project/source directory (normal dev run)
        p_src = exe_parent_dir() / fname
        if p_src.exists():
            parsed = parse_text(p_src.read_text(encoding='utf-8'))
            if parsed: return parsed
        # Also check parent of the source/exe directory (handles colors.txt located alongside the parent folder)
        try:
            p_src_parent = exe_parent_dir().parent / fname
            if p_src_parent.exists():
                parsed = parse_text(p_src_parent.read_text(encoding='utf-8'))
                if parsed: return parsed
        except Exception:
            pass
    except Exception:
        pass
    return default_header, default_accent

# active palette names chosen at startup (can be edited via colors.txt)
ACTIVE_HEADER_NAME, ACTIVE_ACCENT_NAME = load_color_selection()

def choose_and_copy(sub="input_audio"):
    base = exe_parent_dir(); folder = base / sub; created_folder = False
    if not folder.exists():
        try: folder.mkdir(parents=True, exist_ok=True); created_folder = True
        except: created_folder = False
    chosen = None
    if TK_AVAILABLE:
        try:
            r = tk.Tk(); r.withdraw()
            p = filedialog.askopenfilename(title="Select audio", filetypes=[("Audio","*.mp3 *.wav *.ogg *.flac"),("All","*.*")])
            r.update(); r.destroy()
            if p: chosen = Path(p)
        except: chosen = None
    if chosen is None:
        e = input(f"Audio path (Enter to cancel). Supported: {SUPPORTED}\n> ").strip()
        if e: chosen = Path(e)
    if not chosen or not chosen.exists() or chosen.suffix.lower() not in SUPPORTED:
        return None, [], created_folder
    dest = folder / chosen.name
    if dest.exists(): dest = folder / f"{chosen.stem}_{int(time.time()*1000)}{chosen.suffix}"
    try: shutil.copy2(str(chosen), str(dest))
    except Exception as ex: print("Copy failed:", ex); return None, [], created_folder
    return dest, [dest], created_folder

# --- Display + keyboard (small, efficient)
class Display:
    def __init__(self):
        tw, th = shutil.get_terminal_size(); self.term_width, self.term_height = tw, th
        base_w = max(36, min(80, tw-6)); base_h = max(12, min(28, th-10))
        w = min(tw-4, int(base_w*1.7)); h = min(th-8, int(base_h*1.4))
        self.width, self.height = max(40, w), max(12, h)
        self.canvas = [" " * self.width for _ in range(self.height)]
        self.hide, self.show, self.clear = "\33[?25l", "\33[?25h", "\33[2J\33[H"
        print(self.hide + self.clear, end="", flush=True)
        header_palette = PALETTES.get(ACTIVE_HEADER_NAME) or []
        accent_palette = PALETTES.get(ACTIVE_ACCENT_NAME) or []
        print((header_palette[0] if len(header_palette)>0 else "") + "=" * self.term_width + RESET)
        print((accent_palette[2] if len(accent_palette)>2 else "") + "SPIRAL".center(self.term_width) + RESET)
        print((header_palette[0] if len(header_palette)>0 else "") + "=" * self.term_width + RESET)
        print((accent_palette[1] if len(accent_palette)>1 else "") + "Status: Ready | Controls: SPACE=Play R=Reset Q=Quit" + RESET + "\n", flush=True)
    def goto(self,x,y): return f"\33[{y+6};{x+2}H"
    def _colorize(self,row,mask,base):
        out=[]; header_palette = PALETTES.get(ACTIVE_HEADER_NAME) or []
        accent_palette = PALETTES.get(ACTIVE_ACCENT_NAME) or []
        npal=len(header_palette); gpal=len(accent_palette)
        for i,ch in enumerate(row):
            if ch==" ":
                out.append(" ")
            elif mask and mask[i]=="1":
                out.append(accent_palette[(base+(i//4))%gpal]+ch+RESET)
            else:
                out.append(header_palette[(base+(i//6))%npal]+ch+RESET)
        return "".join(out)
    def update(self, rows, masks=None):
        parts=[]; gw=self.goto; buf=self.canvas; base=int(time.time()*COLOR_SPEED)
        for y,row in enumerate(rows):
            if y>=self.height: break
            if row!=buf[y]:
                mask = masks[y] if masks else None
                parts.append(gw(0,y)+self._colorize(row,mask,base+(y//3)))
                buf[y]=row
        if parts: print("".join(parts), end="", flush=True)
    def status(self,s,p=""):
        tw=self.term_width
        accent_palette = PALETTES.get(ACTIVE_ACCENT_NAME) or []
        print("\33[4;1H"+" "*(tw-1), end="")
        print("\33[4;1H"+(accent_palette[2] if len(accent_palette)>2 else "")+s[:tw-1]+RESET, end="", flush=True)
        if p:
            print("\33[5;1H"+" "*(tw-1), end="")
            print("\33[5;1H"+(accent_palette[1] if len(accent_palette)>1 else "")+p[:tw-1]+RESET, end="", flush=True)
    def cleanup(self): print(self.show + self.clear, end="", flush=True)

class Kbd:
    def __init__(self):
        self.old=None; self.stdin_tty=sys.stdin.isatty()
        if not WINDOWS and self.stdin_tty and termios:
            try: self.old = termios.tcgetattr(sys.stdin.fileno()); tty.setcbreak(sys.stdin.fileno())
            except: self.old=None
    def get(self):
        try:
            if WINDOWS and msvcrt:
                if msvcrt.kbhit(): return msvcrt.getwch().lower()
                return None
            if not self.stdin_tty or select is None: return None
            r,_,_ = select.select([sys.stdin],[],[],0)
            if r: return sys.stdin.read(1).lower()
            return None
        except: return None
    def cleanup(self):
        if not WINDOWS and self.old and termios:
            try: termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old)
            except: pass

# --- Visualizer (compact, efficient)
class Visualizer:
    GRAD = [" ",".",":","-","=","+","*","█","▓","█"]
    SPECTRUM = ["░","▒","▓","█"]
    SHAPES = ["mandala","spiral","checker","rings","waves"]
    def __init__(self,w,h):
        self.w,self.h=w,h; self.cx,self.cy=w//2,h//2
        self.frame=0; self.shape_idx=0; self.prev_intensity=0.0; self.cool=0
        self.prev_levels=None; self.smooth_alpha=0.55
        self.pulse_timer=0.0; self.pulse_duration=0.28
        self.star_lives={}; self.star_fade_frames=max(1,int(self.pulse_duration*30)); self.star_glyphs=["✦","*","."]
        # speaker state (driven by audio overall level)
        self.speaker_scale = 0.0
        self.speaker_target = 0.0
        self.speaker_smooth = 0.22
        self.speaker_level_multiplier = 2.6
        self.speaker_decay = 0.06
        self.speaker_thickness = 2
        self.speaker_glyph = "●"
    def blank(self): return [list(" "*self.w) for _ in range(self.h)], [list("0"*self.w) for _ in range(self.h)]
    def _set(self, rows,x,y,ch):
        if 0<=x<self.w and 0<=y<self.h: rows[y][x]=ch
    def _set_spec(self, rows,mask,x,y,ch):
        if 0<=x<self.w and 0<=y<self.h: rows[y][x]=ch; mask[y][x]="1"
    def _iter_region(self,cx,cy,r,fn):
        r_i=int(math.ceil(r)); y0=max(0,cy-r_i); y1=min(self.h-1,cy+r_i)
        x0=max(0, cx-r_i); x1=min(self.w-1,cx+r_i)
        for yy in range(y0,y1+1):
            yd=yy-cy
            for xx in range(x0,x1+1):
                xd=xx-cx; dist=math.hypot(xd,yd)
                if dist<=r: fn(xx,yy,xd,yd,dist)
    def _mandala(self,rows,cx,cy,scale,levels):
        gl=len(self.GRAD)-1; bass=float(np.mean(levels[:max(1,len(levels)//6)])) if len(levels) else 0.2
        spokes=max(6,int(6+bass*18)); rings=max(3,int(3+bass*8)); hi=levels[len(levels)//2] if len(levels)>1 else 0.2
        def d(xx,yy,xd,yd,dist):
            a=math.atan2(yd,xd); v=0.5*(math.cos(dist*(rings*0.6))+0.5*math.cos(a*(spokes/2))); v*=(0.7+0.6*hi)
            idx=int((v+1.0)/2.0*gl); self._set(rows,xx,yy,self.GRAD[max(0,min(gl,idx))])
        self._iter_region(cx,cy,scale,d)
    def _spiral(self,rows,cx,cy,scale,levels):
        gl=len(self.GRAD)-1; mid=float(np.mean(levels[len(levels)//4:len(levels)//2])) if len(levels)>4 else 0.2
        turns=2+int(4*mid); freq=3+int(6*mid)
        def d(xx,yy,xd,yd,dist):
            theta=math.atan2(yd,xd); s=math.sin(theta*turns + dist*0.6*freq)
            v=(s+math.cos(dist*0.8))*(0.6+0.6*mid); idx=int((v+1.0)/2.0*gl); self._set(rows,xx,yy,self.GRAD[max(0,min(gl,idx))])
        self._iter_region(cx,cy,scale,d)
    def _checker(self,rows,cx,cy,scale,levels):
        bass=float(levels[0]) if len(levels)>0 else 0.0
        cell=max(2,int(max(2,scale//6)+int(bass*6))); alt=len(levels)>2 and levels[2]>0.4
        half=int(scale); phase=self.frame*0.12; freq=0.12
        def d(xx,yy,xd,yd,dist):
            sx=((xx+half)//cell)&1; sy=((yy+half)//cell)&1; base=sx^sy
            diag=xd+yd; m=1 if math.sin(diag*freq + phase)>0 else 0
            c=base^m; ch="#" if c else "."; 
            if alt and ((xx+yy)%(cell*2)==0): ch="*"
            self._set(rows,xx,yy,ch)
        self._iter_region(cx,cy,scale,d)
    def _rings(self,rows,cx,cy,scale,levels):
        gl=len(self.GRAD)-1; low=float(np.mean(levels[:max(1,len(levels)//6)])) if len(levels) else 0.2
        rings=4+int(8*low)
        def d(xx,yy,xd,yd,dist):
            v=math.sin(dist*(rings*0.9)-self.frame*0.08); idx=int((v+1.0)/2.0*gl); self._set(rows,xx,yy,self.GRAD[max(0,min(gl,idx))])
        self._iter_region(cx,cy,scale,d)
    def _waves(self,rows,cx,cy,scale,levels):
        gl=len(self.GRAD)-1; mid=float(np.mean(levels[len(levels)//3:len(levels)//3*2])) if len(levels)>3 else 0.3
        freq=0.08+mid*0.25; phase=self.frame*0.12
        def d(xx,yy,xd,yd,dist):
            v=math.sin(xd*freq + math.cos(yd*freq*0.8) + phase); idx=int((v+1.0)/2.0*gl); self._set(rows,xx,yy,self.GRAD[max(0,min(gl,idx))])
        self._iter_region(cx,cy,scale,d)
    def _draw_speakers(self, rows, mask):
        base_rx = max(8, int(self.w * 0.12)); base_ry = max(8, int(self.h * 0.33))
        s = max(0.0, 1.0 + self.speaker_scale); rx = max(3, int(base_rx * s)); ry = max(3, int(base_ry * s))
        cy = int(self.h * 0.5); left_cx = - (rx // 2); right_cx = self.w - 1 + (rx // 2)
        glyph = self.speaker_glyph; thickness = max(1, int(self.speaker_thickness))
        def draw_half(cx, side):
            x0 = max(0, cx - rx); x1 = min(self.w - 1, cx + rx); y0 = max(0, cy - ry); y1 = min(self.h - 1, cy + ry)
            cutoff = -0.02 if side=="left" else 0.02
            inner_rx = max(1, rx - thickness); inner_ry = max(1, ry - thickness)
            for yy in range(y0, y1+1):
                for xx in range(x0, x1+1):
                    nx = (xx - cx) / float(rx); ny = (yy - cy) / float(ry)
                    v = nx*nx + ny*ny
                    if v <= 1.0 and ((side=="left" and nx>cutoff) or (side=="right" and nx< -cutoff)):
                        ix = (xx - cx) / float(inner_rx); iy = (yy - cy) / float(inner_ry)
                        if ix*ix + iy*iy >= 1.0:
                            rows[yy][xx] = glyph; mask[yy][xx] = "1"
        draw_half(left_cx,"left"); draw_half(right_cx,"right")
    def draw_pattern(self,rows,mask,levels):
        name=self.SHAPES[self.shape_idx]; base_scale=max(6,min(int(self.h*0.42),int(self.w*0.18))); gap=int(base_scale*2.8)
        offs=[0] if self.w<gap*2 else ([-gap,0,gap] if self.w>=gap*3 else [-gap//2,gap//2])
        for off in offs:
            cx=self.cx+off; scale=base_scale
            if name=="mandala": self._mandala(rows,cx,self.cy,scale,levels)
            elif name=="spiral": self._spiral(rows,cx,self.cy,scale,levels)
            elif name=="checker": self._checker(rows,cx,self.cy,scale,levels)
            elif name=="rings": self._rings(rows,cx,self.cy,scale,levels)
            else: self._waves(rows,cx,self.cy,scale,levels)
    def add_spectrum(self,rows,mask,levels,bar_slots=None):
        if bar_slots is None: bar_slots=min(self.w-6,len(levels))
        if bar_slots<=0: return
        L=np.asarray(levels,dtype=float)
        if self.prev_levels is None or self.prev_levels.shape!=L.shape: self.prev_levels=L.copy()
        a=self.smooth_alpha; smooth=a*L+(1-a)*self.prev_levels; self.prev_levels=smooth
        src=np.linspace(0.0,smooth.size-1.0,smooth.size); tgt=np.linspace(0.0,smooth.size-1.0,bar_slots)
        mapped=np.interp(tgt,src,smooth)
        maxh=self.h-3; chars=self.SPECTRUM; clen=len(chars); base_x=3
        for i,v in enumerate(mapped):
            bar_h=int(v*maxh); x=base_x+i
            for j in range(bar_h):
                y=self.h-2-j; ch=chars[min(j*clen//max(1,maxh),clen-1)]; self._set_spec(rows,mask,x,y,ch)
    def _add_star_at(self,x,y):
        if 0<=x<self.w and 0<=y<self.h and (x,y) not in self.star_lives:
            self.star_lives[(x,y)]=self.star_fade_frames; return True
        return False
    def _decay_stars(self):
        if not self.star_lives: return
        tod=[] 
        for k in list(self.star_lives):
            self.star_lives[k]-=1
            if self.star_lives[k]<=0: tod.append(k)
        for k in tod: del self.star_lives[k]
    def _draw_stars(self,rows,mask):
        for (x,y),life in list(self.star_lives.items()):
            if 0<=x<self.w and 0<=y<self.h and rows[y][x]==" ":
                frac=life/float(max(1,self.star_fade_frames))
                g=self.star_glyphs[0] if frac>0.66 else (self.star_glyphs[1] if frac>0.33 else self.star_glyphs[2])
                rows[y][x]=g; mask[y][x]="1"
    def generate(self,levels):
        rows,mask=self.blank()
        self._draw_speakers(rows,mask)
        self.draw_pattern(rows,mask,levels)
        self.add_spectrum(rows,mask,levels,bar_slots=min(self.w-6,len(levels)))
        self._draw_stars(rows,mask)
        intensity=float(np.mean(levels)) if len(levels) else 0.0
        self.prev_intensity=self.prev_intensity*0.82 + intensity*0.18
        if self.cool>0: self.cool-=1
        self._decay_stars()
        if self.pulse_timer>0: self.pulse_timer=max(0.0,self.pulse_timer-(1.0/30.0))
        if self.speaker_scale > self.speaker_target: self.speaker_scale = max(self.speaker_target, self.speaker_scale - self.speaker_decay)
        else: self.speaker_scale = self.speaker_target
        self.frame+=1
        return ["".join(r)[:self.w] for r in rows], ["".join(m)[:self.w] for m in mask]

# --- Audio (compact, robust)
class Audio:
    LOW_BAND_ATTENUATION=1.0
    def __init__(self,path):
        self.path=path; self.sr=22050; self.data=None; self.duration=0.0; self.t=0.0; self.playing=False; self.start=0.0
        self._band_cache={}; self._energy_mean=0.0; self._energy_alpha=0.08
        self._last_pulse_time=0.0; self._pulse_cooldown=0.25; self._sensitivity=1.6; self._pending_pulse=False
        self._low_energy=0.0; self._sustain_spawn_interval=0.22; self._last_sustain_spawn=0.0; self._sustain_threshold=0.9
        self._pulse_count_since_shape=0; self._shape_change_every=4
        self._vis_max = 1e-9; self._overall_level = 0.0
        try:
            pygame.mixer.pre_init(frequency=self.sr,size=-16,channels=2,buffer=1024); pygame.mixer.init()
        except: pass
        self._load()
    def _load(self):
        try:
            self.data,sr=librosa.load(self.path,sr=self.sr,mono=True); self.sr=sr
            self.duration=len(self.data)/float(self.sr)
            try: pygame.mixer.music.load(self.path)
            except: pass
        except Exception as e:
            print("Audio load error:",e); self.data=None; self.duration=0.0
    def _get_band_indices(self,bands,win):
        key=(bands,win,int(self.sr))
        if key in self._band_cache: return self._band_cache[key]
        freqs=np.fft.rfftfreq(win,1.0/self.sr); fmin=20.0; nyq=self.sr/2.0
        edges=np.logspace(math.log10(fmin),math.log10(max(fmin+1.0,nyq)),num=bands+1)
        starts=np.searchsorted(freqs,edges[:-1]); ends=np.searchsorted(freqs,edges[1:])
        slices=[(int(s),int(e)) for s,e in zip(starts,ends)]; self._band_cache[key]=slices; return slices
    def get_frequency_data(self,bands=64,fmin=20.0):
        if self.data is None or not self.playing:
            self._energy_mean*=0.995; self._low_energy*=0.995; self._overall_level*=0.98
            return np.zeros(bands)
        pos=int(self.t*self.sr); win=4096 if len(self.data)>16384 else 2048
        if pos+win>=len(self.data): return np.zeros(bands)
        window=self.data[pos:pos+win]*np.hanning(win); mag=np.abs(np.fft.rfft(window))
        slices=self._get_band_indices(bands,win); levels=np.zeros(bands,dtype=float)
        for i,(s,e) in enumerate(slices):
            if e> s: levels[i]=mag[s:e].mean()
            else:
                c=(s+e)//2; levels[i]=mag[c] if c<mag.size else 0.0
        low_count=min(3,levels.size)
        if low_count>0 and self.LOW_BAND_ATTENUATION!=1.0: levels[:low_count]*=self.LOW_BAND_ATTENUATION
        maxv=levels.max()
        if maxv>0: levels/=(maxv+1e-12)
        levels=np.power(levels,0.9)
        # overall visual level (smoothed)
        try: raw_overall = float(np.mean(mag)) if mag.size else float(np.mean(levels)) if levels.size else 0.0
        except: raw_overall = float(np.mean(levels)) if levels.size else 0.0
        self._vis_max = max(self._vis_max * 0.995, raw_overall, 1e-9)
        cur = raw_overall / (self._vis_max + 1e-12)
        self._overall_level = 0.15 * cur + 0.85 * self._overall_level
        # low-frequency bookkeeping (unchanged)
        now=time.time()
        try:
            low_bins=max(1,min(8,mag.size//8))
            raw_low=float(mag[:low_bins].mean()) if mag.size>=low_bins else float(levels[:low_count].mean())
        except:
            raw_low=float(levels[:low_count].mean()) if levels.size else 0.0
        self._energy_mean=self._energy_alpha*raw_low + (1.0-self._energy_alpha)*self._energy_mean
        self._low_energy=0.15*raw_low + 0.85*self._low_energy
        if (now-self._last_pulse_time)>self._pulse_cooldown and raw_low>max(1e-12,self._sensitivity*self._energy_mean):
            self._last_pulse_time=now; self._pending_pulse=True; self._pulse_count_since_shape+=1
        return levels
    def pop_pulse(self):
        if getattr(self,"_pending_pulse",False): self._pending_pulse=False; return True
        return False
    def bass_sustained(self,rel_threshold=None):
        if rel_threshold is None: rel_threshold=self._sustain_threshold
        ref=max(1e-12,self._energy_mean); return (self._low_energy>=rel_threshold*ref)
    def should_change_shape_now(self):
        if self._pulse_count_since_shape>=self._shape_change_every:
            self._pulse_count_since_shape=0; return True
        return False
    def play(self):
        if not self.playing and self.t<self.duration:
            try: pygame.mixer.music.play(loops=0,start=self.t)
            except:
                try: pygame.mixer.music.play()
                except: pass
            self.playing=True; self.start=time.time()-self.t
    def pause(self):
        try: pygame.mixer.music.pause()
        except: pass
        self.playing=False
    def toggle(self): self.pause() if self.playing else self.play()
    def reset(self):
        try: pygame.mixer.music.stop()
        except: pass
        self.t=0.0; self.playing=False
    def update_time(self):
        if self.playing:
            self.t=time.time()-self.start
            if self.t>=self.duration: self.playing=False

# --- Main loop (tight)
def main():
    dest=None; created_files=[]; created_folder=False
    if len(sys.argv)==2:
        p=Path(sys.argv[1]); 
        if not p.exists(): print("File not found:",p); return
        dest=p
    else:
        dest,created_files,created_folder=choose_and_copy("input_audio")
        if dest is None: print("No audio selected. Exiting."); return
    print("Using audio:",dest)
    disp=Display(); kbd=Kbd(); audio=Audio(str(dest)); viz=Visualizer(disp.width,disp.height)
    def _cleanup():
        for f in created_files:
            try:
                if f.exists(): f.unlink()
            except: pass
        try:
            folder=exe_parent_dir()/ "input_audio"
            if created_folder and folder.exists() and not any(folder.iterdir()): folder.rmdir()
        except: pass
    atexit.register(_cleanup)
    fps=28.0; delay=1.0/fps; frame=0

    # sensitivity tuning for speakers
    SPEAKER_THRESHOLD = 0.28   # only above this does speaker move
    SPEAKER_GAMMA = 2.2

    try:
        while True:
            t0=time.time(); audio.update_time()
            bands=min(128,max(24,disp.width-6))
            levels=audio.get_frequency_data(bands=bands)
            # speaker: threshold + nonlinear mapping to reduce sensitivity
            vol = getattr(audio, "_overall_level", 0.0)
            if vol <= SPEAKER_THRESHOLD:
                vol_adj = 0.0
            else:
                norm = (vol - SPEAKER_THRESHOLD) / (1.0 - SPEAKER_THRESHOLD)
                vol_adj = max(0.0, min(1.0, norm)) ** SPEAKER_GAMMA
            target = vol_adj * viz.speaker_level_multiplier
            viz.speaker_target = (1.0 - viz.speaker_smooth) * viz.speaker_target + viz.speaker_smooth * target
            if viz.speaker_scale < viz.speaker_target: viz.speaker_scale = viz.speaker_target
            else: viz.speaker_scale = max(viz.speaker_target, viz.speaker_scale - viz.speaker_decay)

            now=time.time()
            if audio.pop_pulse():
                viz.pulse_timer=viz.pulse_duration
                min_stars=max(6,int((viz.w*viz.h)*0.002)); max_attempts=max(200,int((viz.w*viz.h)*0.02))
                added=0; attempts=0
                while added<min_stars and attempts<max_attempts:
                    if viz._add_star_at(random.randrange(0,viz.w), random.randrange(0,viz.h)): added+=1
                    attempts+=1
                if added<min_stars:
                    for yy in range(viz.h):
                        if added>=min_stars: break
                        for xx in range(viz.w):
                            if added>=min_stars: break
                            if viz._add_star_at(xx,yy): added+=1
                extra=int(min(max(0,(viz.w*viz.h)//400),8))
                for _ in range(extra): viz._add_star_at(random.randrange(0,viz.w), random.randrange(0,viz.h))
                if audio.should_change_shape_now(): viz.shape_idx=(viz.shape_idx+1)%len(Visualizer.SHAPES)
            if audio.bass_sustained():
                if (now-audio._last_sustain_spawn)>=audio._sustain_spawn_interval:
                    audio._last_sustain_spawn=now
                    sustain_count=max(3,int((viz.w*viz.h)*0.0008)); added=0; attempts=0; max_attempts=max(80,sustain_count*20)
                    while added<sustain_count and attempts<max_attempts:
                        if viz._add_star_at(random.randrange(0,viz.w), random.randrange(0,viz.h)): added+=1
                        attempts+=1

            rows,mask=viz.generate(list(levels) if isinstance(levels,np.ndarray) else list(levels))
            disp.update(rows,mask)
            if frame%4==0:
                status=f"Status: {'PLAYING' if audio.playing else 'PAUSED'} | Shape: {Visualizer.SHAPES[viz.shape_idx].upper()}"
                if audio.duration>0:
                    p=min(1.0,audio.t/audio.duration); bl=38; filled=int(bl*p)
                    prog="█"*filled+"░"*(bl-filled); disp.status(status,f"[{prog}]")
                else: disp.status(status,"")
            k=kbd.get()
            if k:
                if k in ("q","\x03","\x04"): break
                if k==" ": audio.toggle()
                elif k=="r": audio.reset(); viz.star_lives.clear()
            frame+=1; elapsed=time.time()-t0; to_sleep=delay-elapsed
            if to_sleep>0: time.sleep(to_sleep)
    except KeyboardInterrupt:
        pass
    finally:
        disp.cleanup(); kbd.cleanup()
        try: pygame.mixer.quit()
        except: pass
        _cleanup()
        print("Thanks for using the visualizer")

if __name__=="__main__": main()

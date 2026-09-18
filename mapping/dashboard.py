"""Native desktop viewer. No synthetic source and no direct camera access."""
import json
import math
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from core import grid_pixel, occupancy_rgb


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.rgb = self.depth = self.local = self.grid = None
        self.meta = None
        self.path = []
        self.markers = []
        self.marker_reason = '탐지 위치 대기'
        self.pose = None
        self.tracking = 'WAITING'
        self.reason = '카메라와 ROS 입력 대기'
        self.model = 'NO_MODEL'
        self.rows = []
        self.times = {}
        self.error = ''
        self.source = 'LIVE D435i / RTAB-Map'
        self.recorded = False
        self.performance = {}


def map_image(state):
    if state.grid is None or state.meta is None:
        return None
    im = Image.fromarray(occupancy_rgb(state.grid))
    # Render at increased native resolution without aspect distortion.
    scale = max(1, min(4, int(800/max(im.size))))
    im = im.resize((im.width*scale, im.height*scale), Image.Resampling.NEAREST)
    d = ImageDraw.Draw(im)
    def point(p):
        u, v = grid_pixel(p[0], p[1], state.meta)
        return u*scale, v*scale
    pts = [point(p) for p in state.path]
    if len(pts) > 1:
        d.line(pts, fill=(57, 160, 245), width=2)
    if state.tracking in ('TRACKING', 'RECORDED'):
        for m in state.markers:
            x, y = point(m['xyz'])
            from console_ui import font
            names = dict(person='사람',fire='화염',smoke='연기',door='문',staircase='계단')
            label = names.get(m['label'],m['label'])
            if m.get('depth_m') is not None:
                label += f" {m['depth_m']:.1f}m"
            face = font(22,True)
            width = int(d.textlength(label,font=face))+12
            tx,ty = max(0,min(x+10,im.width-width)),max(0,min(y-26,im.height-34))
            d.ellipse((x-8,y-8,x+8,y+8),fill=(255,80,110),outline='white',width=2)
            d.rectangle((tx,ty,tx+width,ty+33),fill='#101927')
            d.text((tx+5,ty),label,font=face,fill='#ff728d')
        if state.pose:
            x, y = point(state.pose)
            a = state.pose[3]-state.meta['origin'][2]
            d.ellipse((x-5, y-5, x+5, y+5), fill=(40, 225, 175))
            d.line((x, y, x+18*math.cos(a), y-18*math.sin(a)), fill=(40, 225, 175), width=3)
    return im


class Dashboard:
    def __init__(self, state, session, stop, stale=1.5):
        self.state, self.session, self.stop, self.stale = state, Path(session), stop, stale
        self.root = tk.Tk()
        self.root.title('실내 인식 — 현장 기록')
        self.root.geometry('1360x900')
        self.root.minsize(1000, 720)
        self.root.configure(bg='#101927')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Escape>', lambda _: self.close())
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview', background='#1b293b', foreground='#edf3fa', fieldbackground='#1b293b', rowheight=24)
        header = tk.Frame(self.root, bg='#101927')
        header.pack(fill='x', padx=22, pady=(18, 8))
        tk.Label(header, text='실내 인식', bg='#101927', fg='#edf3fa',
                 font=('Malgun Gothic', 16, 'bold')).pack(side='left')
        tk.Label(header, text='  /  현장 기록', bg='#101927', fg='#a5abb1', font=('Malgun Gothic', 11)).pack(side='left')
        tk.Button(header, text='저장하고 종료  Esc', command=self.close, bg='#c56841', fg='white').pack(side='right', padx=5)
        tk.Button(header, text='현재 결과 저장', command=self.snapshot, bg='#267bb0', fg='white').pack(side='right', padx=5)
        tk.Button(header, text='상세 정보', command=self.show_details, bg='#343a40', fg='white').pack(side='right', padx=5)
        self.status = tk.Label(self.root, anchor='w', bg='#1b293b', fg='#ffd086', font=('Malgun Gothic', 11), padx=12, pady=10)
        self.status.pack(fill='x', padx=22, pady=6)
        panel = tk.Frame(self.root, bg='#101927')
        panel.pack(fill='both', expand=True, padx=22)
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        panel.rowconfigure(1, weight=2)
        panel.rowconfigure(3, weight=1)
        self.views = {}
        for name, title, row, col in [('rgb', '카메라', 0, 0),
            ('map', '누적 지도', 0, 1),
            ('depth', '깊이 영상  ·  0.2–6 m', 2, 0),
            ('local', '현재 시야의 장애물  ·  누적 지도 아님', 2, 1)]:
            tk.Label(panel, text=title, anchor='w', bg='#101927', fg='#c2d3e9', font=('Malgun Gothic', 10)).grid(row=row, column=col, sticky='ew', padx=5, pady=(8, 4))
            view = tk.Label(panel, text='실제 데이터 없음', bg='#172336', fg='#8498b4', width=1, height=1)
            view.grid(row=row+1, column=col, sticky='nsew', padx=5)
            self.views[name] = view
        self.details = tk.Label(self.root, anchor='w', bg='#101927', fg='#aec2dc', font=('Malgun Gothic', 10))
        self.details.pack(fill='x', padx=26, pady=7)
        self.table = ttk.Treeview(self.root, columns=('class', 'confidence', 'distance', 'quality'), show='headings', height=4)
        for name, title in zip(self.table['columns'], ('클래스', '신뢰도', '광축 거리 Z', '깊이 상태')):
            self.table.heading(name, text=title)
        self.table.pack(fill='x', padx=26, pady=(0, 8))
        tk.Label(self.root, text='지도  —  주황: 장애물    밝은색: 관측된 빈 공간    회색: 미관측    파랑: 이동 궤적',
                 bg='#101927', fg='#859ab4').pack(pady=(0, 12))
        self.root.after(100, self.tick)

    def tick(self):
        if self.stop.is_set():
            self.root.destroy()
            return
        now = time.monotonic()
        with self.state.lock:
            s = self.state
            tracking = 'RECORDED' if s.recorded else (s.tracking if now-s.times.get('tracking', -1e9) < self.stale else 'STALE')
            # Hide stale poses and markers without discarding the last genuine map.
            if not s.recorded and (tracking != 'TRACKING' or now-s.times.get('pose', -1e9) > self.stale):
                s.pose = None
                s.markers = []
            if now-s.times.get('detections', -1e9) > self.stale*2:
                s.markers = []
            s.tracking = tracking
            camera_age = now-s.times.get('rgb', -1e9)
            camera = '저장 영상' if s.recorded else ('영상 수신' if camera_age < self.stale else '영상 없음 / 중단됨')
            model = s.model if s.recorded or now-s.times.get('model', -1e9) < self.stale*2 else '탐지 노드 입력 없음'
            source = '저장 기록' if s.recorded else ('녹화 데이터 재생' if 'TUM' in s.source else ('입력 대기' if 'OFFLINE' in s.source else s.source.split(' · ')[0]))
            health = {'TRACKING': '위치추정 정상', 'LOST': '위치추정 중단', 'STALE': '입력 중단',
                      'WAITING': '입력 대기', 'NOT_RUNNING': '누적 맵핑 미실행', 'RECORDED': '기록 열람'}.get(tracking, tracking)
            model_short = '모델 없음' if 'NO_MODEL' in model else ('탐지 준비' if 'READY' in model else ('탐지 중' if 'RUNNING' in model else model.split(':')[0]))
            self.status.config(text=f'{source}    ·    {health}    ·    {model_short}')
            pose = '카메라 위치 미확정' if not s.pose else f'카메라 map 좌표: x={s.pose[0]:.2f}, y={s.pose[1]:.2f}, z={s.pose[2]:.2f}m'
            self.details.config(text=s.error or f'{pose}    ·    기록 {self.session.name}', fg='#ff9c86' if s.error else '#aec2dc')
            images = {'rgb': s.rgb, 'depth': s.depth, 'local': s.local, 'map': map_image(s)}
            for name, im in images.items():
                view = self.views[name]
                if im is None:
                    continue
                im = Image.fromarray(im) if isinstance(im, np.ndarray) else im.copy()
                age_key = 'rgb' if name == 'rgb' else ('map' if name == 'map' else 'rgb')
                if not s.recorded and now-s.times.get(age_key, -1e9) > (5 if name == 'map' else self.stale):
                    ImageDraw.Draw(im).text((10, 10), 'STALE / LAST RECEIVED FRAME', fill=(255, 100, 80))
                im.thumbnail((max(80, view.winfo_width()), max(60, view.winfo_height())))
                photo = ImageTk.PhotoImage(im)
                view.configure(image=photo, text='')
                view.image = photo
            self.table.delete(*self.table.get_children())
            rows = s.rows if s.recorded or now-s.times.get('detections', -1e9) < self.stale*2 else []
            for row in rows:
                distance = row['depth_m']
                self.table.insert('', 'end', values=(row['label'], f"{row['confidence']:.2f}",
                    '미확정' if distance is None else f'{distance:.2f} m', row['distance_status']))
        self.root.after(33, self.tick)

    def show_details(self):
        from tkinter import messagebox
        with self.state.lock:
            messagebox.showinfo('입력 및 기록 정보', f'{self.state.source}\n\n{self.state.reason}\n\n모델: {self.state.model}\n\n저장: {self.session}\n\nF450 비행 제어 없음. 화염·연기의 자체 거리는 미확정입니다.')

    def snapshot(self):
        try:
            with self.state.lock:
                target = self.session / ('snapshot_' + str(time.time_ns()))
                target.mkdir(parents=True)
                for name in ('rgb', 'depth', 'local'):
                    data = getattr(self.state, name)
                    if data is not None:
                        Image.fromarray(data).save(target/(name+'.png'))
                im = map_image(self.state)
                if im:
                    im.save(target/'map_with_trajectory.png')
                (target/'state.json').write_text(json.dumps(dict(tracking=self.state.tracking,
                    times_monotonic=self.state.times, pose=self.state.pose, detections=self.state.rows), ensure_ascii=False, indent=2), encoding='utf-8')
        except Exception as e:
            with self.state.lock:
                self.state.error = '저장 실패: ' + str(e)

    def close(self):
        self.snapshot()
        self.stop.set()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# Keep the existing State and map projection contract; use the operations layout.
from console_ui import ConsoleDashboard as Dashboard

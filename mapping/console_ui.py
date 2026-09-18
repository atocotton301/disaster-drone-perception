"""Responsive native operations console; only displays supplied sensor state."""
import datetime
import math
from functools import lru_cache
from pathlib import Path
import time
import os
import json
import copy
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageOps

BG = '#07151c'
PANEL = '#091a22'
BORDER = '#24414c'
WHITE = '#edf3f6'
MUTED = '#92a9b5'

@lru_cache(maxsize=64)
def font(size, bold=False):
    candidates = ['C:/Windows/Fonts/malgunbd.ttf' if bold else 'C:/Windows/Fonts/malgun.ttf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
                  '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for p in candidates:
        if Path(p).exists(): return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@lru_cache(maxsize=1024)
def text_tile(value, size, color, bold):
    """Cache rasterized labels; CJK shaping every refresh dominated UI time."""
    face = font(size, bold)
    x0, y0, x1, y1 = face.getbbox(value)
    tile = Image.new('RGBA', (max(1,x1-x0), max(1,y1-y0)))
    ImageDraw.Draw(tile).text((-x0,-y0), value, font=face, fill=color)
    return tile, x0, y0

def point_cloud_preview(depth, K):
    """Camera-frame perspective view computed from measured depth, no synthetic scene."""
    y, x = np.mgrid[0:depth.shape[0]:5, 0:depth.shape[1]:5]
    z = depth[::5, ::5]
    valid = np.isfinite(z) & (z > .2) & (z < 6)
    z, x, y = z[valid], x[valid], y[valid]
    cloud = Image.new('RGB', (560, 320), '#061118')
    if not len(z): return np.array(cloud)
    k = np.asarray(K).reshape(3, 3)
    xyz = np.column_stack(((x-k[0,2])*z/k[0,0], -(y-k[1,2])*z/k[1,1], z))
    a = -.28
    xx = xyz[:,0]*math.cos(a)+xyz[:,2]*math.sin(a)
    zz = -xyz[:,0]*math.sin(a)+xyz[:,2]*math.cos(a)+1.8
    u, v = 280+xx*270/zz, 160-xyz[:,1]*270/zz
    # Rasterize in NumPy instead of thousands of Python/Pillow calls per frame.
    inside = (u >= 0) & (u < 560) & (v >= 0) & (v < 320)
    order = np.flatnonzero(inside)[np.argsort(zz[inside])]
    pixel = v[order].astype(int)*560+u[order].astype(int)
    _, first = np.unique(pixel, return_index=True)
    nearest = order[first]
    c = np.minimum(1., z[nearest]/4)
    colors = np.column_stack((40+180*c, 245-100*c, 250*(1-c))).astype(np.uint8)
    rendered = np.array(cloud)
    rendered.reshape(-1,3)[pixel[first]] = colors
    return rendered

class ConsoleDashboard:
    def __init__(self, state, session, stop, stale=1.5):
        self.state, self.session, self.stop, self.stale = state, Path(session), stop, stale
        self.root = tk.Tk()
        self.root.title('실내 재난 탐색 | RGB · SLAM 관제')
        self.root.geometry('1500x844')
        self.root.minsize(1000, 600)
        self.root.configure(bg=BG)
        controls=tk.Frame(self.root,bg=PANEL)
        controls.pack(fill='x')
        tk.Label(controls,text='저장 기록 열람' if state.recorded else '탐색 중 · 종료하면 지도와 기록을 저장합니다',
            font=('Sans',12),bg=PANEL,fg=WHITE).pack(side='left',padx=14,pady=10)
        self.finish_button=tk.Button(controls,text='기록 닫기' if state.recorded else '탐색 종료 · 지도 저장',
            command=self.close,bg='#ae3e43',fg='white',font=('Sans',13,'bold'),padx=16,pady=5)
        self.finish_button.pack(side='right',padx=12,pady=5)
        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.image_id = self.canvas.create_image(0,0,anchor='nw')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.bind('<Escape>', lambda _: self.close())
        self.root.bind('<F11>', lambda _: self.root.attributes('-fullscreen', not self.root.attributes('-fullscreen')))
        self.root.bind('<Control-s>', lambda _: self.snapshot())
        self.events = []
        self.last_event = None
        self.last_tick = None
        self.video = None
        self.record_started = time.monotonic()
        self.record_frames = 0
        if not state.recorded and os.environ.get('DISASTER_RECORD_VIDEO','1') == '1':
            import cv2
            self.session.mkdir(parents=True, exist_ok=True)
            self.video = cv2.VideoWriter(str(self.session/'console_live.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 4, (1672,940))
            if not self.video.isOpened():
                raise RuntimeError('Cannot record console video')
            self.video_log = (self.session/'console_video.jsonl').open('w',encoding='utf-8',buffering=1)
        self.root.after(80, self.tick)

    def render(self, snapshot=None):
        from dashboard import map_image
        s = snapshot if snapshot is not None else self.state
        im = Image.new('RGB', (1672,940), BG)
        d = ImageDraw.Draw(im)
        def text(x,y,value,size=18,color=WHITE,bold=False):
            tile, dx, dy = text_tile(str(value), size, color, bold)
            im.paste(tile, (x+dx,y+dy), tile)
        def box(x,y,w,h,title):
            d.rounded_rectangle((x,y,x+w,y+h), radius=4, fill=PANEL, outline=BORDER)
            text(x+17,y+10,title,19,bold=True)
        def picture(data,rect,empty='영상 입력 대기'):
            x,y,w,h=rect
            d.rectangle((x,y,x+w,y+h),fill='#0c2029')
            if data is None:
                text(x+20,y+h//2-12,empty,18,MUTED)
                return
            src=Image.fromarray(data) if isinstance(data,np.ndarray) else data.copy()
            src=ImageOps.contain(src.convert('RGB'),(w,h))
            im.paste(src,(x+(w-src.width)//2,y+(h-src.height)//2))
        now=time.monotonic()
        recorded=s.recorded or 'REPLAY' in s.source or 'TUM' in s.source
        active=s.recorded or now-s.times.get('detections',-1e9)<self.stale*2
        rows=s.rows if active else []
        people=sum(r.get('label')=='person' for r in rows)
        fires=sum(r.get('label')=='fire' for r in rows)
        tracking=s.tracking if s.recorded or now-s.times.get('tracking',-1e9)<self.stale else 'STALE'
        if not s.recorded and (tracking!='TRACKING' or now-s.times.get('pose',-1e9)>self.stale):
            s.pose=None
            s.markers=[]
        health={'TRACKING':'정상','RECORDED':'저장 기록','LOST':'추정 중단','STALE':'입력 대기','WAITING':'입력 대기','NOT_RUNNING':'대기'}.get(tracking,tracking)
        status_color='#13d47e' if tracking=='TRACKING' else '#ffab4a'
        text(28,15,'실내 재난 탐색',34,bold=True)
        text(305,32,'객체 인식 및 2D 맵핑 시스템',20)
        badge='기록 재생' if recorded else '실시간 입력'
        d.rounded_rectangle((1105,7,1475,72),radius=4,outline=BORDER,fill=PANEL)
        alert = bool(people or fires)
        message = f'사람 {people}명 · 화염 {fires}건' if alert else badge
        d.ellipse((1125,25,1150,50),fill='#ff9329' if alert else '#349bca')
        text(1168,13,message,24,'#ff9329' if alert else '#86d6f4',True)
        text(1168,44,'화염 감지 · 거리 미확정' if fires else '사람이 감지되었습니다.' if people else '영상 · 공간 정보 통합 화면',15,MUTED)
        text(1495,16,'시스템 시간',15,MUTED)
        text(1495,44,datetime.datetime.now().strftime('%m-%d %H:%M:%S'),17)

        box(18,80,973,646,'카메라 영상 (RGB)')
        picture(s.rgb,(30,121,949,592))
        d.rounded_rectangle((40,675,390,706),radius=5,fill='#10212a')
        dims=f'{s.rgb.shape[1]} × {s.rgb.shape[0]}' if s.rgb is not None else '입력 없음'
        text(50,679,f'RGB  {dims}  |  {badge}',16)
        if not s.recorded and now-s.times.get('rgb',-1e9)>self.stale:
            text(45,134,'영상 수신 대기 · 마지막 프레임',18,'#ffab4a')

        local_only = s.grid is None and s.local is not None and 'SLAM 미실행' in s.source
        box(1001,80,651,421,'2D 장애물 투영 (카메라 기준)' if local_only else '2D 공간 지도 (SLAM)')
        picture(s.local if local_only else map_image(s),(1019,122,439,365),'지도 입력 대기')
        legend=[('탐색 영역','#dee7e9'),('미탐색 영역','#313d4f'),('벽 / 장애물','#f3a041'),('카메라 위치','#28e1af'),('사람 위치','#ff506e'),('이동 경로','#39a0f5')]
        if local_only:
            legend=[('관측 장애물','#f3a041'),('나머지: 미확정','#313d4f')]
        for i,(label,c) in enumerate(legend):
            y=136+i*35
            if i in (3,4): d.ellipse((1484,y+3,1502,y+21),fill=c,outline='white')
            else: d.rounded_rectangle((1479,y+5,1513,y+22),radius=2,fill=c)
            text(1525,y,label,17)
        if s.meta:
            text(1480,410,f"격자 {s.meta['resolution']*100:g} cm",17,MUTED)
        text(1480,446,'누적 지도 아님' if local_only else 'map 좌표계',16,MUTED)
        box(1001,510,324,216,'깊이 영상 (Depth)')
        picture(s.depth,(1013,548,298,165),'깊이 입력 대기')
        box(1334,510,318,216,'포인트 클라우드')
        picture(getattr(s,'pointcloud',None),(1345,548,294,165),'깊이 좌표 입력 대기')

        box(18,737,349,182,'장치 상태')
        for i,(label,value) in enumerate([('컴퓨터','Jetson Orin NX'),('비전 센서','공개 RGB-D 기록' if recorded else 'Intel RealSense D435i'),('입력 모드',badge)]):
            y=784+i*37
            text(37,y,label,17,MUTED);text(164,y,value,17)
        box(377,737,475,182,'위치 추정 (RGB-D SLAM)')
        d.ellipse((765,754,778,767),fill=status_color)
        text(789,748,health,15,status_color)
        pose=s.pose if tracking in ('TRACKING','RECORDED') else None
        for i,axis in enumerate('XYZ'):
            y=783+i*35
            text(407,y,axis,17,MUTED)
            text(457,y,f'{pose[i]:.2f} m' if pose else '—',18)
        text(630,783,'좌표 기준',16,MUTED);text(730,783,'map',17)
        text(630,823,'Yaw',17,MUTED)
        text(730,823,f'{math.degrees(pose[3]):.1f}°' if pose else '—',18)
        text(630,863,'저장된 관측' if s.recorded else getattr(s,'marker_reason','위치 수신 상태'),13,MUTED)
        box(862,737,313,182,'탐지 현황 (YOLOv8)')
        ready=not any(k in s.model for k in ('NO_MODEL','MODEL_ERROR')) and active
        shown_classes = [('person','사람'),('fire','화염')] if os.environ.get('DISASTER_PERSON_FIRE') == '1' else [('person','사람'),('fire','화염'),('smoke','연기'),('door','문'),('staircase','계단')]
        for i,(cls,label) in enumerate(shown_classes):
            y=777+i*(48 if len(shown_classes)==2 else 27)
            d.rectangle((884,y+6,891,y+13),fill='#5dbee5')
            text(905,y,label,16)
            text(1095,y,str(sum(r.get('label')==cls for r in rows)) if ready else '—',17,bold=True)
        box(1185,737,467,182,'최근 이벤트')
        event=(tracking,s.model,bool(people),bool(fires))
        if event!=self.last_event:
            self.events.append((datetime.datetime.now().strftime('%H:%M:%S'), '화염 감지' if fires else '사람 감지' if people else health))
            self.last_event=event
        for i,(tm,msg) in enumerate(self.events[-4:]):
            text(1204,782+i*27,tm,15,MUTED);text(1290,782+i*27,msg,16)
        text(25,921,'F11 전체 화면   ·   Ctrl+S 화면 저장   ·   Esc 종료',12,MUTED)
        perf = getattr(s, 'performance', {})
        if perf:
            fields=[]
            for key,label,unit in [('capture_hz','입력','Hz'),('inference_ms','추론','ms'),('frame_age_ms','프레임 지연','ms'),('ui_hz','화면','Hz')]:
                if perf.get(key) is not None:
                    fields.append(f'{label} {perf[key]:.1f}{unit}')
            text(750,921,' · '.join(fields),12,MUTED)
        else:
            text(1215,921,'RGB · Depth · 2D SLAM 통합 관제',12,MUTED)
        return im

    def tick(self):
        tick_started = time.monotonic()
        if self.stop.is_set():
            self.root.destroy();return
        with self.state.lock:
            if self.last_tick is not None:
                self.state.performance['ui_hz'] = 1/max(tick_started-self.last_tick, 1e-6)
            self.last_tick = tick_started
            snapshot = copy.copy(self.state)
            snapshot.times = dict(self.state.times)
            snapshot.performance = dict(getattr(self.state, 'performance', {}))
        # PIL text drawing and resize must not block camera acquisition on state.lock.
        self.frame=self.render(snapshot)
        if self.video is not None:
                import cv2
                elapsed=time.monotonic()-self.record_started
                target=int(elapsed*4)
                if target > self.record_frames:
                    frame=cv2.cvtColor(np.array(self.frame),cv2.COLOR_RGB2BGR)
                    for _ in range(min(target-self.record_frames,40)):
                        self.video.write(frame)
                        self.record_frames+=1
                    self.video_log.write(json.dumps({'elapsed_s':elapsed,'video_frame':self.record_frames,'tracking':self.state.tracking,'map_stamp':self.state.meta.get('stamp') if self.state.meta else None,'keyframes':len(self.state.path),'pose':self.state.pose})+'\n')
        w,h=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height())
        shown=ImageOps.contain(self.frame,(w,h),Image.Resampling.BILINEAR)
        self.photo=ImageTk.PhotoImage(shown)
        self.canvas.itemconfigure(self.image_id,image=self.photo)
        self.canvas.coords(self.image_id,(w-shown.width)//2,(h-shown.height)//2)
        self.root.after(max(1, 50-int((time.monotonic()-tick_started)*1000)),self.tick)

    def snapshot(self):
        self.session.mkdir(parents=True,exist_ok=True)
        with self.state.lock:
            self.render().save(self.session/'console_snapshot.png')

    def close(self):
        self.finish_button.configure(state='disabled',text='지도 저장 중…')
        self.root.update_idletasks()
        try:self.snapshot()
        except Exception as e:self.state.error='화면 캡처 저장 실패: '+str(e)
        finally:
            self.stop.set();self.finish_video();self.root.destroy()

    def finish_video(self):
        if self.video is not None:
            self.video.release();self.video_log.close();self.video=None

    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.finish_video()

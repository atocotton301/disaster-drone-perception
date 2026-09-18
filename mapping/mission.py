"""Start → record → stop and save → review workflow, one session per area."""
import argparse
import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageOps
from core import load_config
from mission_result import finalize_session

BG='#07151c';FG='#edf3f6'
DEFAULT_AREA='한성대 낙상관'

def open_result_window(session):
    """A result has its own process/window, independent of the next mission."""
    return subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--result-only',str(Path(session).resolve())])

def notice(title, detail):
    root=tk.Tk();root.title(title);root.geometry('700x230');root.configure(bg=BG)
    tk.Label(root,text=title,font=('Sans',24,'bold'),bg=BG,fg=FG).pack(pady=(40,14))
    tk.Label(root,text=detail,font=('Sans',14),bg=BG,fg='#aacbd8',wraplength=640).pack()
    root.protocol('WM_DELETE_WINDOW',lambda:None);root.update()
    return root

def show_result(session):
    session=Path(session)
    summary=finalize_session(session)
    root=tk.Tk();root.title('탐색 종료 · 저장된 지도');root.geometry('1300x930');root.configure(bg=BG)
    title='지도 저장 완료' if summary['has_map'] else '기록 저장 완료 · 누적 지도 없음'
    if not summary.get('ended'):title='종료 상태 미확인 · 남아 있는 기록 열람'
    tk.Label(root,text=title,font=('Sans',22,'bold'),bg=BG,fg=FG).pack(pady=(12,3))
    tk.Label(root,text=str(session),font=('Sans',10),bg=BG,fg='#a5c0cd',wraplength=1200).pack()
    canvas=tk.Canvas(root,bg=BG,highlightthickness=0);canvas.pack(fill='both',expand=True,padx=12,pady=8)
    original=Image.open(session/'mission_map.png').convert('RGB');image_id=canvas.create_image(0,0,anchor='nw')
    def resize(event):
        shown=ImageOps.contain(original,(max(1,event.width),max(1,event.height)),Image.Resampling.BILINEAR)
        canvas.photo=ImageTk.PhotoImage(shown)
        canvas.itemconfigure(image_id,image=canvas.photo);canvas.coords(image_id,(event.width-shown.width)//2,(event.height-shown.height)//2)
    canvas.bind('<Configure>',resize)
    bar=tk.Frame(root,bg=BG);bar.pack(fill='x',padx=16,pady=(0,14))
    video=session/'console_live.mp4'
    def play_video():
        if os.name=='nt':os.startfile(str(video))
        else:subprocess.Popen(['xdg-open',str(video)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tk.Button(bar,text='녹화 영상 새 창' if video.exists() else '이번 탐색 녹화 없음',command=play_video,
        state='normal' if video.exists() else 'disabled',padx=12,pady=9).pack(side='left',padx=6)
    def open_folder():
        if os.name=='nt':os.startfile(str(session))
        else:subprocess.Popen(['xdg-open',str(session)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tk.Button(bar,text='저장 폴더 열기',command=open_folder,padx=15,pady=9).pack(side='right',padx=6)
    boundary=tk.BooleanVar(value=True)
    def toggle_view():
        nonlocal original
        boundary.set(not boundary.get())
        original=Image.open(session/('mission_map.png' if boundary.get() else 'mission_map_raw.png')).convert('RGB')
        switch.configure(text='원본 지도 보기' if boundary.get() else '장애물 경계 보기')
        from types import SimpleNamespace
        resize(SimpleNamespace(width=canvas.winfo_width(),height=canvas.winfo_height()))
    switch=tk.Button(bar,text='원본 지도 보기',command=toggle_view,padx=12,pady=9)
    switch.pack(side='right',padx=6)
    tk.Button(bar,text='지도 창 닫기',command=root.destroy,bg='#2175a8',fg='white',padx=18,pady=9).pack(side='right',padx=6)
    root.mainloop()

def choose_action(config):
    root=tk.Tk();root.title('실내 재난 탐색 · 시작 및 저장');root.geometry('1050x720');root.configure(bg=BG)
    action=[]
    tk.Label(root,text='실내 재난 탐색',font=('Sans',30,'bold'),bg=BG,fg=FG).pack(pady=(55,12))
    tk.Label(root,text='시작 → 탐지·누적 맵핑 → 종료·저장 → 구역 지도 확인',font=('Sans',17),bg=BG,fg='#96cadd').pack(pady=8)
    tk.Label(root,text='탐색 구역 이름',font=('Sans',15),bg=BG,fg=FG).pack(pady=(40,8))
    name=tk.StringVar(value=DEFAULT_AREA)
    entry=tk.Entry(root,textvariable=name,font=('Sans',21),justify='center',width=30);entry.pack(ipady=8)
    tk.Label(root,text='시작 전까지 카메라 탐지·지도 기록은 실행하지 않습니다.\n종료하면 탐색별 폴더에 지도와 탐지 기록을 저장합니다.',font=('Sans',14),bg=BG,fg='#a6bac5',justify='center').pack(pady=24)
    def start():action.append(('start',name.get().strip() or DEFAULT_AREA));root.destroy()
    def review():
        selected=filedialog.askdirectory(parent=root,initialdir=config['runs_dir'],title='저장된 탐색 날짜 폴더 선택')
        if not selected:return
        if not (Path(selected)/'manifest.json').is_file():
            messagebox.showerror('탐색 기록 없음','manifest.json이 들어 있는 탐색 날짜 폴더를 선택하세요.',parent=root);return
        open_result_window(selected)
    tk.Button(root,text='탐색 시작',command=start,bg='#13845e',fg='white',font=('Sans',22,'bold'),width=24,pady=15).pack(pady=10)
    tk.Button(root,text='저장된 지도 다시 열기',command=review,font=('Sans',15),width=32,pady=10).pack(pady=12)
    tk.Label(root,text='지도는 실제로 관측하고 위치 추정에 성공한 영역만 포함합니다.',bg=BG,fg='#c4b690',font=('Sans',12)).pack(pady=15)
    root.mainloop()
    return action[0] if action else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('live_slam_config.json'))
    parser.add_argument('--review',type=Path,help='Open a saved session before the start menu')
    parser.add_argument('--result-only',type=Path,help='Independent saved-map window')
    args=parser.parse_args()
    if args.result_only:
        show_result(args.result_only);return
    config=load_config(args.config)
    if args.review:open_result_window(args.review)
    while True:
        action=choose_action(config)
        if action is None:return
        if action[0]=='review':
            try:show_result(action[1])
            except Exception as e:
                root=tk.Tk();root.withdraw();messagebox.showerror('기록 열기 실패',str(e),parent=root);root.destroy()
        else:
            from run import main as run_session
            result=run_session(['--config',str(args.config.resolve()),'--source','ros','--area-name',action[1],'--show-result'])
            if result:
                root=tk.Tk();root.withdraw();messagebox.showerror('탐색 실행 또는 저장 실패','실행 로그와 탐색 폴더를 확인하세요. 기록은 삭제하지 않았습니다.',parent=root);root.destroy()

if __name__=='__main__':main()

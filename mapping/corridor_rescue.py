"""Illustrative engineering-building corridor, fallen person, visibility event and map.
Not a surveyed Hansung floor plan. Person event comes from rendered semantic visibility,
not YOLO. Camera pose and occupied cells are estimated by OpenCV RGB-D odometry.
"""
from pathlib import Path
import json, math, subprocess, time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
import virtual_test as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'simulation/corridor_rescue'
W, H = 480, 360
K = np.array([[410.,0,239.5],[0,410.,179.5],[0,0,1]],np.float32)
base.W, base.H, base.K = W, H, K
N, FPS = 225, 15
FONT = r'C:\Windows\Fonts\malgun.ttf'
BOLD = r'C:\Windows\Fonts\malgunbd.ttf'


def boxes():
    objects = [([-1.7,0,-.1,1.7,18,0],[194,196,192],'floor'),
        ([-1.7,0,2.9,1.7,18,3.0],[229,230,226],'ceiling'),
        ([-1.7,0,0,-1.5,18,2.9],[225,222,212],'wall'),
        ([1.5,0,0,1.7,18,2.9],[225,222,212],'wall'),
        ([-1.5,17.9,0,1.5,18,2.9],[218,220,212],'wall')]
    for y in (2.,5.,9.5,13.5):
        for side in (-1,1):
            x = side*1.49
            objects += [([x-.02,y,0,x+.02,y+1.,2.15],[114,143,146],'door'),
                ([x-.03,y-.06,0,x+.03,y,2.22],[79,88,86],'frame'),
                ([x-.03,y+1.,0,x+.03,y+1.06,2.22],[79,88,86],'frame'),
                ([x-.03,y,2.15,x+.03,y+1,2.22],[79,88,86],'frame')]
    return objects


def people():
    # Rounded, non-graphic prone figure: head, torso, limbs and shoes.
    return [([.15,7.95,.24],[.16,.19,.16],[196,155,121]),
        ([.15,7.47,.22],[.29,.40,.17],[47,86,129]),
        ([.15,7.03,.18],[.25,.18,.14],[47,63,77]),
        ([-.01,6.62,.13],[.105,.43,.11],[50,61,71]),
        ([.32,6.57,.13],[.105,.43,.11],[50,61,71]),
        ([-.01,6.16,.11],[.12,.15,.1],[41,43,43]),
        ([.32,6.10,.11],[.12,.15,.1],[41,43,43]),
        ([-.22,7.43,.12],[.085,.35,.075],[47,86,129]),
        ([.50,7.35,.12],[.085,.35,.075],[47,86,129]),
        ([-.22,7.06,.10],[.075,.105,.065],[196,155,121]),
        ([.50,6.99,.10],[.075,.105,.065],[196,155,121])]


def rotation(yaw=0., pitch=.16):
    right=np.array([math.cos(yaw),-math.sin(yaw),0.])
    level=np.array([math.sin(yaw),math.cos(yaw),0.])
    forward=level*math.cos(pitch)+np.array([0.,0.,-math.sin(pitch)])
    down=np.cross(forward,right)
    return np.column_stack((right,down,forward))


v,u=np.mgrid[:H,:W]
RAYS=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones((H,W))),-1)


def render(pos,R):
    direction=RAYS@R.T
    inv=np.divide(1.,direction,out=np.full_like(direction,1e12),where=np.abs(direction)>1e-12)
    best=np.full((H,W),np.inf)
    labels=np.full((H,W),-1,np.int16)
    objects=boxes()
    for i,(b,_,_) in enumerate(objects):
        a,c=(np.array(b[:3])-pos)*inv,(np.array(b[3:])-pos)*inv
        near=np.minimum(a,c).max(-1); far=np.maximum(a,c).min(-1)
        t=np.where(near>.01,near,far)
        ok=(far>=np.maximum(near,0))&(t>.01)&(t<best)
        best[ok],labels[ok]=t[ok],i
    bodies=people()
    for i,(center,radius,_) in enumerate(bodies):
        origin=(pos-np.array(center))/radius
        ray=direction/np.array(radius)
        aa=(ray*ray).sum(-1); bb=2*(ray*origin).sum(-1); cc=(origin*origin).sum()-1
        discriminant=bb*bb-4*aa*cc
        t=(-bb-np.sqrt(np.maximum(discriminant,0)))/(2*aa)
        ok=(discriminant>=0)&(t>.01)&(t<best)
        best[ok],labels[ok]=t[ok],100+i
    depth=np.where(np.isfinite(best),best,0).astype(np.float32)
    hit=pos+direction*depth[...,None]
    rgb=np.full((H,W,3),210,np.uint8)
    for i,(_,color,material) in enumerate(objects):
        mask=labels==i
        x,y,z=hit[...,0],hit[...,1],hit[...,2]
        if material=='floor':
            gx=np.mod(x+.3,.6); gy=np.mod(y,.6)
            seam=(gx<.012)|(gy<.012)
            tex=.95+.025*np.sin(x*71+y*67)
            tex[seam]=.70
        elif material=='wall':
            tex=.985+.012*np.sin(y*55+z*43)
            tex=np.where(z<.12,.55,tex)
            tex=np.where((z>.85)&(z<.91),.72,tex)
        elif material=='door':
            tex=.93+.035*np.sin(z*30)+.02*np.sin(y*40)
            # Frosted window inset and horizontal handle.
            tex=np.where((z>1.15)&(z<1.85),1.25,tex)
        else:
            tex=np.ones((H,W))*.96
        rgb[mask]=np.clip(np.array(color)*tex[mask,None],0,255).astype(np.uint8)
    for i,(center,radius,color) in enumerate(bodies):
        mask=labels==100+i
        normal=(hit-np.array(center))/(np.array(radius)**2)
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
        light=.70+.30*np.clip(normal@np.array([-.3,-.4,.86]),0,1)
        rgb[mask]=np.clip(np.array(color)*light[mask,None],0,255).astype(np.uint8)
    depth[(depth<.2)|(depth>6)]=0
    return rgb,depth,labels>=100


def point(x,y):
    # Blueprint-like horizontal main corridor and right-side wing.
    return 804+y*23., 267+(x+1.5)*28.


def board(rgb,depth,mask,mapper,event,index,status):
    im=Image.new('RGB',(1280,800),'#202427'); d=ImageDraw.Draw(im)
    f=lambda size,b=False: ImageFont.truetype(BOLD if b else FONT,size)
    d.text((24,18),'공학관 복도 탐색',font=f(27,True),fill='#eeeeeb')
    d.text((24,57),'제공 도면에서 착안한 가상 공간 · 실제 한성대 공학관의 실측 지도가 아닙니다',font=f(16),fill='#bac1c3')
    d.rounded_rectangle((988,16,1254,80),radius=6,fill='#30383b')
    d.text((1004,26),'사람 발견  1명' if event else '탐색 중',font=f(24,True),fill='#ffb286' if event else '#dce4e5')
    d.text((24,100),'카메라 시야',font=f(18),fill='#d5dcdd')
    visual=Image.fromarray(rgb).resize((744,558),Image.Resampling.BILINEAR)
    vd=ImageDraw.Draw(visual)
    if event and mask.any():
        yy,xx=np.where(mask)
        box=(max(0,int(xx.min()*744/W)-8),max(0,int(yy.min()*558/H)-8),
             min(743,int(xx.max()*744/W)+8),min(557,int(yy.max()*558/H)+8))
        vd.rectangle(box,outline='#ff994f',width=3)
        label='쓰러진 사람 · 가상 발견'
        label_y=max(0,box[1]-30)
        vd.rectangle((box[0],label_y,min(743,box[0]+244),label_y+28),fill='#22292b')
        vd.text((box[0]+6,label_y+3),label,font=f(16,True),fill='#ffb98c')
    im.paste(visual,(24,130))
    d.text((804,100),'지도 · 발견 위치',font=f(18),fill='#d5dcdd')
    d.rectangle((794,130,1254,594),fill='#e7e8e3')
    # Schematic rooms inspired by the supplied horizontal corridor layout.
    for n in range(4):
        left=810+n*92
        d.rectangle((left,173,left+85,257),fill='#d1d5d1',outline='#8b9691',width=2)
        d.text((left+8,205),f'강의실 {n+1}',font=f(14),fill='#4c5954')
        d.rectangle((left,361,left+85,426),fill='#d1d5d1',outline='#8b9691',width=2)
    d.rectangle((804,267,1218,351),fill='#fafaf5',outline='#8b9691',width=2)
    d.rectangle((1183,267,1218,555),fill='#fafaf5',outline='#8b9691',width=2)
    d.text((1055,508),'우측 연결 복도',font=f(14),fill='#64716b')
    d.text((816,448),'회색 선: 가상 배치도',font=f(14),fill='#68746f')
    d.text((816,474),'주황 점: RGB-D 관측 장애물',font=f(14),fill='#a2642f')
    rr,cc=np.where(mapper.grid==100)
    for row,col in zip(rr,cc):
        wx=-5+(col+.5)*.05; wy=-2+(row+.5)*.05+.8
        px,py=point(wx,wy)
        if 795<px<1253 and 131<py<593:
            d.point((px,py),fill='#d28645')
    pts=[point(p[0],p[1]+.8) for p in mapper.path]
    if len(pts)>1: d.line(pts,fill='#347aa3',width=3)
    if pts:
        x,y=pts[-1]; d.ellipse((x-5,y-5,x+5,y+5),fill='#328a85')
    if event:
        x,y=point(event['map_xyz'][0],event['map_xyz'][1]+.8)
        d.ellipse((x-9,y-9,x+9,y+9),fill='#bf4f35',outline='white',width=2)
        d.line((x+9,y,x+35,y-35),fill='#bf4f35',width=2)
        d.rounded_rectangle((x+20,y-69,x+170,y-36),radius=4,fill='#bf4f35')
        d.text((x+29,y-64),'사람 1명',font=f(17,True),fill='white')
        d.text((805,618),f"발견 시각  {event['time_s']:.1f}초",font=f(18,True),fill='#ffb58a')
        d.text((805,650),'발견 위치를 지도에 기록했습니다',font=f(16),fill='#d9dedd')
    else:
        d.text((805,618),'복도를 따라 내부를 확인하고 있습니다',font=f(16),fill='#d9dedd')
    d.text((24,708),f'가상 카메라 이동  {index/FPS:04.1f}초  ·  '+('위치추정 정상' if status!='LOST' else '위치추정 중단'),font=f(17),fill='#b8c5c6')
    d.text((24,744),'인물 판정은 가상 객체의 가시성·깊이에 따른 시나리오 이벤트입니다. 실제 YOLO 탐지 결과가 아닙니다.',font=f(15),fill='#9caeb1')
    d.text((805,744),'자동비행·자동복귀 기능 없음',font=f(14),fill='#9caeb1')
    return np.array(im)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mapper=base.EstimatedMapper()
    mapper.pose[:3,:3]=rotation()
    mapper.pose[2,3]=1.2
    writer=cv2.VideoWriter(str(OUT/'render.avi'),cv2.VideoWriter_fourcc(*'MJPG'),FPS,(1280,800))
    if not writer.isOpened(): raise RuntimeError('Encoder unavailable')
    event=None; rows=[]
    try:
        for i in range(N):
            t=i/(N-1)
            pos=np.array([.10*math.sin(t*math.pi),.8+4.55*t,1.2])
            R=rotation(.018*math.sin(t*math.pi*2))
            rgb,depth,mask=render(pos,R)
            status=mapper.update(rgb,depth)
            valid=mask&(depth>.2)
            if event is None and status=='TRACKING' and valid.sum()>100 and np.median(depth[valid])<3.8:
                yy,xx=np.where(valid); k=len(xx)//2; z=float(depth[yy[k],xx[k]])
                optical=np.array([(xx[k]-K[0,2])*z/K[0,0],(yy[k]-K[1,2])*z/K[1,1],z])
                world=mapper.pose[:3,:3]@optical+mapper.pose[:3,3]
                event=dict(person_id='sim_person_01',count=1,time_s=i/FPS,frame=i,
                    optical_depth_m=z,map_xyz=world.tolist(),
                    method='SIMULATED_SEMANTIC_VISIBILITY_NOT_YOLO')
                print('DISCOVERY',json.dumps(event),flush=True)
            frame=board(rgb,depth,mask,mapper,event,i,status)
            writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
            if i in (0,N//2,N-1): Image.fromarray(frame).save(OUT/f'preview_{i:03d}.png')
            rows.append(dict(frame=i,status=status,estimated_xyz=mapper.pose[:3,3].tolist()))
            if i%45==0: print(f'{i}/{N} {status}',flush=True)
    finally: writer.release()
    if event is None: raise RuntimeError('No visibility discovery was triggered')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i',str(OUT/'render.avi'),
        '-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',
        str(OUT/'corridor_rescue.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    (OUT/'render.avi').unlink()
    (OUT/'scenario_report.json').write_text(json.dumps(dict(
        scenario='Illustrative engineering-building corridor inspired by supplied plan; not surveyed Hansung geometry',
        person_count=1,event=event,actual_yolo=False,virtual_rgbd=True,
        pose_backend='OpenCV RgbdICPOdometry',loop_closure=False,
        frames=N,fps=FPS,lost_frames=sum(r['status']=='LOST' for r in rows),trajectory=rows),indent=2),encoding='utf-8')
    print('COMPLETE',OUT/'corridor_rescue.mp4')


if __name__=='__main__': main()

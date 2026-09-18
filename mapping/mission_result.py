"""Durable per-mission exports from measured map and recorded detections only."""
import json
import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from core import grid_pixel

NAMES={'person':'사람','fire':'화염','smoke':'연기','door':'문','staircase':'계단'}
COLORS={'person':'#e63e64','fire':'#ff792c','smoke':'#835fc3','door':'#0a9793','staircase':'#917329'}

def records(path):
    if not path.exists():return []
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:out.append(json.loads(line))
        except json.JSONDecodeError:continue
    return out

def summarize_observations(events):
    """Spatial bins are observation locations, never unique-person counts."""
    bins={}
    for event in events:
        for marker in event.get('map_markers',[]):
            xyz=marker.get('xyz',[])
            if len(xyz)!=3 or not np.isfinite(xyz).all():continue
            label=marker['label']
            key=(label,round(xyz[0]/.35),round(xyz[1]/.35))
            if key not in bins:
                bins[key]=dict(label=label,xyz=list(xyz),first_stamp=event['stamp'],
                    last_stamp=event['stamp'],observations=0)
            row=bins[key];row['last_stamp']=event['stamp'];row['observations']+=1
            row['xyz']=list(xyz)
    return list(bins.values())

def representative_snapshot(events):
    """One timestamp, supported by two preceding frames; no identity/count claims.

    Matching is one-to-one within each comparison frame. It only rejects
    transient projected positions; it is not a persistent person tracker.
    """
    frames=sorted(events,key=lambda e:e['stamp'])
    best=dict(stamp=None,markers=[])
    best_rank=(0,0)
    for i,event in enumerate(frames):
        if i<2:continue
        neighbors=frames[i-2:i]
        if event['stamp']-neighbors[0]['stamp']>1.0:continue
        current=[m for m in event.get('map_markers',[]) if len(m.get('xyz',[]))==3 and np.isfinite(m['xyz']).all()]
        supported=set(range(len(current)))
        for prior in neighbors:
            candidates=[]
            for a,m in enumerate(current):
                for b,n in enumerate(prior.get('map_markers',[])):
                    if m['label']!=n['label'] or len(n.get('xyz',[]))!=3:continue
                    distance=np.linalg.norm(np.array(m['xyz'][:2])-n['xyz'][:2])
                    if distance<=.5:candidates.append((distance,a,b))
            used_a=set();used_b=set()
            for _,a,b in sorted(candidates):
                if a not in used_a and b not in used_b:
                    used_a.add(a);used_b.add(b)
            supported &= used_a
        markers=[m for j,m in enumerate(current) if j in supported]
        rank=(sum(m['label']=='person' for m in markers),len(markers))
        if markers and rank>=best_rank:
            best=dict(stamp=event['stamp'],markers=markers);best_rank=rank
    return best

def boundary_grid(grid):
    """Display the main connected free region's occupied boundary.

    Disconnected free pockets and occupied interiors are omitted in this view
    only. This is not wall reconstruction or an update to measured occupancy.
    """
    import cv2
    free=(grid>=0)&(grid<50)
    count,components,stats,_=cv2.connectedComponentsWithStats(free.astype(np.uint8),connectivity=4)
    if count>1:
        free=components==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA])))
    padded=np.pad(free,1,constant_values=False)
    adjacent=(padded[:-2,1:-1]|padded[2:,1:-1]|padded[1:-1,:-2]|padded[1:-1,2:])
    shown=np.full_like(grid,-1);shown[free]=grid[free]
    shown[(grid>=50)&adjacent]=grid[(grid>=50)&adjacent]
    return shown

def build_summary(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    shutdown=json.loads((root/'shutdown.json').read_text(encoding='utf-8')) if (root/'shutdown.json').exists() else {}
    events=records(root/'detections.jsonl');health=records(root/'tracking.jsonl')
    raw_events=records(root/'detections_raw.jsonl') if (root/'detections_raw.jsonl').exists() else events
    observations=summarize_observations(events)
    poses=records(root/'poses.jsonl')
    result=dict(schema_version=2,session=root.name,area_name=manifest.get('area_name','실내 탐색'),
        started=manifest.get('start_time'),ended=shutdown.get('finished'),
        source=manifest.get('source'),replay=manifest.get('replay'),
        has_map=(root/'map.npz').exists(),forced_termination=shutdown.get('forced_termination',[]),
        detection_frames={k:sum(any(d['label']==k for d in e.get('detections',[])) for e in raw_events) for k in NAMES},
        observation_locations=observations,pose_samples=len(poses),
        last_tracking=health[-1]['status'] if health else 'UNKNOWN',
        marker_note='관측 당시 추정 위치. 현재 사람 위치나 고유 인원 수가 아니며, 이후 루프 폐합 보정과 차이가 있을 수 있음.')
    result['representative_snapshot']=representative_snapshot(events)
    result['raw_max_simultaneous']={k:max((sum(d['label']==k for d in e.get('detections',[])) for e in raw_events),default=0) for k in NAMES}
    result['snapshot_note']='동일 시점의 대표 관측 장면. 이전 두 프레임의 0.5m 이내 위치 지지 필요. 전체 고유 인원 수가 아님.'
    result['boundary_note']='가장 큰 연결된 빈 영역과 인접 장애물 경계만 표시. 분리된 영역과 장애물 내부는 회색으로 생략. 원본 map.npz 보존; 벽 검출/지도 보정 아님.'
    if result['has_map']:
        with np.load(root/'map.npz',allow_pickle=False) as data:
            grid=data['occupancy'];meta=json.loads(str(data['metadata']))
            if grid.ndim!=2 or grid.shape!=(meta['height'],meta['width']):raise ValueError('Invalid saved map dimensions')
            result['map']=dict(meta,known_cells=int((grid>=0).sum()),obstacle_cells=int((grid>=50).sum()))
    mapped_stamps={e['stamp'] for e in events if e.get('map_markers')}
    result['detected_frames_without_map_position']=sum(bool(e.get('detections')) and e['stamp'] not in mapped_stamps for e in raw_events)
    return result

def draw_result(root,summary,boundary=True):
    from console_ui import font
    root=Path(root);image=Image.new('RGB',(1500,1050),'#f3f6fa');d=ImageDraw.Draw(image)
    d.rectangle((0,0,1500,118),fill='#16394e')
    d.text((36,20),summary['area_name']+' · 저장된 탐색 지도',font=font(32,True),fill='white')
    d.text((36,72),f"{summary.get('started','')} → {summary.get('ended') or '종료 시각 미확인'}",font=font(17),fill='#d6e4ec')
    if summary['has_map']:
        with np.load(root/'map.npz',allow_pickle=False) as z:
            grid=z['occupancy'];meta=json.loads(str(z['metadata']))
        if boundary:grid=boundary_grid(grid)
        rgb=np.full((*grid.shape,3),(201,209,215),np.uint8)
        rgb[(grid>=0)&(grid<50)]=(249,252,255);rgb[grid>=50]=(34,89,132)
        raw=Image.fromarray(np.flipud(rgb));scale=min(1060/raw.width,735/raw.height)
        shown=raw.resize((max(1,round(raw.width*scale)),max(1,round(raw.height*scale))),Image.Resampling.NEAREST)
        left=24+(1080-shown.width)//2;top=140+(750-shown.height)//2
        image.paste(shown,(left,top));d=ImageDraw.Draw(image)
        def pixel(xyz):
            x,y=grid_pixel(xyz[0],xyz[1],meta);return left+x*scale,top+y*scale
        path_file=root/'optimized_path.json'
        path=json.loads(path_file.read_text())['keyframe_positions'] if path_file.exists() else []
        if len(path)<2:path=[p['xyz_yaw'][:3] for p in records(root/'poses.jsonl')]
        if len(path)>1:d.line([pixel(p) for p in path],fill='#168dec',width=3)
        if path:
            x,y=pixel(path[-1]);d.ellipse((x-7,y-7,x+7,y+7),fill='#13b897',outline='white',width=2)
        labels=[]
        for row in summary['representative_snapshot']['markers']:
            x,y=pixel(row['xyz'])
            if not (left<=x<=left+shown.width and top<=y<=top+shown.height):continue
            color=COLORS.get(row['label'],'#222222')
            d.ellipse((x-7,y-7,x+7,y+7),fill=color,outline='white',width=2)
            if not any(abs(x-lx)<85 and abs(y-ly)<32 for lx,ly in labels):
                caption=NAMES.get(row['label'],row['label']);face=font(17,True)
                tw=d.textlength(caption,font=face)+10
                tx=max(24,min(x+10,1100-tw));ty=max(130,min(y-24,870))
                d.rectangle((tx,ty,tx+tw,ty+27),fill='#16394e')
                d.text((tx+5,ty),caption,font=face,fill='white');labels.append((x,y))
        meters=max(1,int(180/max(scale/meta['resolution'],1)))
        length=meters/meta['resolution']*scale
        if length<=300:
            d.line((46,920,46+length,920),fill='#16394e',width=4)
            d.text((46,932),f'{meters} m · 격자 {meta["resolution"]*100:.0f} cm',font=font(18),fill='#16394e')
    else:
        d.text((130,380),'저장된 누적 지도가 없습니다',font=font(32,True),fill='#a63e34')
        d.text((130,438),'위치 추정이 성공한 구간이 있는지 기록을 확인하세요.',font=font(21),fill='#394957')
    d.text((1130,144),'탐색 기록',font=font(25,True),fill='#16394e')
    y=196
    for label,color in [('관측된 빈 공간','#f9fcff'),('미확인·내부 생략' if boundary else '미확인 영역','#c9d1d7'),('벽·장애물 경계' if boundary else '벽·장애물 원본','#225984'),('카메라 경로','#168dec')]:
        d.rectangle((1134,y+6,1158,y+24),fill=color,outline='#879cab')
        d.text((1171,y),label,font=font(20),fill='#263b49');y+=42
    y+=22;d.text((1130,y),'대표 관측 장면',font=font(23,True),fill='#16394e');y+=42
    for k,name in NAMES.items():
        count=sum(o['label']==k for o in summary['representative_snapshot']['markers'])
        d.ellipse((1134,y+6,1153,y+25),fill=COLORS[k])
        d.text((1166,y),f'{name}  {count}'+('명' if k=='person' else '개'),font=font(20),fill='#263b49');y+=40
    stamp=summary['representative_snapshot']['stamp']
    when=datetime.datetime.fromtimestamp(stamp).strftime('%H:%M:%S') if stamp is not None else '지속 관측 없음'
    d.text((1130,y+18),f'선택 장면: {when}',font=font(17),fill='#263b49')
    d.text((1130,y+49),'3프레임 위치 일치 기준',font=font(16),fill='#775c37')
    d.text((1130,y+79),'전체 인원 수·현재 위치 아님',font=font(16),fill='#775c37')
    d.text((1130,y+111),f"위치 미확정: {summary['detected_frames_without_map_position']}프레임",font=font(16),fill='#775c37')
    d.text((32,984),'주 탐색 영역 경계 · 분리된 영역과 장애물 내부는 회색으로 생략합니다. 전체는 원본 보기에서 확인하세요.' if boundary else '원본 누적 점유 지도 · 표시되지 않은 구역은 안전이 확인된 구역이 아닙니다.',font=font(18),fill='#4d6270')
    if summary['forced_termination']:
        d.text((32,1015),'일부 처리가 강제 종료됨: 기록 검토 필요',font=font(18,True),fill='#b04430')
    else:
        d.text((32,1015),'탐지 기록의 위치는 이후 지도 보정과 차이가 있을 수 있습니다.',font=font(16),fill='#667d8b')
    return image

def finalize_session(root):
    root=Path(root);summary=build_summary(root)
    png=root/'mission_map.tmp.png';draw_result(root,summary).save(png);png.replace(root/'mission_map.png')
    png=root/'mission_map_raw.tmp.png';draw_result(root,summary,boundary=False).save(png);png.replace(root/'mission_map_raw.png')
    temp=root/'mission_summary.tmp.json';temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(root/'mission_summary.json')
    return summary

import ast, io, json, threading, time, unittest
from pathlib import Path
from types import SimpleNamespace as NS
import cv2
import numpy as np

source=Path(__file__).resolve().parents[1]/'ros2/viewer_backend.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Viewer');cls.bases=[]
transform=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='xyz_transform')
class MissingTransform(Exception):pass
ns=dict(json=json,time=time,np=np,cv2=cv2,Time=lambda **kw:kw['nanoseconds'],TransformException=MissingTransform)
exec(compile(ast.Module(body=[transform,cls],type_ignores=[]),str(source),'exec'),ns)

class DetectionRetry(unittest.TestCase):
    def setUp(self):
        self.node=object.__new__(ns['Viewer'])
        self.node.s=NS(lock=threading.RLock(),tracking='TRACKING',times={'tracking':time.monotonic()},rows=[],markers=[])
        self.node.c={'stale_s':1.5};self.node.events=io.StringIO();self.node.raw_events=io.StringIO();self.node.pending_detections=[]
        self.node.camera_info=NS(header=NS(frame_id='optical'),k=[100,0,50,0,100,50,0,0,1],d=[0]*5,distortion_model='plumb_bob')
        self.calls=[];self.available=False
        def lookup(target,source,stamp):
            self.calls.append((target,source,stamp))
            if not self.available:raise MissingTransform()
            return NS(transform=NS(rotation=NS(x=0,y=0,z=0,w=1),translation=NS(x=1,y=2,z=0)))
        self.node.tf=NS(lookup_transform=lookup)
        self.msg=NS(data=json.dumps(dict(stamp=10.,frame_id='optical',detections=[dict(label='person',bbox=[40,40,60,60],depth_m=2.)])))

    def test_late_transform_resolves_original_image_timestamp(self):
        self.node.detections(self.msg)
        self.assertEqual(self.node.s.markers,[])
        self.assertEqual(len(self.node.pending_detections),1)
        self.available=True;self.node.resolve_detections()
        self.assertEqual(self.node.s.markers[0]['xyz'],[1.,2.,2.])
        self.assertTrue(all(c==('map','optical',10_000_000_000) for c in self.calls))
        self.assertFalse(self.node.pending_detections)

    def test_lost_tracking_never_creates_marker_even_with_transform(self):
        self.available=True;self.node.s.tracking='LOST'
        self.node.detections(self.msg)
        self.assertFalse(self.node.s.markers)
        self.assertFalse(self.calls)
        self.assertIn('person',self.node.raw_events.getvalue())

import math
import unittest
import numpy as np
from PIL import Image, ImageDraw
from console_ui import font, text_tile, point_cloud_preview


class ConsolePixels(unittest.TestCase):
    def test_cached_korean_label_keeps_position_and_pixels(self):
        expected=Image.new('RGB',(600,100),'#07151c')
        actual=expected.copy()
        ImageDraw.Draw(expected).text((10,10),'카메라 영상 (RGB)',font=font(19,True),fill='#edf3f6')
        tile,dx,dy=text_tile('카메라 영상 (RGB)',19,'#edf3f6',True)
        actual.paste(tile,(10+dx,10+dy),tile)
        # Alpha composition may differ by one gray level depending on Pillow build.
        self.assertLessEqual(np.abs(np.array(actual,dtype=int)-np.array(expected,dtype=int)).max(),1)

    def test_vectorized_cloud_matches_reference_depth_render(self):
        depth=np.random.default_rng(8).uniform(.3,5,(50,60)).astype(np.float32)
        K=np.array([[70,0,30],[0,70,25],[0,0,1.]])
        expected=Image.new('RGB',(560,320),'#061118');draw=ImageDraw.Draw(expected)
        y,x=np.mgrid[0:50:5,0:60:5];z=depth[::5,::5].ravel()
        x,y=x.ravel(),y.ravel()
        xx=(x-30)*z/70;yy=-(y-25)*z/70;a=-.28
        rotated_x=xx*math.cos(a)+z*math.sin(a)
        rotated_z=-xx*math.sin(a)+z*math.cos(a)+1.8
        u,v=280+rotated_x*270/rotated_z,160-yy*270/rotated_z
        for i in np.argsort(rotated_z)[::-1]:
            if 0<=u[i]<560 and 0<=v[i]<320:
                c=min(1.,z[i]/4)
                draw.point((int(u[i]),int(v[i])),fill=(int(40+180*c),int(245-100*c),int(250*(1-c))))
        np.testing.assert_array_equal(point_cloud_preview(depth,K),np.array(expected))

import json,tempfile,unittest
from pathlib import Path
import numpy as np
from mission_result import finalize_session,summarize_observations,representative_snapshot,boundary_grid

class MissionExports(unittest.TestCase):
    def test_moving_people_are_one_snapshot_not_historical_locations(self):
        frames=[]
        for i in range(8):
            markers=[dict(label='person',xyz=[i*.15,0,0]),dict(label='person',xyz=[i*.15,2,0])]
            if i==6:markers.append(dict(label='person',xyz=[9,9,0]))
            frames.append(dict(stamp=10+i*.2,map_markers=markers))
        self.assertGreater(len(summarize_observations(frames)),2)
        snapshot=representative_snapshot(frames)
        self.assertEqual(len(snapshot['markers']),2)
        self.assertEqual(snapshot['stamp'],frames[-1]['stamp'])
        self.assertEqual(snapshot['markers'],frames[-1]['map_markers'])

    def test_one_previous_marker_cannot_support_two_people(self):
        one=dict(label='person',xyz=[1,1,0])
        extra=dict(label='person',xyz=[1.1,1,0])
        frames=[dict(stamp=1+i*.1,map_markers=[one]) for i in range(2)]
        frames.append(dict(stamp=1.2,map_markers=[one,extra]))
        self.assertEqual(len(representative_snapshot(frames)['markers']),1)
        frames[-1]['stamp']=9
        self.assertEqual(representative_snapshot(frames)['markers'],[])

    def test_boundary_hides_interior_without_inventing_free_space(self):
        grid=np.full((8,8),-1,np.int8)
        grid[1:7,1:3]=0;grid[1:7,3:7]=100
        before=grid.copy();shown=boundary_grid(grid)
        np.testing.assert_array_equal(grid,before)
        np.testing.assert_array_equal((shown>=0)&(shown<50),(grid>=0)&(grid<50))
        self.assertTrue((shown[1:7,3]==100).all())
        self.assertTrue((shown[1:7,4:7]==-1).all())

    def test_boundary_omits_disconnected_free_pocket_behind_wall(self):
        grid=np.full((12,12),100,np.int8)
        grid[1:10,1:4]=0;grid[5:7,8:10]=0
        shown=boundary_grid(grid)
        self.assertTrue((shown[5:7,8:10]==-1).all())
        self.assertTrue((shown[1:10,1:4]==0).all())
        self.assertFalse(((shown==0)&(grid!=0)).any())
    def session(self,folder):
        p=Path(folder)
        (p/'manifest.json').write_text(json.dumps(dict(area_name='실내 테스트',start_time='2026-09-16T10:00:00',source='ros')),encoding='utf-8')
        (p/'shutdown.json').write_text(json.dumps(dict(finished='2026-09-16T10:01:00',forced_termination=[])))
        return p
    def test_no_map_does_not_create_fabricated_occupancy(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.session(folder);result=finalize_session(p)
            self.assertFalse(result['has_map']);self.assertFalse((p/'map.npz').exists())
            self.assertTrue((p/'mission_map.png').exists())
    def test_export_preserves_map_and_groups_observations_not_people(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.session(folder)
            grid=np.full((20,20),-1,np.int8);grid[4:12,4:12]=0;grid[4,4:12]=100
            meta=dict(width=20,height=20,resolution=.05,origin=[0,0,0],stamp=1,frame_id='map')
            np.savez(p/'map.npz',occupancy=grid,metadata=json.dumps(meta))
            before=(p/'map.npz').read_bytes()
            events=[dict(stamp=s,detections=[dict(label='person')],map_markers=[dict(label='person',xyz=[.3,.3,0])]) for s in (1.,2.)]
            (p/'detections.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n{"incomplete":')
            result=finalize_session(p)
            self.assertTrue(result['has_map']);self.assertEqual(result['detection_frames']['person'],2)
            self.assertEqual(len(result['observation_locations']),1)
            self.assertEqual(result['observation_locations'][0]['observations'],2)
            self.assertEqual((p/'map.npz').read_bytes(),before)
            self.assertEqual(json.loads((p/'mission_summary.json').read_text(encoding='utf-8'))['session'],p.name)
            self.assertFalse((p/'mission_map.tmp.png').exists())

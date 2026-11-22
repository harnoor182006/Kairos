import json,sys,unittest,tempfile,subprocess
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from assemble import assemble_prediction,sanitize_highlights
from schema import validate_output
from fpv_select import importance_curve,select_highlights
import numpy as np
def run_pipeline(video, output, timeout):
    env = __import__("os").environ.copy()
    env["PYTHONPATH"] = str(ROOT / "code")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    subprocess.run(
        [sys.executable, str(ROOT / "code" / "run_video.py"),
         str(video), str(output)],
        cwd=ROOT,
        env=env,
        check=False,
        timeout=timeout
    )
class ProjectTests(unittest.TestCase):
 def test_assembly_valid(self):
  p=assemble_prediction("TEST","sample",30.0,raw_highlights=[{"gsegment":[0,8],"score":0.8},{"segment":[10,18],"score":0.5}])
  ok,err,warn=validate_output(p,expected_video_id="sample")
  self.assertTrue(ok,err); self.assertTrue(p["highlights"])
 def test_selector_budget_and_minimum(self):
  curve=np.linspace(0,1,90,dtype=np.float32)
  h=select_highlights(curve,90.0)
  self.assertTrue(h); self.assertTrue(all(e-s>=4.05 for s,e in [x["segment"] for x in h]))
  self.assertLessEqual(sum(e-s for s,e in [x["segment"] for x in h]),90*.15+0.1)
 def test_corrupt_video_fallback_writes_json(self):
  with tempfile.TemporaryDirectory() as d:
   v=Path(d)/"broken.mp4"; v.write_text("not video")
   out=Path(d)/"out.json"
   run_pipeline(v, out, 60)
   data=json.loads(out.read_text()); ok,err,_=validate_output(data,expected_video_id="broken")
   self.assertTrue(ok,err)
 def test_manifest_missing_video_placeholder(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); videos=d/"videos"; videos.mkdir(); outputs=d/"outputs"
   manifest=d/"manifest.csv"; manifest.write_text("video_id,duration_sec,dataset\nmissing_id,60,fpvsum\n",encoding="utf-8")
   subprocess.run([sys.executable,str(ROOT/"batch.py"),str(videos),str(outputs),"--manifest",str(manifest),"--fill-missing"],check=True,timeout=60)
   data=json.loads((outputs/"missing_id.json").read_text()); ok,err,_=validate_output(data,expected_video_id="missing_id",manifest_duration=60)
   self.assertTrue(ok,err)
 def test_synthetic_video_pipeline(self):
  with tempfile.TemporaryDirectory() as d:
   v=Path(d)/"synthetic.mp4"; out=Path(d)/"out.json"
   subprocess.run(["ffmpeg","-y","-v","error","-f","lavfi","-i","testsrc2=size=320x240:rate=15:duration=30","-f","lavfi","-i","sine=frequency=500:duration=30","-shortest","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(v)],check=True,timeout=60)
   run_pipeline(v, out, 120)
   data=json.loads(out.read_text()); ok,err,_=validate_output(data,expected_video_id="synthetic")
   self.assertTrue(ok,err)
 def test_cooking_route_builds_missing_epic_cache(self):
  import pipeline
  import fpv_pipeline
  class CookingRouter:
   def route_and_label(self,frames,audio):
    return "cooking",0.9,{"cooking":0.9},np.zeros((0,64),dtype=np.float32)
  with tempfile.TemporaryDirectory() as d:
   video=Path(d)/"kitchen.mp4"
   video.touch()
   with patch.object(pipeline,"probe_duration",return_value=30.0), \
        patch.object(pipeline,"extract_motion_curve",return_value=[0.1]*30), \
        patch.object(pipeline,"decode_video_frames",return_value=np.zeros((0,224,224,3),dtype=np.uint8)), \
        patch.object(pipeline,"extract_audio_1hz_features",return_value=(np.zeros(30),np.zeros(30))), \
        patch.object(pipeline,"SigLIPRouter",return_value=CookingRouter()), \
        patch.object(pipeline,"_load_epic_cache",side_effect=[None,{"duration_sec":np.array([30.0])}]), \
        patch.object(pipeline,"_create_epic_cache") as create_cache, \
        patch("epic_actions.decode_epic_actions",return_value=[]) as decode, \
        patch("epic_select.select_epic_highlights",return_value=[{"segment":[0,6],"score":0.5}]), \
        patch.object(fpv_pipeline,"run_fpv_branch") as fpv:
    result=pipeline.process_video(video)
   self.assertEqual(result["label"]["name"],"cooking")
   self.assertEqual(result["actions"],[])
   create_cache.assert_called_once_with(video,"kitchen")
   decode.assert_called_once()
   fpv.assert_not_called()
   self.assertTrue(validate_output(result,expected_video_id="kitchen")[0])
 def test_fpv_uses_importance_windows_when_kts_shots_exceed_budget(self):
  from fpv_pipeline import run_fpv_branch
  from videomae_fpv import FPV_CLASSES
  class Router:
   def category_prompt_curve(self,embeddings,label):
    return np.zeros(len(embeddings),dtype=np.float32)
  class VideoModel:
   enabled=False
   def predict(self,frames):
    return {name:0.0 for name in FPV_CLASSES}
  n=30
  with patch("builtins.print"):
   _,_,highlights,_,_=run_fpv_branch(
    Path("unused.mp4"),
    float(n),
    router=Router(),
    video_model=VideoModel(),
    frames=np.zeros((n,224,224,3),dtype=np.uint8),
    audio_flux=np.zeros(n,dtype=np.float32),
    highband=np.zeros(n,dtype=np.float32),
    motion=[0.0]*n,
    embeddings=np.ones((n,8),dtype=np.float32),
    sig_label="parkour",
    sig_score=0.8,
    sig_probs={"parkour":1.0},
   )
  self.assertTrue(highlights)
  total=sum(e-s for s,e in [x["segment"] for x in highlights])
  self.assertLessEqual(total,0.148*n+1e-3)
  self.assertTrue(all(e-s>=4.05 for s,e in [x["segment"] for x in highlights]))
if __name__=="__main__":unittest.main()


class TestKTSAndFPVModels(unittest.TestCase):
    def test_kts_segments_cover_timeline_and_minimum(self):
        import numpy as np
        from kts import kts_segments
        # Synthetic features with a distinct change every 8 seconds.
        x=np.vstack([np.tile([1.,0.],(8,1)),np.tile([0.,1.],(8,1)),np.tile([1.,0.],(8,1))])
        segs=kts_segments(x,min_len_sec=4.05,max_len_sec=12,penalty=0.05,fps=1.0)
        self.assertTrue(segs)
        self.assertGreaterEqual(segs[0][0],0)
        self.assertAlmostEqual(segs[-1][1],len(x),delta=0.1)
        self.assertTrue(all(e-s>=4.0 for s,e in segs))
    def test_videomae_label_alias_mapping(self):
        from videomae_fpv import LABEL_TO_CLASS, norm_label
        self.assertEqual(LABEL_TO_CLASS[norm_label("Mountain biking")], "bike")
        self.assertEqual(LABEL_TO_CLASS[norm_label("Rock climbing")], "rockclimbing")
        self.assertEqual(LABEL_TO_CLASS[norm_label("Scuba diving")], "scuba")
        self.assertNotIn(norm_label("playing basketball"), LABEL_TO_CLASS)

    def test_epic_official_vocabulary_and_action_mapping(self):
        from schema import load_supported_classes
        import numpy as np
        ROOT=Path(__file__).resolve().parents[1]
        vocab=load_supported_classes(ROOT/"supported_classes.json")
        self.assertEqual(len(vocab["verbs"]),98)
        self.assertEqual(len(vocab["nouns"]),301)
        with np.load(ROOT/"weights/epic_priors.npz",allow_pickle=False) as data:
            self.assertEqual(len(data["action_verb"]),3806)
            self.assertEqual(len(data["action_noun"]),3806)
            self.assertEqual(data["prior"].shape,(98,301))

    def test_video_mae_optional_without_checkpoint(self):
        from videomae_fpv import VideoMAEFpvClassifier, FPV_CLASSES
        model=VideoMAEFpvClassifier(model_path='__missing_checkpoint__')
        self.assertFalse(model.enabled)
        out=model.predict([])
        self.assertEqual(set(out),set(FPV_CLASSES))
        self.assertTrue(all(v==0 for v in out.values()))

    def test_kts_knapsack_respects_budget_and_no_overlap(self):
        from kts import knapsack_highlights
        segs=[[0,5],[6,11],[12,17],[18,23]]
        scores=[0.2,0.9,0.5,0.8]
        out=knapsack_highlights(segs,scores,100,budget_ratio=0.15,min_len=4.05)
        chosen=[x["segment"] for x in out]
        self.assertLessEqual(sum(e-s for s,e in chosen),15.0+1e-3)
        self.assertTrue(all(e-s>=4.0 for s,e in chosen))
        self.assertTrue(all(chosen[i][1]<=chosen[i+1][0] for i in range(len(chosen)-1)))

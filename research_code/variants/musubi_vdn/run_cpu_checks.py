"""Tiny CPU/reference-code checks only. No real models or GPU calls."""
import sys,os,json,time,unittest
from pathlib import Path
root=Path(__file__).resolve().parent
if sys.platform=='win32':sys.path.insert(0,str(root.parent/'gpu_training/.test_deps'))
sys.path.insert(0,str(root));os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
started=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/'tests')))
report={'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'seconds':time.perf_counter()-started,'scope':'tiny CPU tensors, pinned Musubi target statements, full H3 shapes on meta device; no GPU/model loads','full_CUDA_preflight_completed':False}
(root/'cpu_check_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
raise SystemExit(0 if result.wasSuccessful() else 1)

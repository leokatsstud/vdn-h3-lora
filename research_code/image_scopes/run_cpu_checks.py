"""Bound local CPU checks to tiny tensors and two threads; never load real weights."""
from pathlib import Path
import os,sys,unittest,time,json
ROOT=Path(__file__).resolve().parent
SINGLE_PROCESS='--single-process' in sys.argv
if sys.platform=='win32':
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ('total','available','pagefile','available_pagefile','virtual','available_virtual','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    if memory.available<(1 if SINGLE_PROCESS else 3)*2**30:raise SystemExit('CPU checks deferred: insufficient free RAM for selected test mode')
if (ROOT/'.test_deps').exists():sys.path.insert(0,str(ROOT/'.test_deps'))
elif sys.platform=='win32' and (ROOT.parent/'gpu_training/.test_deps').exists():sys.path.insert(0,str(ROOT.parent/'gpu_training/.test_deps'))
sys.path.insert(0,str(ROOT));os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
def main():
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
    if SINGLE_PROCESS:
        def flatten(group):
            for item in group:
                if isinstance(item,unittest.TestSuite):yield from flatten(item)
                else:yield item
        tests=list(flatten(suite))
        for test in tests:
            if 'test_replica_cpu.ReplicaCPU' in test.id() or test.id().endswith('test_four_ranks_equal_mean_loss_single_adam_update'):
                method=getattr(type(test),test._testMethodName);method.__unittest_skip__=True;method.__unittest_skip_why__='Multiprocess proof deferred: local RAM limit'
        suite=unittest.TestSuite(tests)
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'passed':result.wasSuccessful(),'seconds':time.perf_counter()-started,'torch':sys.modules['torch'].__version__,'threads':2,'full_vdn_gpu_tested':False,'scope':'tiny CPU tensors; real 2/4-rank Gloo tests only when memory permits; no model weights loaded','single_process':SINGLE_PROCESS}
    if sys.platform=='win32':
        class ProcessMemory(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(s,ctypes.c_size_t) for s in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        pm=ProcessMemory();pm.cb=ctypes.sizeof(pm);ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),ctypes.byref(pm),pm.cb)
        report['peak_working_set_mib']=pm.peak_ws/2**20
    (ROOT/'cpu_check_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()

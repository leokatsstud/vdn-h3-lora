import datetime,json,tempfile,unittest
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from stylelib.distributed import average_adapter_gradients

def replica_worker(rank,folder):
    store=dist.FileStore(str(Path(folder)/'rendezvous'),2)
    dist.init_process_group('gloo',store=store,rank=rank,world_size=2,timeout=datetime.timedelta(seconds=60))
    frozen=torch.nn.Parameter(torch.tensor([7.]),requires_grad=False)
    a=torch.nn.Parameter(torch.ones(3));b=torch.nn.Parameter(torch.ones(5))
    a.grad=torch.full_like(a,float(1+2*rank));b.grad=torch.full_like(b,float(3-rank))
    average_adapter_gradients([a,b],2,max_bucket_bytes=16)
    opt=torch.optim.SGD([a,b],lr=.1);opt.step()
    Path(folder,f'rank{rank}.json').write_text(json.dumps({'a':a.detach().tolist(),'b':b.detach().tolist(),'frozen':frozen.item()}))
    dist.destroy_process_group()

class ReplicaCPU(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),'CPU Gloo unavailable')
    def test_only_adapter_gradients_are_averaged_and_replicas_update_equally(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(replica_worker,args=(d,),nprocs=2,join=True)
            rows=[json.loads(Path(d,f'rank{i}.json').read_text()) for i in (0,1)]
            self.assertEqual(rows[0],rows[1]);self.assertEqual(rows[0]['frozen'],7)
            for x in rows[0]['a']:self.assertAlmostEqual(x,.8,places=6)
            for x in rows[0]['b']:self.assertAlmostEqual(x,.75,places=6)

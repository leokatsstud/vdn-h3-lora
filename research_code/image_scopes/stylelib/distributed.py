"""Replicated frozen model; communicate only LoRA gradients, not66GB of base weights."""
import copy,datetime,os
import torch
import torch.distributed as dist

def setup(device):
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world>1:
        if world!=2:raise ValueError('This optional replica mode is implemented for exactly2GPUs')
        torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=900))
        return rank,world,f'cuda:{local}'
    return 0,1,device

def any_rank(value,device,world):
    if world==1:return bool(value)
    flag=torch.tensor(int(bool(value)),device=device,dtype=torch.int32);dist.all_reduce(flag,op=dist.ReduceOp.MAX);return bool(flag.item())

def average_adapter_gradients(parameters,world,max_bucket_bytes=16*2**20,group=None):
    if world==1:return
    bucket=[];size=0
    def flush():
        if not bucket:return
        flat=torch.cat([p.grad.reshape(-1) for p in bucket]);dist.all_reduce(flat,group=group);flat.div_(world)
        cursor=0
        for p in bucket:
            count=p.numel();p.grad.copy_(flat[cursor:cursor+count].view_as(p));cursor+=count
    for p in parameters:
        if p.grad is None:raise ValueError('A declared LoRA parameter has no gradient')
        n=p.grad.numel()*p.grad.element_size()
        if bucket and size+n>max_bucket_bytes:flush();bucket=[];size=0
        bucket.append(p);size+=n
    flush()

def gather(value,world):
    if world==1:return [copy.deepcopy(value)]
    values=[None]*world;dist.all_gather_object(values,value);return values

def barrier(world):
    if world>1:dist.barrier()

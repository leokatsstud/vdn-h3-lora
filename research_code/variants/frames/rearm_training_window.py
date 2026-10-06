"""Rearm only the image-training elapsed-time field in a committed state."""
import argparse,copy
from pathlib import Path
import torch
from stylelib.common import atomic_torch_save

def equal(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
            if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b

def reset_training_window(saved,expected_step=5037):
    if saved['step']!=expected_step or saved.get('fresh_experiment_seconds')!=3600:
        raise ValueError('Unexpected source checkpoint')
    if len(saved['style'])!=600 or len(saved['optimizer']['state'])!=600:
        raise ValueError('Adapter/Adam state scope differs')
    result=copy.deepcopy(saved)
    result['fresh_window_elapsed_seconds']=0.
    restored=copy.deepcopy(result)
    restored['fresh_window_elapsed_seconds']=saved['fresh_window_elapsed_seconds']
    if not equal(restored,saved):raise ValueError('Learning state changed')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--expected-step',type=int,default=5037)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Output checkpoint already exists')
    saved=torch.load(args.source,map_location='cpu',weights_only=True)
    result=reset_training_window(saved,args.expected_step)
    atomic_torch_save(result,args.out)
    if not equal(result,torch.load(args.out,map_location='cpu',weights_only=True)):
        raise ValueError('Serialized resume is not exact')

if __name__=='__main__':main()

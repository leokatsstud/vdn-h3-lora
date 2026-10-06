"""Every retained image exactly once per shuffled epoch, independent of source."""
import random
from .common import fingerprint

class FrameEpochSampler:
    def __init__(self,pool,seed):
        if not pool:raise ValueError('Empty sampling pool')
        self.pool=pool;self.rng=random.Random(seed)
        self.identity=fingerprint([e['record']['id'] for e in pool])
        self.order=list(range(len(pool)));self.rng.shuffle(self.order)
        self.cursor=0;self.cycles=0
    def take(self):
        if self.cursor==len(self.order):
            self.rng.shuffle(self.order);self.cursor=0;self.cycles+=1
        i=self.order[self.cursor];self.cursor+=1
        return self.pool[i]
    def state_dict(self):
        return {'version':2,'policy':'all_frames_shuffled_epoch','identity':self.identity,
                'rng':self.rng.getstate(),'order':list(self.order),'cursor':self.cursor,'cycles':self.cycles}
    def load_state_dict(self,state):
        if state['version']!=2 or state['policy']!='all_frames_shuffled_epoch' or state['identity']!=self.identity:raise ValueError('Frame epoch policy/dataset differs')
        if sorted(state['order'])!=list(range(len(self.pool))) or not 0<=state['cursor']<=len(self.pool):raise ValueError('Invalid epoch permutation/cursor')
        self.rng.setstate(state['rng']);self.order=list(state['order']);self.cursor=state['cursor'];self.cycles=state['cycles']

"""Equal source-video exposure; no replacement until each source/frame cycle finishes."""
import random
from .common import fingerprint

class SourceBalancedSampler:
    def __init__(self,pool,seed):
        self.pool=pool;self.rng=random.Random(seed);self.groups={}
        for i,e in enumerate(pool):
            r=e['record'];key=r.get('source_video_id',r['id']);self.groups.setdefault(key,[]).append(i)
        if not self.groups:raise ValueError('Empty sampling pool')
        self.identity=fingerprint([(e['record']['id'],e['record'].get('source_video_id',e['record']['id'])) for e in pool])
        self.source_order=sorted(self.groups);self.rng.shuffle(self.source_order);self.cursor=0;self.cycles=0
        self.frame_orders={key:list(values) for key,values in self.groups.items()};self.frame_cursors={key:0 for key in self.groups}
        for values in self.frame_orders.values():self.rng.shuffle(values)
    def take(self):
        if self.cursor==len(self.source_order):
            self.rng.shuffle(self.source_order);self.cursor=0;self.cycles+=1
        source=self.source_order[self.cursor];self.cursor+=1
        if self.frame_cursors[source]==len(self.frame_orders[source]):
            self.rng.shuffle(self.frame_orders[source]);self.frame_cursors[source]=0
        i=self.frame_orders[source][self.frame_cursors[source]];self.frame_cursors[source]+=1
        return self.pool[i]
    def state_dict(self):
        return {'version':1,'identity':self.identity,'rng':self.rng.getstate(),'source_order':list(self.source_order),'cursor':self.cursor,'cycles':self.cycles,'frame_orders':{k:list(v) for k,v in self.frame_orders.items()},'frame_cursors':dict(self.frame_cursors)}
    def load_state_dict(self,state):
        if state['version']!=1 or state['identity']!=self.identity:raise ValueError('Sampler dataset/policy changed')
        if sorted(state['source_order'])!=sorted(self.groups) or not 0<=state['cursor']<=len(self.groups):raise ValueError('Invalid source cycle')
        if set(state['frame_orders'])!=set(self.groups) or set(state['frame_cursors'])!=set(self.groups):raise ValueError('Invalid sampler groups')
        for key,order in state['frame_orders'].items():
            if sorted(order)!=sorted(self.groups[key]) or not 0<=state['frame_cursors'][key]<=len(order):raise ValueError('Invalid frame cycle')
        self.rng.setstate(state['rng']);self.source_order=list(state['source_order']);self.cursor=state['cursor'];self.cycles=state['cycles']
        self.frame_orders={k:list(v) for k,v in state['frame_orders'].items()};self.frame_cursors=dict(state['frame_cursors'])

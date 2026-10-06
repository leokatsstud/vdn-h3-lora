"""Preserve and trim only events beyond an atomic checkpoint's committed step."""
import json,time
from pathlib import Path

def reconcile_events(path,committed_step):
    path=Path(path)
    if not path.exists():
        if committed_step:raise ValueError('Resume ledger missing')
        return 0
    kept=[];discarded=[];last=0
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError:discarded.append(line);continue
        step=row['step']
        if step>committed_step:discarded.append(line);continue
        if step!=last+1:raise ValueError('Committed event ledger has gaps/duplicates')
        last=step;kept.append(line)
    if last!=committed_step:raise ValueError('Checkpoint step is not represented by the ledger')
    if discarded:
        suffix=str(time.time_ns());path.with_name('events_uncommitted_'+suffix+'.jsonl').write_text('\n'.join(discarded)+'\n',encoding='utf-8')
        temp=path.with_suffix('.tmp');temp.write_text(('\n'.join(kept)+'\n') if kept else '',encoding='utf-8');temp.replace(path)
    return len(discarded)

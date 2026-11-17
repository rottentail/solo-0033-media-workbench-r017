"""Compact waveform interpolation of explicitly marked damage."""
import numpy as np

CONTEXT = 512


def read(handle, first, count):
    handle.seek(first*8)
    raw = handle.read(count*8)
    if len(raw) != count*8:
        raise ValueError('incomplete repair context')
    pcm = np.frombuffer(raw, dtype='<f4').reshape(-1, 2).astype(np.float64)
    if not np.isfinite(pcm).all():
        raise ValueError('nonfinite repair context')
    return pcm


def restore(left, right, length):
    """Reuse a nearby voiced cycle with a smooth boundary bridge."""
    mono = np.mean(left,axis=1)
    energy = float(np.dot(mono,mono))
    if energy<1e-16:
        return np.zeros((length,2),dtype='<f4')
    scores=[]
    for lag in range(32, min(320,len(mono)-32)):
        a,b=mono[lag:],mono[:-lag]
        scale=max(float(np.linalg.norm(a)*np.linalg.norm(b)),1e-20)
        scores.append(float(np.dot(a,b))/scale)
    period=32+int(np.argmax(scores)) if scores else max(1,len(mono))
    cycle=mono[-period:]
    generated=np.resize(cycle,length)
    phase=np.linspace(0,1,length+2)[1:-1]
    boundary=(1-phase)*float(np.mean(left[-1]))+phase*float(np.mean(right[0]))
    blend=np.sin(np.pi*phase)**2
    repaired=generated*blend+boundary*(1-blend)
    return np.column_stack((repaired,repaired)).astype('<f4')


def patches(handle, intervals):
    restored=[]
    for first,stop in intervals:
        left=read(handle,first-CONTEXT,CONTEXT)
        right=read(handle,stop,CONTEXT)
        restored.append((first,stop,restore(left,right,stop-first)))
    return restored


def blocks(handle, clip, restored, first, stop, block_frames):
    origin=clip['start_ms']*48
    for cursor in range(first,stop,block_frames):
        end=min(stop,cursor+block_frames)
        pcm=read(handle,origin+cursor,end-cursor)
        for left,right,patch in restored:
            a,b=max(origin+cursor,left),min(origin+end,right)
            if a<b: pcm[a-origin-cursor:b-origin-cursor]=patch[a-left:b-left]
        yield (pcm*clip['gain']).astype('<f4')

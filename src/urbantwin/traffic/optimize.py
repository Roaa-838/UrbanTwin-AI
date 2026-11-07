"""Small, transparent GA/PSO optimizers for two continuous green durations."""
from __future__ import annotations
from typing import Callable
import numpy as np


def _bounded_int(v,lo,hi):
    return tuple(int(x) for x in np.clip(np.rint(v),lo,hi).astype(int))


def genetic_algorithm(score: Callable, bounds: tuple[int,int], population: int=6,
                      generations: int=3, seed: int=42) -> dict:
    lo,hi=bounds
    if not (0<lo<hi and population>=4 and generations>=1):raise ValueError("Invalid GA settings")
    rng=np.random.default_rng(seed)
    pool=rng.integers(lo,hi+1,size=(population,2))
    history=[]
    best=(float("inf"),None)
    for gen in range(generations):
        values=np.array([float(score(_bounded_int(v,lo,hi))) for v in pool])
        ranked=np.argsort(values)
        if values[ranked[0]]<best[0]:best=(float(values[ranked[0]]),_bounded_int(pool[ranked[0]],lo,hi))
        history.append({"iteration":gen,"best_score":best[0],"best_plan":best[1]})
        elites=pool[ranked[:max(2,population//3)]].copy()
        next_pool=list(elites)
        while len(next_pool)<population:
            a=elites[rng.integers(0,len(elites))];b=elites[rng.integers(0,len(elites))]
            child=np.array([a[0],b[1]],dtype=int)
            if rng.random()<.7:child[rng.integers(0,2)]+=int(rng.integers(-8,9))
            next_pool.append(np.clip(child,lo,hi))
        pool=np.array(next_pool,dtype=int)
    return {"plan":best[1],"objective":best[0],"history":history}


def particle_swarm(score: Callable,bounds: tuple[int,int],particles: int=6,
                   iterations: int=3,seed: int=42) -> dict:
    lo,hi=bounds
    if not (0<lo<hi and particles>=2 and iterations>=1):raise ValueError("Invalid PSO settings")
    rng=np.random.default_rng(seed)
    x=rng.uniform(lo,hi,size=(particles,2));v=np.zeros_like(x)
    personal=x.copy();personal_val=np.full(particles,np.inf)
    best_val=float("inf");best_plan=None;best_position=x[0].copy();history=[]
    for iteration in range(iterations):
        for i in range(particles):
            candidate=_bounded_int(x[i],lo,hi)
            cost=float(score(candidate))
            if cost<personal_val[i]:personal_val[i]=cost;personal[i]=x[i].copy()
            if cost<best_val:best_val=cost;best_plan=candidate;best_position=x[i].copy()
        history.append({"iteration":iteration,"best_score":best_val,"best_plan":best_plan})
        v=.65*v+1.35*rng.random((particles,2))*(personal-x)+1.35*rng.random((particles,2))*(best_position-x)
        x=np.clip(x+v,lo,hi)
    return {"plan":best_plan,"objective":best_val,"history":history}

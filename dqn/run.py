"""Minimal train/evaluate interface; defaults never start a long run implicitly."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from . import config as cfg
from .agent import DQN

def epsilon(step):
    if step < cfg.EPS_DECAY_STEPS:
        return cfg.EPS_START + step / cfg.EPS_DECAY_STEPS * (cfg.EPS_END - cfg.EPS_START)
    return cfg.EPS_END + min(1.,(step-cfg.EPS_DECAY_STEPS)/150000) * (cfg.EPS_FINAL-cfg.EPS_END)

def evaluate(model, env, episodes, seed):
    returns=[]
    for episode in range(episodes):
        state,_=env.reset(seed=seed+episode+1); score=0.; ended=False
        while not ended:
            action=int(np.argmax(model(np.asarray(state,dtype=np.float32)[None,:], training=False).numpy()[0]))
            state,reward,terminated,truncated,_=env.step(action)
            ended=terminated or truncated; score+=float(reward)
        returns.append(score)
    return returns

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['train','evaluate'])
    parser.add_argument('--episodes',type=int,default=30)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--checkpoint',type=Path,default=Path('runs/final.keras'))
    args=parser.parse_args()
    if args.episodes<1:parser.error('--episodes must be positive')
    import gymnasium as gym
    random.seed(args.seed);np.random.seed(args.seed);tf.random.set_seed(args.seed)
    env=gym.make('LunarLander-v3')
    args.checkpoint.parent.mkdir(parents=True,exist_ok=True)
    try:
        if args.mode=='evaluate':
            model=tf.keras.models.load_model(args.checkpoint)
            scores=evaluate(model,env,args.episodes,args.seed)
            report=dict(environment='LunarLander-v3',seed_start=args.seed+1,
                        episodes=args.episodes,returns=scores,mean=float(np.mean(scores)),
                        std=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        training_seeds=1)
            out=args.checkpoint.with_suffix('.evaluation.json')
        else:
            agent=DQN(env.observation_space.shape[0],env.action_space.n)
            steps=0;scores=[]
            for episode in range(args.episodes):
                state,_=env.reset(seed=args.seed if episode==0 else None)
                ended=False;score=0.
                while not ended:
                    action=agent.select(state,epsilon(steps))
                    nxt,reward,terminated,truncated,_=env.step(action);steps+=1
                    # Time-limit truncation ends rollout but still permits bootstrapping.
                    agent.remember(state,action,reward,nxt,terminated)
                    if steps>cfg.WARMUP_STEPS and steps%cfg.TRAIN_FREQ==0:agent.learn(steps)
                    if steps%cfg.TARGET_UPDATE_INTERVAL==0:agent.update_target_model(hard=True)
                    score+=float(reward);state=nxt;ended=terminated or truncated
                scores.append(score)
                print(json.dumps(dict(episode=episode+1,return_=score,steps=steps)))
            agent.model.save(args.checkpoint)
            report=dict(returns=scores,steps=steps,seed=args.seed,
                        config={k:v for k,v in vars(cfg).items() if k.isupper()},
                        note='2026 corrected implementation; fixed episode count, no historical best-average selection')
            out=args.checkpoint.with_suffix('.training.json')
        out.write_text(json.dumps(report,indent=2));print(out)
    finally:env.close()
if __name__=='__main__':main()

"""Isolated 2x2 factorial experiment; never updates deployment registry/splits."""
import argparse, copy, gc, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import optuna
from src.common import ROOT, read_json, write_json, seed_all, digest
from src.data.dataset import load_split, encode_answers
from src.models.scorer import create_model
from src.evaluation.metrics import metrics

GROUPS = {'hard_ce':(False,False),'soft_ce':(True,False),
          'hard_ce_emd':(False,True),'soft_ce_emd':(True,True)}
SEEDS = [42,123,2026]

def targets(frame, soft):
    a=torch.tensor(frame.score.to_numpy(),dtype=torch.long)
    y=F.one_hot(a,3).float()
    if soft:
        b=pd.to_numeric(frame.score2,errors='coerce').to_numpy()
        if np.any(np.isfinite(b)&((b<0)|(b>2)|(b!=np.floor(b)))):
            raise ValueError('Invalid finite Score2; refusing to coerce rater annotations')
        valid=np.isfinite(b)&(b>=0)&(b<=2)&(b==np.floor(b))
        other=F.one_hot(torch.tensor(np.where(valid,b,a.numpy()),dtype=torch.long),3).float()
        y=(y+other)/2
    return y

def loss_fn(logits, target, emd_weight=0.):
    ce=-(target*F.log_softmax(logits.float(),-1)).sum(-1).mean()
    # Squared CDF distance on the two nontrivial ordinal thresholds, normalized.
    emd=((logits.float().softmax(-1).cumsum(-1)[:,:-1]-target.cumsum(-1)[:,:-1])**2).mean()
    return ce+emd_weight*emd

def dataset(frame,tok,p,cfg,soft):
    enc=encode_answers(tok,frame.answer.tolist(),p,cfg['max_length'])
    return TensorDataset(enc['input_ids'],enc['attention_mask'],targets(frame,soft),torch.tensor(frame.score.tolist()))

@torch.inference_mode()
def evaluate(model,dl,dev,weight):
    model.eval(); truth=[]; pred=[]; prob=[]; total=0.; n=0
    for ids,mask,y,a in dl:
        logits=model(input_ids=ids.to(dev),attention_mask=mask.to(dev)).logits
        total+=float(loss_fn(logits,y.to(dev),weight))*len(a); n+=len(a)
        truth+=a.tolist(); pred+=logits.argmax(-1).cpu().tolist(); prob+=logits.softmax(-1).cpu().tolist()
    return {**metrics(truth,pred),'loss':total/n},dict(true=truth,pred=pred,probabilities=prob)

def fit(p,g,cfg,folder,save=False,pilot=False):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists(): return read_json(folder/'summary.json')
    seed_all(cfg['seed']); torch.set_num_threads(4)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    soft,emd=GROUPS[g]; w=cfg['emd_weight'] if emd else 0.
    tok=AutoTokenizer.from_pretrained(ROOT/cfg['pretrained'],local_files_only=True)
    tr,va=load_split(p,'train'),load_split(p,'validation')
    dl=DataLoader(dataset(tr,tok,p,cfg,soft),batch_size=cfg['batch_size'],shuffle=True,
                  generator=torch.Generator().manual_seed(cfg['seed']))
    vl=DataLoader(dataset(va,tok,p,cfg,soft),batch_size=cfg['batch_size']*2)
    model=create_model('classification',cfg).to(dev)
    witness=model.distilbert.transformer.layer[-1].attention.q_lin.weight.detach().cpu().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sch=get_linear_schedule_with_warmup(opt,int(len(dl)*cfg['epochs']*.1),len(dl)*cfg['epochs'])
    best=-float('inf'); bad=0; hist=[]; start=time.time(); steps=0
    run=dict(prompt=p,group=g,config=cfg,device=str(dev),gpu=torch.cuda.get_device_name(0) if dev.type=='cuda' else None,
             torch=torch.__version__,soft_target='0.5 onehot(Score1)+0.5 onehot(Score2); missing Score2 -> Score1',
             emd_definition='mean squared CDF distance over K-1 thresholds',train_rows=len(tr),validation_rows=len(va))
    write_json(folder/'run.json',run)
    for ep in range(cfg['epochs']):
        model.train(); total=0.; seen=0
        for ids,mask,y,a in dl:
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dev.type=='cuda' and torch.cuda.is_bf16_supported()):
                logits=model(input_ids=ids.to(dev),attention_mask=mask.to(dev)).logits
                loss=loss_fn(logits,y.to(dev),w)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sch.step()
            total+=float(loss.detach())*len(a); seen+=len(a); steps+=1
            if pilot and steps==3: break
        val,preds=evaluate(model,vl,dev,w)
        hist.append(dict(epoch=ep+1,train_loss=total/seen,validation=val,steps=steps,seconds=time.time()-start))
        write_json(folder/'history.json',hist)
        q=val['qwk'] if val['qwk'] is not None else -1.
        if q>best:
            best=q; bad=0; bestepoch=ep+1; bestmetrics=val
            write_json(folder/'validation_predictions.json',{'ids':va.id.tolist(),**preds})
            if save:
                model.save_pretrained(folder/'checkpoint'); tok.save_pretrained(folder/'checkpoint')
                write_json(folder/'checkpoint/run.json',run)
        else: bad+=1
        print('EPOCH',folder.name,ep+1,q,flush=True)
        if pilot or bad>=cfg['patience']: break
    change=float((model.distilbert.transformer.layer[-1].attention.q_lin.weight.detach().cpu()-witness).abs().max())
    assert change>0 and steps>0
    result=dict(prompt=p,group=g,seed=cfg['seed'],config=cfg,validation_qwk=best,validation=bestmetrics,
                best_epoch=bestepoch,optimizer_steps=steps,encoder_weight_change=change,seconds=time.time()-start,path=str(folder))
    write_json(folder/'summary.json',result)
    del model,opt,sch,dl,vl; gc.collect()
    if dev.type=='cuda': torch.cuda.empty_cache()
    return result

def protected():
    paths=list((ROOT/'data/processed').glob('prompt*/*.jsonl'))+[ROOT/'data/processed/split_manifest.json',ROOT/'models/registry.json']
    registry=read_json(ROOT/'models/registry.json')
    paths += [ROOT / x['path'] / 'model.safetensors' for x in registry.values()]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[read_json(p) for p in (out/'repeats').glob('*/summary.json')]
    if not records:return
    rows=[]
    for r in records:
        row={k:r[k] for k in ['prompt','group','seed','validation_qwk','seconds','best_epoch']}
        if (Path(r['path'])/'test.json').exists(): row.update({'test_'+k:v for k,v in read_json(Path(r['path'])/'test.json')['metrics'].items() if k in ['qwk','mae','rmse','accuracy']})
        rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    agg=df.groupby(['prompt','group']).agg({'validation_qwk':['mean','std'],'seconds':'mean'})
    agg.to_csv(out/'stability.csv')
    plots=out/'figures';plots.mkdir(exist_ok=True)
    for p in [3,4]:
        fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
        for g in GROUPS:
            values=[read_json(f)['validation_qwk'] for f in sorted((out/'search').glob(f'p{p}_{g}_t*/summary.json'))]
            if values:ax.plot(range(1,len(values)+1),np.maximum.accumulate(values),label=g)
        ax.set(xlabel='Shared candidate index',ylabel='Best validation QWK',title=f'Prompt {p}: equal search budgets');ax.legend()
        fig.savefig(plots/f'p{p}_search_budget.png',dpi=200);plt.close(fig)
    for p in [3,4]:
        subset=df[df.prompt==p]
        for metric in ['validation_qwk','test_qwk','test_mae','test_rmse','test_accuracy']:
            if metric not in subset:continue
            fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
            for i,g in enumerate(GROUPS):
                v=subset.loc[subset.group==g,metric].dropna()
                if len(v):
                    ax.bar(i,v.mean(),yerr=v.std(ddof=1) if len(v)>1 else 0,color=['#4477AA','#66CCEE','#228833','#CCBB44'][i],capsize=4)
                    ax.scatter(np.linspace(i-.12,i+.12,len(v)),v,color='black',s=15)
            ax.set_xticks(range(4),GROUPS,rotation=15);ax.set_ylabel(metric);ax.set_title(f'Prompt {p}: individual seeds, mean and SD (n=3)')
            for ext in ['png','pdf','svg']:fig.savefig(plots/f'p{p}_{metric}.{ext}',dpi=200)
            plt.close(fig)
    for r in records:
        h=read_json(Path(r['path'])/'history.json')
        for name,vals in [('loss',[v['train_loss'] for v in h]),('validation_qwk',[v['validation']['qwk'] for v in h])]:
            fig,ax=plt.subplots(figsize=(5,3),layout='constrained');ax.plot(range(1,len(h)+1),vals,marker='o',label='Training' if name=='loss' else 'Validation')
            if name=='loss':ax.plot(range(1,len(h)+1),[v['validation']['loss'] for v in h],marker='s',label='Validation')
            ax.legend();ax.set(xlabel='Epoch',ylabel=name,title=f"P{r['prompt']} {r['group']} seed {r['seed']}")
            fig.savefig(plots/f"p{r['prompt']}_{r['group']}_s{r['seed']}_{name}.png",dpi=180);plt.close(fig)
        if (Path(r['path'])/'test.json').exists():
            cm=np.array(read_json(Path(r['path'])/'test.json')['metrics']['confusion_matrix'])
            fig,ax=plt.subplots(figsize=(4,4),layout='constrained');ax.imshow(cm,cmap='Blues')
            for (a,b),v in np.ndenumerate(cm):ax.text(b,a,str(v),ha='center',va='center')
            ax.set(xticks=[0,1,2],yticks=[0,1,2],xlabel='Predicted Score',ylabel='Score1',title=f"P{r['prompt']} {r['group']} s{r['seed']}")
            fig.savefig(plots/f"p{r['prompt']}_{r['group']}_s{r['seed']}_confusion.png",dpi=180);plt.close(fig)
    text='# 软标签与序数损失实验\n\n'+df.to_string(index=False)+'\n\n'+agg.to_string()
    text+='\n\n四组均使用相同的20组候选超参数、种子42、早停规则和最大训练预算；每组各自以验证集QWK选择配置。三种种子重复后冻结全部选择，随后才读取测试集。测试集曾在旧项目被评价过，因此本轮不是全新盲测。原有划分和部署模型不替换。误差棒为3种种子的样本标准差，不是置信区间。软标签仅用于训练；评价锚定Score1。\n'
    (out/'report.md').write_text(text,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(ROOT/'configs/training.json');cfg.update(epochs=4,patience=2,emd_weight=1.)
    before=protected()
    if (out/'preserved.json').exists(): assert before==read_json(out/'preserved.json')
    else:write_json(out/'preserved.json',before)
    # Same candidate table for all groups and prompts, frozen before fitting.
    if not (out/'candidates.json').exists():
        sampler=optuna.samplers.RandomSampler(seed=20261103);study=optuna.create_study(sampler=sampler)
        candidates=[]
        for i in range(20):
            t=study.ask();c=copy.deepcopy(cfg)
            c.update(learning_rate=t.suggest_float('learning_rate',1e-5,5e-5,log=True),batch_size=t.suggest_categorical('batch_size',[16,32]),epochs=t.suggest_categorical('epochs',[3,4]),weight_decay=t.suggest_float('weight_decay',0.,.1),dropout=t.suggest_categorical('dropout',[.1,.2,.3]),max_length=t.suggest_categorical('max_length',[128,192,256]))
            candidates.append(c);study.tell(t,0.)
        write_json(out/'candidates.json',candidates)
    candidates=read_json(out/'candidates.json')
    audit=[]
    for p in [3,4]:
        for split in ['train','validation']:
            frame=load_split(p,split);b=pd.to_numeric(frame.score2,errors='coerce')
            targets(frame,True)
            audit.append(dict(prompt=p,split=split,n=len(frame),missing_score2=int(b.isna().sum()),rater_disagreements=int(((b!=frame.score)&b.notna()).sum())))
    write_json(out/'rater_audit.json',audit)
    if args.pilot:
        for g in GROUPS:fit(3,g,cfg,out/'pilot'/g,pilot=True)
        assert protected()==before;return
    selected={};repeats=[]
    for p in [3,4]:
        for g in GROUPS:
            study=optuna.create_study(study_name=f'p{p}_{g}',storage='sqlite:///'+str(out/'optuna.db').replace('\\','/'),direction='maximize',load_if_exists=True,sampler=optuna.samplers.RandomSampler(seed=20261103),pruner=optuna.pruners.NopPruner())
            # Idempotent filesystem results allow safe resume after interruption.
            results=[]
            for i,c in enumerate(candidates):
                r=fit(p,g,c,out/'search'/f'p{p}_{g}_t{i:02d}')
                results.append(r)
                if not any(t.state==optuna.trial.TrialState.COMPLETE and t.user_attrs.get('candidate_index')==i for t in study.trials):
                    distributions={k:optuna.distributions.CategoricalDistribution(sorted(set(v[k] for v in candidates))) for k in ['learning_rate','batch_size','epochs','weight_decay','dropout','max_length']}
                    study.add_trial(optuna.trial.create_trial(params={k:c[k] for k in distributions},distributions=distributions,value=r['validation_qwk'],user_attrs={'candidate_index':i,'sampling':'shared Optuna RandomSampler candidate table'}))
            best=max(results,key=lambda r:r['validation_qwk']);selected[f'{p}_{g}']=best
            for seed in SEEDS:
                c=copy.deepcopy(best['config']);c['seed']=seed
                repeats.append(fit(p,g,c,out/'repeats'/f'p{p}_{g}_s{seed}',save=True))
            report(out)
    champions={str(p):max(GROUPS,key=lambda g:np.mean([r['validation_qwk'] for r in repeats if r['prompt']==p and r['group']==g])) for p in [3,4]}
    freeze={'selected':selected,'champions':champions,'repeats':repeats,'preserved':before,'test_access_before_freeze':False,
            'selection_rule':'best validation QWK per group/prompt, ties first candidate; champion by mean validation QWK across seeds'}
    if (out/'selection_frozen.json').exists():assert read_json(out/'selection_frozen.json')==freeze
    else:write_json(out/'selection_frozen.json',freeze)
    frozen_hash=digest(out/'selection_frozen.json')
    for r in repeats:
        folder=Path(r['path'])
        if (folder/'test.json').exists():continue
        p=r['prompt'];frame=load_split(p,'test');tok=AutoTokenizer.from_pretrained(folder/'checkpoint',local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(folder/'checkpoint',local_files_only=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        dl=DataLoader(dataset(frame,tok,p,r['config'],False),batch_size=32)
        m,preds=evaluate(model,dl,next(model.parameters()).device,0.)
        write_json(folder/'test.json',{'metrics':m,'ids':frame.id.tolist(),**preds,'selection_sha256':frozen_hash})
        del model;gc.collect();torch.cuda.empty_cache()
    assert protected()==before
    report(out);write_json(out/'complete.json',{'finished':True,'test_selection_sha256':frozen_hash,'protected_files_verified':True})

if __name__=='__main__':main()

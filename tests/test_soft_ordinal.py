import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from src.training.soft_ordinal import targets, loss_fn

def test_two_raters_disagreement_and_missing():
    f=pd.DataFrame({'score':[0,1,2],'score2':[2,1,float('nan')]})
    assert torch.allclose(targets(f,True),torch.tensor([[.5,0,.5],[0,1,0],[0,0,1.]]))
    assert torch.allclose(targets(f,False),F.one_hot(torch.tensor([0,1,2]),3).float())

def test_invalid_rater_rejected():
    with pytest.raises(ValueError): targets(pd.DataFrame({'score':[1],'score2':[1.5]}),True)

def test_hard_ce_equivalence_and_finite_gradient():
    logits=torch.tensor([[2.,1.,0.]],requires_grad=True)
    y=torch.tensor([[0.,1.,0.]])
    assert torch.allclose(loss_fn(logits,y),F.cross_entropy(logits,torch.tensor([1])))
    loss_fn(logits,y,1.).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0

def test_ordinal_farther_errors_cost_more():
    y=torch.tensor([[1.,0.,0.]])
    near=torch.tensor([[0.,10.,0.]])
    far=torch.tensor([[0.,0.,10.]])
    assert loss_fn(far,y,1)-loss_fn(far,y,0)>loss_fn(near,y,1)-loss_fn(near,y,0)

"""仅核对输入形状，不生成分析结果。"""
from .io import unique
def summarize(v,root):
    unique(v['nodes'],'id');n=len(v['nodes']);m=len(v['bottom_ids'])
    if len(v['predictions'])!=n or len(v['covariance'])!=n or any(len(r)!=n for r in v['covariance']):raise ValueError('误差维度错误')
    if len(set(v['bottom_ids']))!=m:raise ValueError('底层身份重复')
    return {'bottoms':m,'predictions':n,'queries':len(v['queries'])}

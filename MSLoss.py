import torch as th
import torch.nn as nn
import torch.nn.functional as F

def dilation3d(X, conn=26):
    assert conn in [6,26] , "3d connectivity only 8 or 26"
    if conn == 26:
        return  F.max_pool3d(X, 3, 1, 1)
    if conn == 6:
        p1 = F.max_pool3d(X, (3,1,1), (1,1,1), (1,0,0))
        p2 = F.max_pool3d(X, (1,3,1), (1,1,1), (0,1,0))
        p3 = F.max_pool3d(X, (1,1,3), (1,1,1), (0,0,1))
        return th.max(th.max(p1,p2),p3)
    
def erosion3d(X, conn=26):
    return -dilation3d(-X, conn=conn)

def opening3d(X, conn_e=26, conn_d=26):
    return dilation3d(erosion3d(X, conn=conn_e), conn=conn_d)

def closing3d(X, conn_e=26, conn_d=26):
    return erosion3d(dilation3d(X, conn=conn_d), conn=conn_e)

def skeletonization3d(X, iter_, persistent=True):
    X1 = opening3d(X, conn_d=26, conn_e=6)
    skel = F.relu(X-X1)
    for i in range(iter_):
        Xe = erosion3d(X, conn=6)
        X1 = opening3d(Xe, conn_d=26, conn_e=6)
        delta = (F.relu(Xe-X1))
        if persistent: 
            skel = X * dilation3d(skel, conn=26) + skel
        skel = skel + (delta - skel * delta)
        X = Xe
    #if persistent: skel.clamp_(0,1)
    if persistent: skel[skel>1] = skel[skel>1]/skel[skel>1].detach()
    
    return skel

def persistent_skeletonization3d(X, niter, niter2=None):
    sX = skeletonization3d(X, niter, persistent=True)
    if niter2 is None: niter2 = niter//2
    sX = skeletonization3d(sX, niter2, persistent=True)
    return sX

def vs_score(v,s, eps=1e-12, mode='bmean'):
    """vs/s"""
    dim = list(range(1,len(s.shape)))
    if mode == 'mean':
        return ( (v*s).sum() + eps ) / ( (s).sum() + eps )
    if mode == 'max':
        return ( (v*s).max() + eps ) / ( (s).max() + eps )
    if mode == 'bmean':
        return (( (v*s).sum(dim=dim) + eps ) / ( s.sum(dim=dim) + eps )).mean()
    if mode == 'bmax':
        return (( (v*s).amax(dim=dim) + eps ) / ( s.amax(dim=dim) + eps )).max()
    if mode == 'bmeanmax':
        return (( (v*s).amax(dim=dim) + eps ) / ( s.amax(dim=dim) + eps )).mean()
    
class MSLoss(nn.Module):
    
    def __init__(self, k=5, eps=1e-12, bin_tresh=0.7):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.k = k
        self.eps = eps
        self.bin_tresh = bin_tresh
        
    def forward(self, Vlogit, Vcl, Vedt):
        Vp = self.sigmoid(Vlogit)
        
        Vp_clone = (Vp.clone().detach() > self.bin_tresh).float()
        # homology patching!
        pad = [3,3,3,3,3,3,0,0,0,0]
        Vp_clone = F.pad(Vp_clone, pad, mode='constant', value=1.0)
        Sp_clone = persistent_skeletonization3d(Vp_clone, self.k)[:,:,3:-3,3:-3,3:-3]
        Sp = Sp_clone*Vp
        
        # inclusion
        tprec = vs_score(Vp, Vcl, eps=self.eps, mode='bmean')
        tsens = vs_score(Vcl, Sp, eps=self.eps, mode='bmean')
        dice_loss = 1.0 - ( 2.0 * tprec * tsens + self.eps ) / ( tprec + tsens + self.eps )
        
        loss = dice_loss
        return loss
        

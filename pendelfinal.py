import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random

dtype = torch.float
device = torch.device("cpu")


def polar2kartesisch(x,l): # x = [phi,phi']
    phi = x[0]
    vphi = x[1]
    return [l*np.sin(phi), -l * np.cos(phi), l * np.cos(phi) * vphi, l *np.sin(phi) * vphi]


def kartesisch2polar(kart,l): # x = [phi,phi']
    
    return np.array([np.arctan2(kart[0],-kart[1]),(-kart[2]*kart[1]+kart[3]*kart[0])/np.sqrt(kart[0]*kart[0]+kart[1]*kart[1])])


def eulerkartesisch(kart, g,l,m,h): # kart = [x,y,vx,vy]
    
    ll = kart[0]*kart[0]+kart[1]*kart[1]
    vv = kart[2]*kart[2]+kart[3]*kart[3]

    return np.array([
        kart[0] + h * kart[2], 
        kart[1] + h * kart[3],
        kart[2] + h * (-vv*kart[0] - g * kart[0]*(-kart[1]) ) / ll / m,
        kart[3] + h * (-vv*kart[1] - g * kart[0]*( kart[0]) ) / ll / m
        ])


#[u,v] = [phi,phi'], [u',v'] = [v, -g/l*sin(u)]
def rk4(X, g, l, m, h):
    phi0 = X[0]
    v0 = X[1]
    
    #Rungekutta-4
    k1 = h*v0
    l1 = -(g/l)*h*np.sin(phi0)
    
    k2 = h*(v0 + 0.5*l1)
    l2 = -(g/l)*h*np.sin(phi0 + 0.5*k1)
    
    k3 = h*(v0 + 0.5*l2)
    l3 = -(g/l)*h*np.sin(phi0 + 0.5*k2)
    
    k4 = h*(v0 + l3)
    l4 = -(g/l)*h*np.sin(phi0 + k3)
    
    phineu = phi0 + 1/6*(k1 + 2*k2 + 2*k3 + k4) 
    vneu = v0 + 1/6*(l1 + 2*l2 + 2*l3 + l4) 
    
    return np.array([phineu,vneu])



###Neuronales Netz


D_in,H1,H2,H3,D_out = 8,20,20,20,4 

model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1,H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2,H3),
        torch.nn.ReLU(),
        torch.nn.Linear(H3,D_out),
       )

loss_fn = torch.nn.MSELoss(reduction='sum')

SE = 5000
iteration = 0
optimizer = torch.optim.Adam(model.parameters()) #,lr=learning_rate)

# Parameter
g = 9.81
l = 1
m = 1 
h = 0.1
t = 0 

### Daten generieren
ndata = 5000
alldata = np.zeros( (ndata, 12) )

for i in range(ndata):
    phi0 = (random.random()-0.5) * math.pi # Startauslenkun in [-pi/2,pi/2]
    v0 = 2*(random.random()-0.5) * math.pi  # Start-Geschwindigkeit in [-1, 1]
    
    alldata[i,0:4]  = polar2kartesisch(np.array([phi0,v0]),l)
    alldata[i,4:8]  = eulerkartesisch(polar2kartesisch(np.array([phi0,v0]),l),
                                             g, l, m, h)
    alldata[i,8:12] = polar2kartesisch(rk4(np.array([phi0,v0]),
                                         g,l,m,h),l) - alldata[i,0:4]


ntrain = 2*ndata//3
ntest = ndata - ntrain
traindata = alldata[:ntrain,:]
testdata = alldata[ntrain:,:]

traindata_in_torch = torch.Tensor(traindata[:,:8])
traindata_out_torch = torch.Tensor(traindata[:,8:])

testdata_in_torch = torch.Tensor(testdata[:,:8])
testdata_out_torch = torch.Tensor(testdata[:,8:])


for t in range(SE):
    y_pred = model(traindata_in_torch)
    loss = loss_fn(y_pred,traindata_out_torch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    if iteration%100 == 0:
        losstest = loss_fn(model(testdata_in_torch),testdata_out_torch)
        print(iteration+1,loss.item()/ntrain,losstest.item()/ntest)

    iteration=iteration+1

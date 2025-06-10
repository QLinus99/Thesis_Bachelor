import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random

from scipy.integrate import RK45
 
dtype = torch.float
device = torch.device("cpu")


# die Rechte Seite der 'exakten' pendel ODE
# phi'' = -g/l sin(phi')
# geschrieben als System
# [phi,phi']' = [phi, -g/l sin(phi') ]


def pendel_exact(phi):
    return np.array([phi[0],-g/l*np.sin(phi[1])])



# phi0 Startauslenkung, phi0 <<1
# Zeitpunkt t 
# l: Länge des Pendels
# phiab: Geschwindigkeit, phi' in [0, sqrt(2*g*l*(1-cos(0.1)))]

### Die Pendelparameter
g = 9.81
l = 1.00  # laenge
m = 1.0   # masse


# u= (x_1,x_2,v_1,v_2)
def u(phi0, t ):
    
    startx2 = np.cos(phi0)*l
    startx1 = np.sign(phi0)*np.sqrt(l * l - startx2**2.0)
#    if(phi0 >= 0):
#        startx1 = math.sqrt(math.pow(l,2) - math.pow(startx2,2))
#    else:
#        startx1 = -math.sqrt(math.pow(l,2) - math.pow(startx2,2))
        
    x1 = startx1*np.cos(t*np.sqrt(g/l)) #x1 zum Zeitpunkt t
    x2 = np.sqrt(l*l - x1**2.0)
#    if(x1 >= 0):
#        x2 = math.sqrt(math.pow(l,2) - math.pow(x1,2)) #x2 zumZeitpunkt t
#    else:
#        x2 = math.sqrt(math.pow(l,2) - math.pow(x1,2))
        
        
    v1 =  -startx1* np.sqrt(g/l) *np.sin(t*np.sqrt(g/l))     #v1 = x1'
    v2 = -(x1*v1)/x2   #v2 = x2' 
    
    u = np.array([x1, x2, v1, v2])
    return u




#m Masse des Körpers
def Lagrange(phi0,t):
    vektoru = u(phi0,t)
    
    x2 = vektoru[1]
    v1 = vektoru[2]
    v2 = vektoru[3]
    
    lag = m*(v1**2 + v2**2 + m*g*x2)/(2*l**2)
    
    return lag
    

def matrixA(phi0,t):
    
    lag = Lagrange(phi0,t)
    A  = np.array( ((0,0,-1,0), (0,0,0,-1), (2*lag/m,0,0,0) , (0,2*lag/m,0,0)) )
    
    return A



# u' + A*u = b <=> u' = b - A*u =:f  
def f(phi0,t ,h, wert):
    
    vektoru = u(phi0,t)
    
    b = np.array([0, 0, 0, g])
    A1 = matrixA(phi0,t)    #Matrix A zumZeitpunkt t
    A2 = matrixA(phi0,t+h)  #Matrix A zum Zeitpunkt t+h
        
    
    if(wert == 1):
        fout = b - np.dot(A1,vektoru)
    else:
        uexplizit = vektoru + h*(b - np.dot(A1,vektoru))
        fout = b - np.dot(A2,uexplizit)
    
    return fout
    
    
# Zeitschritt h
def heun(phi0,t, h):
    
    vektoru = u(phi0,t )
    
    
    ft = f(phi0, t , h, 1)   #f zum Zeitpunkt t
    fth = f(phi0,t+h,h, 0) #f zum Zeitpunkt t+h
    
    
    uheun = vektoru + 0.5*h*(ft + fth)
    
    return uheun


def exakt(phi0,t , h):
    
    
    t=t+h
    
    startx2 = math.cos(phi0)*l
    if(phi0 >= 0):
        startx1 = math.sqrt(math.pow(l,2) - math.pow(startx2,2))
    else:
        startx1 = -math.sqrt(math.pow(l,2) - math.pow(startx2,2))
        
    x1 = startx1*math.cos(t*math.sqrt(g/l)) #x1 zum Zeitpunkt t+h
    x2 = math.sqrt(math.pow(l,2) - math.pow(x1,2)) #x2 zumZeitpunkt t+h
        
        
    v1 = -startx1* math.sqrt(g/l) *math.sin(t*math.sqrt(g/l))        #v1 = x1'
    v2 = -(x1*v1)/x2   #v2 = x2' 
    
    u = np.array([x1, x2, v1, v2])
    return u



#----------------------------------------------------------------------------------------------------
    
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

SE = 50000
iteration = 0
optimizer = torch.optim.Adam(model.parameters()) #,lr=learning_rate)


### Trainingsdaten erstellen
h = 0.1
Ntrain = 1000
phi_np = 0.01*np.random.random(Ntrain)
t_np   = np.random.random(Ntrain)

xtrain_np = np.zeros( (Ntrain,8) )
for i in range(Ntrain):
    xtrain_np[i,0:4] = u(phi_np[i],t_np[i])
    xtrain_np[i,4:8] = heun(phi_np[i],t_np[i],h)

ytrain_np = np.zeros( (Ntrain,4) )
for i in range(Ntrain):
    ytrain_np[i,:] = exakt(phi_np[i],t_np[i],h)

x_train = torch.Tensor(xtrain_np)
y_train = torch.Tensor(ytrain_np)


# Train
for t in range(SE):
    
    
    
    y_pred = model(x_train)
    loss = loss_fn(y_pred,y_train)
    if iteration%100 == 0:
        print(iteration+1,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    iteration=iteration+1



    
    



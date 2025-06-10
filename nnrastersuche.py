import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

dtype = torch.float
device = torch.device("cpu")


# phi0 Startauslenkung
# v0 Startgeschwindigkeit
# l: Länge des Pendels

def sign(x):
    
    if(x == 0):
        return 0
    else:
        return x/abs(x)


# u= (x_1,x_2,v_1,v_2)
def u(phi0, v0, l ,m, t): # in Koordinaten
    
    g = 9.81 # Fallbeschleunigung
    
    if(t == 0):
        
        x1 = np.sin(phi0)*l
        x2 = -np.sqrt(l**2 - x1**2)
        
        v2 = -sign(phi0)*np.sin(phi0)*v0
        v1 = sign(v0)*np.sqrt(v0**2 - v2**2)
    
    
    else:
        vektor = output(phi0, v0, g, l, m, t)
        
        x1 = vektor[0]
        x2 = vektor[1]
        v1 = vektor[2]
        v2 = vektor[3]
        
        
    u = np.array([x1, x2, v1, v2])
    return u



#m Masse des Körpers
def Lagrange(phi0, v0, l ,t , m):
    
    g = 9.81
    vektoru = u(phi0, v0, l ,m ,t)
    
    x2 = vektoru[1]
    v1 = vektoru[2]
    v2 = vektoru[3]
    
    lag = m*(math.pow(v1,2) + math.pow(v2,2) + g*x2)/(2*math.pow(l,2))
    
    return lag
    

def matrixA(phi0, v0, l ,t , m):
    
    
    lag = Lagrange(phi0, v0, l ,t , m)
    A  = np.array( ((0,0,-1,0), (0,0,0,-1), (2*lag/m,0,0,0) , (0,2*lag/m,0,0)) )
    
    return A



# u' + A*u = b <=> u' = b - A*u =:f  
def f(phi0,v0, l ,t , m, h, wert):
    
    g = 9.81
    vektoru = u(phi0, v0, l ,m ,t)
    
    b = np.array([0, 0, 0, g])
    A1 = matrixA(phi0, v0, l ,t , m)    #Matrix A zumZeitpunkt t
    A2 = matrixA(phi0, v0,l ,t+h , m)  #Matrix A zum Zeitpunkt t+h
        
    
    if(wert == 1):
        fout = b - np.dot(A1,vektoru)
    else:
        uexplizit = vektoru + h*(b - np.dot(A1,vektoru))
        fout = b - np.dot(A2,uexplizit)
    
    return fout
    
    
# Zeitschritt h
def heun(phi0,v0, l ,t , m, h):
    
    vektoru = u(phi0,v0, l ,m ,t )
    
    
    ft = f(phi0, v0, l ,t , m, h, 1)   #f zum Zeitpunkt t
    fth = f(phi0, v0, l ,t+h, m, h, 0) #f zum Zeitpunkt t+h
    
    
    uheun = vektoru + 0.5*h*(ft + fth)
    
    return uheun


def polar2lagrange(x,l): # x = [phi,phi']
    phi = x[0]
    vphi = x[1]
    return [l*np.sin(phi), -l * np.cos(phi), l * np.cos(phi) * vphi, l *np.sin(phi) * vphi]


#
# (Vx, Vy) = (-y, x) * phi' => (Vx,Vy)*(-y,x) = l^2 * phi'
# phi' = (-Vx*y + Vy*x)/l^2

def lagrange2polar(lag,l): # x = [phi,phi']
    
    return np.array([np.arctan2(lag[0],-lag[1]),(-lag[2]*lag[1]+lag[3]*lag[0])/np.sqrt(lag[0]*lag[0]+lag[1]*lag[1])])

# [U,V]
## V = (vx,vy)   GEschw.
## U = (ux,uy)  Pos.
# U' = V
# U'' = V' = F/m
## Kraft F = < (0,-g), tau> * tau/m , wobau tau der Tangentialvektor ist
# F = -g * tau_y * (taux, tauy)/m
# tau = (-y, x) / sqrt(x^2 + y^2) normierter Tangentialvektor 
# => F = -g * x  * (-y, x)  / (x^2 + y^2)/m

def heunLagrange(lag, g,l,m,h): # lag = [U,V]
    
    ll = lag[0]*lag[0]+lag[1]*lag[1]
    vv = lag[2]*lag[2]+lag[3]*lag[3]

    return np.array([
        lag[0] + h * lag[2], 
        lag[1] + h * lag[3],
        lag[2] + h * (-vv*lag[0] - g * lag[0]*(-lag[1]) ) / ll / m,
        lag[3] + h * (-vv*lag[1] - g * lag[0]*( lag[0]) ) / ll / m
        ])
    


#[phi',phi''] = [(-g/l)*sin(u),phi']
def rk4(X, g, l, m, h):
    phi0 = X[0]
    v0 = X[1]
    
#    return np.array([phi0+h*v0,v0-h*g/l*np.sin(phi0)])
    
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


#[u',v'] = [v,(-g/l)*sin(u)]
def output(phi0, v0, g, l, m, h):
    
    
    #Rungekutta-4
    k1 = h*v0
    l1 = -(g/l)*h*np.sin(phi0)
    
    k2 = h*(v0 + 0.5*l1)
    l2 = -(g/l)*h*np.sin(phi0 + 0.5*k1)
    
    k3 = h*(v0 + 0.5*l2)
    l3 = -(g/l)*h*np.sin(phi0 + 0.5*k2)
    
    k4 = h*(v0 + l3)
    l4 = -(g/l)*h*np.sin(phi0 + k3)
    
    phineu = phi0 + 1/6*(k1 + 2*k2 + 2*k3 + k4) #Winkel zum Zeitpunkt h
    vneu = v0 + 1/6*(l1 + 2*l2 + 2*l3 + l4) #v zum Zeitpunkt h
    
    
    return u(phineu, vneu, l ,m, 0)
    

#print(output(0, 0 ,9.81, 1, 1, 0.1))

def testrungekutta(phi0, v0, l, m, h):
    
    g = 9.81
    #x1,x2
    xachse = np.linspace(-10, 100, 100)
    phitest = np.linspace(-2*np.pi, 2*np.pi, 100, endpoint=True)
    outx1 = np.zeros(100)
    outx2 = np.zeros(100)
    nullf = np.zeros(100)

    for i in range(0, 99 ) :
        outx1[i] = output(phitest[i], v0, g, l, m, h)[0]
        outx2[i] = output(phitest[i], v0, g, l, m, h)[1]
        
        
    plt.scatter(phitest, outx1, s=5)
    plt.scatter(phitest, outx2, s=5)
    plt.plot(phitest, outx1, linestyle='solid', color='blue', label ='x1')
    plt.plot(phitest, outx2, linestyle='solid', color='red', label = 'x2')
    plt.plot(xachse, nullf, color= 'black')
    plt.legend(loc="upper right")
    plt.title("h = "  +str(h))
    plt.xlabel('phi')
    plt.ylabel('phi + h')
    startx, endx = -8, 8
    starty, endy = -15, 15
    plt.axis([startx, endx, starty, endy])
    plt.show()
    
    #v1,v2
    xachse = np.linspace(-10, 100, 100)
    phitest = np.linspace(-2*np.pi, 2*np.pi, 100, endpoint=True)
    outv1 = np.zeros(100)
    outv2 = np.zeros(100)
    nullf = np.zeros(100)

    for i in range(0, 99 ) :
        outv1[i] = output(phitest[i], v0, g, l, m, h)[2]
        outv2[i] = output(phitest[i], v0, g, l, m, h)[3]
        

    plt.scatter(phitest, outv1, s=5)
    plt.scatter(phitest, outv2, s=5)
    plt.plot(phitest, outv1, linestyle='solid', color='blue', label ='v1')
    plt.plot(phitest, outv2, linestyle='solid', color='red', label = 'v2')
    plt.plot(xachse, nullf, color= 'black')
    plt.legend(loc="upper right")
    plt.title("h = "  +str(h))
    plt.xlabel('phi')
    plt.ylabel('v + h')
    startx, endx = -10, 10
    starty, endy = -0.25, 0.25
    plt.axis([startx, endx, starty, endy])
    plt.show()    
    
    
    
def testheun(phi0,v0, l ,t , m, h):
    
    g = 9.81
    #x1,x2
    xachse = np.linspace(-10, 100, 100)
    phitest = np.linspace(-2*np.pi, 2*np.pi, 100, endpoint=True)
    outx1 = np.zeros(100)
    outx2 = np.zeros(100)
    nullf = np.zeros(100)

    for i in range(0, 99 ) :
        outx1[i] = heun(phitest[i],v0, l ,t , m, h)[0]
        outx2[i] = heun(phitest[i],v0, l ,t , m, h)[1]
        
        
    plt.scatter(phitest, outx1, s=5)
    plt.scatter(phitest, outx2, s=5)
    plt.plot(phitest, outx1, linestyle='solid', color='blue', label ='x1')
    plt.plot(phitest, outx2, linestyle='solid', color='red', label = 'x2')
    plt.plot(xachse, nullf, color= 'black')
    plt.legend(loc="upper right")
    plt.title("h = "  +str(h))
    plt.xlabel('phi')
    plt.ylabel('phi + h')
    startx, endx = -8, 8
    starty, endy = -5, 5
    plt.axis([startx, endx, starty, endy])
    plt.show()
    
    
    
    #v1,v2
    xachse = np.linspace(-10, 100, 100)
    phitest = np.linspace(-2*np.pi, 2*np.pi, 100, endpoint=True)
    outv1 = np.zeros(100)
    outv2 = np.zeros(100)
    nullf = np.zeros(100)

    for i in range(0, 99 ) :
        outv1[i] = heun(phitest[i], v0, g, l, m, h)[2]
        outv2[i] = heun(phitest[i], v0, g, l, m, h)[3]
        
        
    plt.scatter(phitest, outv1, s=5)
    plt.scatter(phitest, outv2, s=5)
    plt.plot(phitest, outv1, linestyle='solid', color='blue', label ='v1')
    plt.plot(phitest, outv2, linestyle='solid', color='red', label = 'v2')
    plt.plot(xachse, nullf, color= 'black')
    plt.legend(loc="upper right")
    plt.title("h = "  +str(h))
    plt.xlabel('phi')
    plt.ylabel('v + h')
    startx, endx = -10, 10
    starty, endy = -5, 5
    plt.axis([startx, endx, starty, endy])
    plt.show()    




#----------------------------------------------------------------------------------------------------


D_in,H1,D_out = 8,11,4


model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1,D_out),
       )

loss_fn = torch.nn.MSELoss(reduction='sum')

SE = 5000
iteration = 0
optimizer = torch.optim.Adam(model.parameters(),lr =0.1) #,lr=learning_rate)


# Parameter
g = 1
l = 1
m = 1 
h = 0.1
t = 0 



### Data generation
ndata = 5000
alldata = np.zeros( (ndata, 12) )

for i in range(ndata):
    phi0 = 2.0*(random.random()-0.5) * math.pi/2. # Startauslenkun in [-pi/4,pi/4]
    v0 = (random.random()-0.5) * 2.   # Start-Geschwindigkeit in [-0.1, 0,1]
    
    alldata[i,0:4]  = polar2lagrange(np.array([phi0,v0]),l)
    alldata[i,4:8]  = heunLagrange(polar2lagrange(np.array([phi0,v0]),l),
                                             g, l, m, h)
    alldata[i,8:12] = polar2lagrange(rk4(np.array([phi0,v0]),
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



def simulation_rk4(h,N):
    l = 1.0
    g = 1.0
    m = 1.0
    
    x = np.array([3.1415/3., 0.0])
    X = np.zeros((N+1,2))
    L = np.zeros((N+1,4))

    X[0,:] = x
    L[0,:] = polar2lagrange(x, l)

    T = np.linspace(0,N*h,N+1)
        

    for n in range(N):
        x = rk4(x,g,l,m,h)
        X[n+1,:]=x
        L[n+1,:] = polar2lagrange(x, l)

  #  plt.plot(T,L[:,0]**2+L[:,1]**2,label='rk4')
    #plt.plot(T,L[:,0],label='rk4')
    #plt.plot(T,L[:,1],label='rk4')
    #plt.plot(X[:,0],X[:,1],label='rk4 (phi,Dphi)')
    #plt.plot(L[:,0],L[:,1], label ='rk4')
    #plt.plot(L[:,2],L[:,3],label='rk4')
    plt.plot(T,L[:,3],label='rk4')
    plt.xlabel('t')
    plt.ylabel('vy')
    
    
def simulation_heun(h,N):
    l = 1.0
    g = 1.0
    m = 1.0
    
    x = np.array([3.1415/3., 0.0])
    X = np.zeros((N+1,2))
    L = np.zeros((N+1,4))

    X[0,:] = x
    L[0,:] = polar2lagrange(x, l)

    T = np.linspace(0,N*h,N+1)
        
    lag = polar2lagrange(x,l)
    
    for n in range(N):
        lag = heunLagrange(lag,g,l,m,h)
        x = lagrange2polar(lag,l)
#        x = rk4(x,g,l,m,h)
        X[n+1,:]=x
        L[n+1,:]=lag

#    plt.plot(T,L[:,0]**2+L[:,1]**2,label='heun')
    #plt.plot(T,L[:,0],label='euler')
    #plt.plot(T,L[:,1],label='euler')
    #plt.plot(L[:,2],L[:,3],label='euler')
    #plt.plot(L[:,0],L[:,1],label='euler')
    plt.plot(T,L[:,3],label='euler')


def simulation_nn(h,N):
    l = 1.0
    g = 1.0
    m = 1.0
    
    x = np.array([3.1415/3., 0.0])
    X = np.zeros((N+1,2))
    L = np.zeros((N+1,4))

    X[0,:] = x
    L[0,:] = polar2lagrange(x, l)

    T = np.linspace(0,N*h,N+1)
        
    lag = polar2lagrange(x,l)
    
    for n in range(N):
        
        lagneu = heunLagrange(lag,g,l,m,h)
        minput = np.concatenate((lag,lagneu))
        
        ### ist es vielleicht doch besser
        ### direkt das Ergebnis zu lernen, lag = model..
        ### und oben alldata[8:12] = exakte RK4-Ergebnis
        lag = lag + model(torch.Tensor(minput)).detach().numpy()
        
        x = lagrange2polar(lag,l)
        #x = rk4(x,g,l,m,h)
        X[n+1,:]=x
        L[n+1,:]=lag

#    plt.plot(T,L[:,0]**2+L[:,1]**2,label='nn')
    #plt.plot(T,L[:,0],label='nn')
    #plt.plot(T,L[:,1],label='nn')
    #plt.plot(L[:,2],L[:,3],label='nn')
    plt.plot(L[:,0],L[:,1],label='nn')
    #plt.plot(T,L[:,3],label='nn')
    plt.xlabel('x')
    plt.ylabel('y')

h=0.1
N=400

#simulation_heun(h,N)
#simulation_rk4(h,N)
simulation_nn(h,N)

plt.legend()
plt.show()

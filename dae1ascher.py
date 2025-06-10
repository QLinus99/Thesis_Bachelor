import numpy as np
import math
import matplotlib.pyplot as plt
import random
    
def matrixA():
    
    return np.matrix( ((1, 0, 0), (0, 1, 0), (0, 0, 0)) )
    

def matrixB(lamda,t):
    
     return np.matrix( ((1/(2-t)-lamda, 0, (t-2)*lamda), ((lamda-1)/(t-2), 1, 1-lamda) , (-t-2, 4-math.pow(t,2), 0 )) )
    
    
def vektorf(t):
    
    return (np.matrix( (  (math.exp(t)*((3-t)/(2-t)), 2*math.e**t, -(math.pow(t,2) +t -2)*math.e**t)      ) )).transpose()
    
def korrekt(t):
    
    return (np.matrix( (  math.pow(math.e,t), math.pow(math.e,t), -math.pow(math.e,t)/(2-t)  )   )).transpose()


# Startwert: t0, Schrittweite: h, g2 zu optimierender Parameter 
def naechstez(t0, h, lamda, g2, z1vor, z2vor):
    
    t2 = t0 + 2*h
    
    return np.dot(np.linalg.inv(matrixA()*((1+g2)/h) + matrixB(lamda,t2)), vektorf(t2) + np.dot(matrixA() , (1/h)*((1+2*g2)*z1vor - g2*z2vor))    )
    

# 0.5*sum(|| z(t2) - korrekt(t2) ||^2 _2)
def Error(t0, h, lamda, g2, z1vor, z2vor):
        
    t2 = t0 + 2*h
    wert = naechstez(t0, h, lamda, g2, z1vor, z2vor) - korrekt(t2)
        
    summand = (math.pow(wert[0],2) + math.pow(wert[1],2) + math.pow(wert[2],2))
    
    return 0.5*summand


def plotError(t0, h, lamda, z1vor, z2vor):
    listx = []
    k=-30
    while(k<30):
        k=k+0.5
        listx.append(k)
        
    listy = []
    for x in listx:
        listy.append(Error(t0, h, lamda, x, z1vor, z2vor))
        
    plt.plot(listx,listy)
    startx, endx = -5,5
    starty, endy = 0, 1000
    plt.axis([startx, endx, starty, endy])
    plt.xlabel("h=" + str(h) + "  ,lamda =" + str(lamda))
    plt.show()
    
    
'''   
z_n+2 = (A*((1+g2)/h) + B)^-1  *  (f(t_n+2) + A*(1/h)*(1+2*g2)*z_n+1 -g2*z_n)

z_n+2' = (A*((1+g2)/h) + B)^-1 * A*(1/h)*(2*z_n+1 -z_n) - (A*((1+g2)/h) + B)^-1 * A*(1/h) * (A*((1+g2)/h) + B)^-1 * (f(t_n+2) + A*(1/h)*(1+2*g2)*z_n+1 -g2*z_n)
'''

# E= 0.5*|z(t2)- korrekt(t2)|²_2
# GE = z_0*Gz_0-Gz_0*korrekt_0(t) + z_1*Gz_1-Gz_1*korrekt_1(t) + z_2*Gz_2-Gz_2*korrekt_2(t) 
def gradientError(t0, h, lamda, g2, z1vor, z2vor):
    
    ableitung = 0
    
    t1 = t0 + h
    t2 = t0 + 2*h
    inverse = np.linalg.inv(matrixA()*((1+g2)/h) + matrixB(lamda,t2))
    
    ableitung = np.dot(inverse, np.dot(matrixA() , (1/h)*(2*z1vor-z2vor))) - np.dot(np.dot(np.dot(inverse, (1/h)*matrixA()), inverse), vektorf(t2) + np.dot(matrixA() , (1/h)*((1+2*g2)*z1vor - g2*z2vor)))
      
    ab0 = ableitung[0]
    ab1 = ableitung[1]
    ab2 = ableitung[2]
        
    z = naechstez(t0, h, lamda, g2, z1vor, z2vor)
        
    z0 = z[0]
    z1 = z[1]
    z2 = z[2]
        
    kor0 = korrekt(t2)[0]
    kor1 = korrekt(t2)[1]
    kor2 = korrekt(t2)[2]
        
    return z0*ab0-ab0*kor0 + z1*ab1-ab1*kor1 + z2*ab2-ab2*kor2                                  


def gradientenverfahren(t0, h ,lamda, z1vor, z2vor):
    
    g2 = 0.5  #g2-Start
    
    while(abs(gradientError(t0, h, lamda, g2, z1vor, z2vor)) > 0.0001):
        
        GE = gradientError(t0, h, lamda, g2, z1vor, z2vor)
        d = -GE/abs(GE) #Abstiegsrichtung
        alpha = 1  # Start-Schrittweite             
                       
        
        while(Error(t0, h, lamda, float(g2 + alpha*d), z1vor, z2vor) > Error(t0, h, lamda, g2, z1vor, z2vor) + 0.2*alpha*GE*d): #Armijo-Bedingung
            alpha = 0.5*alpha


        g2 = float(g2 + alpha*d)
        
        
    return g2

def Fehlermin(N, lamda):
    
    h = 1/N
    
    z2vor = korrekt(0)
    z1vor = korrekt(h)
    
    for n in range(N-1):
        
        t = n/N
        
        g2 = gradientenverfahren(t, h ,lamda, z1vor, z2vor)
        
        naechsterwert = naechstez(t, h, lamda, g2, z1vor, z2vor)   
        
        z2vor = z1vor
        z1vor = naechsterwert
        
    return (abs(naechsterwert - korrekt(1))).transpose()



print("Fehler-Ascher")
for lamda in [1, 10, 100]:
        
        for N in [10, 20, 40]:
            
            h=1/N
            
            print("Lamda = " + str(lamda) + "  , N = " +str(N), Fehlermin(N, lamda))












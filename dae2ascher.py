import numpy as np
import math
import matplotlib.pyplot as plt
import random

'''
->   z(t) = (x_1(t),x_2(t),y(t))
    
Modellgleichung: A*z'(t) + B*z(t) = f(t)   <=> A*z'(t) = -B*z(t) + f(t)

BDF2(Allgemein):   
    
     A((1+g_2)*z(t_2) - (1+g_2)*z_(t_1) + g_2*z_(t_0)) = h*(-B*z(t_2) + f(t_2))

<=>  ( A*((1+g_2)/h) + B ) * z(t_2) = f(t_2) + (1/h)*((1+g_2)*z(t_1) - g_2*z(t_0))

<=> z(t_2) = ( A*((1+g_2)/h) + B )^(-1) *( f(t_2) + (1/h)*((1+g_2)*z(t_1) - g_2*z(t_0)))
'''
    
    
def matrixA(t):
    
    return np.matrix( ((1, -t), (0, 0)) )
    

def matrixB(t):
    
     return np.matrix( ( (1 , -(1 + t)), (0 , 1 )  ) )
    
    
def vektorf(t):
    
    return (np.matrix( (  ( 0 , np.sin(t))      ) )).transpose()
    
def korrekt(t):
    
    return (np.matrix( (  np.exp(-t) + t*np.sin(t), np.sin(t)  )   )).transpose()


# Startwert: t0, Schrittweite: h, g2 zu optimierender Parameter 
def nächstez(t0, h, g2, z1vor, z2vor):
    t2 = t0 + 2*h
    
    
    
    return np.dot(np.linalg.inv(matrixA(t2)*((1+g2)/h) + matrixB(t2)), vektorf(t2) + np.dot(matrixA(t2) , (1/h)*((1+2*g2)*z1vor - g2*z2vor))    )
    

# 0.5*sum(|| z(t2) - korrekt(t2) ||² _2)
def Error(t0, h, g2, z1vor, z2vor):
    
        
    t2 = t0 + 2*h
    wert = nächstez(t0, h, g2, z1vor, z2vor) - korrekt(t2)
        
    summand = (math.pow(wert[0],2) + math.pow(wert[1],2) + math.pow(wert[2],2))
    
    return 0.5*summand


def plotError(t0, h, z1vor, z2vor):
    listx = []
    k=-30
    while(k<30):
        k=k+0.5
        listx.append(k)
        
    listy = []
    for x in listx:
        listy.append(Error(t0, h, x, z1vor, z2vor))
        
    plt.plot(listx,listy)
    startx, endx = -5,5
    starty, endy = 0, 1000
    plt.axis([startx, endx, starty, endy])
    plt.xlabel("h=" + str(h))
    plt.show()
    
    
'''   
z_n+2 = (A*((1+g2)/h) + B)^-1  *  (f(t_n+2) + A*(1/h)*(1+2*g2)*z_n+1 -g2*z_n)

z_n+2' = (A*((1+g2)/h) + B)^-1 * A*(1/h)*(2*z_n+1 -z_n) - (A*((1+g2)/h) + B)^-1 * A*(1/h) * (A*((1+g2)/h) + B)^-1 * (f(t_n+2) + A*(1/h)*(1+2*g2)*z_n+1 -g2*z_n)
    
    
'''


# E= 0.5*|z(t2)- korrekt(t2)|²_2
# GE = z_0*Gz_0-Gz_0*korrekt_0(t) + z_1*Gz_1-Gz_1*korrekt_1(t) + z_2*Gz_2-Gz_2*korrekt_2(t) 

def gradientError(t0, h, g2, z1vor, z2vor):
    
    ableitung = 0
    
    t1 = t0 + h
    t2 = t0 + 2*h
    inverse = np.linalg.inv(matrixA(t2)*((1+g2)/h) + matrixB(t2))
    
    ableitung = np.dot(inverse, np.dot(matrixA(t2) , (1/h)*(2*z1vor-z2vor))) - np.dot(np.dot(np.dot(inverse, (1/h)*matrixA(t2)), inverse), vektorf(t2) + np.dot(matrixA(t2) , (1/h)*((1+2*g2)*z1vor - g2*z2vor)))
    
    ab0 = ableitung[0]
    ab1 = ableitung[1]
        
    z = nächstez(t0, h, g2, z1vor, z2vor)
        
    z0 = z[0]
    z1 = z[1]
        
    kor0 = korrekt(t2)[0]
    kor1 = korrekt(t2)[1]
        
    return z0*ab0-ab0*kor0 + z1*ab1-ab1*kor1                                  


def gradientenverfahren(t0, h , z1vor, z2vor):
    
    g2 = 0.5  #g2-Start
    
    while(abs(gradientError(t0, h, g2, z1vor, z2vor)) > 0.0001):
        
        GE = gradientError(t0, h, g2, z1vor, z2vor)
        d = -GE/abs(GE)
        alpha = 1  # Start-Schrittweite             
                       
        
        while(Error(t0, h, float(g2 + alpha*d), z1vor, z2vor) > Error(t0, h, g2, z1vor, z2vor) + 0.2*alpha*GE*d): #Armijo-Bedingung
            alpha = 0.5*alpha


        g2 = float(g2 + alpha*d)
        
        
    return g2


#print(plotError([0], 0.1, 1))
#print(plotError([0], 0.4, 10))
#print(gradientenverfahren([0], 0.4 ,10,korrekt(0),korrekt(1/10)))

def Fehlermin(N):
    
    h = 1/N
    
    z2vor = korrekt(0)
    z1vor = korrekt(h)
    
    for n in range(N-1):
        
        t = n/N
        
        g2 = gradientenverfahren(t, h, z1vor, z2vor)
        
        nächsterwert = nächstez(t, h, g2, z1vor, z2vor)   
        
        z2vor = z1vor
        z1vor = nächsterwert
        
    return (abs(nächsterwert - korrekt(1))).transpose()



print("Fehler-Ascher")
        
for N in [10, 20, 40]:
            
            h=1/N
            
            print(" N = " +str(N), Fehlermin(N))



'''
(1 0 0)   (x_1')    (  lamda-1/(2-t)    0   (2-t)*lamda )   (x_1)   ( (3-t)/(2-t)*exp(t) )
(0 1 0) * (x_2') =  ( (1-lamda)/(t-2)  -1      lamda-1  ) * (x_2) + (     2*exp(t)       )
(0 0 0)   (y'  )    (     t+2          t²-4      0      )   (y  )   ( -(t² +t-2)*exp(t)  )


                                     =:                                      =:
                            
                                 [ A B C ]                                 [p]
                                 [ D E F ]                                 [q]
                                 [ G H I ]                                 [r]

Eulerimp =>

(1)  x_1_n = x_1_n-1 + h*A*x_1_n + h*C*y_n + h*p
(2)  x_2_n = x_2_n-1 + h*D*x_1_n + h*E*x_2_n + h*F*y_n + h*q
(3)  0   = h*G*x_1_n + h*H*x_2_n + h*r

<=>

( 1- h*A   0      -h*C  )  (x_1_n)    ( x_1_n-1 + h*p )
( -h*D    1-h*E   -h*F  )* (x_2_n) =  ( x_2_n-1 + h*q )
( -h*G    -h*H     0    )  (y_n)      (       h*r     )


'''

import math
import numpy as np
import matplotlib.pyplot as plt
import random

def A(t,lamda):
    return lamda - 1/(2-t)

def C(t,lamda):
    return lamda*(2-t)

def D(t,lamda):
    return (1-lamda)/(t-2)

def E(t,lamda):
    return -1

def F(t,lamda):
    return lamda-1

def G(t,lamda):
    return t+2

def H(t,lamda):
    return math.pow(t,2) -4

def p(t):
    return math.exp(t)*((3-t)/(2-t))

def q(t):
    return math.exp(t)*2

def r(t):
    return -(math.pow(t,2) + t -2)*math.exp(t)
    


# N als Anzahl Schritte
def eulerimp(N,lamda):
    
    t=1/N   # Startwert Zeit
    h=1/N   # Schrittweite
    
    Loesung = np.zeros([N+1, 3])
    Loesung[0][0] = 1            #Startwert x_1
    Loesung[0][1] = 1            #Startwert x_2
    Loesung[0][2] = 0            #Startwert y
    
    besetztezeileindex = 0
    
    for n in range(N):
        
        t=(n+1)/N
        
        matrixlinks = np.array([[1 - h*A(t,lamda), 0, -h*C(t,lamda)],
                                 [-h*D(t,lamda), 1-h*E(t,lamda), -h*F(t,lamda)],
                                 [-h*G(t,lamda), -h*H(t,lamda), 0]])
                                
        matrixrechts = np.array([[Loesung[besetztezeileindex][0] + h*p(t)],
                                  [Loesung[besetztezeileindex][1] + h*q(t)],
                                   [h*r(t)] ])
        
        
        loesung = np.array(np.linalg.solve(matrixlinks,matrixrechts))
        Loesung[besetztezeileindex + 1] = loesung.T
        
        besetztezeileindex = besetztezeileindex + 1
        

    return Loesung[N]
        

def exakt():
    return np.array([[math.exp(1) , math.exp(1), -math.exp(1)]])


print("Fehler-EulerImp :")

for lamda in [1, 10, 100]:
        print()
        for N in [10, 20, 40]:
            
            print("Lambda = ",lamda," ,N = " ,N, ", Fehler: ", abs(eulerimp(N,lamda)-exakt()))








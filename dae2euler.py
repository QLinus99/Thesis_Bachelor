import math
import numpy as np
import matplotlib.pyplot as plt
import random
    


# N als Anzahl Schritte
def eulerimp(N):
    
    t=1/N   # Startwert Zeit
    h=1/N   # Schrittweite
    
    Lösung = np.zeros([N+1, 2])
    Lösung[0][0] = 1            #Startwert x
    Lösung[0][1] = 1            #Startwert y
    
    besetztezeileindex = 0
    
    for n in range(N):
        
        t=(n+1)/N
        
        
        Lösung[besetztezeileindex + 1][1] = np.sin(t)
        Lösung[besetztezeileindex + 1][0] = (1/(1+h)) * ( np.sin(t)*(t + h*(1+t) ) + Lösung[besetztezeileindex ][0] - t*Lösung[besetztezeileindex ][1] )
        
        besetztezeileindex = besetztezeileindex + 1
        

    return Lösung[N]
        

def exakt():
    return np.array([[np.exp(-1) -np.sin(-1) , np.sin(1)]])



print("Fehler-EulerImp :")

for N in [10, 20, 40]:
            
    print(" ,N = " ,N, ", Fehler: ", abs(eulerimp(N)-exakt()))































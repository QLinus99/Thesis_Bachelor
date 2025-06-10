import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math

# g Erdbeschleunigung
# l Länge des Fadenpendels
# m Masse des Körpers
# phi0 Winkel zum Zeitpunkt t=0
# phimax = pi/3



def phi(t, g, l, phi0):
    
    return (math.pi/3) * math.sin(math.sqrt(g/l) * t + phi0)

def ab1phi(t, g, l, phi0):
    
    return (math.pi/3) * math.sqrt(g/l) * math.cos(math.sqrt(g/l) * t + phi0)

def ab2phi(t, g, l, phi0):
    
    return -(g/l) * math.sin(phi(t, g, l, phi0))

def x(t, g, l, phi0):
    
    return l*math.sin(phi(t, g, l, phi0))

def ab1x(t, g, l, phi0):
    
    return l*ab1phi(t, g, l, phi0) * math.cos(phi(t, g, l, phi0))

def ab2x(t, g, l, phi0):
    
    return - l*math.pow(ab1phi(t, g, l, phi0),2) * math.sin(phi(t, g, l, phi0)) + l * ab2phi(t, g, l, phi0) * math.cos(phi(t, g, l, phi0))

                                                                                                                       
def y(t, g, l, phi0):
    
    return -l*math.cos(phi(t, g, l, phi0))
    
def ab1y(t, g, l, phi0):
    
    return l*ab1phi(t, g, l, phi0) * math.sin(phi(t, g, l, phi0))

def ab2y(t, g, l, phi0):
    
    return l*math.pow(ab1phi(t, g, l, phi0),2) * math.cos(phi(t, g, l, phi0)) + l * ab2phi(t, g, l, phi0) * math.sin(phi(t, g, l, phi0))


def Lagrange(t, g, l, m, phi0):
    
    return (m/(2*math.pow(l,2))) * math.pow(ab1x(t, g, l, phi0) ,2) + math.pow(ab1y(t, g, l, phi0), 2) + m*g*y(t ,g, l, phi0)

    
    
class MeinNetz(nn.Module):
    
       
    
    # Konstruktor
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.lin1 = nn.Linear(8, 8)
        self.lin2 = nn.Linear(8, 4)
        
    # Aktivierungsfunktionen
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        return x
    
    # Berechnungen
    def num_flat_features(self, x):
        size = x.size(x)[1:]
        num = 1
        for i in range(size):
            num = num*i
        
        return num
    
    
netz = MeinNetz()
#print(netz)



# Hier Eingabe: l, m, h,  phistart
l = 1
m = 1
phistart = math.pi/4
h = 0.1
    
#h ist Schrittweite im Heunverfahren
g = 9.81                                #Erdbeschleunigung
phi0 = phi(0, g, l, phistart)           # Startauslenkung
lag1 = Lagrange(0, g, l, m, phi0)
lag2 = Lagrange(0+h, g, l, m, phi0)
    

#Input
x0 = x(0, g, l, phi0)
y0 = y(0, g, l, phi0)
abx0 = ab1x(0, g, l, phi0)
aby0 = ab1y(0, g, l, phi0)
    
xheun = x0 + 0.5*h*(abx0 + x(h, g, l, phi0))
yheun = y0 + 0.5*h*(aby0 + y(h, g, l, phi0))
xabheun = abx0 + h/(2*m) * (2*lag1*x0 + 2*lag1*(abx0 + h*2*lag2*x(h, g, l, phi0)))
yabheun = aby0 + 1/(2*h) * (g-(2*lag1*y0)/m) + (g-(2*lag1) * (aby0 + h*(g-2*lag2*y(h, g, l, phi0))))




# Netz Lernen

x = [1, 0, 0, 0, 1, 1, 1, 1]
input = Variable(torch.Tensor([x for _ in range(8)]))

out = netz(input)
print(out)
    
#gewünschte Ausgabe
x = [0, 1, 1, 1]
target = Variable(torch.Tensor([x for _ in range(4)]))
criterion = nn.MSELoss()
loss = criterion(out, target)
#print(loss)


netz.zero_grad()
loss.backward()
optimizer = optim.SGD(netz.parameters(), lr = 0.1)
optimizer.step()




        
    


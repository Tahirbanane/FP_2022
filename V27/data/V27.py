import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import scipy.constants as const
import pandas as pd

I, B = np.genfromtxt('bfeld.txt', unpack = True)

def Mag(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

par, cov = curve_fit(Mag, I, B)

errpar = np.sqrt(np.diag(cov))


print(par)
print(errpar)

x = np.linspace(0, 5)

plt.plot(I, B, 'x', color='r', label='Messwerte')
plt.plot(x, Mag(x, *par), '-', color='green', label='Ausgleichskurve')
plt.xlim(0,5)
plt.xlabel(r'Stromstärke I / A')
plt.ylabel(r'Magnetfeldstärke B / mT')
plt.ylim(0, 500)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('Magnetfeld.pdf')
plt.close()

BRot = Mag(5,par[0],par[1], par[2], par[3]) 
FBRot = np.sqrt( (5**3 * errpar[0])**2  +  (5**2 *errpar[1])**2  +  (5*errpar[2])**2 + errpar[3]**2)  
BBlau = Mag(3.2,par[0],par[1], par[2], par[3])  
FBBlau = np.sqrt( (3.2**3 * errpar[0])**2  +  (3.2**2 *errpar[1])**2  +  (3.2*errpar[2])**2 + errpar[3]**2)  
#from PIL import Image
#
#pic_array = np.array(Image.open("IMG_0040.JPG"))
#print(pic_array[100])
#
#img = Image.open('IMG_0040.jpg')
#imgGray = img.convert('L')
#imgGray.save('test_gray.jpg')

#-------- Spektrallinien -----------------------------------------------

h = 6.626 * 10**(-34)
c = 299792458
mu = 9.274 * 10**(-24)
Lambda1 = 643.8 * 10**(-9)          #Rote Wellenlänge
Lambda2 = 480 * 10**(-9)            #Blaue Wellenlänge


def Lam(e,E,F):                                                     #Wellenlängenverschiebung
    return (1/2) * (e / E) * F 

        #------------------ Rot ----------------------------------------

#S = gross delta s
#s = klein delta s

L1 = 4.8916 *10**(-11) 
L2 = 2.6952 *10**(-11) 

Sr1, sr1 = np.genfromtxt('rot1.txt', unpack = True)
Sb1, sb1 = np.genfromtxt('blau1.txt', unpack = True)

Sr1 = unp.uarray(Sr1,10)  #unsicherheit von 10 pixeln
sr1 = unp.uarray(sr1,10)  #unsicherheit von 10 pixeln


Sb1 = unp.uarray(Sb1,10)  #unsicherheit von 10 pixeln
sb1 = unp.uarray(sb1,10)  #unsicherheit von 10 pixeln


LambdaRot = np.mean(Lam(sr1,Sr1,L1))                                    #Wellenlängenverschiebung Rot
LambdaBlau = np.mean(Lam(sb1,Sb1,L2))  

gRot = LambdaRot * ( (h * c) / (mu * BRot*10**(-3) * Lambda1**2))
gBlau = LambdaBlau * ( (h * c) / (mu * BBlau*10**(-3) * Lambda2**2))

def Rel_Abw(u_i, u):
   return abs(u_i - u)/u

print(f'BRot: {BRot} pm {FBRot}')
print(f'LambdaRot: {LambdaRot}')
print(f'gRot: {gRot}')

print(f'BBlau: {BBlau} pm {FBBlau}')
print(f'LambdaBlau: {LambdaBlau}')
print(f'gBlau: {gBlau}')

print(f'relative abweichung grot: {Rel_Abw(0.94 , 1)}')


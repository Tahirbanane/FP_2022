import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
import sympy
import statistics
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.signal import find_peaks
import scipy.constants as const
import pandas as pd


T1, I1, t1 = np.genfromtxt('1,5erSchritte.txt', unpack=True)
T2, I2, t2 = np.genfromtxt('2erSchritte.txt', unpack=True)

T1 += 273.15 #Celsius in Kelvin
T2 += 273.15 #Celsius in Kelvin

print(np.where(np.max(I1)==I1))
print(I1,I1[71],T1[71])

print(np.where(np.max(I2)==I2))

def gerade(x,m,b):
    return m*x+b

def ef(x,a,b,c):
    return a*np.exp(c*x)+b

par2, cov2 = curve_fit(gerade, np.append([T2[1:9]],T2[25:48]), np.log(np.append([I2[1:9]],I2[25:48])))
error2 = np.sqrt(np.diag(cov2))

I2_u = [I2[:5]]

#print(np.append([I2[:5]],(I2[27:49])))

plt.figure()
plt.plot(T2,I2,'.', label = '2er Schritte')
plt.plot(T2,np.exp(gerade(T2,par2[0],par2[1])),'--',label = '2er Fit')

plt.xlabel('T / K')
plt.ylabel(r'I / $10^{-11}$' + 'A')
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('IT2.png')


par1, cov1 = curve_fit(gerade, np.append([T1[12:20]],T1[33:50]), np.log(np.append([I1[12:20]],I1[33:50])))
error1 = np.sqrt(np.diag(cov1))

plt.figure()
plt.plot(T1,I1,'.', label = '1,5er Schritte')
plt.plot(T1,np.exp(gerade(T1,par1[0],par1[1])),'--',label = '1,5er Fit')

plt.xlabel('T / K')
plt.ylabel(r'I / $10^{-11}$' + 'A')
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('IT1.png')


plt.figure()
plt.plot(T1,I1-np.exp(gerade(T1,par1[0],par1[1])),'--',label = 'Ohne Untergrund')

plt.xlabel('T / K')
plt.ylabel(r'I / $10^{-11}$' + 'A')
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('IT1_ohneU.png')

plt.figure()
plt.plot(T2,I2-np.exp(gerade(T2,par2[0],par2[1])),'--',label = 'Ohne Untergrund 2')

plt.xlabel('T / K')
plt.ylabel(r'I / $10^{-11}$' + 'A')
plt.tight_layout()
plt.legend(loc='best')

#plt.show()
plt.savefig('IT2_ohneU.png')
plt.close()


def diff(k):
    j = k
    for i in range(len(j)-1):
        j[i] = k[i+1]-k[i] 
    return ufloat(np.mean(j[:-1]), np.std(j[:-1]))

print((diff(T1)))
print((diff(T2)))


#-------------- Austrittsarbeit --------------


parW, covW = curve_fit(gerade, 1/(T1[20:33]), np.log(I1[20:33]))
errorW = np.sqrt(np.diag(covW))

plt.figure()
plt.plot(1/(T1[20:33]),(I1[20:33]),'.')
plt.plot(1/(T1[20:33]),gerade(T1[20:33],*parW),'--')

plt.yscale('log')
plt.xlabel('1/T / K')
plt.ylabel(r'log(I) / $10^{-11}$' + 'A')
plt.tight_layout()
plt.legend(loc='best')


#plt.show()
#plt.savefig('IT2_ohneU.png')






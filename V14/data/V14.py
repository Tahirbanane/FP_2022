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
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
import math
from numpy.linalg import inv

hoehe = np.genfromtxt('1_I1.txt', unpack = True)

x=np.arange(1,201)
plt.plot(x, hoehe, color='k')
plt.plot([150.5,150.5],[0,hoehe.max()],color='steelblue', ls = '--',label="Photopeak")
plt.plot([107,107],[0,hoehe[107]],color='seagreen', ls = '--',label="Comptonkante")
plt.xlabel(r'Channel')
plt.ylabel(r'Anzahl der Ereignisse')
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig('spektrum.pdf')
plt.clf()

d_s = 3
d_nd = 2*np.sqrt(2)
d_hd = 3*np.sqrt(2)

print('______________________WUERFEL_1_____________________')

N1, t1 = np.genfromtxt('wuerfel1.txt', unpack=True)
#print(f'N: {N1}')
#print(f't:  {t1}')
sigma_N1 = np.sqrt(N1)
sigma_I1 = sigma_N1/t1
I1 = N1/t1
I_1 = unp.uarray(I1,sigma_I1)
#I1_s = []
print(f'Intensitäten:  {I_1}')


print('______________________WUERFEL_2_____________________')

N2, t2 = np.genfromtxt('wuerfel2.txt', unpack=True)
sigma_N2 = np.sqrt(N2)
sigma_I2 = sigma_N2/t2
I2 = N2/t2
I_2 = unp.uarray(I2,sigma_I2)
#print(f'Intensitäten:  {I_2}')

mu2 = [
unp.log((I_1[0])/ I_2[0])/d_s,
unp.log((I_1[1])/ I_2[1])/d_s,
unp.log((I_1[2])/ I_2[2])/d_s,
unp.log((I_1[3])/ I_2[3])/d_s,
unp.log((I_1[4])/ I_2[4])/d_s,
unp.log((I_1[5])/ I_2[5])/d_s,
unp.log((I_1[6])/ I_2[6])/d_nd,
unp.log((I_1[7])/ I_2[7])/d_hd,
unp.log((I_1[8])/ I_2[8])/d_nd,
unp.log((I_1[9])/ I_2[9])/d_nd,
unp.log((I_1[10])/ I_2[10])/d_hd,
unp.log((I_1[11])/ I_2[11])/d_nd,]

mu_2 = [0.04830237364789899, 0.03236999947060798,0.09574796217814617, 0.05477386169433759, 0.15218036532471896, 0.05871592458816722,0.15723682240494055, 0.10296499760634775, 0.224142699461198]
mumit = np.mean(mu_2)

print(f'mu 2: {mu2}')
print(f'mumit: {mumit} pm {np.std(mu_2)}')

print('______________________WUERFEL_3_____________________')

N3, t3 = np.genfromtxt('wuerfel3.txt', unpack=True)
sigma_N3 = np.sqrt(N3)
sigma_I3 = sigma_N3/t3
I3 = N3/t3
I_3 = unp.uarray(I3,sigma_I3)

mu3 = [
unp.log((I_1[0])/  I_3[0])/d_s,
unp.log((I_1[1])/  I_3[1])/d_s,
unp.log((I_1[2])/  I_3[2])/d_s,
unp.log((I_1[3])/  I_3[3])/d_nd,
unp.log((I_1[4])/  I_3[4])/d_hd,
unp.log((I_1[5])/  I_3[5])/d_nd]

mu_3 = np.mean(mu3)

print(f'mu 3: {mu3}')
print(f'Intensitäten: {I_3}')
print(f'mu 3 mit: {mu_3} pm ')

print('______________________WUERFEL_4_____________________')

N4, t4 = np.genfromtxt('wuerfel4.txt', unpack=True)
sigma_N4 = np.sqrt(N4)
sigma_I4 = sigma_N4/t4
I4 = N4/t4
I_4 = unp.uarray(I4,sigma_I4)
I_44 = unp.log(I_1/I_4)

print(I_44)

sig = np.sqrt((sigma_I1/I1)**2 + (sigma_I4/I4)**2)
w = 1/sig

s = np.sqrt(2)
A = np.matrix([[1,1,1,0,0,0,0,0,0],  #1
               [0,0,0,1,1,1,0,0,0],  #2
               [0,0,0,0,0,0,1,1,1],  #3
               [1,0,0,1,0,0,1,0,0],  #4
               [0,1,0,0,1,0,0,1,0],  #5
               [0,0,1,0,0,1,0,0,1],  #6
               [0,s,0,0,0,s,0,0,0],  #7
               [s,0,0,0,s,0,0,0,s],  #8
               [0,0,0,s,0,0,0,s,0],  #9
               [0,s,0,s,0,0,0,0,0],  #10
               [0,0,s,0,s,0,s,0,0],  #11
               [0,0,0,0,0,s,0,s,0]]) #12

W = np.matrix([[25.00248873,0,0,0,0,0,0,0,0,0,0,0],
               [0,24.6284284,0,0,0,0,0,0,0,0,0,0],
               [0,0,25.65005396,0,0,0,0,0,0,0,0,0],
               [0,0,0,30.20628824,0,0,0,0,0,0,0,0],
               [0,0,0,0,13.58411268,0,0,0,0,0,0,0],
               [0,0,0,0,0,32.77695568,0,0,0,0,0,0],
               [0,0,0,0,0,0,28.23666601,0,0,0,0,0],
               [0,0,0,0,0,0,0,21.62563331,0,0,0,0],
               [0,0,0,0,0,0,0,0,24.32966509,0,0,0],
               [0,0,0,0,0,0,0,0,0,28.95341285,0,0],
               [0,0,0,0,0,0,0,0,0,0,17.20233187,0],
               [0,0,0,0,0,0,0,0,0,0,0,23.15372575]])

A_T = np.transpose(A)

B = np.dot(A_T,W)

C = np.dot(B,A)

V = np.linalg.inv(C)

D = np.dot(V,A_T)
E = np.dot(D,W)
mu4 = np.dot(E,I_44)
print(f'mu4: {mu4}')
print(f'gewichtselemente: {w}')
 
print(f'Intensitäten: {I_4}')
 
#print(f'varianz: {V}')

#--------------------Vergleich mit Literaturwerten--------------------#

Eisen = 0.578
Aluminium = 0.202
Blei = 1.245
Blei_Exp = 1.05
Messing = 0.620
Delrin = 0.118
mu2g=0.1029 
mu3g = 0.9827 

#print(f"\n")
#print(f"Abweichung zu Eisen: \n {Abk_W4 - Eisen} \n")
#print(f"Abweichung zu Aluminium: \n {Abk_W4 - Aluminium} \n")
#print(f"Abweichung zu Blei: \n {Abk_W4 - Blei} \n")
#print(f"Abweichung zu Blei experimentell: \n {Abk_W4 - Blei_Exp} \n")
#print(f"Abweichung zu Messing: \n {Abk_W4 - Messing} \n")
#print(f"Abweichung zu Delrin: \n {Ab - Delrin} \n")
#
def Rel_Abw(u_i, u):
   return abs(u_i - u)/u

print(f"Relative Abweichung mu_2: \n {Rel_Abw(mu2g, Delrin)} \n")
print(f"Relative Abweichung mu_3: \n {Rel_Abw(mu3g, Blei)} \n")
print('--------------------Würfel 4 abweichung------------------')
#print(f"Relative Abweichung mu_1: \n {Rel_Abw(2.011, Blei)} \n")
#print(f"Relative Abweichung mu_2: \n {Rel_Abw(0.5621, Eisen)} \n")
#print(f"Relative Abweichung mu_3: \n {Rel_Abw(2.2843, Blei)} \n")
#print(f"Relative Abweichung mu_6: \n {Rel_Abw(4.7007, Blei)} \n")
#print(f"Relative Abweichung mu_7: \n {Rel_Abw(3.66125, Blei)} \n")
#print(f"Relative Abweichung mu_8: \n {Rel_Abw(2.779, Blei)} \n")
#print(f"Relative Abweichung mu_9: \n {Rel_Abw(3.64122, Blei)} \n")

print(f"Relative Abweichung mu_1: \n {Rel_Abw(0.1825, Aluminium)} \n")
print(f"Relative Abweichung mu_2: \n {Rel_Abw(0.6057, Messing)} \n")
print(f"Relative Abweichung mu_3: \n {Rel_Abw(0.3663, Aluminium)} \n")
print(f"Relative Abweichung mu_5: \n {Rel_Abw(1.2781, Blei)} \n")
print(f"Relative Abweichung mu_7: \n {Rel_Abw(0.1432, Delrin)} \n")
print(f"Relative Abweichung mu_8: \n {Rel_Abw(0.9774, Blei)} \n")

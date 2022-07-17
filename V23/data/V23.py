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


x, y = np.genfromtxt('Wasserstoffatom_1.dat', unpack = True)

plt.figure()
plt.plot(x, y)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
#plt.legend()
plt.savefig('../pic/Wasserstoffatom.pdf')
plt.close()


#ineffizentes einlesen von daten
x_2300_0,   y_2300_0     = np.genfromtxt('2300Hz/0.dat', unpack = True)
x_2300_10,  y_2300_10    = np.genfromtxt('2300Hz/10.dat', unpack = True)
x_2300_20,  y_2300_20    = np.genfromtxt('2300Hz/20.dat', unpack = True)
x_2300_30,  y_2300_30    = np.genfromtxt('2300Hz/30.dat', unpack = True)
x_2300_40,  y_2300_40    = np.genfromtxt('2300Hz/40.dat', unpack = True)
x_2300_50,  y_2300_50    = np.genfromtxt('2300Hz/50.dat', unpack = True)
x_2300_60,  y_2300_60    = np.genfromtxt('2300Hz/60.dat', unpack = True)
x_2300_70,  y_2300_70    = np.genfromtxt('2300Hz/70.dat', unpack = True)
x_2300_80,  y_2300_80    = np.genfromtxt('2300Hz/80.dat', unpack = True)
x_2300_90,  y_2300_90    = np.genfromtxt('2300Hz/90.dat', unpack = True)
x_2300_100, y_2300_100   = np.genfromtxt('2300Hz/100.dat', unpack = True)
x_2300_110, y_2300_110   = np.genfromtxt('2300Hz/110.dat', unpack = True)
x_2300_120, y_2300_120   = np.genfromtxt('2300Hz/120.dat', unpack = True)
x_2300_130, y_2300_130   = np.genfromtxt('2300Hz/130.dat', unpack = True)
x_2300_140, y_2300_140   = np.genfromtxt('2300Hz/140.dat', unpack = True)
x_2300_150, y_2300_150   = np.genfromtxt('2300Hz/150.dat', unpack = True)
x_2300_160, y_2300_160   = np.genfromtxt('2300Hz/160.dat', unpack = True)
x_2300_170, y_2300_170   = np.genfromtxt('2300Hz/170.dat', unpack = True)
x_2300_180, y_2300_180   = np.genfromtxt('2300Hz/180.dat', unpack = True)

x_3700_0,   y_3700_0     = np.genfromtxt('3700Hz/0.dat', unpack = True)
x_3700_10,  y_3700_10    = np.genfromtxt('3700Hz/10.dat', unpack = True)
x_3700_20,  y_3700_20    = np.genfromtxt('3700Hz/20.dat', unpack = True)
x_3700_30,  y_3700_30    = np.genfromtxt('3700Hz/30.dat', unpack = True)
x_3700_40,  y_3700_40    = np.genfromtxt('3700Hz/40.dat', unpack = True)
x_3700_50,  y_3700_50    = np.genfromtxt('3700Hz/50.dat', unpack = True)
x_3700_60,  y_3700_60    = np.genfromtxt('3700Hz/60.dat', unpack = True)
x_3700_70,  y_3700_70    = np.genfromtxt('3700Hz/70.dat', unpack = True)
x_3700_80,  y_3700_80    = np.genfromtxt('3700Hz/80.dat', unpack = True)
x_3700_90,  y_3700_90    = np.genfromtxt('3700Hz/90.dat', unpack = True)
x_3700_100, y_3700_100   = np.genfromtxt('3700Hz/100.dat', unpack = True)
x_3700_110, y_3700_110   = np.genfromtxt('3700Hz/110.dat', unpack = True)
x_3700_120, y_3700_120   = np.genfromtxt('3700Hz/120.dat', unpack = True)
x_3700_130, y_3700_130   = np.genfromtxt('3700Hz/130.dat', unpack = True)
x_3700_140, y_3700_140   = np.genfromtxt('3700Hz/140.dat', unpack = True)
x_3700_150, y_3700_150   = np.genfromtxt('3700Hz/150.dat', unpack = True)
x_3700_160, y_3700_160   = np.genfromtxt('3700Hz/160.dat', unpack = True)
x_3700_170, y_3700_170   = np.genfromtxt('3700Hz/170.dat', unpack = True)
x_3700_180, y_3700_180   = np.genfromtxt('3700Hz/180.dat', unpack = True)

x_4963_0,   y_4963_0     = np.genfromtxt('4963Hz/0.dat', unpack = True)
x_4963_10,  y_4963_10    = np.genfromtxt('4963Hz/10.dat', unpack = True)
x_4963_20,  y_4963_20    = np.genfromtxt('4963Hz/20.dat', unpack = True)
x_4963_30,  y_4963_30    = np.genfromtxt('4963Hz/30.dat', unpack = True)
x_4963_40,  y_4963_40    = np.genfromtxt('4963Hz/40.dat', unpack = True)
x_4963_50,  y_4963_50    = np.genfromtxt('4963Hz/50.dat', unpack = True)
x_4963_60,  y_4963_60    = np.genfromtxt('4963Hz/60.dat', unpack = True)
x_4963_70,  y_4963_70    = np.genfromtxt('4963Hz/70.dat', unpack = True)
x_4963_80,  y_4963_80    = np.genfromtxt('4963Hz/80.dat', unpack = True)
x_4963_90,  y_4963_90    = np.genfromtxt('4963Hz/90.dat', unpack = True)
x_4963_100, y_4963_100   = np.genfromtxt('4963Hz/100.dat', unpack = True)
x_4963_110, y_4963_110   = np.genfromtxt('4963Hz/110.dat', unpack = True)
x_4963_120, y_4963_120   = np.genfromtxt('4963Hz/120.dat', unpack = True)
x_4963_130, y_4963_130   = np.genfromtxt('4963Hz/130.dat', unpack = True)
x_4963_140, y_4963_140   = np.genfromtxt('4963Hz/140.dat', unpack = True)
x_4963_150, y_4963_150   = np.genfromtxt('4963Hz/150.dat', unpack = True)
x_4963_160, y_4963_160   = np.genfromtxt('4963Hz/160.dat', unpack = True)
x_4963_170, y_4963_170   = np.genfromtxt('4963Hz/170.dat', unpack = True)
x_4963_180, y_4963_180   = np.genfromtxt('4963Hz/180.dat', unpack = True)

x_7400_0,   y_7400_0     = np.genfromtxt('7400Hz/0.dat', unpack = True)
x_7400_10,  y_7400_10    = np.genfromtxt('7400Hz/10.dat', unpack = True)
x_7400_20,  y_7400_20    = np.genfromtxt('7400Hz/20.dat', unpack = True)
x_7400_30,  y_7400_30    = np.genfromtxt('7400Hz/30.dat', unpack = True)
x_7400_40,  y_7400_40    = np.genfromtxt('7400Hz/40.dat', unpack = True)
x_7400_50,  y_7400_50    = np.genfromtxt('7400Hz/50.dat', unpack = True)
x_7400_60,  y_7400_60    = np.genfromtxt('7400Hz/60.dat', unpack = True)
x_7400_70,  y_7400_70    = np.genfromtxt('7400Hz/70.dat', unpack = True)
x_7400_80,  y_7400_80    = np.genfromtxt('7400Hz/80.dat', unpack = True)
x_7400_90,  y_7400_90    = np.genfromtxt('7400Hz/90.dat', unpack = True)
x_7400_100, y_7400_100   = np.genfromtxt('7400Hz/100.dat', unpack = True)
x_7400_110, y_7400_110   = np.genfromtxt('7400Hz/110.dat', unpack = True)
x_7400_120, y_7400_120   = np.genfromtxt('7400Hz/120.dat', unpack = True)
x_7400_130, y_7400_130   = np.genfromtxt('7400Hz/130.dat', unpack = True)
x_7400_140, y_7400_140   = np.genfromtxt('7400Hz/140.dat', unpack = True)
x_7400_150, y_7400_150   = np.genfromtxt('7400Hz/150.dat', unpack = True)
x_7400_160, y_7400_160   = np.genfromtxt('7400Hz/160.dat', unpack = True)
x_7400_170, y_7400_170   = np.genfromtxt('7400Hz/170.dat', unpack = True)
x_7400_180, y_7400_180   = np.genfromtxt('7400Hz/180.dat', unpack = True)

print(np.amax(y_2300_0))

max_2300 = [
   np.amax( y_2300_0 ),
   np.amax( y_2300_10),
   np.amax( y_2300_20),    
   np.amax( y_2300_30),
   np.amax( y_2300_40),
   np.amax( y_2300_50),
   np.amax( y_2300_60),
   np.amax( y_2300_70),
   np.amax( y_2300_80),
   np.amax( y_2300_90),
   np.amax(y_2300_100),
   np.amax(y_2300_110),
   np.amax(y_2300_120),
   np.amax(y_2300_130),
   np.amax(y_2300_140),
   np.amax(y_2300_150),
   np.amax(y_2300_160),
   np.amax(y_2300_170),
   np.amax(y_2300_180)
]

#korrektur da nicht immer die gleiche stelle die größte stelle war

max_2300 = [
    y_2300_0 [19],
    y_2300_10[19],
    y_2300_20[19],    
    y_2300_30[19],
    y_2300_40[19],
    y_2300_50[19],
    y_2300_60[19],
    y_2300_70[19],
    y_2300_80[19],
    y_2300_90[19],
   y_2300_100[19],
   y_2300_110[19],
   y_2300_120[19],
   y_2300_130[19],
   y_2300_140[19],
   y_2300_150[19],
   y_2300_160[19],
   y_2300_170[19],
   y_2300_180[19]
]

print(np.where(y_2300_180 == 2000))

max_3700 = [
    np.amax(y_3700_0),
    np.amax(y_3700_10 ),
    np.amax(y_3700_20 ),    
    np.amax(y_3700_30 ),
    np.amax(y_3700_40 ),
    np.amax(y_3700_50 ),
    np.amax(y_3700_60 ),
    np.amax(y_3700_70 ),
    np.amax(y_3700_80 ),
    np.amax(y_3700_90 ),
    np.amax(y_3700_100),
    np.amax(y_3700_110),
    np.amax(y_3700_120),
    np.amax(y_3700_130),
    np.amax(y_3700_140),
    np.amax(y_3700_150),
    np.amax(y_3700_160),
    np.amax(y_3700_170),
    np.amax(y_3700_180)
]

max_4963 = [
    np.amax(y_4963_0),
    np.amax(y_4963_10 ),
    np.amax(y_4963_20 ),    
    np.amax(y_4963_30 ),
    np.amax(y_4963_40 ),
    np.amax(y_4963_50 ),
    np.amax(y_4963_60 ),
    np.amax(y_4963_70 ),
    np.amax(y_4963_80 ),
    np.amax(y_4963_90 ),
    np.amax(y_4963_100),
    np.amax(y_4963_110),
    np.amax(y_4963_120),
    np.amax(y_4963_130),
    np.amax(y_4963_140),
    np.amax(y_4963_150),
    np.amax(y_4963_160),
    np.amax(y_4963_170),
    np.amax(y_4963_180)
]

max_7400 = [
    np.amax(y_7400_0),
    np.amax(y_7400_10 ),
    np.amax(y_7400_20 ),    
    np.amax(y_7400_30 ),
    np.amax(y_7400_40 ),
    np.amax(y_7400_50 ),
    np.amax(y_7400_60 ),
    np.amax(y_7400_70 ),
    np.amax(y_7400_80 ),
    np.amax(y_7400_90 ),
    np.amax(y_7400_100),
    np.amax(y_7400_110),
    np.amax(y_7400_120),
    np.amax(y_7400_130),
    np.amax(y_7400_140),
    np.amax(y_7400_150),
    np.amax(y_7400_160),
    np.amax(y_7400_170),
    np.amax(y_7400_180)
]

# ----------- nicht genutzt von hier----------------------
#r = np.linspace(0, 1, len(max_2300))
#theta =  np.pi * r
#
#print(theta,np.flip(theta))
#
#plt.figure()
#plt.polar(theta, max_2300/np.amax(max_2300), color='orange')
#plt.polar(np.flip(theta)+np.pi, max_2300/np.amax(max_2300), color='orange')
#plt.title('2300Hz')
#plt.savefig('../pic/2300.pdf')
#
#plt.figure()
#plt.polar(theta, max_3700/np.amax(max_3700), color='orange')
#plt.polar(np.flip(theta)+np.pi, max_3700/np.amax(max_3700), color='orange')
#plt.title('3700Hz')
#plt.savefig('../pic/3700.pdf')
#
#plt.figure()
#plt.polar(theta, max_4963/np.amax(max_4963), color='orange')
#plt.polar(np.flip(theta)+np.pi, max_4963/np.amax(max_4963), color='orange')
#plt.title('4963Hz')
#plt.savefig('../pic/5000.pdf')
#
#plt.figure()
#plt.polar(theta, max_7400/np.amax(max_7400) , color='orange')
#plt.polar(np.flip(theta)+np.pi, max_7400/np.amax(max_7400) , color='orange')
#plt.title('7400Hz')
#plt.savefig('../pic/7400.pdf')
##plt.show()

# ----------- nicht genutzt bis hier----------------------

arrays = [max_2300,max_3700,max_4963,max_7400]

fig,ax = plt.subplots(2,2,figsize=(6.4,6.4),dpi=300,subplot_kw={'projection': 'polar'})
count = 0
k = 0
name_array = ['2288Hz', '3682Hz','4963Hz','7410Hz']

print(name_array[0])

for i in (0,1):
    for j in (0,1):
        rho = np.concatenate([arrays[count],arrays[count][::-1],arrays[count],arrays[count][::-1]])
        r = np.linspace(1, 2, len(rho))
        theta = 2 * np.pi * r
        theorie = [abs(np.cos(theta-np.pi/2)),abs(1/3*(3*np.cos(theta-np.pi/2)**2-1))/0.6,abs(np.cos(theta-np.pi/2)*(5*np.cos(theta-np.pi/2)**2-3))/2,
                    abs(63*np.cos(theta-np.pi/2)**5-70*np.cos(theta-np.pi/2)**3+15*np.cos(theta-np.pi/2))/7.5]
        #ax[i,j].set_rlim(0,35)
        ax[i,j].set_rticks([5, 15, 25])  # Less radial ticks
        ax[i,j].set_title('Resonanzstelle ~' + name_array[k], va='bottom')
        #ax[i,j].set_title_position(-22.5)     
        ax[i,j].set_rlabel_position(-22.5) 
        ax[i,j].plot(theta,theorie[k], '--')
        ax[i,j].plot(theta,rho/np.amax(rho))
        count = count+1
        k = k+1
fig.tight_layout()
fig.savefig('../pic/all.pdf')


#-------------Aufspaltung der Zustände-------------



x3, y3 = np.genfromtxt('3mm.dat', unpack = True)
x6, y6 = np.genfromtxt('6mm.dat', unpack = True)
x9, y9 = np.genfromtxt('9mm.dat', unpack = True)

plt.figure()
plt.plot(x3, y3, label = 'Ring = 3mm')
plt.plot(x6, y6, label = 'Ring = 6mm')
plt.plot(x9, y9, label = 'Ring = 9mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.legend(loc='best')
plt.tight_layout()
plt.grid()
#plt.legend()
plt.savefig('../pic/3mm.pdf')

#diff3 = np.amax(y3[np.arange(0,440,1)])
#print(diff3)

diff3 = abs(x3[np.where(np.amax(y3) == y3 )] - x3[np.where(12.888 == y3 )] )
diff6 = abs(x6[np.where(np.amax(y6) == y6 )] - x6[np.where(12.681 == y6 )])
diff9 = abs(x9[np.where(np.amax(y9) == y9 )] - x9[np.where(8.523  == y9 )])
 


plt.figure()
plt.plot([3,6,9], [diff3,diff6,diff9], 'x')
plt.plot([3,6,9], [diff3,diff6,diff9])
plt.ylabel(r'Frequenzdifferenz in Hz')
plt.xlabel(r'Ringdicke in mm')
plt.tight_layout()
plt.grid()
#plt.legend()
plt.savefig('../pic/aufspaltung.pdf')




#-------------Winkelabhängigkeit-------------

#code dreist von https://github.com/komrozik/FP2021/blob/main/V23-Quanten_Analogien/plot.py übernommen
def cre_polar(data,name):
    plt.close()
    #rho = np.concatenate([data,data[::-1]])
    theta = np.linspace(0, np.pi, len(data))
    a=np.arange(0,185,10)
    plt.polar(theta,data/max(data),color = 'orange')
    plt.polar(theta+np.pi,np.flip(data)/max(data) ,color = 'orange')
    plt.savefig("../pic/polar_"+f"{name}.pdf")
    plt.close()

x00_9,  y00_9 = np.genfromtxt('2300Hz_9mm/0.dat', unpack = True)
x10_9,  y10_9 = np.genfromtxt('2300Hz_9mm/10.dat', unpack = True)
x20_9,  y20_9 = np.genfromtxt('2300Hz_9mm/20.dat', unpack = True)
x30_9,  y30_9 = np.genfromtxt('2300Hz_9mm/30.dat', unpack = True)
x40_9,  y40_9 = np.genfromtxt('2300Hz_9mm/40.dat', unpack = True)
x50_9,  y50_9 = np.genfromtxt('2300Hz_9mm/50.dat', unpack = True)
x60_9,  y60_9 = np.genfromtxt('2300Hz_9mm/60.dat', unpack = True)
x70_9,  y70_9 = np.genfromtxt('2300Hz_9mm/70.dat', unpack = True)
x80_9,  y80_9 = np.genfromtxt('2300Hz_9mm/80.dat', unpack = True)
x90_9,  y90_9 = np.genfromtxt('2300Hz_9mm/90.dat', unpack = True)
x100_9, y100_9 = np.genfromtxt('2300Hz_9mm/100.dat', unpack = True)
x110_9, y110_9 = np.genfromtxt('2300Hz_9mm/110.dat', unpack = True)
x120_9, y120_9 = np.genfromtxt('2300Hz_9mm/120.dat', unpack = True)
x130_9, y130_9 = np.genfromtxt('2300Hz_9mm/130.dat', unpack = True)
x140_9, y140_9 = np.genfromtxt('2300Hz_9mm/140.dat', unpack = True)
x150_9, y150_9 = np.genfromtxt('2300Hz_9mm/150.dat', unpack = True)
x160_9, y160_9 = np.genfromtxt('2300Hz_9mm/160.dat', unpack = True)
x170_9, y170_9 = np.genfromtxt('2300Hz_9mm/170.dat', unpack = True)
x180_9, y180_9 = np.genfromtxt('2300Hz_9mm/180.dat', unpack = True)


max_9 = [
   np.amax(y00_9),
   np.amax(y10_9),
   np.amax(y20_9),    
   np.amax(y30_9),
   np.amax(y40_9),
   np.amax(y50_9),
   np.amax(y60_9),
   np.amax(y70_9),
   np.amax(y80_9),
   np.amax(y90_9),
   np.amax(y100_9),
   np.amax(y110_9),
   np.amax(y120_9),
   np.amax(y130_9),
   np.amax(y140_9),
   np.amax(y150_9),
   np.amax(y160_9),
   np.amax(y170_9),
   np.amax(y180_9)
]

max_9 = [
    y00_9[461],
    y10_9[461],
    y20_9[461],    
    y30_9[461],
    y40_9[461],
    y50_9[461],
    y60_9[461],
    y70_9[461],
    y80_9[461],
    y90_9[461],
   y100_9[461],
   y110_9[461],
   y120_9[461],
   y130_9[461],
   y140_9[461],
   y150_9[461],
   y160_9[461],
   y170_9[461],
   y180_9[461]
]

#a=np.arange(0,185,10)#winkel
cre_polar(max_9,"9mm")


x_10, y_10 = np.genfromtxt('Wasserstoffmolekuel_10mm.dat', unpack = True)
x_20, y_20 = np.genfromtxt('Wasserstoffmolekuel_20mm.dat', unpack = True)

x00_15,  y00_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/0.dat', unpack = True)
x10_15,  y10_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/10.dat', unpack = True)
x20_15,  y20_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/20.dat', unpack = True)
x30_15,  y30_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/30.dat', unpack = True)
x40_15,  y40_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/40.dat', unpack = True)
x50_15,  y50_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/50.dat', unpack = True)
x60_15,  y60_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/60.dat', unpack = True)
x70_15,  y70_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/70.dat', unpack = True)
x80_15,  y80_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/80.dat', unpack = True)
x90_15,  y90_15 =  np.genfromtxt('Wasserstoffmolekuel_15mm_angle/90.dat', unpack = True)
x100_15, y100_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/100.dat', unpack = True)
x110_15, y110_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/110.dat', unpack = True)
x120_15, y120_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/120.dat', unpack = True)
x130_15, y130_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/130.dat', unpack = True)
x140_15, y140_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/140.dat', unpack = True)
x150_15, y150_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/150.dat', unpack = True)
x160_15, y160_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/160.dat', unpack = True)
x170_15, y170_15 = np.genfromtxt('Wasserstoffmolekuel_15mm_angle/170.dat', unpack = True)
x180_15, y180_15 = np.genfromtxt('Wasserstoffmolekuel_15mm.dat', unpack = True)

plt.close()
plt.figure()
plt.plot(x_20, y_20, label = '20mm Blende')
plt.plot(x180_15, y180_15, label = '15mm Blende')
plt.plot(x_10, y_10, label = '10mm Blende')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.legend()
plt.savefig('../pic/101520.pdf')

max_294 = [
   np.amax(y00_15 [94]),
   np.amax(y10_15 [94]),
   np.amax(y20_15 [94]),    
   np.amax(y30_15 [94]),
   np.amax(y40_15 [94]),
   np.amax(y50_15 [94]),
   np.amax(y60_15 [94]),
   np.amax(y70_15 [94]),
   np.amax(y80_15 [94]),
   np.amax(y90_15 [94]),
   np.amax(y100_15[94]),
   np.amax(y110_15[94]),
   np.amax(y120_15[94]),
   np.amax(y130_15[94]),
   np.amax(y140_15[94]),
   np.amax(y150_15[94]),
   np.amax(y160_15[94]),
   np.amax(y170_15[94]),
   np.amax(y180_15[94])
]

max_298 = [
    np.amax(y00_15 [99]),
    np.amax(y10_15 [99]),
    np.amax(y20_15 [99]),    
    np.amax(y30_15 [99]),
    np.amax(y40_15 [99]),
    np.amax(y50_15 [99]),
    np.amax(y60_15 [99]),
    np.amax(y70_15 [99]),
    np.amax(y80_15 [99]),
    np.amax(y90_15 [99]),
    np.amax(y100_15[99]),
    np.amax(y110_15[99]),
    np.amax(y120_15[99]),
    np.amax(y130_15[99]),
    np.amax(y140_15[99]),
    np.amax(y150_15[99]),
    np.amax(y160_15[99]),
    np.amax(y170_15[99]),
    np.amax(y180_15[99])
]

max_369 = [
   np.amax(y00_15 [170]),
   np.amax(y10_15 [170]),
   np.amax(y20_15 [170]),    
   np.amax(y30_15 [170]),
   np.amax(y40_15 [170]),
   np.amax(y50_15 [170]),
   np.amax(y60_15 [170]),
   np.amax(y70_15 [170]),
   np.amax(y80_15 [170]),
   np.amax(y90_15 [170]),
   np.amax(y100_15[170]),
   np.amax(y110_15[170]),
   np.amax(y120_15[170]),
   np.amax(y130_15[170]),
   np.amax(y140_15[170]),
   np.amax(y150_15[170]),
   np.amax(y160_15[170]),
   np.amax(y170_15[170]),
   np.amax(y180_15[170])
]

cre_polar(max_294,"max_294")
cre_polar(max_298,"max_298")
cre_polar(max_369,"max_369")

#------------festkörper------------

x_2_16, y_2_16   = np.genfromtxt('festkoerper/2x50mm_16mmBlende.dat', unpack = True) 
x_4_16, y_4_16   = np.genfromtxt('festkoerper/4x50mm_3x16mmBlende.dat', unpack = True)
x_10_16, y_10_16 = np.genfromtxt('festkoerper/10x50mm_9x16mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x_2_16, y_2_16   , label = '2x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_16mm/2_16.pdf')

plt.figure()
plt.plot(x_4_16, y_4_16   , label = '4x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_16mm/4_16.pdf')

plt.figure()
plt.plot(x_10_16, y_10_16 , label = '10x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_16mm/10_16.pdf')

#------------10mm-----------
x_2_10, y_2_10   = np.genfromtxt('festkoerper/2x50mm_10mmBlende.dat', unpack = True) 
x_4_10, y_4_10   = np.genfromtxt('festkoerper/4x50mm_3x10mmBlende.dat', unpack = True)
x_10_10, y_10_10 = np.genfromtxt('festkoerper/10x50mm_9x10mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x_2_10, y_2_10   , label = '2x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_10mm/2_10.pdf')

plt.figure()
plt.plot(x_4_10, y_4_10   , label = '4x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_10mm/4_10.pdf')

plt.figure()
plt.plot(x_10_10, y_10_10 , label = '10x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_10mm/10_10.pdf')

#---------13mm--------

x_2_13, y_2_13   = np.genfromtxt('festkoerper/2x50mm_13mmBlende.dat', unpack = True) 
x_4_13, y_4_13   = np.genfromtxt('festkoerper/4x50mm_3x13mmBlende.dat', unpack = True)
x_10_13, y_10_13 = np.genfromtxt('festkoerper/10x50mm_9x13mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x_2_13, y_2_13   , label = '2x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_13mm/2_13.pdf')

plt.figure()
plt.plot(x_4_13, y_4_13   , label = '4x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_13mm/4_13.pdf')

plt.figure()
plt.plot(x_10_13, y_10_13 , label = '10x50mm')
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_13mm/10_13.pdf')


#-------Fehlstellen--------

x_9_10_75,  y_9_10_75   = np.genfromtxt('festkoerper/9x50mm_75mm_9x16mmBlende.dat', unpack = True) #wurde von 10mm zu 16mm blende in den daten geändert und damit es zu keinen problemen kommt wurde der variablenname biebehalten
x_9_10_625, y_9_10_625 = np.genfromtxt('festkoerper/9x50mm_50plus25halbemm_9x16mmBlende.dat', unpack = True)
x_9_10_375, y_9_10_375 = np.genfromtxt('festkoerper/9x50mm_37komma5mm_9x16mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x_9_10_75, y_9_10_75)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_stoerstelle/75mm.pdf')

plt.figure()
plt.plot(x_9_10_625, y_9_10_625)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_stoerstelle/62,5mm.pdf')

plt.figure()
plt.plot(x_9_10_375, y_9_10_375)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/festkoerper_stoerstelle/37,5mm.pdf')

#--------Zylinderwechsel---------

x_5_16, y_5_16 = np.genfromtxt('festkoerper/5x50mm_5x75mm_9x16mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x_5_16, y_5_16)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/abwechselnd50u75.pdf')

x50, y50  = np.genfromtxt('festkoerper/50mm.dat', unpack = True)
plt.figure()
plt.plot(x50, y50)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/nur50.pdf')

x75, y75  = np.genfromtxt('festkoerper/75mm.dat', unpack = True)

plt.figure()
plt.plot(x75, y75)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/nur75.pdf')

#---------13 u 16 abwechselnd--------

x13_16, y13_16  = np.genfromtxt('festkoerper/8x75mm_4x16mmBlende_3x13mmBlende.dat', unpack = True)

plt.figure()
plt.plot(x13_16, y13_16)
plt.xlabel(r'Frequenz in Hz')
plt.ylabel(r'Amplitude')
plt.tight_layout()
plt.grid()
plt.savefig('../pic/13u16.pdf')

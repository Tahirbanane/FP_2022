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


#from PIL import Image
#
#pic_array = np.array(Image.open("IMG_0040.JPG"))
#print(pic_array[100])
#
#img = Image.open('IMG_0040.jpg')
#imgGray = img.convert('L')
#imgGray.save('test_gray.jpg')
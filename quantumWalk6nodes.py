#!/usr/bin/python3
#https://github.com/mgarbellini/Quantum-Walks-Time-Dependent-Hamiltonians/blob/main/Code/Schroedinger_Solver.py
import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping, shgo, dual_annealing
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import sqrtm

global dimension
global H

def generate_hamiltonian():

    laplacian = np.empty([dimension, dimension],dtype=complex)
    laplacian.fill(0)

    laplacian[0,0]=1
    laplacian[0,1]=-1

    laplacian[1,0]=-1
    laplacian[1,1]=3
    laplacian[1,3]=-1
    laplacian[1,4]=-1

    laplacian[2,2]=1
    laplacian[2,3]=-1

    laplacian[3,2]=-1
    laplacian[3,3]=2
    laplacian[3,5]=-1

    laplacian[4,1]=-1
    laplacian[4,4]=1

    laplacian[5,3]=-1
    laplacian[5,5]=1

    hamiltonian=laplacian#.transpose()
    return hamiltonian

#routine to implement schroedinger equation. returns d/dt(psi)
#for the ODE solver
def schrodinger_equation(t, y):
    derivs = []
    psi = 0
    for i in range(dimension):
        for j in range(dimension):
            psi += H[i,j]*y[j]
        derivs.append(-1j*psi)
        psi = 0

    return derivs

#schroedinger equation solver. returns psi(t)
def solve_schrodinger_equation(time):

    y0 = np.empty(dimension, dtype=complex)
    y0.fill(1/(np.sqrt(dimension)))

    sh_solved = solve_ivp(schrodinger_equation, [0., time], y0, method='RK45')
    psi_t = np.empty(dimension,dtype=complex)
    for i in range(dimension):
        psi_t[i] = sh_solved.y[i, len(sh_solved.y[i])-1]

    normalization = np.dot(np.conj(psi_t), psi_t)
    print(str(st)+";"+str(linalg.norm(psi_t[0]))+";"+str(linalg.norm(psi_t[1]))+";"+str(linalg.norm(psi_t[2]))+";"+str(linalg.norm(psi_t[3]))+";"+str(linalg.norm(psi_t[4]))+";"+str(linalg.norm(psi_t[5])))
    return psi_t,normalization.real

dimension = 6
H=generate_hamiltonian()
print('H')
H=H.transpose()
print(H)
evals,evecs=linalg.eig(H)
print(evals)
print('EVECS')
print(evecs)
i=0
zz=np.empty([dimension,dimension],dtype=np.double)
zz.fill(0)
phi=[]
for aD in range(0,dimension):
  aPhi=np.empty([dimension])
  aPhi.fill(0)
  aPhi[0]=evecs[0][aD]
  aPhi[1]=evecs[1][aD]
  aPhi[2]=evecs[2][aD]
  aPhi[3]=evecs[3][aD]
  aPhi[4]=evecs[4][aD]
  aPhi[5]=evecs[5][aD]
  phi.append(aPhi)
  zz=zz+np.outer(aPhi,aPhi)
print('PHI')
for ap in phi:
  print("..",ap)
print(zz)
nu=sqrtm(zz)
nuInv=linalg.inv(nu)
H=H.transpose()
H=nu.dot(H.dot(nuInv))
print(H)
for i in range(0,160):
  st=float(i)/float(5)
  rez,norm=solve_schrodinger_equation(st)

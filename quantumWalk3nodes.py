#!/usr/bin/python3
#https://github.com/mgarbellini/Quantum-Walks-Time-Dependent-Hamiltonians/blob/main/Code/Schroedinger_Solver.py
import sys
import time
import numpy as np
from scipy import linalg
from scipy.optimize import minimize, basinhopping, shgo, dual_annealing
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import sqrtm

#useful global variables, shouldn't be too inefficient
global dimension
global H

#routine to generate loop hamiltonian + oracle state
def generate_hamiltonian():

    laplacian = np.empty([dimension, dimension],dtype=complex)
    laplacian.fill(0)

    laplacian[0,0]=1
    laplacian[0,1]=-1
    laplacian[0,2]=-1
    laplacian[1,0]=-1
    laplacian[1,1]=1
    laplacian[1,2]=-1
    laplacian[2,2]=2
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
#    print(st,'Normalization:',normalization.real,psi_t)
    print(st,linalg.norm(psi_t[0]),linalg.norm(psi_t[1]),linalg.norm(psi_t[2]))#,normalization)
    return psi_t,normalization.real
    #return normalization.real

dimension = 3
H=generate_hamiltonian()
print('pseudo H')
print(H)
H=H.transpose()
evals,evecs=linalg.eig(H)
print("eigenvalues",evals)
print("eigenvectors")
for ev in evecs:
  print("vec",ev)
i=0
zz=np.empty([dimension,dimension],dtype=np.double)
zz.fill(0)
phi=[]
for aD in range(0,dimension):
  aPhi=np.empty([3])
  aPhi.fill(0)
  aPhi[0]=evecs[0][aD]
  aPhi[1]=evecs[1][aD]
  aPhi[2]=evecs[2][aD]
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
for i in range(0,50):
  st=float(i)/float(10)
  rez,norm=solve_schrodinger_equation(st)

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
    laplacian[0,2]=-1
    laplacian[0,3]=-1

    laplacian[1,1]=2
    laplacian[1,0]=-1

    laplacian[2,2]=1
    laplacian[2,1]=-1

    laplacian[3,3]=1
    laplacian[3,1]=-1

    laplacian[4,5]=-1

    laplacian[5,5]=1

    hamiltonian=laplacian
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
    normalization = np.abs(np.dot(np.conj(psi_t), psi_t))**2
#    norm=np.dot(np.conj(psi_t), psi_t)
    sigma=linalg.norm(psi_t[0])+linalg.norm(psi_t[1])+linalg.norm(psi_t[2])+linalg.norm(psi_t[3])+linalg.norm(psi_t[4])+linalg.norm(psi_t[5])
    print(st,linalg.norm(psi_t[0])/sigma,linalg.norm(psi_t[1])/sigma,linalg.norm(psi_t[2])/sigma,linalg.norm(psi_t[3])/sigma,linalg.norm(psi_t[4])/sigma,linalg.norm(psi_t[5])/sigma)#,normalization)#,normalization)
    return psi_t,normalization.real

dimension = 6
H=generate_hamiltonian()
print('pseudo H')
print(H)
_,eigenvectors=linalg.eig(H)
eigenvectors=eigenvectors
print("!!",eigenvectors,"!!")
result = np.zeros((dimension, dimension),dtype=complex)
for i in range(eigenvectors.shape[1]):
  outer_product = np.outer(eigenvectors[:, i], eigenvectors[:, i])
  result += outer_product
print(result)
print('square')
print(sqrtm(result))
nu=sqrtm(result)
nuInv=linalg.inv(nu)
H=H.transpose()
H=nu.dot(H.dot(nuInv))
print(H)
for i in range(0,600):
  st=float(i)/float(10)
  rez,norm=solve_schrodinger_equation(st)

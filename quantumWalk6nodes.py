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
    print(st,'Normalization:',normalization.real,psi_t)
    print(st,linalg.norm(psi_t[0]),linalg.norm(psi_t[1]),linalg.norm(psi_t[2]))
    return psi_t,normalization.real
    #return normalization.real


#zz=np.empty([3,3])
#zz.fill(0)
#zz[0,0]=2
#zz[0,2]=1
#zz[1,1]=2
#zz[1,2]=1
#zz[2,0]=1
#zz[2,1]=1
#zz[2,2]=2
#print(zz)
#nu=sqrtm(zz)
#nuInv=linalg.inv(nu)
#print(nu)
#print(nuInv)
dimension = 6
H=generate_hamiltonian()
print('H')
#print(H)
H=H.transpose()
print(H)
evals,evecs=linalg.eig(H)
print(evals)
print('EVECS')
print(evecs)
i=0
zz=np.empty([dimension,dimension],dtype=np.double)
zz.fill(0)
phi1=np.empty([dimension])
phi1.fill(0)
phi1[0]=evecs[0][0]
phi1[1]=evecs[1][0]
phi1[2]=evecs[2][0]
phi1[3]=evecs[3][0]
phi1[4]=evecs[4][0]
phi1[5]=evecs[5][0]
phi2=np.empty([dimension])
phi2.fill(0)
phi2[0]=evecs[0][1]
phi2[1]=evecs[1][1]
phi2[2]=evecs[2][1]
phi2[3]=evecs[3][1]
phi2[4]=evecs[4][1]
phi2[5]=evecs[5][1]
phi3=np.empty([dimension])
phi3.fill(0)
phi3[0]=evecs[0][2]
phi3[1]=evecs[1][2]
phi3[2]=evecs[2][2]
phi3[3]=evecs[3][2]
phi3[4]=evecs[4][2]
phi3[5]=evecs[5][2]

phi4=np.empty([dimension])
phi4.fill(0)
phi4[0]=evecs[0][3]
phi4[1]=evecs[1][3]
phi4[2]=evecs[2][3]
phi4[3]=evecs[3][3]
phi4[4]=evecs[4][3]
phi4[5]=evecs[5][3]
phi5=np.empty([dimension])
phi5.fill(0)
phi5[0]=evecs[0][4]
phi5[1]=evecs[1][4]
phi5[2]=evecs[2][4]
phi5[3]=evecs[3][4]
phi5[4]=evecs[4][4]
phi5[5]=evecs[5][4]
phi6=np.empty([dimension])
phi6.fill(0)
phi6[0]=evecs[0][5]
phi6[1]=evecs[1][5]
phi6[2]=evecs[2][5]
phi6[3]=evecs[3][5]
phi6[4]=evecs[4][5]
phi6[5]=evecs[5][5]

print(phi1)
print(phi2)
print(phi3)
print(phi4)
print(phi5)
print(phi6)



if np.amin(phi3)!=0:
  phi3=phi3/np.amax(phi3)
if np.amin(phi2)!=0:
  phi2=phi2/np.amax(phi2)
if np.amin(phi1)!=0:
  phi1=phi1/np.amax(phi1)

if np.amin(phi4)!=0:
  phi4=phi4/np.amax(phi4)
if np.amin(phi5)!=0:
  phi5=phi5/np.amax(phi5)
if np.amin(phi6)!=0:
  phi6=phi6/np.amax(phi6)


print('PHI')
print(phi1)
print(phi2)
print(phi3)
print('PHI',phi4,phi5,phi6)
zz=zz+np.outer(phi1,phi1)
zz=zz+np.outer(phi2,phi2)
zz=zz+np.outer(phi3,phi3)

zz=zz+np.outer(phi4,phi4)
zz=zz+np.outer(phi5,phi5)
zz=zz+np.outer(phi6,phi6)
print(zz)
nu=sqrtm(zz)
nuInv=linalg.inv(nu)
H=H.transpose()
H=nu.dot(H.dot(nuInv))
print(H)
normMin=999999999999999999999999999999999999.0
normMax=0.0
for i in range(0,80):
  st=float(i)/float(20)
  rez,norm=solve_schrodinger_equation(st)
#  print("..",norm,normMin,normMax)
  if norm<normMin:
    normMin=norm
  if norm>normMax:
    normMax=norm
#print("delta",normMin,normMax,abs(normMin-normMax))

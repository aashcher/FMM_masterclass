# © Alexey A. Shcherbakov, 2025

import numpy as np
from matplotlib import pyplot as plt

from fmm1d_mc import *

np.set_printoptions(precision=2)

wavelength = 0.6328
polarization = 'p'
angle_incidence = 30
epsilon_substrate = 2.25
epsilon_superstrate = 1.0

problem = DiffractionGratingProblem(wavelength, polarization, angle_incidence, epsilon_substrate, epsilon_superstrate)

nx = 100
nz = 50

number_harmonics = 20
method = FourierMethod(number_harmonics)

V_pw = method.plane_wave('from above')

############################################

period = 1.2
fill_factors = np.array([0.45, 0.55], dtype=float)
phc_epsilons = np.array([1.5**2, 1.0], dtype=complex)
slab_thickness = 0.24
grating = PhotonicCrystalSlab(period, fill_factors, phc_epsilons, slab_thickness)
fmm = FourierModalMethod(number_harmonics)

S, betas, M2, E_eig, H_eig = fmm.get_smatrix_phc_slab(problem, grating)
V_out = fmm.diffract(S, V_pw)

F, G, Z = fmm.local_field(problem, grating, V_pw, betas, E_eig, H_eig, M2, nx, nz)
V = V_pw * np.exp(-1j*np.sqrt(problem.epsilon_superstrate - problem.kx_inc**2)*grating.depth*2.0*np.pi/problem.wavelength)
V[0,:] = V_out[1,:]
Fa, Ga, Za = method.layer_field(problem, grating.depth, V, wavelength/grating.period, epsilon_superstrate, nx, nz)
V[1,:] = V_out[0,:]
V[0,:] = 0.0
Fb, Gb, Zb = method.layer_field(problem, grating.depth, V, wavelength/grating.period, epsilon_substrate, nx, nz)

F_tot = np.concatenate([Fb, F])
F_tot = np.concatenate([F_tot, Fa])
G_tot = np.concatenate([Gb, G])
G_tot = np.concatenate([G_tot, Ga])
Z_tot = np.concatenate([Zb, Z])
Z_tot = np.concatenate([Z_tot, Za])

###########################################
# 2d field
height = slab_thickness
x = np.linspace(0,period*(1.0-1.0/nx),nx)
y = np.linspace(-height+height*0.5/nz,height*(2.0-0.5/nz),3*nz)
X,Y = np.meshgrid(x,y)

#plt.pcolor(X,Y,np.imag(G_tot)) # +G*G+Z*Z
plt.pcolor(X,Y,np.abs(F_tot*F_tot)) # 
#plt.pcolor(X,Y,np.abs(G_tot*G_tot+Z_tot*Z_tot)) # 

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()

plt.show()

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 13:49:33 2022
This code distorts framework and rotates molecules in complex structures
@author: Popoola
"""
#%%Import Modules
import numpy as np
import pickle 
import pandas as pd
import re
from itertools import islice
from os.path import exists
import os
import subprocess
import os.path
import math
np.set_printoptions(formatter={'float':           "{0:0.6f}".format})

#%%Parameters
nAtom = 60
# nAtom = 6
nSteps = 10
#%%Get inpu files
#Read input files: StructA, StructB and input
home = "D:/Downloads/Polarization_reversal_code/distortion_code_python"
with open(home+"/alatA", 'w') as f1, open(home+"/alatB", 'w') as f2:
        fA = open(home+"/StructA", 'r'); fB = open(home+"/StructB", 'r')
        f1.write("".join(islice(fA, 3))); f2.write("".join(islice(fB, 3))) #islice(iterable, stop)
        fA.close(); fB.close()
        
with open(home+"/CoordA", 'w') as g1, open(home+"/CoordB", 'w') as g2:
        gA = open(home+"/StructA", 'r'); gB = open(home+"/StructB", 'r')
        g1.write("".join(islice(gA, 3, nAtom+3, 1))); g2.write("".join(islice(gB, 3, nAtom+3, 1))) #islice(iterable, stop)
        gA.close(); gB.close()
        
vectA = np.genfromtxt(home+'/alatA',skip_header=False,dtype='unicode')
vectA = vectA.astype("float32")
vectB = np.genfromtxt(home+'/alatB',skip_header=False,dtype='unicode')
vectB = vectB.astype("float32")

np.testing.assert_array_equal(vectA, vectB) #To make sure vectA == vectB
alat = vectA
a1 = alat[0,:]; a2 = alat[1,:]; a3 = alat[2,:]

a = np.linalg.norm(a1);  b = np.linalg.norm(a2); c = np.linalg.norm(a3) #Lattice parameters

#angle between b and c
b_dot_c = a2.dot(a3); mag_b = np.linalg.norm(a2); mag_c = np.linalg.norm(a3)
cos_alpha = b_dot_c/(mag_b*mag_c)
alpha = np.arccos(cos_alpha)
alpha = np.degrees(alpha)
alpha = round(alpha, 2)

#Angle between a and c
a_dot_c = a1.dot(a3); mag_a = np.linalg.norm(a1); mag_c = np.linalg.norm(a3)
cos_beta = a_dot_c/(mag_a*mag_c)
beta = (np.arccos(cos_beta))*180/np.pi
beta = round(beta, 2)

#Angle between a and b
a_dot_b = a1.dot(a2); mag_a = np.linalg.norm(a1); mag_b = np.linalg.norm(a2)
cos_gamma = a_dot_b/(mag_a*mag_b)
gamma = (np.arccos(cos_gamma))*180/np.pi
gamma = round(gamma, 2)


print('alpha is ', alpha)
print('beta is ', beta)
print('gamma is ', gamma)
if alpha == beta == gamma == 90.00:
    print("The structure is orthorhombic")

if alpha != 90.00 or beta != 90.00 or gamma != 90.00:
    print("The structure is Monoclinic")

if alpha != beta != gamma != 90.00:
    print("The structure is Triclinic")
#%%Distortion
alat_inv = np.linalg.inv(alat)

fileA = np.genfromtxt(home+'/CoordA',skip_header=False,dtype='unicode')
labA = fileA[:,0]; clustA = fileA[:,1]; pos_redA = fileA[:,3:6]; labSymbA = fileA[:,8]
fileB = np.genfromtxt(home+'/CoordB',skip_header=False,dtype='unicode')
labB = fileB[:,0]; clustB = fileB[:,1]; pos_redB = fileB[:,3:6]; labSymbB = fileB[:,8]
clustB = clustB.astype("int"); clustA = clustA.astype("int")
pos_redA = pos_redA.astype("float32"); pos_redB = pos_redB.astype("float32")
pos_cartA = np.zeros((nAtom, 3)); pos_cartB = np.zeros((nAtom, 3))
for i in range(nAtom):
    pos_cartA[i,:] = alat.dot(pos_redA[i,:])
    pos_cartB[i,:] = alat.dot(pos_redB[i,:])
    
distort = np.zeros((nAtom, 3))
dr_save = np.zeros((nAtom, 3))
for iCountB in range(nAtom):
    if clustB[iCountB] == 0:
        rr_Min = 1000.00
        for iCountA in range(nAtom):
            if clustA[iCountB] == 0 and labSymbA[iCountA] == labSymbA[iCountB]:
                dr =  pos_redA[iCountA, :] - pos_redB[iCountB, :]
                # dr_save[iCountB,:] = dr
# print(dr_save)
                for j in range(3):
                    if abs(dr[j]) > 0.5:
                        dr[j] = dr[j] - math.copysign(1, dr[j])
                # dr_save[iCountB,:] = dr
# print(dr_save)
                
                dr_cart = np.matmul(alat, (pos_redB[iCountB, :]+dr)) - np.matmul(alat, pos_redB[iCountB, :])
                rr = dr_cart.dot(dr_cart)
                if rr < rr_Min:
                    distort[iCountB,:] = dr
                    rr_Min = rr
# print(distort)
# distort = np.transpose(distort)                   
# distort = distort.reshape(nAtom,3)
distort = distort/nSteps
#%%Create intermediate, but distortive structures
for iStep in range(nSteps+1):
    print("step", iStep)
    "****Create .cif files to store coordinates along the path*****"
    with open(home+"/step"+str(iStep)+'.cif', 'w') as save_D:
        save_D.write('#CIF file generated from distortion of frameworks for hyrids##\n')
        save_D.write('##############################################################\n\n')
        save_D.write('_cell_length_a              ' + str('%.8f' %a) + '\n')
        save_D.write('_cell_length_b              ' + str('%.8f' %b) + '\n')
        save_D.write('_cell_length_c              ' + str('%.8f' %c) + '\n')
        save_D.write('_cell_angle_alpha           ' + str('%.4f' %alpha) + '\n')
        save_D.write('_cell_angle_beta            ' + str('%.4f' %beta) +'\n')
        save_D.write('_cell_angle_gamma           ' + str('%.4f' %gamma) + '\n')
        save_D.write('_space_group_IT_number      ' + str(1) + '\n')
        save_D.write('loop_\n')
        save_D.write('_space_group_symop_operation_xyz\n')
        save_D.write("'x, y, z'\n")
        save_D.write('loop_\n')
        save_D.write('_atom_site_label\n')
        save_D.write('_atom_site_occupancy\n')
        save_D.write('_atom_site_fract_x\n')
        save_D.write('_atom_site_fract_y\n')
        save_D.write('_atom_site_fract_z\n')
        save_D.write('_atom_site_adp_type\n')
        save_D.write('_atom_site_U_iso_or_equiv\n')
        save_D.write('_atom_site_type_symbol\n')
        
        
        "****Distortion of framework ONLY!*******"
        pos_final = np.zeros((nAtom, 3))
        for iCount in range(nAtom):
            if clustB[iCount] == 0:
                new_pos = pos_redB[iCount, :] + distort[iCount,:]*iStep
                pos_a = new_pos[0]; pos_b = new_pos[1]; pos_c = new_pos[2]             
                save_D.write(str(labB[iCount])+'   \t'+ '1.0'+'   \t'
                              + str('%.8f' %pos_a).replace('[','').replace(']','') +'   \t'
                              + str('%.8f' %pos_b).replace('[','').replace(']','') +'   \t'
                              + str('%.8f' %pos_c).replace('[','').replace(']','') +'   \t'+
                              'Biso' +'   \t'+ str(1.0) +'   \t'+ labSymbB[iCount] +'\n')
                if (iStep == nSteps):
                    pos_final[iCount,:] = new_pos
       
                    
        "****Rotation of molecules in between*******"
# print(pos_final)
# "****Checking for correct mapping of atoms*******"
# for iCountB in range(nAtom):
#     rr_Min = 1000.00
#     for iCountA in range(nAtom):
#         dr =  pos_redA[iCountA, :] - pos_final[iCountB, :]
#         # print(pos_redA[iCountA, :])
#         print(pos_final[iCountB, :])
#         for j in range(3):
#             if abs(dr[j]) > 0.5:
#                 dr[j] = dr[j] - math.copysign(1, dr[j])
#             rr = dr_cart.dot(dr_cart)
#             if rr < rr_Min:
#                 rr_Min = rr
#             #print(rr_Min)



#%%Remove redundant files
if exists(home+"/alatA"):
    os.remove(home+"/alatA")
if exists(home+"/alatB"):
    os.remove(home+"/alatB")
if exists(home+"/CoordA"):
    os.remove(home+"/CoordA")
if exists(home+"/CoordB"):
    os.remove(home+"/CoordB")
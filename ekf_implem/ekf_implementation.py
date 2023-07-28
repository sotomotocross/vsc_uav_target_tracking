from __future__ import print_function
from numpy.core.fromnumeric import size
from numpy.core.numeric import ones
import sys
import time
import json
import os
import csv
import numpy as np
import string
import matplotlib.pyplot as plt
import math

vec = np.zeros((14535, 15))
with open('pred_data.csv') as file1:
    csv_reader1 = csv.reader(file1, delimiter=',')
    with open('vsc_data.csv') as file2:
        csv_reader2 = csv.reader(file2, delimiter=',')
        ln = -1
        for row in csv_reader2:
            if ln >= 0:
                if ln == 0:
                    time_lag = float(row[10])
                vec[ln][0] = float(row[10]) - time_lag
                vec[ln][9] = float(row[8].strip('(').strip(')').strip(','))
                vec[ln][10] = float(row[9].strip('(').strip(')').strip(','))
                vec[ln][11] = float(row[7].split(',')[0].strip('('))
                vec[ln][12] = float(row[7].split(',')[1])
                vec[ln][13] = float(row[7].split(',')[2])
                vec[ln][14] = float(row[7].split(',')[3].strip(')'))
            ln += 1

    ln = -1
    for row in csv_reader1:
        if (ln >= 0 and ln < 14535):
            vec[ln][1] = int(row[5].split(',')[0].strip('('))
            vec[ln][2] = int(row[5].split(',')[1].strip(')'))
            vec[ln][3] = int(row[6].split(',')[0].strip('('))
            vec[ln][4] = int(row[6].split(',')[1].strip(')'))
            vec[ln][5] = int(row[7].split(',')[0].strip('('))
            vec[ln][6] = int(row[7].split(',')[1].strip(')'))
            vec[ln][7] = int(row[8].split(',')[0].strip('('))
            vec[ln][8] = int(row[8].split(',')[1].strip(')'))
        ln += 1


P = 100.0*np.eye(4, dtype=float)
x_est = np.array([0.0, 0.0, 1.0, 1.0],  dtype=float).reshape(4, 1)
Ts = 0.0335

data = np.zeros((14535, 4))

time = vec[:,0].reshape(14535,1)
z1 = vec[:,9].reshape(14535,1)
ekf_position_input = vec[:,9].reshape(14535,1)
print("ekf_position_input :", ekf_position_input)
z2 = vec[:,10].reshape(14535,1)
ekf_velocity_input = vec[:,10].reshape(14535,1)
print("ekf_velocity_input :", ekf_velocity_input)

ekf_position_output_online = vec[:,11].reshape(14535,1)
print("ekf_position_output_online :", ekf_position_output_online)
ekf_velocity_output_online = vec[:,12].reshape(14535,1)
print("ekf_velocity_output_online :", ekf_velocity_output_online)
ekf_frequency_output_online = vec[:,13].reshape(14535,1)
print("ekf_frequency_output_online :", ekf_frequency_output_online)
ekf_offset_output_online = vec[:,14].reshape(14535,1)
print("ekf_offset_output_online :", ekf_offset_output_online)


for i in range(len(z1)):

    PHI_S1 = 0.001 #Try to change these e.g. 0.1
    PHI_S2 = 0.001 #Try to change these e.g. 0.1

    F_est = np.zeros((4, 4), dtype=float)
    F_est [0,1] = 1.0
    F_est [1,0] = -x_est[2]**2
    F_est [1,2] = -2.0*x_est[2]*x_est[0] + 2.0*x_est[2]*x_est[3]
    F_est [1,3] = x_est[2]**2
    # print("F_est: ", F_est)

    PHI_k = np.eye(4,4) + F_est*Ts
    # print("PHI_k: ", PHI_k)

    Q_k = np.array([[0.001, 0.0, 0.0, 0.0],
                    [0.0, 0.5*PHI_S1*(-2.0*x_est[2]*x_est[0]+2.0*x_est[2]*x_est[3])**2*(Ts**2)+0.333*x_est[2]**4*(Ts**3)*PHI_S2, 0.0, 0.5*x_est[2]**2*(Ts**2)*PHI_S2],
                    [0.0, 0.5*PHI_S1*(-2.0*x_est[2]*x_est[0]+2.0*x_est[2]*x_est[3])**2*(Ts**2), PHI_S1*Ts, 0.0],
                    [0.0, 0.5*PHI_S2*(Ts**2)*x_est[2]**2, 0.0, PHI_S2*Ts]], dtype=float).reshape(4,4)
    # print("Q_k: ", Q_k)

    H_k = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0]], dtype=float).reshape(2,4)
    # print("H_k: ", H_k)

    R_k = 0.1*np.eye(2, dtype=float)
    # print("R_k: ", R_k)

    M_k = np.dot(np.dot(PHI_k,P), PHI_k.transpose()) + Q_k
    # print("M_k: ", M_k)
    K_k = np.dot(np.dot(M_k, H_k.transpose()), np.linalg.inv(np.dot(np.dot(H_k, M_k), H_k.transpose()) + R_k) )
    # print("K_k: ", K_k)
    P = np.dot((np.eye(4) - np.dot(K_k, H_k)), M_k)
    # print("P: ", P)

    xdd = -((x_est[2]**2)*x_est[0]) + (x_est[2]**2)*x_est[3]
    # print("xdd: ", xdd)
    xd = x_est[1] + Ts*xdd
    # print("xd: ", xd)
    x = x_est[0] + Ts*xd
    # print("x: ", x)

    x_dash_k = np.array([x, xd, x_est[2], x_est[3]]).reshape(4,1)
    # print("x_dash_k: ", x_dash_k)
    
    z = np.array([z1[i], z2[i]]).reshape(2,1)
        
    x_est = np.dot(PHI_k, x_dash_k) + np.dot(K_k, z - np.dot(H_k, np.dot(PHI_k, x_dash_k)))
    # print("x_est: ", x_est)
    # print("x_est[1] :", x_est[0])
    # print("x_est[2] :", x_est[1])
    # print("x_est[3] :", x_est[2])
    # print("x_est[4] :", x_est[3])
    # data[i] = [x_est[0], x_est[1], x_est[2], x_est[3]]
    data[i] = x_est.reshape(1,4)
    
print("data :", data)
# print("ekf outpus", data)
ekf_position_output_offline = data[:,0].reshape(14535,1)
print("ekf_position_output_offline :", ekf_position_output_offline)
ekf_velocity_output_offline = data[:,1].reshape(14535,1)
print("ekf_velocity_output_offline :", ekf_velocity_output_offline)
ekf_frequency_output_offline = data[:,2].reshape(14535,1)
print("ekf_frequency_output_offline :", ekf_frequency_output_offline)
ekf_offset_output_offline = data[:,3].reshape(14535,1)
print("ekf_offset_output_offline :", ekf_offset_output_offline)


plt.figure(1)
plt.plot(time, ekf_position_input*252.07, 'b')
plt.plot(time, ekf_position_output_offline*252.07, 'g')

plt.figure(2)
plt.plot(time, ekf_velocity_input*252.07, 'b')
plt.plot(time, ekf_velocity_output_offline*252.07, 'g')

plt.figure(3)
plt.plot(time, ekf_frequency_output_offline**2)

plt.figure(4)
plt.plot(time, ekf_offset_output_offline)

plt.figure(5)
plt.plot(time, ekf_position_input*252.07, 'b')
plt.plot(time, ekf_position_output_online*252.07, 'g')

plt.figure(6)
plt.plot(time, ekf_velocity_input*252.07, 'b')
plt.plot(time, ekf_velocity_output_online*252.07, 'g')

plt.figure(7)
plt.plot(time, ekf_frequency_output_online**2)

plt.figure(8)
plt.plot(time, ekf_offset_output_online)

plt.show()



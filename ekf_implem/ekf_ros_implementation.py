import numpy as np

# Definition of Extended Kalman Filter parameters
self.P = 100.0*np.eye(4, dtype=float)
self.x_est = np.array([0.0, 0.0, 1.0, 1.0],  dtype=float).reshape(4,1)

def ekf_estimation(self, e_m, e_m_dot, Ts, P, x_est):
        
        #e_m: centroid error (e_m = x - xd = (u-ud)/ax)
        #e_dot_m : gradient (velocity) after removing the camera velocity
        #Ts: sampling time e.g. Ts = 0.1
        # P initial filter covariance matrix
        #x_est initial estimate to be update by Kalman

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
        
        z = np.array([e_m, e_m_dot]).reshape(2,1)
        
        x_est = np.dot(PHI_k, x_dash_k) + np.dot(K_k, z - np.dot(H_k, np.dot(PHI_k, x_dash_k)))
        print("x_est: ", x_est)

        return x_est # [estimated wave position, estimated wave velocity, estimated frequency] --> you only need x_est[1]: velocity
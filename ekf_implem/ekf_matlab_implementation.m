%load('ekf_data_record_no_motion_session_1.mat')
%load('ekf_data_record_no_motion_session_2.mat')
load('ekf_data_record_control_motion_session_9_09_03_2021.mat')
value_linewidth = 2;
value_fontsize = 12;
width = 720;
height = 480;
moment = 225;
z1 = ekf_position_input;
z2 = ekf_velocity_input;
z = [z1 z2];
Ts = mean(gradient(time));
%Ts=0.1;
P = 100*eye(4,4);
PHI_S1 = 0.001;
PHI_S2 = 0.001;
x_est = [0 0 1 1]';


data = [];
for i = 1:length(z)
    F_est = zeros(4,4);
    F_est (1,2) = 1.0;
    F_est (2,1) = -x_est(3)^2;
    F_est (2,3) = -2.0*x_est(3)*x_est(1)+ 2.0*x_est(3)*x_est(4);
    F_est (2,4) = x_est(3)^2;
    
    PHI_k = eye(4,4) + F_est*Ts;
    
    Q_k = [0.001, 0, 0, 0;
            0, 0.5*PHI_S1*(-2*x_est(3)*x_est(1)+2*x_est(3)*x_est(4))^2*Ts^2+0.333*x_est(3)^4*Ts^3*PHI_S2, 0, 0.5*x_est(3)^2*Ts^2*PHI_S2;
            0, 0.5*PHI_S1*(-2*x_est(3)*x_est(1)+2*x_est(3)*x_est(4))^2*Ts^2, PHI_S1*Ts, 0;
            0, 0.5*PHI_S2*Ts^2*x_est(3)^2, 0, PHI_S2*Ts];
%     Q_k = [0.5 0    0    0   ;
%            0   0.5  0    0   ;
%            0   0    0.01  0 ; 
%            0   0    0    0.01]    ;
%     
    H_k = [1, 0, 0, 0;
           0, 1, 0, 0];
    
    R_k =  [0.1 0;
                0   0.1];
    
    M_k = PHI_k*P*PHI_k'+Q_k;
    K_k = M_k*H_k'*inv(H_k*M_k*H_k' + R_k);
    P = (eye(4,4) - K_k*H_k)*M_k;
    
    xdd = -x_est(3)^2*x_est(1) + x_est(3)^2*x_est(4);
    xd = x_est(2) + Ts*xdd;
    x = x_est(1) + Ts*xd;

    x_dash_k = [x, xd, x_est(3), x_est(4)]';
    
    x_est = PHI_k*x_dash_k + K_k*(z(i,:)' - H_k*PHI_k*x_dash_k);
    
%     x_est(1) = x_dash_k(1) + K_k(1)*(y(i) - H_k*x_dash_k);
%     x_est(2) = x_dash_k(2) + K_k(2)*(y(i) - H_k*x_dash_k);
%     x_est(3) = x_est(3) + K_k(3)*(y(i) - H_k*x_dash_k);
%     x_est(4) = x_est(4) + K_k(4)*(y(i) - H_k*x_dash_k);
    
    data = [data; x_est'];
end

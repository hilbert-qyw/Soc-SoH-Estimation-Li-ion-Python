# ----------------------------------------------------------------------------------------------------
# Program to take string data from sample csv and print 
# on consol and plot the graphs for corresponding values
# ----------------------------------------------------------------------------------------------------
import csv
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
def predictionSoC(SoC, i, t, total):
    A = np.array([[1, (t*100)/total],
                  [0, 1]])
    X = np.array([[SoC],
                  [i]])
    X_predict = A.dot(X)
    return X_predict
def covarianceSoC(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return np.diag(np.diag(cov_matrix))
def OCV_to_SoC(x):
    [k0 ,k1 ,k2 ,k3 ,k4 ,k5 ,a1 ,a2 ,a3 ,a4 ,b1 ,b2 ] = [
        -2.51943946e+00 ,  4.07304942e+01 ,  3.79669044e+01 ,  1.11174735e+01 ,
        2.04417058e+01 ,  2.02933951e+00 , -4.06970001e+00 , -5.43947995e+00 ,
        -1.73243706e-02 , -1.62819294e-02 ,  8.02333838e+00 ,  7.06431073e+00
        ]
    y=k0+k1*(1/(1+np.exp(a1*(x-b1))))+k2*(1/(1+np.exp(a2*(x-b2))))+k3*(1/(1+np.exp(a3*(x-1))))+k4*(1/(1+np.exp(a4*x)))+k5*x
    return y


title=["Current","Voltage","OCV","Coulomb Count","Kalman Filter"]
print(title)
# Initial Conditions
total=10800 # total rated capacity of the battery = 3Ah
t=1 #time difference in measurement
n=2 #State Matrix size
# Process / Estimation Errors
error_est_soc = 1
error_est_i = 0.25
# Uncertainty in the measurement
error_obs_ocv = 1
error_obs_i = 0.25
A = np.array([[1, (t*100)/total],
    [0, 1]])
# Initial Process Covariance Matrix
P = covarianceSoC(error_est_soc, error_est_i)
#Measurement Covariance Matrix
R = covarianceSoC(error_obs_ocv, error_obs_i)
initialize=False
ocvolt=[]
ocvsoc=[]
cousoc=[]
kalsoc=[]


#INITIALIZATION
while True:
    if not initialize:
        with open("D:\\User\\ProjectFY\\Database\\discharge.csv","r") as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            next(lines)
            for row in lines:
                if row[2].isdigit():
                    readtime=int(row[0])/3600
                    ocv=int(row[2])/1000
                    ocvolt.append(ocv)
                    current=(int(row[1])*-1)/1000
                    print(current,end="A,  ")
                    print(ocv,end="V,  ")
                    soc=OCV_to_SoC(ocv)
                    ocvsoc.append(soc)
                    print(soc,end="%,  ")
                    coulombsoc=soc
                    cousoc.append(coulombsoc)
                    print(soc,end="%,  ")
                    kalsoc.append(soc)
                    print(soc)
                    # Initial State Matrix
                    X = np.array([[soc],
                        [current]])
                    initialize = True
                    break


    #PROCESS
    else:
        with open("D:\\User\\ProjectFY\\Database\\discharge.csv","r") as csvfile:
            lines = csv.reader(csvfile, delimiter=',')
            next(lines)
            next(lines)
            for row in lines:
                if row[0].isdigit():
                    readtime=int(row[0])/3600
                    ocv=int(row[2])/1000
                    ocvolt.append(ocv)
                    current=(int(row[1])*-1)/1000
                    print(current,end="A,  ")
                    soc=OCV_to_SoC(ocv)
                    measured = np.array([[soc],
                        [current]])
                    #OCV Method using Function
                    ocvsoc.append(soc)
                    print(ocv,end="V,  ")
                    print(soc,end="%,  ")
                    #Coulomb count Method
                    coulombsoc=coulombsoc+(current*t*100)/total
                    cousoc.append(coulombsoc)
                    print(coulombsoc,end="%,  ")
                    #Kalman filter method
                    X = predictionSoC(X[0][0], X[1][0], t, total)
                    #set off diagonal elements zero
                    # P = np.diag(np.diag(A.dot(P).dot(A.T)))   
                    #not set off diagonal elements zeroa
                    P = A.dot(P).dot(A.T)
                    # Calculating the Kalman Gain
                    H = np.identity(n)
                    S = H.dot(P).dot(H.T) + R
                    K = P.dot(H).dot(inv(S))
                    # Reshape the new data into the measurement space.
                    Y = H.dot(measured).reshape(n, -1)
                    # Update the State Matrix
                    # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
                    X = X + K.dot(Y - H.dot(X))
                    # Update Process Covariance Matrix
                    P = (np.identity(len(K)) - K.dot(H)).dot(P)
                    kalsoc.append(X[0][0])
                    print(X[0][0],end="%,")
                    print()


                else:
                    plt.plot( ocvsoc, ocvolt, color = '#66FF00', linestyle = 'solid',label = "OCV")
                    plt.legend(loc='upper right')
                    plt.plot( cousoc, ocvolt, color = 'r', linestyle = 'solid',label = "Coulomb count")
                    plt.legend(loc='upper right')
                    plt.plot( kalsoc, ocvolt, color = '#306EFF', linestyle = 'solid',label = "Kalman Filter")
                    plt.legend(loc='upper right')
                    plt.xlabel('% SoC')
                    plt.ylabel('Battery Voltage')
                    plt.gca().invert_xaxis()
                    plt.grid(True)
                    plt.title("Discharging")
                    plt.show()
                    plt.tight_layout()
                    quit()
                    
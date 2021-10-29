'''
Program to take string data from port and form a csv file
'''
import csv
import datetime
import serial
from datetime import date, datetime

# current_date_and_time = datetime.datetime.now()
# current_date_and_time_string = str(current_date_and_time)


import numpy as np
from numpy.linalg import inv

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
        -2.51943946e+00,  4.07304942e+01,  3.79669044e+01,  1.11174735e+01,
        2.04417058e+01,  2.02933951e+00, -4.06970001e+00, -5.43947995e+00,
        -1.73243706e-02, -1.62819294e-02,  8.02333838e+00,  7.06431073e+00]
    y=k0+k1*(1/(1+np.exp(a1*(x-b1))))+k2*(1/(1+np.exp(a2*(x-b2))))+k3*(1/(1+np.exp(a3*(x-1))))+k4*(1/(1+np.exp(a4*x)))+k5*x

    return y

# extension = ".csv"
# current_date_and_time_string = current_date_and_time_string.replace(".","-")
# current_date_and_time_string = current_date_and_time_string.replace(":","`")
# current_date_and_time_string = current_date_and_time_string.replace(" ","-Time-")
# file_name =  "BATTERY-DATA-"+current_date_and_time_string + extension

serialPort = serial.Serial(port ="COM5", baudrate=9600) 
#serialPort.close()
#serialPort.open()
title=["Time","Date","Current","Voltage","Temperature1","Temperature2","OCV","Coulomb Count","Kalman Filter"]
print(title)
with open("D:\\User\\ProjectFY\\Database\\Batterydata.csv" , "w",newline="") as csvfile :
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(title)

title=["Process","CycleCount","CycleCapacity"]
print(title)
with open("D:\\User\\ProjectFY\\Database\\Cycledata.csv" , "w",newline="") as csvfile :
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(title)

# Initial Conditions
total=10800 # total rated capacity of the battery = 3Ah
t=1 #time difference in measurement
n=2 #State Matrix size
iold=0
inew=0
cycleCapacity=0
cycledata=[]
cyclecount=0
count=0

# Process / Estimation Errors
error_est_soc = 0.25
error_est_i = 0.25

# Uncertainty in the measurement
error_obs_ocv = 0.25  
error_obs_i = 0.25

A = np.array([[1, (t*100)/total],
    [0, 1]])

# Initial Process Covariance Matrix
P = covarianceSoC(error_est_soc, error_est_i)

#Measurement Covariance Matrix
R = covarianceSoC(error_obs_ocv, error_obs_i)
initialize=False

while True:
    
    if not initialize:

        # Wait until there is data waiting in the serial buffer
        if serialPort.in_waiting > 0:

            # Initial Conditions
            serial = serialPort.readline()
            data=serial.decode('Ascii')
            data=list(data.split(","))
            if len(data)==5:
                data=data[::-1]
                day = date.today()
                today = day.strftime("%d/%m/%Y")
                data.append(today)
                now = datetime.now()
                time = now.strftime("%H:%M:%S.%f")
                data.append(time)
                data=data[::-1]
                data.pop(6)
                data[2]=int(data[2])/1000
                data[3]=int(data[3])/100
                data[4]=int(data[4])/2
                data[5]=int(data[5])/2
                ocv=data[3]
                i=data[2]
                cycleCapacity=i*t
                soc=OCV_to_SoC(ocv)
                data.append(soc)
                data.append(soc)
                data.append(soc)

                # Initial State Matrix
                X = np.array([[soc],
                    [i]])

                print(data)
                with open("D:\\User\\ProjectFY\\Database\\Batterydata.csv", "a",newline="") as csvfile :
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(data)

            initialize = True
            break

    else:
        
        if serialPort.in_waiting > 0:
        
            serial = serialPort.readline()
            data=serial.decode('Ascii')
            data=list(data.split(","))
            if len(data)==5:
                data=data[::-1]
                day = date.today()
                today = day.strftime("%d/%m/%Y")
                data.append(today)
                now = datetime.now()
                time = now.strftime("%H:%M:%S.%f")
                data.append(time)
                data=data[::-1]
                data.pop(6)
                data[2]=int(data[2])/200
                data[3]=int(data[3])/100
                data[4]=int(data[4])/2
                data[5]=int(data[5])/2
                ocv=data[3]
                iold=i
                i=data[2]
                inew=i
                cycleCapacity=cycleCapacity+i*t
                if iold*inew<0:
                    count+=1
                    if count == 2:
                        count=1
                        cyclecount+=1

                    with open("D:\\User\\ProjectFY\\Database\\cycledata.csv", "a",newline="") as csvfile :
                        if count==1:
                            cycledata.append("discharging")
                        if count==2:
                            cycledata.append("charging")
                        cycledata.append(cyclecount)
                        cycledata.append(cycleCapacity)
                        cycleCapacity=0
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(cycledata)

                soc=OCV_to_SoC(ocv)
                measured = np.array([[soc],
                    [i]])

                #OCV Method
                data.append(soc)

                #Coulomb count Method
                soc=soc+(data[2]*100)/total
                data.append(soc) 
                
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
                data.append(X[0][0])

                print(data)
                with open("D:\\User\\ProjectFY\\Database\\Batterydata.csv", "a",newline="") as csvfile :
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(data)

print("Finised") #Transfer Complete

# -*- coding: utf-8 -*-
"""
Created on Thu May  9 12:12:10 2019

@author: Jinjian LI
@Objective: estimation of traffic flow from Floating Car Data with Gaussian Process Regressor.
@Single model: one GPR model for all data. 
@Note:1.Data structure, input features size is 33. It means that 33 Floating Car Data points are treated as input for estimating one point of traffic flow.
      2.Data is each process is save as txt file for other programme to use.
@Step:1.extraction of valuable data from initial raw files.
      2.construction input X based on the selected features
      3.division of training and testing data based on given percentage parameter
      4.input data from file and fix the GPR model for prediction.
@libraries applied: scikit-learn, numpy
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import random
np.random.seed(1)

FileNameExtractFCD="ExtractFCD.txt"
FileNameExtractTF="ExtractTrafficFlowHour.txt"
FileNameAllInputFeaturesX="AllInputFeaturesX.txt"
FileNameTrainingInputFeaturesX="TrainingInputFeaturesX.txt"
FileNameTestingInputFeaturesX="TestingInputFeaturesX.txt"
FileNameTrainingOutputY="TrainingOutputY.txt"
FileNameTestingOutputY="TestingOutputY.txt"
ProportionTraing=0.8

#Step 1. extract the FCD data (second/ten minmuts) from the initial file from Google map.
def ExtractTrafficDurationFromFCDInitialData(FileName):
    file_intial_data = open(FileName, "r")
    TrafficDuration = []
    file_rows = file_intial_data.readlines()
    Num=0
    for row in file_rows:
        Num=Num+1
        row_splited=row.split(";")
        # row_splited = row.split()
        print(row_splited)
        TrafficDuration.append(int(row_splited[3]))
    file_Traffic_duration = open(FileNameExtractFCD, "w")
    for i in range(0, Num):
        file_Traffic_duration.write(str(TrafficDuration[i]))
        if i != Num-1:  ## if it is not the last week or last day.
            file_Traffic_duration.write("\n")
    file_Traffic_duration.close()
    file_intial_data.close()
    return TrafficDuration

# extract the Traffic Flow (TF) data (vehicles/hour) from the initial file from transport management.
# If the totoal day for traffic flow is N days, the one for FCD data should be N+2. 
# Because it needs the two extra days to estimate the fist point and end point of traffic flow. refer to features selection parts.
def ExtractTrafficFlowFromTransportMangement_hour(FileName):
    file_intial_data = open(FileName, "r")
    TrafficVolume = []
    file_rows = file_intial_data.readlines()
    Num=0
    for row in file_rows:
        Num=Num+1
        row_splited = row.split()
        TrafficVolume.append(int(row_splited[2]))
    file_Traffic_flow=open(FileNameExtractTF, "w")
    for i in range(0, Num):
        for j in range(0,6):#the sample time step for Traffic flow is hour but the one for FCD is ten minuts. 
                            #So the linear interpolation is done for Traffic Flow 
            file_Traffic_flow.write(str(TrafficVolume[i]))
            if (i == Num - 1)and(j==5):  ## if it is not the last week or last day.
                print("the last point")
            else:
                file_Traffic_flow.write("\n")
    file_Traffic_flow.close()
    print(Num)
    return TrafficVolume
    





#step 2.construction input X based on the selected features
def ConstructionInputFeaturesX(FileName_FCD):
    file_FCD_Traffic_Duration = open(FileName_FCD, "r")
    TrafficDuration = []
    file_rows = file_FCD_Traffic_Duration.readlines()
    for row in file_rows:
        row_splited = row.split()
        TrafficDuration.append(int(row_splited[0]))
    file_FCD_Traffic_Duration.close()
    
    
    file_InputFeaturesX_all = open(FileNameAllInputFeaturesX, "w")
    Num_Weeks=1
    Num_Days_week = 7
    Num_points_selected=16
    Num_heurs=24
    Num_points_heur=6
    ConstructionPoint = []


    for week in range(1,Num_Weeks+1):
        StartingPoint=week*Num_heurs*Num_points_heur-Num_points_selected
        for point in range(0, Num_Days_week * Num_heurs * Num_points_heur):
            Position_point=StartingPoint
            for i in range(0,Num_points_selected*2+1):
                ConstructionPoint.append(TrafficDuration[Position_point])
                file_InputFeaturesX_all.write(str(TrafficDuration[Position_point]))
                if i!=Num_points_selected*2:
                    file_InputFeaturesX_all.write(" ")
                Position_point=Position_point+1
            StartingPoint=StartingPoint+1
            if week==Num_Weeks and point==Num_Days_week * Num_heurs * Num_points_heur-1:
                print("the last point for X")
            else:
                file_InputFeaturesX_all.write("\n")
    file_InputFeaturesX_all.close()
    return ConstructionPoint

#step 3.division of training and testing data based on given percentage parameter
def divideFeaturesXandYIntoTrainingAndTestingData(FileName_FeaturesX,FileName_Y):
    file_TrainingInputFeaturesX = open(FileNameTrainingInputFeaturesX, "w")
    file_TestingInputFeaturesX = open(FileNameTestingInputFeaturesX, "w")
    file_TrainingOutputY = open(FileNameTrainingOutputY, "w")
    file_TestingOutputY = open(FileNameTestingOutputY, "w")
    file_AllOutputY = open(FileName_Y, "r")
    file_AllInputFeaturesX = open(FileName_FeaturesX, "r")
    file_rows_X = file_AllInputFeaturesX.readlines()
    file_rows_Y = file_AllOutputY.readlines()
    Num_rows_Y=0
    for row in file_rows_X:
        if random.randint(0, 1000) < 1000*ProportionTraing:
            file_TrainingInputFeaturesX.write(row)
            file_TrainingOutputY.write(file_rows_Y[Num_rows_Y])
        else:
            file_TestingInputFeaturesX.write(row)  
            file_TestingOutputY.write(file_rows_Y[Num_rows_Y])
        Num_rows_Y=Num_rows_Y+1
    file_AllInputFeaturesX.close()
    file_AllOutputY.close()
    file_TestingOutputY.close()
    file_TrainingOutputY.close()
    file_TestingInputFeaturesX.close()
    file_TrainingInputFeaturesX.close()

#step 4. input data from file and fix the GPR model for prediction.
def InputFeaturesXandY(FileName_FeaturesX,FileName_Y):
    file_FeaturesX = open(FileName_FeaturesX, "r")
    file_Y = open(FileName_Y, "r")
    file_rows_X = file_FeaturesX.readlines()
    file_rows_Y = file_Y.readlines()
    All_FeaturesX = np.zeros((len(file_rows_X),len(file_rows_X[0].split())))
    All_Y = np.zeros((len(file_rows_Y),len(file_rows_Y[0].split())))
    print('Check X, Y size')
    print(len(All_FeaturesX))
    print(len(All_Y))
    Example_seq=0
    for row in file_rows_X:
        row_splited = row.split()
        Features_seq=0
        for i in row_splited:
            All_FeaturesX[Example_seq,Features_seq]=i
            Features_seq=Features_seq+1
        Example_seq=Example_seq+1
    file_FeaturesX.close()
    
    Example_seq=0
    for row in file_rows_Y:
        row_splited = row.split()
        Y_seq=0
        for i in row_splited:
            All_Y[Example_seq,Y_seq]=i
            Y_seq=Y_seq+1
        Example_seq=Example_seq+1
    file_Y.close()
    
    
    print('all example features: ')
    print(All_FeaturesX)
    return All_FeaturesX, All_Y
    
    
raw_FCD =ExtractTrafficDurationFromFCDInitialData("FCD_initial.txt")
raw_TF =ExtractTrafficFlowFromTransportMangement_hour("Traffic_Flow_initial.txt")
raw_X_all =ConstructionInputFeaturesX(FileNameExtractFCD)
print('test data: ')
divideFeaturesXandYIntoTrainingAndTestingData(FileNameAllInputFeaturesX,FileNameExtractTF)
TraingFeaturesX,TraingY=InputFeaturesXandY('TrainingInputFeaturesX.txt','TrainingOutputY.txt')

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(TraingFeaturesX, TraingY)
TestFeaturesX,TestY=InputFeaturesXandY('TestingInputFeaturesX.txt','TestingOutputY.txt')
y_pred, sigma = gp.predict(TestFeaturesX, return_std=True)
print(y_pred)

print(sum(abs(y_pred-TestY))/len(y_pred))

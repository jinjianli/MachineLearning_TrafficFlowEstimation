Traffic flow estimation with Machine Learning algorithms
@author: Jinjian LI
@Objective: estimation of traffic flow from Floating Car Data with Gaussian Process Regressor.
@Single model: one GPR model for all data. 
@Note:1.Data structure, input features size is 33. It means that 33 Floating Car Data points are treated as input for estimating one point of traffic flow.
      2.Data is each process is save as txt file for other programme to use.
@Step:1.extraction of valuable data from initial raw files.
      2.construction input X based on the selected features
      3.division of training and testing data based on given percentage parameter
      4.input data from file and fix the GPR model for prediction.
@libraries applied: scikit-learn, numpy,GPR

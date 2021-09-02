# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:24:10 2020

@author: utahlab
"""
import numpy as np
import pandas as pd
import math
import re
import statistics
import random
from random import shuffle
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, ion, show

########################  RUN   ###############################################
""""
# Run the code by assigning 1 to one of the parameters in the following lines

"""
# Hyper Parameter for the simple perceptron 
hyper_param = 0 
# Hyper Parameter for the decay_rate perceptron 
hyper_param_decay = 0
# Hyper Parameter for the average perceptron 
hyper_param_ave = 0
# Hyper Parameter for the margin perceptron 
hyper_param_mar = 1
# majority base line classifier
Majority_baseline = 0



########################  Inputs   ############################################

Train = pd.read_csv('train.csv',header=None)
Test = pd.read_csv('test.csv',header=None)
Fold1 = pd.read_csv('fold1.csv',header=None)
Fold2 = pd.read_csv('fold2.csv',header=None)
Fold3 = pd.read_csv('fold3.csv',header=None)
Fold4 = pd.read_csv('fold4.csv',header=None)
Fold5 = pd.read_csv('fold5.csv',header=None)
F1 = pd.concat([Fold2,Fold3,Fold4,Fold5],axis = 0,ignore_index=True)
F2 = pd.concat([Fold1,Fold3,Fold4,Fold5],axis = 0,ignore_index=True)
F3 = pd.concat([Fold1,Fold2,Fold4,Fold5],axis = 0,ignore_index=True)
F4 = pd.concat([Fold1,Fold2,Fold3,Fold5],axis = 0,ignore_index=True)    
F5 = pd.concat([Fold1,Fold2,Fold3,Fold4],axis = 0,ignore_index=True) 
F = [F1,F2,F3,F4,F5]
ff = [Fold1,Fold2,Fold3,Fold4,Fold5]

eta = [1,0.1,0.01]
Mu = [1,0.1,0.01]
random.seed(24)
orig = [random.uniform(-0.01,0.01) for i in range(207)]

########################  Simple_Perceptron   #################################
def sim_perc(training_data,E,Epoch):
    w = orig[0:206]
    b = orig[-1]
    W = list()
    B = list()
    for epoch in range(Epoch):
        F1 = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
        for i in range(len(F1)):
            temp = F1.iloc[i][0]*np.matmul(w,np.transpose(F1.iloc[i][1:])) + b
            W.append(w)
            B.append(b)
            if (temp) < 0:
                w = w + E*F1.iloc[i][0]*F1.iloc[i][1:]
                b = b + E*F1.iloc[i][0]            
    return (W), (B)

########################  Simple_Perceptron2   #################################
def sim_perc2(training_data,w,b,E,ctr=0):
    F1 = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
    for i in range(len(F1)):
        temp = F1.iloc[i][0]*np.matmul(w,np.transpose(F1.iloc[i][1:])) + b
        if (temp) < 0:
            w = w + E*F1.iloc[i][0]*F1.iloc[i][1:]
            b = b + E*F1.iloc[i][0] 
            ctr+=1            
    return w, b , ctr
########################  decayR_Perceptron   #################################             
    
def dcy_perc(training_data,w,b,E,Epoch):
    ctr = 0
    for epoch in range(Epoch):
        Shf = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
        for i in range(len(Shf)):
            temp = Shf.iloc[i][0]*np.matmul(w,np.transpose(Shf.iloc[i][1:])) + b
            if (temp) < 0:
                w = w + E*Shf.iloc[i][0]*Shf.iloc[i][1:]
                b = b + E*Shf.iloc[i][0] 
                E = E/(1+ctr)
        ctr += 1
    return w, b

def dcy_perc2(training_data,w,b,E,ctr=0,counter=0):
    E = E/(1+ctr)
    Shf = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
    for i in range(len(Shf)):
        temp = Shf.iloc[i][0]*np.matmul(w,np.transpose(Shf.iloc[i][1:])) + b
        if (temp) < 0:
            w = w + E*Shf.iloc[i][0]*Shf.iloc[i][1:]
            b = b + E*Shf.iloc[i][0] 
            counter+=1
    return w, b, E, counter
########################  Averaged_Perceptron   ###############################
def ave_perc(training_data,w,b,E,Epoch):
    a = 0
    b_a = 0
    for epoch in range(Epoch):
        F1 = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
        for i in range(len(F1)):
            temp = F1.iloc[i][0]*np.matmul(w,np.transpose(F1.iloc[i][1:])) + b
            if (temp) < 0:
                w = w + E*F1.iloc[i][0]*F1.iloc[i][1:]
                b = b + E*F1.iloc[i][0] 
            a = np.array(a) + w
            b_a = b_a + b
                
    return a, b_a   

def ave_perc2(training_data,w,b,E,a,b_a,counter=0):
    F1 = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
    for i in range(len(F1)):
        temp = F1.iloc[i][0]*np.matmul(w,np.transpose(F1.iloc[i][1:])) + b
        if (temp) < 0:
            w = w + E*F1.iloc[i][0]*F1.iloc[i][1:]
            b = b + E*F1.iloc[i][0] 
            counter+=1
        a = np.array(a) + w
        b_a = b_a + b                
    return w, b, a, b_a, counter   
########################  Margin_Perceptron   ###############################
def mar_perc(training_data,w,b,E,Epoch,mu):
    ctr = 0
    for epoch in range(Epoch):
        F1 = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
        for i in range(len(F1)):
            temp = F1.iloc[i][0]*np.matmul(w,np.transpose(F1.iloc[i][1:])) + b
            if (temp) < mu:
                ctr += 1
                w = w + E*F1.iloc[i][0]*F1.iloc[i][1:]
                b = b + E*F1.iloc[i][0] 
                E = E/(1+ctr)
        ctr += 1                
    return w, b 

def mar_perc2(training_data,w,b,E,mu,ctr=0,counter=0):
    E = E/(1+ctr)
    Shf = training_data.sample(frac = 1, axis = 0).reset_index(drop=True) 
    for i in range(len(Shf)):
        temp = Shf.iloc[i][0]*np.matmul(w,np.transpose(Shf.iloc[i][1:])) + b
        if (temp) < mu:
            w = w + E*Shf.iloc[i][0]*Shf.iloc[i][1:]
            b = b + E*Shf.iloc[i][0] 
            counter+=1
    return w, b, E, counter
########################  Majority baseline   #################################
def Majority_base(data):
    vals,counts = np.unique(data[0],return_counts = True)
    Lab = vals[np.argmax(counts)]                
    return Lab

if Majority_baseline == 1:
    training_data = Train
    testing_data = Test     
    predicted_test = [0 for ii in range(len(testing_data))]
    for i in range(len(testing_data)):
        predicted_test[i] = Majority_base(testing_data)
    Acc_test = (np.sum(predicted_test == testing_data[0])/len(testing_data))
    print('The prediction accuracy of testing data is: ',Acc_test*100,'%')
    predicted_train = [0 for ii in range(len(training_data))]
    for i in range(len(training_data)):
        predicted_train[i] = Majority_base(training_data)
    Acc_train = (np.sum(predicted_train == training_data[0])/len(training_data))
    print('The prediction accuracy of training data is: ',Acc_train*100,'%')
            
########################  Hyper_parameters   #################################
if hyper_param == 1:
    results = []
    for j in range(len(eta)):
        for l in range(len(F)):
            training_data = F[l]
            testing_data = ff[l]
            weights, bias = sim_perc(training_data,eta[j],10) 
            predicted = [0 for ii in range(len(testing_data))]
            for i in range(len(testing_data)):
                predicted[i] = np.sign(np.matmul(weights[-1],np.transpose(testing_data.iloc[i][1:]))+bias[-1])
            Acc = (np.sum(predicted == testing_data[0])/len(testing_data))
            print('The prediction accuracy is: ',Acc*100,'%', 'for etta: ', eta[j], 'for Epoch: ', 10)
            results.append([Acc,eta[j],l])
    Mean = [0 for i in range(len(eta))]                        
    Results = pd.DataFrame(results) 
    for i in range(len(eta)):
        j = 5*i
        Mean[i] = statistics.mean(Results.iloc[j:j+5][0])  
        
    Max_acc = eta[np.argmax(Mean)]   

if hyper_param_decay == 1:
    results = []    
    for j in range(len(eta)):
        for l in range(len(F)):
            training_data = F[l]
            testing_data = ff[l]
            weights, bias = dcy_perc(training_data,orig[0:206],orig[-1],eta[j],10) 
            predicted = [0 for i in range(len(testing_data))]      
            for i in range(len(testing_data)):
                predicted[i] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[i][1:]))+bias)
            Acc = (np.sum(predicted == testing_data[0])/len(testing_data))
            results.append([Acc,eta[j],l])
            print('The prediction accuracy is: ',Acc*100,'%', 'for etta: ', eta[j], 'for Epoch: ', 10)
                
    Mean = [0 for i in range(len(eta))]                        
    Results = pd.DataFrame(results) 
    for i in range(len(eta)):
        j = 5*i
        Mean[i] = statistics.mean(Results.iloc[j:j+5][0])  
        
    Max_acc_dcy = eta[np.argmax(Mean)]

if hyper_param_ave == 1:
    results = []    
    for j in range(len(eta)):
        for l in range(len(F)):
            training_data = F[l]
            testing_data = ff[l]
            weights, bias = ave_perc(training_data,orig[0:206],orig[-1],eta[j],10) 
            predicted = [0 for i in range(len(testing_data))]      
            for i in range(len(testing_data)):
                predicted[i] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[i][1:]))+bias)
            Acc = (np.sum(predicted == testing_data[0])/len(testing_data))
            results.append([Acc,eta[j],l])
            print('The prediction accuracy is: ',Acc*100,'%', 'for etta: ', eta[j], 'for Epoch: ', 10)
                
    Mean = [0 for i in range(len(eta))]                        
    Results = pd.DataFrame(results) 
    for i in range(len(eta)):
        j = 5*i
        Mean[i] = statistics.mean(Results.iloc[j:j+5][0])  
        
    Max_acc_ave = eta[np.argmax(Mean)]
    
if hyper_param_mar == 1:
    results = []    
    for j in range(len(eta)):
        for m in range(len(Mu)):
            for l in range(len(F)):
                training_data = F[l]
                testing_data = ff[l]
                weights, bias = mar_perc(training_data,orig[0:206],orig[-1],eta[j],10,Mu[m]) 
                predicted = [0 for i in range(len(testing_data))]      
                for i in range(len(testing_data)):
                    predicted[i] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[i][1:]))+bias)
                Acc = (np.sum(predicted == testing_data[0])/len(testing_data))
                results.append([Acc,eta[j],Mu[m],l])
                print('The prediction accuracy is: ',Acc*100,'%', 'for etta: ', eta[j], 'for Epoch: ', 10, 'for Mu: ', Mu[m])
                
    Mean = [0 for j in range(9)]                        
    Results = pd.DataFrame(results) 
    for J in range(len(Mean)):
        K = 5*J
        Mean[J] = statistics.mean(Results.iloc[range(K,(K+5))][0])
    
    Max_acc_mar = eta[(np.argmax(Mean))//len(eta)]
    Mu_acc_mar = Mu[(np.argmax(Mean))%len(eta)-1]

########################  Train Classifier   ##################################     

if hyper_param == 1: 
    r1 = []
    W = []
    B = []
    for i in range(20):
        training_data = Train
        testing_data = Train
        if i == 0: 
            weights, bias, C = sim_perc2(training_data,orig[0:206],orig[-1],Max_acc,0) 
            W.append([weights])
            B.append([bias])
        else:
            weights, bias, C = sim_perc2(training_data,weights, bias,Max_acc, C)
            W.append([weights])
            B.append([bias])
        predicted = [0 for ii in range(len(testing_data))]
        for t in range(len(testing_data)):
            predicted[t] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[t][1:]))+bias)
        A1 = (np.sum(predicted == testing_data[0])/len(testing_data))
        print('The prediction accuracy is: ',A1*100,'%', 'for etta: ', Max_acc, 'for Epoch: ', i+1)
        r1.append([A1])
    W_test = W[np.argmax(r1)]
    B_test = B[np.argmax(r1)]
    Predicted_test = [0 for ii in range(len(Test))]
    for t in range(len(Test)):
        Predicted_test[t] = int(float(np.sign(np.matmul(W_test,np.transpose(Test.iloc[t][1:]))+B_test)))
    Acc_test = (np.sum(Predicted_test == Test[0])/len(Test))
  
    print('(a) The best hyper-parameter is: ',Max_acc)
    print('(b) Cross-validation accuracy for the best hyper-parameter: ',max(Mean)*100,'%')
    print('(c) Total number of updates the learning algorithm performs on the training set: ',C)
    print('(d) Training accuracy: ',np.sum(r1)/len(r1)*100,'%')
    print('(e) Testing accuracy: ',Acc_test*100,'%')    
    print('(f) Learning curve: ')
    axes = plt.gca()
    axes.set_xlim([1,20])
    x1 = range(1,21)
    plt.plot(x1,r1)
    plt.xticks(x1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
########################      
        
if hyper_param_decay == 1: 
    r1 = []
    W = []
    B = []
    for i in range(20):
        training_data = Train
        testing_data = Train
        if i == 0: 
            weights, bias,E,C = dcy_perc2(training_data,orig[0:206],orig[-1],Max_acc_dcy,0,0) 
            W.append([weights])
            B.append([bias])
        else:
            weights, bias,E,C = dcy_perc2(training_data,weights,bias,E,i,C)
            W.append([weights])
            B.append([bias])
        predicted = [0 for ii in range(len(testing_data))]
        for t in range(len(testing_data)):
            predicted[t] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[t][1:]))+bias)
        A1 = (np.sum(predicted == testing_data[0])/len(testing_data))
        print('The prediction accuracy is: ',A1*100,'%', 'for etta: ', E , 'for Epoch: ', i+1)
        r1.append([A1])
    W_test = W[np.argmax(r1)]
    B_test = B[np.argmax(r1)]
    Predicted_test = [0 for ii in range(len(Test))]
    for t in range(len(Test)):
        Predicted_test[t] = int(float(np.sign(np.matmul(W_test,np.transpose(Test.iloc[t][1:]))+B_test)))
    Acc_test = (np.sum(Predicted_test == Test[0])/len(Test))
  
    print('(a) The best hyper-parameter is: ',Max_acc_dcy)
    print('(b) Cross-validation accuracy for the best hyper-parameter: ',max(Mean)*100,'%')
    print('(c) Total number of updates the learning algorithm performs on the training set: ',C)
    print('(d) Training accuracy: ',np.sum(r1)/len(r1)*100,'%')
    print('(e) Testing accuracy: ',Acc_test*100,'%')    
    print('(f) Learning curve: ')
    axes = plt.gca()
    axes.set_xlim([1,20])
    x1 = range(1,21)
    plt.plot(x1,r1)
    plt.xticks(x1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')    
########################      
        
if hyper_param_ave == 1: 
    r1 = []
    A = []
    B_a = []
    for i in range(20):
        training_data = Train
        testing_data = Train
        if i == 0: 
            weights, bias,a, b_a, C = ave_perc2(training_data,orig[0:206],orig[-1],Max_acc_ave,0,0,0) 
            A.append([a])
            B_a.append([b_a])
        else:
            weights, bias,a, b_a, C = ave_perc2(training_data,weights, bias,Max_acc_ave,a,b_a, C)
            A.append([a])
            B_a.append([b_a])
        predicted = [0 for ii in range(len(testing_data))]
        for t in range(len(testing_data)):
            predicted[t] = np.sign(np.matmul(a,np.transpose(testing_data.iloc[t][1:]))+b_a)
        A1 = (np.sum(predicted == testing_data[0])/len(testing_data))
        print('The prediction accuracy is: ',A1*100,'%', 'for etta: ', Max_acc_ave , 'for Epoch: ', i+1)
        r1.append([A1])
    W_test = A[np.argmax(r1)]
    B_test = B_a[np.argmax(r1)]
    Predicted_test = [0 for ii in range(len(Test))]
    for t in range(len(Test)):
        Predicted_test[t] = int(float(np.sign(np.matmul(W_test,np.transpose(Test.iloc[t][1:]))+B_test)))
    Acc_test = (np.sum(Predicted_test == Test[0])/len(Test))
  
    print('(a) The best hyper-parameter is: ',Max_acc_ave)
    print('(b) Cross-validation accuracy for the best hyper-parameter: ',max(Mean)*100,'%')
    print('(c) Total number of updates the learning algorithm performs on the training set: ',C)
    print('(d) Training accuracy: ',np.sum(r1)/len(r1)*100,'%')
    print('(e) Testing accuracy: ',Acc_test*100,'%')    
    print('(f) Learning curve: ')
    axes = plt.gca()
    axes.set_xlim([1,20])
    x1 = range(1,21)
    plt.plot(x1,r1)
    plt.xticks(x1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy') 
  
########################      
        
if hyper_param_mar == 1: 
    r1 = []
    W = []
    B = []
    for i in range(20):
        training_data = Train
        testing_data = Train
        if i == 0: 
            weights, bias,E,C = mar_perc2(training_data,orig[0:206],orig[-1],Max_acc_mar,Mu_acc_mar,0,0) 
            W.append([weights])
            B.append([bias])
        else:
            weights, bias,E,C = mar_perc2(training_data,weights,bias,E,Mu_acc_mar,i,C)
            W.append([weights])
            B.append([bias])
        predicted = [0 for ii in range(len(testing_data))]
        for t in range(len(testing_data)):
            predicted[t] = np.sign(np.matmul(weights,np.transpose(testing_data.iloc[t][1:]))+bias)
        A1 = (np.sum(predicted == testing_data[0])/len(testing_data))
        print('The prediction accuracy is: ',A1*100,'%', 'for etta: ', E , 'for Epoch: ', i+1)
        r1.append([A1])
    W_test = W[np.argmax(r1)]
    B_test = B[np.argmax(r1)]
    Predicted_test = [0 for ii in range(len(Test))]
    for t in range(len(Test)):
        Predicted_test[t] = int(float(np.sign(np.matmul(W_test,np.transpose(Test.iloc[t][1:]))+B_test)))
    Acc_test = (np.sum(Predicted_test == Test[0])/len(Test))
  
    print('(a) The best hyper-parameter is: etta = ',Max_acc_mar, 'Mu = ', Mu_acc_mar)
    print('(b) Cross-validation accuracy for the best hyper-parameter: ',max(Mean)*100,'%')
    print('(c) Total number of updates the learning algorithm performs on the training set: ',C)
    print('(d) Training accuracy: ',np.sum(r1)/len(r1)*100,'%')
    print('(e) Testing accuracy: ',Acc_test*100,'%')    
    print('(f) Learning curve: ')
    axes = plt.gca()
    axes.set_xlim([1,20])
    x1 = range(1,21)
    plt.plot(x1,r1)
    plt.xticks(x1)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')        
    
 
    
    

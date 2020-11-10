#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[72]:
import pickle
import os
import sys
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import eigh
from scipy.fftpack import fft
import glob
from scipy import linalg as la
from scipy import ndimage
import cv2
from pathlib import Path
from sklearn.metrics import f1_score


# In[138]:

class NN:
    def store_parameters(self,data_set_name):
    #np.save("Mnist.txt")
        parameters={}
        if data_set_name=="MNIST":
            ot=open("MNIST_PAR","wb")

            parameters['weights_matrix']=self.weights_matrix
            parameters['bias']=self.bias
            parameters['mean']=self.mean
            parameters['std']=self.std
            parameters['list_of_nodes']=self.all_layers
            parameters['activationfunc']=self.activationfunc
            pickle.dump(parameters,ot)
            ot.close()
        elif data_set_name=="Cat-Dog":
            ot=open("CATDOG_PAR","wb")

            parameters['weights_matrix']=self.weights_matrix
            parameters['bias']=self.bias
            parameters['mean']=self.mean
            parameters['std']=self.std
            parameters['list_of_nodes']=self.all_layers
            parameters['activationfunc']=self.activationfunc
            pickle.dump(parameters,ot)
            ot.close()

    def restore_parameters(data_set_name):
        if data_set_name=="MNIST":
            it=open("MNIST_PAR",rb)
            parameters=pickle.load(it)
            it.close()
            self.weights_matrix=parameters['weights_matrix']
            self.bias=parameters['bias']
            self.std=parameters['std']
            self.list_of_nodes=parameters['list_of_nodes']
            self.activationfunc=parameters['activationfunc']
            self.mean=parameters['mean']


        elif data_set_name=="Cat-Dog":
            it=open("CATDOG_PAR",rb)
            parameters=pickle.load(it)
            it.close()
            self.weights_matrix=parameters['weights_matrix']
            self.bias=parameters['bias']
            self.std=parameters['std']
            self.list_of_nodes=parameters['list_of_nodes']
            self.activationfunc=parameters['activationfunc']
            self.mean=parameters['mean']

    def load_data(self,path1,nameofset,test):    
        data=[]
        X=[]
        imgnm= []
        rdimg = [] 
        Y=[]
        if nameofset=="Cat-Dog":
            
            cat = glob.glob(path1+'/cat/*.jpg')
            for c_d in cat:
                
                rdimg.append((cv2.imread(c_d, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(1)
            dog=glob.glob(path1+'/dog/*.jpg') 
            for c_d in dog:  
                rdimg.append((cv2.imread(c_d, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(0)
                
        elif nameofset=="MNIST":
            i=glob.glob(path1+'/0/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(0)
            i=glob.glob(path1+'/1/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(1)
            i=glob.glob(path1+'/2/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(2)
            i=glob.glob(path1+'/3/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(3)
            i=glob.glob(path1+'/4/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(4)
            i=glob.glob(path1+'/5/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(5)
            i=glob.glob(path1+'/6/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(6)
            i=glob.glob(path1+'/7/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(7)
            i=glob.glob(path1+'/8/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(8)
            i=glob.glob(path1+'/9/*.jpg')
            for i1 in i:
                rdimg.append((cv2.imread(i1, cv2.IMREAD_GRAYSCALE)).ravel())
                Y.append(9)
            
                
  
       
        Y=self.one_hot_encoding(Y,nameofset)
        
        rdimg=self.preprocessing(rdimg,test)
        
        return Y,rdimg

    # #print((rdimg[0]).shape)


    def __init__(self,list_of_nodes):
        self.no_of_layers=len(list_of_nodes)
        self.all_layers=list_of_nodes
        self.acc=[]
        self.mean=[]
        self.std=[]
        self.f1_macro=[]
        self.f1_micro=[]
    
    def one_hot_encoding(self,Y,nameofset):
        encoded_list = []
        if nameofset=="MNIST":
            for value in Y:
                #print("h")
                i = [0 for _ in range(10)]
                i[value] = 1
                encoded_list.append(i)
            #print(encoded_list)
            Y=np.array(encoded_list)
            return Y
        elif nameofset=="Cat-Dog":
            for value in Y:
                i = [0 for _ in range(2)]
                i[value] = 1
                encoded_list.append(i)
            #print(encoded_list)
            Y=np.array(encoded_list)
            return Y
            
            
    
    def preprocessing(self,rdimg,test):

        if test==0:
            self.mean,self.std   = np.array(rdimg).mean(axis=0),np.array(rdimg).std(axis=0)
            self.std = np.where(self.std==0,1,self.std)
            rdimg  = (rdimg-self.mean)/self.std
            return rdimg
        elif test==1:
            #mean,std   = np.array(rdimg).mean(axis=0),np.array(rdimg).std(axis=0)
            std = np.where(self.std==0,1,self.std)
            rdimg  = (rdimg-self.mean)/self.std
            return rdimg
    
    def softmax_Activation_function(self,net_input,r=0):
        net_input = net_input - net_input.max(axis=1,keepdims=True)
        result = np.exp(net_input)
        result = result / np.sum(result,axis=1,keepdims=True)
        if r==0:
            return result
        else:
            return result*(1-result)
        #return result / result.sum(axis=1,keepdims=True) #softmax r
    
    
    def cross_entropy(self,out_labels,y1,r=0):
        result1=(out_labels-y1)
        out_labels=np.where(y1!=1,out_labels+np.e,out_labels)
        out_labels=np.where(np.logical_and(y1==1,out_labels==0),out_labels+10**-8,out_labels)
        result= -1* np.mean(y1*np.log(out_labels),axis=0,keepdims=True)
        
        if r==0:
            return result
        else:
            return result1
    
    def sigmoid_Activation_function(self,net_input,r=0):
        result = 1.0 / (1.0 + np.exp(-net_input))
        result1 = result * (1 - result)
        if r==0:
            return result
        else:
            return result1
    
     
    def relu_Activation_function(self,net_input,r=0):
        result = np.maximum(0, net_input)
        result1=(np.sign(net_input) >= 0)
        if r==0:
            return result
        else:
            return result1
    
    def swish_Activation_function(self,net_input,r=0):
       # result = net_input / (1.0 + np.exp(-net_input))
        #result1=result+ self.sigmoid_Activation_function(net_input,0) * (1-result)
        result=net_input*self.sigmoid_Activation_function(net_input,0)
        result1=self.sigmoid_Activation_function(net_input,0)
        if r==0:
            return result
        else:
            return result+result1*(1-result)
    
    def tanh_Activation_function(self,net_input,r=0):
        result=2*self.sigmoid_Activation_function(net_input,0)-1
        result1=1-(result)**2
        if r==0:
            return result
        else:
            return result1
    
    def network_init(self,activation_func,mode='gaussian'):  
        print(activation_func)
        self.activationfunc=activation_func       
        num_layers=self.all_layers
        self.weights_matrix = [0 for i in range(len(num_layers)-1)]
        self.bias=[0 for i in range(len(num_layers)-1)]  
        i=0
        for (current_layer_nodes,next_layer_nodes) in zip(num_layers[:-1],num_layers[1:]): 
            self.weights_matrix[i],self.bias[i], = self.initialize_parameters(current_layer_nodes,next_layer_nodes,self.activationfunc[i],mode)
            i+=1


    def initialize_parameters(self,current_layer_nodes,next_layer_nodes,act_function,state='gaussian'): #,next_layer_nodes,current_layer_nodes,act_function):
   
        k=0
        w=[]
        b=[]
#         for (current_layer_nodes,next_layer_nodes) in zip(num_layers[:-1],num_layers[1:]): 
        
        if act_function=='sigmoid' or act_function=='softmax': #or act_function=='tanh':
            if state=='gaussian':
                learning_rate=np.sqrt(2) / np.sqrt(current_layer_nodes + next_layer_nodes)
                w=(np.random.randn(current_layer_nodes,next_layer_nodes)*learning_rate)
                b=(np.random.randn(1,next_layer_nodes)*learning_rate)

            elif state=='uniform':
                learning_rate=np.sqrt(6) / np.sqrt(current_layer_nodes + next_layer_nodes)
                w=(2*learning_rate * np.random.random(current_layer_nodes,next_layer_nodes)-learning_rate )
                b=(2*learning_rate * (np.random.random(1,next_layer_nodes) )-learning_rate)


        elif act_function=='relu' or act_function=='swish' :
            if state=='gaussian':
                learning_rate=2*np.sqrt(1 / (current_layer_nodes * next_layer_nodes) )
                w=(learning_rate * np.random.randn(current_layer_nodes,next_layer_nodes))
                b=(learning_rate * np.random.randn(1,next_layer_nodes))
            elif state=='uniform':
                learning_rate=np.sqrt(12 / (current_layer_nodes * next_layer_nodes) )
                w=(2*learning_rate *np.random.random(current_layer_nodes,next_layer_nodes)-learning_rate)
                b=(2*learning_rate * np.random.random(1,next_layer_nodes)-learning_rate)


        elif act_function=='tanh':  #or act_function=='swish':
            if state=='gaussian':
                learning_rate=4*np.sqrt(2/ (current_layer_nodes * next_layer_nodes) )
                w=(learning_rate * np.random.randn(current_layer_nodes,next_layer_nodes))
                b=(learning_rate * np.random.randn(1,next_layer_nodes))

            elif state=='uniform':
                learning_rate=4*np.sqrt(6/ (current_layer_nodes * next_layer_nodes) )
                w=(2*learning_rate *np.random.random(current_layer_nodes,next_layer_nodes)-learning_rate)
                b=(2*learning_rate * np.random.random(1,next_layer_nodes)-learning_rate)
        return w,b
#         self.activationfunc.append(act_function[k])
#         #print(self.weights_matrix)
#         k=k+1
 
                       
    def mini_batch(self,epochs, mini_batch_size,learning_rate): 
        training_data=self.rdimg
        Y=self.Y
        act_funcs=self.activationfunc
        n = len(training_data)
        for j in range(epochs):
            print("epoch====",str(j))
            print("Epoch====:",str(j))
            indx  = np.arange(Y.shape[0])
            np.random.shuffle(indx)
            training_data,Y = training_data[indx], Y[indx]
#             np.random.shuffle(training_data)
            sgd_batch=[]
            y1=[]
            k=0
            for l in range(int(len(training_data)/mini_batch_size)):
                sgd_batch.append(training_data[k:k+mini_batch_size])
                y1.append(Y[k:k+mini_batch_size])
                k+=mini_batch_size
            k=0 
            for i in sgd_batch:
#                 print(i)
#                 input()
                result=self.forward_propagation(i,y1[k])
                self.backprop(y1[k],learning_rate)
                k+=1
                
    
            result=self.forward_propagation(training_data,Y)
            pred = 1*(result == result.max(axis=1,keepdims=True))
            print("F1- score is(Macro,Micro): ",end=' ')
            a,b=self.F1_score(Y,pred)
            print(a,b)
            self.f1_macro.append(a)
            self.f1_micro.append(b)
            print("Accuracy on the trainset is : ")
            acc1=np.mean((pred==Y).all(axis=1))
            acc1*=100
            print(acc1)
            # self.acc.append(acc1)
            
            
    def testing(self):       
        testlabel=self.Y
        testset=self.rdimg
        result=self.forward_propagation(testset,testlabel)
        print("------------Testing------------ ")
        pred = 1*(result == result.max(axis=1,keepdims=True))
        print("F1- score is (Macro): ",end=' ')
        a,b=self.F1_score(pred,self.Y)
        print(a)
        print("F1- score is (Micro): ",end=' ')
        acc1=np.mean((pred==self.Y).all(axis=1))
        acc1*=100
        print(b)
        # print(pred)
        print("Accuracy on the testset is : ")
        print(acc1)  
    
            
   
    def backprop(self,y1,learning_rate):       
        change = self.cross_entropy(self.netinp_activation[-1],y1,1) * self.softmax_Activation_function(self.net_input[-1],1)
        b_updated = change
        w_updated = np.dot(self.netinp_activation[-2].T,change)/ self.netinp_activation[-2].shape[0] 
        B = np.mean( b_updated ,axis=0, keepdims=True)
        self.weights_matrix[-1]-=learning_rate*w_updated
        self.bias[-1]-=learning_rate*B
        
#         for l in range(self.no_of_layers-2,0,-1):
#             change = np.dot(change,self.weights_matrix[l].T)*(eval("self.{0}_Activation_function(self.net_input[l],1)".format(self.activationfunc[])))
#             b_updated= change
#             w_updated = np.dot(self.netinp_activation[l-1].T,change)/ self.netinp_activation[l-1].shape[0] 
#             #W = np.dot( self.IP[i].T , self.delta[i] ) / self.IP[i].shape[0] #ip[i] isthe activation of previous layer.
#             B = np.mean( b_updated ,axis=0, keepdims=True) 
#             self.weights_matrix[l-1]-=learning_rate*w_updated
#             self.bias[l-1]-=learning_rate*B
    
        for l in range(2, self.no_of_layers):
            change = np.dot(change,self.weights_matrix[-l+1].T)* (self.sigmoid_Activation_function(self.net_input[-l],1))#(eval("self.{0}_Activation_function(self.net_input[l],1)".format(self.activationfunc[l])))
            b_updated= change
            w_updated = np.dot(self.netinp_activation[-l-1].T,change)/ self.netinp_activation[-l-1].shape[0] 
            #W = np.dot( self.IP[i].T , self.delta[i] ) / self.IP[i].shape[0] #ip[i] isthe activation of previous layer.
            B = np.mean( b_updated ,axis=0, keepdims=True) 
            self.weights_matrix[-l]-=learning_rate*w_updated
            self.bias[-l]-=learning_rate*B

        #return b_updated,w_updated

    def F1_score(self,testlabel,predictions):
        return ((f1_score(testlabel, predictions, average='macro')),(f1_score(testlabel, predictions, average='micro')))    

   
        
    def forward_propagation(self,input_matrix,y1):        
        self.netinp_activation=[]       
        self.net_input=[]
        self.net_input.append(input_matrix)
        self.netinp_activation.append(input_matrix)
#        print(self.weights_matrix)
#         print(self.bias)
        for i in range(self.no_of_layers-1): 
#             print(np.dot(self.netinp_activation[i],self.weights_matrix[i]))
            result = np.dot(self.netinp_activation[i],self.weights_matrix[i])+self.bias[i]  #weights equal to the no of layers-1

#             print(self.bias[i])
#             print(self.netinp_activation[i])
#             print(result)
           
            
            
            self.net_input.append(result)           
            if self.activationfunc[i]=='sigmoid': 
#                 print("ppppp")
                output_val=self.sigmoid_Activation_function(result)
            elif self.activationfunc[i]=='softmax':
                output_val=self.softmax_Activation_function(result)
            elif self.activationfunc[i]=='tanh':
                output_val=self.tanh_Activation_function(result)
            elif self.activationfunc[i]=='swish':
                output_val=self.swish_Activation_function(result)
            elif self.activationfunc[i]=='relu':
                output_val=self.relu_Activation_function(result)
            self.netinp_activation.append(output_val)
        #print(self.netinp_activation)
        #result=self.cross_entropy(self.netinp_activation[-1],y1)
#         print(self.netinp_activation[i])
#         input()
        return self.netinp_activation[-1]
 


# In[ ]:


if __name__=='__main__':
    array_of_arguments=sys.argv


    if array_of_arguments[1]=="--test-data":
        

        # #print("wbdj")
        test_path=array_of_arguments[2]
        #test_label=array_of_arguments[4]

        if array_of_arguments[4]=="MNIST":
            
            print("-------------Test------------")
            # net=NN([784,30,20,10])
            # net.Y,net.rdimg=net.load_data("MNIST","MNIST",0)
            # net.network_init(["sigmoid","sigmoid","softmax"])
            
            # net.mini_batch(1,30,0.01) 
            # net.store_parameters("MNIST")

            it=open("MNIST_PAR","rb")
            parameters=pickle.load(it)
            it.close()
            weights_matrix=parameters['weights_matrix']
            bias=parameters['bias']
            std=parameters['std']
            list_of_nodes=parameters['list_of_nodes']
            activationfunc=parameters['activationfunc']
            mean=parameters['mean']
            # print("jhbhjvjvkv")
            # print(mean)
            # 
            tt=NN(list_of_nodes)
            tt.activationfunc=activationfunc

            tt.weights_matrix=weights_matrix
            tt.mean=mean
            tt.std=std
            tt.bias=bias
            
            tt.Y,tt.rdimg=tt.load_data(test_path,"MNIST",1)
            
            tt.testing()
        elif array_of_arguments[4]=="Cat-Dog":

            # net=NN([40000,30,20,2])
            # net.Y,net.rdimg=net.load_data("Cat-Dog","Cat-Dog",0)
            # net.network_init(["sigmoid","sigmoid","softmax"])
            
            # net.mini_batch(1,30,0.01) 
            # net.store_parameters("Cat-Dog")

            it=open("CATDOG_PAR","rb")
            parameters=pickle.load(it)
            it.close()
            weights_matrix=parameters['weights_matrix']
            bias=parameters['bias']
            std=parameters['std']
            list_of_nodes=parameters['list_of_nodes']
            activationfunc=parameters['activationfunc']
            mean=parameters['mean']

            tt=NN(list_of_nodes)
            tt.activationfunc=activationfunc

            tt.weights_matrix=weights_matrix
            tt.mean=mean
            tt.std=std
            tt.bias=bias
            
            tt.Y,tt.rdimg=tt.load_data(test_path,"Cat-Dog",1)
            
            tt.testing()
    
    elif array_of_arguments[1]=="--train-data":
        print("--------Training----------")
        list_of_nodes=array_of_arguments[8]
        #act=array_of_arguments[10]
        train_path=array_of_arguments[2]
        test_path=array_of_arguments[4]
        k=9
        i=1
        # print((list_of_nodes)[1:])
        while(1):
            if str(array_of_arguments[k])[-1]==']':
                i+=1
                break
            i+=1
            k+=1
        actv=[]
        for i1 in range(i):
            actv.append("sigmoid")
        actv.append("softmax")
        # print(array_of_arguments[9])
        if array_of_arguments[6]=="MNIST":
            k=9
            i=1
            listofnodes=[784]
            listofnodes.append(int(list_of_nodes[1:]))
            while(1):
                if array_of_arguments[k][-1]==']':
                    listofnodes.append(int(array_of_arguments[k][:-1]))
                    i+=1
                    break
                listofnodes.append(int(array_of_arguments[k]))
                i+=1
                k+=1
            listofnodes.append(10)
           

            # print(actv)
            # print(listofnodes)
            net=NN(listofnodes)
            # print(train_path)

            net.Y,net.rdimg=net.load_data(train_path,"MNIST",0)
            indx  = np.arange(net.Y.shape[0])
            np.random.shuffle(indx)
            net.rdimg,net.Y = net.rdimg[indx], net.Y[indx]
            # test=net.rdimg[35000:]
            # testl=net.Y[35000:]
            # net.Y=net.Y[:35000]
            # net.rdimg=net.rdimg[:35000]
           
            net.network_init(actv)
            #print(net.weights_matrix)
            net.mini_batch(600,30,0.01)   
            #net.store_parameters("MNIST")       
            net.Y,net.rdimg=net.load_data(test_path,"MNIST",1)
            # net.Y=testl
            # net.rdimg=test
            net.testing()
                        
    
        elif array_of_arguments[6]=="Cat-Dog":
            k=9
            i=1
            listofnodes=[40000]
            listofnodes.append(int(list_of_nodes[1:]))
            while(1):
                if array_of_arguments[k][-1]==']':
                    listofnodes.append(int(array_of_arguments[k][:-1]))
                    i+=1
                    break
                listofnodes.append(int(array_of_arguments[k]))
                i+=1
                k+=1
            listofnodes.append(2)
            # print(listofnodes)

            net=NN(listofnodes)
            net.Y,net.rdimg=net.load_data(train_path,"Cat-Dog",0)
            # net.Y=net.Y[:14000]
            # net.rdimg=net.rdimg[:14000]

            net.network_init(actv)
            net.mini_batch(200,40,0.01)   
            #net.store_parameters("Cat-Dog")
            net.Y,net.rdimg=net.load_data(test_path,"Cat-Dog",1)
            # net.Y=net.Y[14000:]
            # net.rdimg=net.rdimg[14000:]

            net.testing()


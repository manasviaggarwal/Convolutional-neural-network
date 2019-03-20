#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import operator
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import pandas as pd
import os
import sys
import gzip
import glob
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
from yellowbrick.text import TSNEVisualizer
# from google.colab import drive
# drive.mount('/content/gdrive')

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)



class CNN1:
    def one_hot_encoding(self,Y):
        encoded_list = []
        # print(Y[1])
        #if nameofset=="MNIST":
        for value in Y:
            # print(value)
            # break



            i = [0 for _ in range(10)]
            i[value[0]] = 1
            encoded_list.append(i)
        Y=np.array(encoded_list)
        # print(Y)
        return Y
#      def F1_score(self,testlabel,predictions):
#         return ((f1_score(testlabel, predictions, average='macro')),(f1_score(testlabel, predictions, average='micro')))  
    def unpickle(self,file):
        #import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
    def load_main(self,path1,nameofset,v):
        if v==0:
            train_set=pd.DataFrame()
            train_label=pd.DataFrame()
            if nameofset=="CIFAR-10":
                c=os.getcwd()
                path1=c+"/data/CIFAR-10/"
                for file in glob.glob(path1+"*"):
                    if "data_batch" in file:        
                        dict1=self.unpickle(file)
                        train_set1=(pd.DataFrame(dict1['data']))
                        train_set=pd.concat([train_set,train_set1])
                        train_set1=(pd.DataFrame(dict1['labels']))
                        train_label=pd.concat([train_label,train_set1])
                train_label=np.array(train_label)
                self.train_labels=train_label 
                train_label=self.one_hot_encoding(train_label)
                train_set=np.array(train_set)
                self.trainset=train_set
                self.trainlabel=train_label
                #print(len(train_label))
                x = tf.placeholder(tf.float32, shape=[None, 32*32*3], name='X')
                x_image = tf.reshape(x, [-1, 32, 32, 3])
                
                y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
                y_true_cls = tf.argmax(y_true, dimension=1)
                return x,x_image,y_true,y_true_cls

            elif nameofset=="Fashion-MNIST":
                kind='train'
                c=os.getcwd()
                c=os.getcwd()
                path1=c+"/data/Fashion-MNIST/"
                
                labels_path = os.path.join(path1,'%s-labels-idx1-ubyte.gz'% kind)
                images_path = os.path.join(path1,'%s-images-idx3-ubyte.gz'% kind)
                print(labels_path)
                with gzip.open(labels_path, 'rb') as lbpath:
                    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

                with gzip.open(images_path, 'rb') as imgpath:
                    images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)


                images=pd.DataFrame(images)
                labels=pd.DataFrame(labels)
                train_set=(images)
                train_label=(labels)
                train_set=np.array(train_set)
                print("------")
                print(len(train_label))
                train_label=np.array(train_label)
                self.train_labels=train_label
    #             train_set=self.normalize(np.array(train_set),0)
                train_label=self.one_hot_encoding(train_label)

    #             np.array(train_set).reshape(-1,28,28,1)
                self.trainset=train_set
                self.trainlabel=train_label
                x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
                x_image = tf.reshape(x, [-1, 28, 28, 1])
                y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
                #print(x_image)
                #y_true_cls=tf.get_variable("y_true_cls",shape=y_true.shape)
                y_true_cls = tf.argmax(y_true, dimension=1)
                return x,x_image,y_true,y_true_cls
        elif v==1:
            if nameofset=="CIFAR-10":
                c=os.getcwd()
                test_set=pd.DataFrame()
                test_label=pd.DataFrame()
                c=os.getcwd()
                path1=c+"/data/CIFAR-10/"
                for file in glob.glob(path1+"*"):
                    if "test_batch" in file:        
                        dict1=self.unpickle(file)

                        test_set1=(pd.DataFrame(dict1['data']))
                        test_set=pd.concat([test_set,test_set1])
                        test_set1=(pd.DataFrame(dict1['labels']))
                        test_label=pd.concat([test_label,test_set1])
                test_label=np.array(test_label) 
                self.test_labels=test_label
                test_label=self.one_hot_encoding(test_label)
                test_set=np.array(test_set)
        
                self.testset=test_set
                self.testlabel=test_label
                
            elif nameofset=="Fashion-MNIST":
                kind='t10k'
                c=os.getcwd()
                
                path1=c+"/data/Fashion-MNIST/Newfolder/"
                
                labels_path = os.path.join(path1,'%s-labels-idx1-ubyte.gz'% kind)
                images_path = os.path.join(path1,'%s-images-idx3-ubyte.gz'% kind)
                with gzip.open(labels_path, 'rb') as lbpath:
                    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

                with gzip.open(images_path, 'rb') as imgpath:
                    images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)


                images=pd.DataFrame(images)
                labels=pd.DataFrame(labels)
                test_set=(images)
                test_label=(labels)
                test_set=np.array(test_set)
                self.test_labels=test_label
                test_label=np.array(test_label)
                print(len(test_set))

                test_label=self.one_hot_encoding(test_label)


                self.testset=test_set
                self.testlabel=test_label
                #print((test_label).shape)




    def new_conv_layer(self,input, num_input_channels, filter_size, num_filters, name):

        with tf.variable_scope(name) as scope:
            shape = [filter_size, filter_size, num_input_channels, num_filters]
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
            biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
            layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
            layer += biases
            return layer, weights

    def new_pool_layer(self,input, name):

        with tf.variable_scope(name) as scope:
            layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return layer


    def new_relu_layer(self,input, name):

        with tf.variable_scope(name) as scope:
            layer = tf.nn.relu(input)
            return layer

    def new_fc_layer(self,input, num_inputs, num_outputs, name):

        with tf.variable_scope(name) as scope:
            weights = tf.Variable(tf.truncated_normal([num_inputs, num_outputs], stddev=0.05))
            biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
            layer = tf.matmul(input, weights) + biases
            return layer

    def initialization_layers_test(self,test_path,dataset):
           
 
             self.load_main(test_path,dataset,1)
             with tf.Session() as sess:
                if dataset=="Fashion-MNIST":
                    new_saver = tf.train.import_meta_graph('my-test-model_FM.meta')
                    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
                    a,b,c=sess.run([tf.get_default_graph().get_tensor_by_name("accuracy/Mean:0"),tf.get_default_graph().get_tensor_by_name("ArgMax:0"),tf.get_default_graph().get_tensor_by_name("Softmax/ArgMax:0")],feed_dict={tf.get_default_graph().get_tensor_by_name("X:0"):self.testset,tf.get_default_graph().get_tensor_by_name("y_true:0"):self.testlabel})
                 # print(np.array(b))
                    print("ACCURACY IS:",end=' ')
                    print(np.array(a))
                    print("F1-MICRO: ",end=' ')

                    f1_macro=(f1_score(np.array(c), np.array(b), average='macro'))
                    f1_micro=(f1_score(np.array(c), np.array(b), average='micro'))
                    print(f1_micro)
                    print("F1-MACRO: ",end=' ')
                    print(f1_macro)
                elif dataset=="CIFAR-10":
                    new_saver = tf.train.import_meta_graph('my-test-model_CF.meta')
                    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
                    a,b,c=sess.run([tf.get_default_graph().get_tensor_by_name("accuracy/Mean:0"),tf.get_default_graph().get_tensor_by_name("ArgMax:0"),tf.get_default_graph().get_tensor_by_name("Softmax/ArgMax:0")],feed_dict={tf.get_default_graph().get_tensor_by_name("X:0"):self.testset,tf.get_default_graph().get_tensor_by_name("y_true:0"):self.testlabel})
                 # print(np.array(b))
                    print("ACCURACY IS:",end=' ')
                    print(np.array(a))
                    print("F1-MICRO: ",end=' ')
                    f1_macro=(f1_score(np.array(c), np.array(b), average='macro'))
                    f1_micro=(f1_score(np.array(c), np.array(b), average='micro'))
                    print(f1_micro)
                    print("F1-MACRO: ",end=' ')
                    print(f1_macro)

                 # op = sess.graph.get_operations()
                 # #print(op.name)
                 # for op in sess.graph.get_operations():
                 #     print(op.name)
                 # a,b,c=sess.run([tf.get_default_graph().get_tensor_by_name("accuracy/Mean:0"),tf.get_default_graph().get_tensor_by_name("ArgMax:0"),tf.get_default_graph().get_tensor_by_name("Softmax/ArgMax:0")],feed_dict={tf.get_default_graph().get_tensor_by_name("X:0"):self.testset,tf.get_default_graph().get_tensor_by_name("y_true:0"):self.testlabel})
                 # # print(np.array(b))
                 # print("ACCURACY IS:",end=' ')
                 # print(np.array(a))
                 # print("F1-MICRO: ",end=' ')

                 # f1_macro=(f1_score(np.array(c), np.array(b), average='macro'))
                 # f1_micro=(f1_score(np.array(c), np.array(b), average='micro'))
                 # print(f1_micro)
                 # print("F1-MACRO: ",end=' ')
                 # print(f1_macro)

                 # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='some_scope'):
                 #     print(i) 
                 # #print(sess.run(tf.get_default_graph().get_tensor_by_name("y_pred:0")))
                 # print(sess.run(y_pred))
    

    def initialization_layers_train(self,train_path,test_path,dataset,no_of_layers,filter_sizes):
            global n1,n2
            # y_pred_cl=tf.get_variable("y_pred_cl",0)

            x,x_image,y_true,y_true_cls=self.load_main(train_path,dataset,0)
            self.load_main(test_path,dataset,1)
            # print(len(x_image))
           # y_true_cls=tf.Variable(y_true_cls)
            if dataset=="Fashion-MNIST":
                layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=1, filter_size=filter_sizes[0], num_filters=64, name ="conv1")
           
            elif dataset=="CIFAR-10":
                layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=3, filter_size=filter_sizes[0], num_filters=20, name ="conv1")

            layer_pool1 = self.new_pool_layer(layer_conv1, name="pool1")
            layer_pool1=tf.nn.local_response_normalization(layer_pool1)
            layer_relu1 = self.new_relu_layer(layer_pool1, name="relu1")
            
            layer_pools=[]
            layer_relus=[]
            layer_convs=[]
            weight_convs=[]
            
            layer_convs.append(layer_conv1)
            layer_pools.append(layer_pool1)
            layer_relus.append(layer_relu1)
            weight_convs.append(weights_conv1)
            n2=5
            for k1 in range(1,no_of_layers):
                namee="conv"+str(k1+1)
                layer_conv1, weights_conv1 = self.new_conv_layer(input=layer_relus[k1-1], num_input_channels=64, filter_size=filter_sizes[k1], num_filters=64, name= namee)
                name2="pool"+str(k1+1)
                name1="relu"+str(k1+1)
                
                layer_pool1 = self.new_pool_layer(layer_conv1, name=name2)
                layer_pool1=tf.nn.local_response_normalization(layer_pool1)
                layer_relu1 = self.new_relu_layer(layer_pool1, name=name1)
                
                layer_convs.append(layer_conv1)
                layer_pools.append(layer_pool1)
                layer_relus.append(layer_relu1)
                weight_convs.append(weights_conv1)
                n2=10
            
            num_features = layer_relu1.get_shape()[1:4].num_elements()
            layer_flat = tf.reshape(layer_relu1, [-1, num_features])


            
            layer_fc1 = self.new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=512, name="fc1")
            layer_relu4 = self.new_relu_layer(layer_fc1, name="relu"+str(no_of_layers+1))

            layer_fc3 = self.new_fc_layer(layer_relu4, num_inputs=512, num_outputs=192, name="fc3")
            layer_relu3 = self.new_relu_layer(layer_fc3, name="relu"+str(no_of_layers+2))
       

            
            layer_fc2 = self.new_fc_layer(input=layer_relu3, num_inputs=192, num_outputs=10, name="fc2")

            
            with tf.variable_scope("Softmax"):
                y_pred =(tf.nn.softmax(layer_fc2))
                
                y_pred_cls = tf.argmax(y_pred, dimension=1)
                
                # y_pred_cl=y_pred_cls
            

            with tf.name_scope("cross_ent"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
                cost = tf.reduce_mean(cross_entropy)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

            with tf.variable_scope("accuracy"):
                # print("-----------")
                # print(y_pred_cls)
                correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            num_epochs = 10
            batch_size = 100

            trainset=self.trainset
            testset=self.testset
            trainlabel=self.trainlabel
            testlabel=self.testlabel
            f=open("fm_allembeddings.txt","w")
            tsne=TSNEVisualizer()
            with tf.Session() as sess:
                global train_acc
                global f1_mi
                global f1_ma
                train_acc=[]
                f1_mi=[]
                f1_ma=[]
                sess.run(tf.global_variables_initializer())
                trainset1=self.trainset
                trainlabels11=self.train_labels
#                print(len(trainset1))
                trainlabel1=self.trainlabel
                for i in range(0,4):
                        f.write("\n\n                   Percentage of traindata ::: ")
                        f.write(str((1+i)*10))
                        f.write("\n")
                        # indx  = np.arange(trainlabel1.shape[0])
                        # np.random.shuffle(indx)
                        # trainset,trainlabel = trainset1[indx], trainlabel1[indx]
                        # trainset1, trainlabel1=trainset1[indx], trainlabel1[indx]
                        # trainlabels1=trainlabels11[indx]
                        n=len(trainset1)
                        n=int(n*(i+1)/10)
                        # print(n)
                        trainset,trainlabel = trainset1, trainlabel1
                        trainlabels1=trainlabels11
                        testset,testlabel =trainset[n:,:],trainlabel[n:,:]
                        testlabels1=trainlabels11[n:,:]
                        # print(len(testlabel))                        
                        trainset,trainlabel =trainset[0:n,:],trainlabel[0:n,:]
                        trainlabels1=trainlabels1[0:n,:]
                        print("----------------------------------------------")
                        # print(len(trainlabel1),len(trainset),len(testset))
#                   testset,testlabel =trainset1[n:,:],trainlabel1[n:,:] 
                        # print(trainlabel)
                    # writer.add_graph(sess.graph)
                        # print("testlabel:",str(testlabels1[4]))
                        # print("testsset",str(testset[4]))
                        # print("testlabel",str(testlabel[4]))
                        # print("set",str(trainset1[n+4]))
                        # print("lab",str(trainlabel1[n+4]))
                        # break
                        for epoch in range(num_epochs):
                            start_time = time.time()
                            train_accuracy = 0
                            k=0
                            f1_macro=0
                            f1_micro=0
                            a1=0
                        # batch_size=int(len(trainlabel))

                            for batch in range(0, int(len(trainlabel)/batch_size)):
                                # print("--")
                                x_batch  = trainset[k:k+batch_size]
                                y_true_batch = trainlabel[k:k+batch_size]
                                k=k+batch_size
                                feed_dict_train = {x: x_batch, y_true: y_true_batch}
                                sess.run(optimizer, feed_dict=feed_dict_train)
                                a,y_p,y,t=sess.run([accuracy,y_pred_cls,y_true_cls,layer_relu3], feed_dict=feed_dict_train)
                                a1+=a
#                                 a1+=a
                            print("acc :",str(epoch),"---",str(a1/int(len(trainlabel)/batch_size)))
          
                        vali_accuracy,l = sess.run([accuracy,layer_relu3] ,feed_dict={x:testset, y_true:testlabel})
                        # print(np.array(l))
                        kmeans = KMeans(n_clusters=10, random_state=0).fit(np.array(l))
                # yy=kmeans.predict(np.array(l))
                # plt.scatter()
                        labels = kmeans.labels_
                        # print(labels)
                        f.write("\n\nEMBEDDINGS ARE : \n\n")
                        f.write(str(np.array(l)))
                        f.write("\n\n")
                        mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
                        # print(mydict)
                        # print(len(mydict[9]))

                        # print(len(mydict[8]))

                        # print(len(mydict[7]))

                        # print(len(mydict[6]))
                        dictlist = []
                #print(mydict)
                        mydict1={}
                        for key in mydict:
                            mydict1[key]=list(mydict[key])
                        # print(len(mydict[0]))
                        
                        cluster_label={}
                        acc=0.0
        #                print(testlabels1[0][0])
                        lab=[i for i in range(10)]  
                        while(len(lab)!=0):
                            #iidx=max(mydict.items(), key=operator.itemgetter(1))[0]  #idx is the cluster no.
                            iidx=max(mydict1,key=mydict1.get)
                            cluster_label[iidx]=0
                            keys1=mydict1[iidx]
        #                    print(keys1)
                            dict1={}
                            print(keys1)
                            for i in (keys1):
                                if testlabels1[i][0] not in dict1:
                                    dict1[testlabels1[i][0]]=0
                                    
                                dict1[testlabels1[i][0]]+=1
                            idx1=max(dict1.items(), key=operator.itemgetter(1))[0]  #idx1 is the label of maximum occuring samples in idx cluste
                            while(idx1 not in lab and len(dict1)!=0):
                                del dict1[idx1]
                                idx1=max(dict1.items(), key=operator.itemgetter(1))[0] 
                            # print(dict1[idx1])
                            # print(dict1)
                            #print(cluster_label)
                            acc+=(dict1[idx1])
                            cluster_label[iidx]=idx1
                            # print(idx1)
                            lab.remove(idx1)
                            del mydict1[iidx]
                        print("Acc: ")
                        print(acc)
                        print(acc/len(testset))
                        f.write("\nACCURACY AFTER CLUSTERING IS:   ")
                        f.write(str(acc))
                        f.write("\n\n")
                        # tsne.fit(np.array(l),labels)
                        # tsne.poof()
                        # tsne.save()
 


print("--------Training----------")
cn=CNN1()
cn.initialization_layers_train("f","f","Fashion-MNIST",2,[6,3])

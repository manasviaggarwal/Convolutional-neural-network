#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# x_image=[]
# y_true=[]
# y_true_cls=[]

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
        print(Y)
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
                train_label=self.one_hot_encoding(train_label)
                train_set=np.array(train_set)
                self.trainset=train_set
                self.trainlabel=train_label
                print(len(train_label))
                x = tf.placeholder(tf.float32, shape=[None, 32*32*3], name='X')
                x_image = tf.reshape(x, [-1, 32, 32, 3])
                
                y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
                y_true_cls = tf.argmax(y_true, dimension=1)
                return x,x_image,y_true,y_true_cls

            elif nameofset=="Fashion-MNIST":
                kind='train'
                c=os.getcwd()
                path1=c+"/data/Fashion-MNIST/"
                labels_path = os.path.join(path1,'%s-labels-idx1-ubyte.gz'% kind)
                images_path = os.path.join(path1,'%s-images-idx3-ubyte.gz'% kind)
                with gzip.open(labels_path, 'rb') as lbpath:
                    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

                with gzip.open(images_path, 'rb') as imgpath:
                    images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)


                images=pd.DataFrame(images)
                labels=pd.DataFrame(labels)
                train_set=(images)
                train_label=(labels)
                train_set=np.array(train_set)
                print(len(train_label))
                train_label=np.array(train_label)
    #             train_set=self.normalize(np.array(train_set),0)
                train_label=self.one_hot_encoding(train_label)

    #             np.array(train_set).reshape(-1,28,28,1)
                self.trainset=train_set
                self.trainlabel=train_label
                x = tf.placeholder(tf.float32, shape=[None, 28*28], name='X')
                x_image = tf.reshape(x, [-1, 28, 28, 1])
                y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
                print(x_image)
                y_true_cls = tf.argmax(y_true, dimension=1)
                return x,x_image,y_true,y_true_cls
        elif v==1:
            if nameofset=="CIFAR-10":
                c=os.getcwd()
                test_set=pd.DataFrame()
                test_label=pd.DataFrame()
                path1=c+"/data/CIFAR-10/"
                for file in glob.glob(path1+"*"):
                    if "test_batch" in file:        
                        dict1=self.unpickle(file)

                        test_set1=(pd.DataFrame(dict1['data']))
                        test_set=pd.concat([test_set,test_set1])
                        test_set1=(pd.DataFrame(dict1['labels']))
                        test_label=pd.concat([test_label,test_set1])
                test_label=np.array(test_label) 
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
                test_label=np.array(test_label)
                print(len(test_set))

                test_label=self.one_hot_encoding(test_label)


                self.testset=test_set
                self.testlabel=test_label
                print((test_label).shape)



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

    # def initialization_layers_test(self,test_path,dataset):
        
    #         new_saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    #         new_saver.restore(sess, tf.train.latest_checkpoint('./'))


    #         x,x_image,y_true,y_true_cls=self.load_main(train_path,dataset,0)
    #         self.load_main(test_path,dataset,1)
    #         # print(len(x_image))
    #         if dataset=="Fashion-MNIST":
    #             layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=1, filter_size=filter_sizes[0], num_filters=5, name ="conv1")
           
    #         elif dataset=="CIFAR-10":
    #             layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=3, filter_size=filter_sizes[0], num_filters=5, name ="conv1")

    #         layer_pool1 = self.new_pool_layer(layer_conv1, name="pool1")
    #         layer_relu1 = self.new_relu_layer(layer_pool1, name="relu1")
            
    #         layer_pools=[]
    #         layer_relus=[]
    #         layer_convs=[]
    #         weight_convs=[]
            
    #         layer_convs.append(layer_conv1)
    #         layer_pools.append(layer_pool1)
    #         layer_relus.append(layer_relu1)
    #         weight_convs.append(weights_conv1)
            
    #         for k1 in range(1,no_of_layers):
    #             layer_conv1, weights_conv1 = self.new_conv_layer(input=layer_relus[k1-1], num_input_channels=5, filter_size=filter_sizes[k1], num_filters=15, name= "conv2")
    #             name2="pool"+str(k1+1)
    #             name1="relu"+str(k1+1)
                
    #             layer_pool1 = self.new_pool_layer(layer_conv1, name=name2)
    #             layer_relu1 = self.new_relu_layer(layer_pool1, name=name1)
                
    #             layer_convs.append(layer_conv1)
    #             layer_pools.append(layer_pool1)
    #             layer_relus.append(layer_relu1)
    #             weight_convs.append(weights_conv1)

            
    #         num_features = layer_relu1.get_shape()[1:4].num_elements()
    #         layer_flat = tf.reshape(layer_relu1, [-1, num_features])
            
    #         layer_fc1 = self.new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
    #         layer_relu3 = self.new_relu_layer(layer_fc1, name="relu"+str(no_of_layers+1))
            
    #         layer_fc2 = self.new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")

            
    #         with tf.variable_scope("Softmax"):
    #             y_pred = tf.nn.softmax(layer_fc2)
    #             y_pred_cls = tf.argmax(y_pred, dimension=1)

    #         with tf.name_scope("cross_ent"):
    #             cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
    #             cost = tf.reduce_mean(cross_entropy)

    #         with tf.name_scope("optimizer"):
    #             optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    #         with tf.name_scope("accuracy"):
    #             correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    #             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                

    #         num_epochs = 2
    #         batch_size = 100

    #         trainset=self.trainset
    #         testset=self.testset
    #         trainlabel=self.trainlabel
    #         testlabel=self.testlabel
            
    #         with tf.Session() as sess:
    #             sess.run(tf.global_variables_initializer())
    #             # writer.add_graph(sess.graph)
    #             for epoch in range(num_epochs):

    #                 start_time = time.time()
    #                 train_accuracy = 0
    #                 k=0
    #                 for batch in range(0, int(len(trainlabel)/batch_size)):
    #                     x_batch  = trainset[k:k+batch_size]
    #                     y_true_batch = trainlabel[k:k+batch_size]
    #                     k=k+batch_size
    #                     feed_dict_train = {x: x_batch, y_true: y_true_batch}
    #                     sess.run(optimizer, feed_dict=feed_dict_train)
    #                     train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
    #                 train_accuracy /= int(len(trainlabel)/batch_size)
    #                 vali_accuracy = sess.run(accuracy , feed_dict={x:testset, y_true:testlabel})
    #                 end_time = time.time()
    #                 print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
    #                 print("\tAccuracy:")
    #                 print ("\t- Training Accuracy:\t{}".format(train_accuracy))
    #                 print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
    #             saver = tf.train.Saver()
    #             saver.save(sess, 'my-test-model')

    def initialization_layers_train(self,train_path,test_path,dataset,no_of_layers,filter_sizes):
            global n1,n2
            x,x_image,y_true,y_true_cls=self.load_main(train_path,dataset,0)
            self.load_main(test_path,dataset,1)
            # print(len(x_image))
            if dataset=="Fashion-MNIST":
                layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=1, filter_size=10, num_filters=n1, name ="conv1")
           
            elif dataset=="CIFAR-10":
                layer_conv1, weights_conv1 = self.new_conv_layer(input=x_image, num_input_channels=3, filter_size=10, num_filters=n1, name ="conv1")

            layer_pool1 = self.new_pool_layer(layer_conv1, name="pool1")
            layer_relu1 = self.new_relu_layer(layer_pool1, name="relu1")
            
            layer_pools=[]
            layer_relus=[]
            layer_convs=[]
            weight_convs=[]
            
            layer_convs.append(layer_conv1)
            layer_pools.append(layer_pool1)
            layer_relus.append(layer_relu1)
            weight_convs.append(weights_conv1)
            
            for k1 in range(1,no_of_layers):
                namee="conv"+str(k1+1)
                layer_conv1, weights_conv1 = self.new_conv_layer(input=layer_relus[k1-1], num_input_channels=n1, filter_size=10, num_filters=n2, name= namee)
                name2="pool"+str(k1+1)
                name1="relu"+str(k1+1)
                
                layer_pool1 = self.new_pool_layer(layer_conv1, name=name2)
                layer_relu1 = self.new_relu_layer(layer_pool1, name=name1)
                
                layer_convs.append(layer_conv1)
                layer_pools.append(layer_pool1)
                layer_relus.append(layer_relu1)
                weight_convs.append(weights_conv1)
                n1=n2
            
            num_features = layer_relu1.get_shape()[1:4].num_elements()
            layer_flat = tf.reshape(layer_relu1, [-1, num_features])
            
            layer_fc1 = self.new_fc_layer(layer_flat, num_inputs=num_features, num_outputs=128, name="fc1")
            layer_relu3 = self.new_relu_layer(layer_fc1, name="relu"+str(no_of_layers+1))
            
            layer_fc2 = self.new_fc_layer(input=layer_relu3, num_inputs=128, num_outputs=10, name="fc2")

            
            with tf.variable_scope("Softmax"):
                y_pred = tf.nn.softmax(layer_fc2)
                y_pred_cls = tf.argmax(y_pred, dimension=1)

            with tf.name_scope("cross_ent"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
                cost = tf.reduce_mean(cross_entropy)

            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

            with tf.name_scope("accuracy"):
                # print("-----------")
                # print(y_pred_cls)
                correct_prediction = tf.equal(y_pred_cls, y_true_cls)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            num_epochs = 100
            batch_size = 100

            trainset=self.trainset
            testset=self.testset
            trainlabel=self.trainlabel
            testlabel=self.testlabel
            
            with tf.Session() as sess:
                global train_acc
                global f1_mi
                global f1_ma
                train_acc=[]
                f1_mi=[]
                f1_ma=[]
                sess.run(tf.global_variables_initializer())
                # writer.add_graph(sess.graph)
                for epoch in range(num_epochs):

                    start_time = time.time()
                    train_accuracy = 0
                    k=0
                    f1_macro=0
                    f1_micro=0

                    for batch in range(0, int(len(trainlabel)/batch_size)):
                        x_batch  = trainset[k:k+batch_size]
                        y_true_batch = trainlabel[k:k+batch_size]
                        k=k+batch_size
                        feed_dict_train = {x: x_batch, y_true: y_true_batch}
                        sess.run(optimizer, feed_dict=feed_dict_train)
                        a,y_p,y=sess.run([accuracy,y_pred_cls,y_true_cls], feed_dict=feed_dict_train)
                        train_accuracy += a
                        # print(y_true_cls)
                        # print(y_pred_cls)
                        f1_macro+=(f1_score(np.array(y_p), np.array(y), average='macro'))
                        f1_micro+=(f1_score(np.array(y_p), np.array(y), average='micro'))
                    train_accuracy /= int(len(trainlabel)/batch_size)
                    
                    train_acc.append(train_accuracy)

                    f1_macro /= int(len(trainlabel)/batch_size)
                    f1_micro /= int(len(trainlabel)/batch_size)

                    f1_ma.append(f1_macro)
                    f1_mi.append(f1_micro)

                    vali_accuracy = sess.run(accuracy ,feed_dict={x:testset, y_true:testlabel})
                    # vali_accuracy = sess.run(accuracy , feed_dict={x:testset, y_true:testlabel})
                    end_time = time.time()
                    print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
                    print("\tAccuracy:")

                    print("\tf1:")
                    print(f1_macro)

                    print ("\t- Training Accuracy:\t{}".format(train_accuracy))
                    print ("\t- Validation Accuracy:\t{}".format(vali_accuracy))
                saver = tf.train.Saver()
                saver.save(sess, 'my-test-model')


    
# In[ ]:

train_acc=[]
f1_mi=[]
f1_ma=[]
n1=0
n2=0
if __name__=='__main__':
    array_of_arguments=sys.argv


    if array_of_arguments[1]=="--test-data":
        

        # #print("wbdj")
        test_path=array_of_arguments[2]
        #test_label=array_of_arguments[4]

        if array_of_arguments[4]=="Fashion-MNIST":
            
            print("-------------Test------------")
            it=open("MNIST_PAR","rb")
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
            
            tt.Y,tt.rdimg=tt.load_data(test_path,"MNIST",1)
            
            tt.testing()
        elif array_of_arguments[4]=="CIFAR-10":
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
        #if array_of_arguments[6]=="MNIST":
        k=9
        i=1
        listofnodes=[]
        listofnodes.append(int(list_of_nodes[1:]))
        while(1):
            if array_of_arguments[k][-1]==']':
                listofnodes.append(int(array_of_arguments[k][:-1]))
                i+=1
                break
            listofnodes.append(int(array_of_arguments[k]))
            i+=1
            k+=1
        # for i1 in range(4):
        n1=5
        n2=15
        # cn=CNN1()
        # cn.initialization_layers_train(train_path,test_path,array_of_arguments[6],len(listofnodes),listofnodes)
        t=[]
        fmi=[]
        fma=[]
        # for i1 in range(4):
        cn=CNN1()
        cn.initialization_layers_train(train_path,test_path,array_of_arguments[6],2,[4,4])

        t.append(train_acc)
        fmi.append(f1_mi)
        fma.append(f1_ma)
        del cn

        n1=10
        n2=20
       
        cn=CNN1()
        cn.initialization_layers_train(train_path,test_path,array_of_arguments[6],2,[4,4,4])
        t.append(train_acc)
        fmi.append(f1_mi)
        fma.append(f1_ma)
        del cn

        n1=15
        n2=25

        cn=CNN1()
        cn.initialization_layers_train(train_path,test_path,array_of_arguments[6],2,[4,4,4,4])
        t.append(train_acc)
        fmi.append(f1_mi)
        fma.append(f1_ma)
        del cn

        n1=20
        n2=30

        cn=CNN1()
        cn.initialization_layers_train(train_path,test_path,array_of_arguments[6],2,[4,4,4,4,4])
        t.append(train_acc)
        fmi.append(f1_mi)
        fma.append(f1_ma)
        del cn

        legends=['5,15','10,20','15,25','20,30']
        cwds=os.getcwd()
        epochs=[i for i in range(100)]
        for i in range(4):
            plt.plot(epochs,t[i])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(legends)
        plt.savefig(cwds+'/C_task2_A')
        plt.show()
        plt.clf()
        for i in range(4):
            plt.plot(epochs,fma[i])
        plt.xlabel('Epochs')
        plt.ylabel('f1_macro')
        plt.legend(legends)
        plt.savefig(cwds+'/C_task2_MA')
        plt.show()
        plt.clf()
        for i in range(4):
            plt.plot(epochs,fmi[i])
        plt.xlabel('Epochs')
        plt.ylabel('f1_micro')
        plt.legend(legends)
        plt.savefig(cwds+'/C_task2_MI')
        plt.show()
        plt.clf()


    

    


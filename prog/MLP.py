# -*- coding: cp936 -*-
import random
import numpy as np
import loader

class MLP(object):
    def __init__(self,size):
        self.layer_num=len(size)
        self.size=size
        self.b=[np.random.randn(y,1) for y in size[1:]]
        self.w=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]

    def forward(self,a):
        for b,w in zip(self.b,self.w):
            a=self.sigmoid(np.dot(w,a)+b)
        return a

    def bp(self,x,y):
        #forward
        a_list=[x]
        z_list=[]
        for b,w in zip(self.b,self.w):
            z=np.dot(w,a_list[-1])+b
            z_list.append(z)
            a=self.sigmoid(z)
            a_list.append(a)
        #backward
        delta_b=[np.zeros(b.shape) for b in self.b]
        delta_w=[np.zeros(w.shape) for w in self.w]
        delta=(a_list[-1]-y)*self.sigmoid_der(z_list[-1])
        #print delta
        delta_b[-1]=delta
        delta_w[-1]=np.dot(delta,a_list[-2].transpose())
        #print delta_w[-1]
        for i in range(2,self.layer_num):
            z=z_list[-i]
            delta=np.dot(self.w[-i+1].transpose(),delta)*self.sigmoid_der(z)
            delta_b[-i]=delta
            delta_w[-i]=np.dot(delta,a_list[-i-1].transpose())
        return (delta_b,delta_w)
        
        
    def SGD(self,training_data,iter,batch_size,eta,test_data=None):
        if test_data:
            test_num=len(test_data)
        for i in range(iter):
            random.shuffle(training_data)
            batchs=[training_data[j:j+batch_size]
                         for j in range(0,len(training_data),batch_size)]
            for batch in batchs:
                self.update(batch,eta)
            if test_data:
                print "iter",i,"precision:",self.evaluate(test_data),"/",test_num
            else:
                print "iter",i,"finish!"
                
    def update(self,batch,eta):
        delta_b=[np.zeros(b.shape) for b in self.b]
        delta_w=[np.zeros(w.shape) for w in self.w]
        for x,y in batch:
            delta_b_re,delta_w_re=self.bp(x,y)
            #print delta_w_re
            delta_b=[b+re_b for b,re_b in zip(delta_b,delta_b_re)]
            delta_w=[w+re_w for w,re_w in zip(delta_w,delta_w_re)]
        self.w=[w-eta*(del_w/len(batch)) for w,del_w in zip(self.w,delta_w)]
        self.b=[b-eta*(del_b/len(batch)) for b,del_b in zip(self.b,delta_b)]
        
    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))

    def sigmoid_der(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def evaluate(self,test_data):
        results=[(np.argmax(self.forward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in results)
        
if __name__=="__main__":
    training_data,validation_data,test_data=loader.load_data_wrapper()
    classifier=MLP([784,30,10])
    classifier.SGD(training_data,30,10,3.0,test_data=test_data)

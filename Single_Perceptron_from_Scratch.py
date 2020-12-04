

import numpy as np
import pandas as pd
import math
from scipy import spatial
import matplotlib.pyplot as plt 





def metrics (output,k):
    total_true = 17528
    total_positives = 0
    true_positives = 0
    for i in range (k):#number of clusters 
        animals,countries,fruits,veggies,number_of_features,greater = 0,0,0,0,0,0
        for key,value in (output.items()):
            if i == value:#finds all the elements that are in the same cluster
                if key <= 49:
                    animals+= 1
                    number_of_features+=1
                if key >= 50 and key <= 210:
                    countries += 1
                    number_of_features+=1
                if key >= 211 and key <= 268:
                    fruits += 1
                    number_of_features+=1
                if key >= 269:
                    veggies += 1
                    number_of_features+=1
                    
        greater = max(animals,countries,fruits,veggies)
       
        if greater>1  :            
            true_positives += math.factorial(greater)/(math.factorial(2)*math.factorial(greater-2))
        else: 
            true_positives += 0
        
        if number_of_features > 1 :
            total_positives += math.factorial(number_of_features)/(math.factorial(2)*math.factorial(number_of_features-2))
        else:
            total_positives +=0
               
    precision = true_positives/total_positives
    recall = true_positives/total_true
    f1 = (2*precision*recall) / (precision+recall)
    print("precision ", precision)
    print("recall: ", recall)
    print("f1: " , f1)
    return precision,recall,f1
    

def Cosine(data,Centroids,output):
    
    for i in range(len(data)):# length of the data
        number =1000000
        for j in range(len(Centroids)):#length of the centroids
            result =0
            cosine_similarity =0
            result = 1 - spatial.distance.cosine(data[i,:] ,Centroids[j,:])
            cosine_similarity = np.sum(abs(result ))
            if cosine_similarity < number:
                number = cosine_similarity
                output.update({i:j})
    return Centroids,output


def Manhattan(data,Centroids,output):
    
    for i in range(len(data)):# length of the data
        number =1000000
        for j in range(len(Centroids)):#length of the centroids
            result =0
            manhattan =0
            result =((data[i,:] - Centroids[j,:]))
            manhattan = np.sum(abs(result ))
            if manhattan < number:
                number = manhattan
                output.update({i:j})
    return Centroids,output


def Euclidian(data,Centroids,output):
    
    for i in range(len(data)):# length of the data
        number =1000000
        for j in range(len(Centroids)):#length of the centroids
            result =0
            euclidian =0
            result =(data[i,:] - Centroids[j,:])
            euclidian = math.sqrt(np.sum(result **2))
            if euclidian < number:
                number = euclidian
                output.update({i:j})
    return Centroids,output

    
def Centroids_calculation(data,Centroids,output,k):
    keys=[]      
    for p in range (k):#number of clusters
        for key, value in output.items():#store the keys that are in the same cluster
            if value == k:#the value of the cluster
                keys.append(key)        
        newclustervalues = data[keys, :] 
        Centroids[p,:] =np.average(newclustervalues, axis=0) 
                
        keys=[]


def l2_norm(datax): 
    for i in range(len(datax)):#calculates l2 for train
            l2norm=np.sqrt(np.sum(datax.iloc[i] * datax.iloc[i]))
            datax.iloc[i] = datax.iloc[i] / (l2norm)
    return datax        



x1 = []
x2 = []
x3 = []
clusters=[1,2,3,4,5,6,7,8,9,10]
for k in range (10):
    
    output = {}
    datax = pd.read_csv(r'.\data.txt',sep=" ", header = None)
    datax=datax.iloc[:,1:]

    Centroids = datax.sample(n=clusters[k])#random instance from the dataset
    Centroids=Centroids.reset_index(drop=True)
    Centroids=Centroids.T.reset_index(drop=True).T
    data = datax
    data=data.reset_index(drop=True)
    data=data.T.reset_index(drop=True).T
    data=data.to_numpy()
    Centroids=Centroids.to_numpy()

    Centroids_norm = l2_norm(datax).sample(n=clusters[k])
    Centroids_norm=Centroids_norm.reset_index(drop=True)
    Centroids_norm=Centroids_norm.T.reset_index(drop=True).T
    data_l2_norm = l2_norm(datax)
    data_l2_norm=data_l2_norm.reset_index(drop=True)
    data_l2_norm=data_l2_norm.T.reset_index(drop=True).T
    data_l2_norm = data_l2_norm.to_numpy()
    Centroids_norm=Centroids_norm.to_numpy()


    for i in range (20):#uncomment one For loop each time. Uncomment to reproduce results using EUCLIDIAN
        Centroids,output = Euclidian(data,Centroids,output)
        Centroids_calculation(data,Centroids,output,clusters[k]) 
    
   # for i in range (20):#uncomment one For loop each time. Uncomment to reproduce results using MANHATTAN
    #    Centroids,output = Manhattan(data,Centroids,output)
     #   Centroids_calculation(data,Centroids,output,clusters[k])
        
    #for i in range (20):#uncomment one For loop each time. Uncomment to reproduce results using COSINE
     #   Centroids,output = Cosine(data,Centroids,output)
      #  Centroids_calculation(data,Centroids,output,clusters[k])
        
    #for i in range (20):#uncomment one For loop each time. Uncomment to reproduce results using EUCLIDIAN WITH NORMALISED DATA
     #   Centroids_norm,output = Euclidian(data_l2_norm,Centroids_norm,output)
      #  Centroids_calculation(data_l2_norm,Centroids_norm,output,clusters[k])
        
        
    #for i in range (20):#uncomment one For loop each time.#uncomment one For loop each time. Uncomment to reproduce results using MANHATTAN WITH NORMALISED DATA
     #   Centroids_norm,output = Manhattan(data_l2_norm,Centroids_norm,output)
      #  Centroids_calculation(data_l2_norm,Centroids_norm,output,clusters[k])
        
        
    print("")
    print("Metrics for cluster, ", clusters[k])
    precision,recall,f1 = metrics(output,clusters[k])
    
    x1.insert(k,precision)
    x2.insert(k,recall)
    x3.insert(k,f1)

    
    

 
# plotting the line points  
plt.plot(clusters, x1, label = "precision")
plt.plot(clusters, x2, label = "recall")
plt.plot(clusters, x3, label = "F1")

 
  
# show a legend on the plot 
plt.legend() 
  
# function to show the plot 
plt.show()    
    
    
    




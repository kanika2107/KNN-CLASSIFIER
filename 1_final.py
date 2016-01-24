import random
import copy
import math

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def average(s):
	return sum(s)*1.0/len(s);

name = raw_input("Enter the name of the dataset file\n");

confusion_matrix=[];
lenth=0;

if(name=="iris.txt"):
	lenth=3;
	for i in range(0,3):
		empty=[];
		for j in range(0,3):
			empty.append(0);
		confusion_matrix.append(empty);

if(name=="banknote.txt"):
	lenth=2;
        for i in range(0,2):
                empty=[];
                for j in range(0,2):
                        empty.append(0);
                confusion_matrix.append(empty);

if(name=="transfusion.data"):
	lenth=2;
        for i in range(0,2):
                empty=[];
                for j in range(0,2):
                        empty.append(0);
                confusion_matrix.append(empty);
			

#print Training_set
#print Test_set
accuracy_knn_1 = [];
for m in range(0,10):
	confusion_matrix=[];
	lenth=0;

	if(name=="iris.txt"):
		lenth=3;
		for i in range(0,3):
			empty=[];
			for j in range(0,3):
				empty.append(0);
			confusion_matrix.append(empty);

	if(name=="banknote.txt"):
		lenth=2;
		for i in range(0,2):
		        empty=[];
		        for j in range(0,2):
		                empty.append(0);
		        confusion_matrix.append(empty);

	if(name=="transfusion.data"):
		lenth=2;
		for i in range(0,2):
		        empty=[];
		        for j in range(0,2):
		                empty.append(0);
		        confusion_matrix.append(empty);
			
	f = open(str(name));
	f = f.readlines();
	random.shuffle(f);
	#print f
	length = len(f);

	Training_set = f[0:length/2];
	Test_set = f[length/2:length];

	#print len(Training_set)
	#print len(Test_set)

	for i in range(0,len(Training_set)):
		Training_set[i] = Training_set[i].split(',');

	for i in range(0,len(Test_set)):
		Test_set[i] = Test_set[i].split(',');
	knn = 1;

	total_number_correct = 0;
	total_number_testcases = len(Test_set);


	for i in range(0,len(Test_set)):
		Training_set_copy = [];
		Training_set_copy = copy.deepcopy(Training_set);
		for j in range(0,len(Training_set)):
			dist = 0;
			for k in range(0,len(Training_set[j])-1):
				dist = dist + (float(Training_set[j][k]) - float(Test_set[i][k]))*(float(Training_set[j][k])-float(Test_set[i][k]));
			Training_set_copy[j].append(dist);
		Training_set_copy = sorted(Training_set_copy,key=lambda s: s[len(s)-1]);
		Training_set_copy = Training_set_copy[0:knn];
		#print Training_set_copy 
		hashi={};
		for k in range(0,knn):
				lent=len(Training_set_copy[k]);
				item=Training_set_copy[k][lent-2];
				if item in hashi.keys():
					hashi[item]+=1;
				else:
					hashi[item]=1;
		#print hashi
		maxi = 0;
		for k in hashi.values():
			if maxi<k:
				maxi=k;

		for k in hashi.keys():
				if hashi[k]==maxi:
					predict=k;
					break;
		#print predict
		if(name=="iris.txt"):
			s1=str(predict);
			if(s1=="Iris-setosa\n"):
				t1=0;
			if(s1=="Iris-versicolor\n"):
				t1=1;
			if(s1=="Iris-virginica\n"):
				t1=2;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-setosa\n"):
				t2=0;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-versicolor\n"):
                                t2=1;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-virginica\n"):
                                t2=2;
			confusion_matrix[t1][t2]+=1;
		if(name=="banknote.txt"):
			predict = predict.strip();
			#print predict;
			t = Test_set[i][len(Test_set[i])-1].strip();
			#s1 = str(predict);
			if(predict=="0"):
				#print "kj";
				t1=0;
			else:
				#print "ggf";
				t1=1;
			if(t=="0"):
                                t2=0;
			else:
				t2=1;
			confusion_matrix[t1][t2]+=1;
		if(name=="transfusion.data"):
			predict = predict.strip();
			t = Test_set[i][len(Test_set[i])-1].strip();

			if(predict=="0"):
				t1=0;
			else:
				t1=1;
			if(t=="0"):
                                t2=0;
			else:
				t2=1;
			confusion_matrix[t1][t2]+=1;






		if(t1==t2):
				#print "kanika"
				total_number_correct+=1;

	#print knn,total_number_correct, total_number_testcases

	accuracy_knn_1.append(total_number_correct/float(total_number_testcases));
	print "For iteration", m;
	for ken in range(0,lenth):
		for ken1 in range(0,lenth):
			print confusion_matrix[ken][ken1],       
		print "\n";                    
	

#print accuracy_knn_1;
avg = average(accuracy_knn_1);
variance = map(lambda x:(x - avg)**2, accuracy_knn_1);
sd = math.sqrt(average(variance));
print "Random sampling with 1 nearest neighbour" ,avg,sd 

accuracy_knn_3=[];
for m in range(0,10):
	confusion_matrix=[];
	lenth=0;

	if(name=="iris.txt"):
		lenth=3;
		for i in range(0,3):
			empty=[];
			for j in range(0,3):
				empty.append(0);
			confusion_matrix.append(empty);

	if(name=="banknote.txt"):
		lenth=2;
		for i in range(0,2):
		        empty=[];
		        for j in range(0,2):
		                empty.append(0);
		        confusion_matrix.append(empty);

	if(name=="transfusion.data"):
		lenth=2;
		for i in range(0,2):
		        empty=[];
		        for j in range(0,2):
		                empty.append(0);
		        confusion_matrix.append(empty);
	f = open(str(name));
	f = f.readlines();
	random.shuffle(f);
	#print f
	length = len(f);

	Training_set = f[0:length/2];
	Test_set = f[length/2:length];

	#print len(Training_set)
	#print len(Test_set)

	for i in range(0,len(Training_set)):
		Training_set[i] = Training_set[i].split(',');

	for i in range(0,len(Test_set)):
		Test_set[i] = Test_set[i].split(',');
	knn = 3;

	total_number_correct = 0;
	total_number_testcases = len(Test_set);

	for i in range(0,len(Test_set)):
		Training_set_copy = [];
		Training_set_copy = copy.deepcopy(Training_set);
		for j in range(0,len(Training_set)):
			dist = 0;
			for k in range(0,len(Training_set[j])-1):
				dist = dist + (float(Training_set[j][k]) - float(Test_set[i][k]))*(float(Training_set[j][k])-float(Test_set[i][k]));
			Training_set_copy[j].append(dist);
		Training_set_copy = sorted(Training_set_copy,key=lambda s: s[len(s)-1]);
		Training_set_copy = Training_set_copy[0:knn];
		#print Training_set_copy 
		hashi={};
		for k in range(0,knn):
				lent=len(Training_set_copy[k]);
				item=Training_set_copy[k][lent-2];
				if item in hashi.keys():
					hashi[item]+=1;
				else:
					hashi[item]=1;
		#print hashi
		maxi = 0;
		for k in hashi.values():
			if maxi<k:
				maxi=k;

		for k in hashi.keys():
				if hashi[k]==maxi:
					predict=k;
					break;
		
		if(name=="iris.txt"):
			s1=str(predict);
			if(s1=="Iris-setosa\n"):
				t1=0;
			if(s1=="Iris-versicolor\n"):
				t1=1;
			if(s1=="Iris-virginica\n"):
				t1=2;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-setosa\n"):
				t2=0;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-versicolor\n"):
                                t2=1;
			if(str(Test_set[i][len(Test_set[i])-1])=="Iris-virginica\n"):
                                t2=2;
			confusion_matrix[t1][t2]+=1;
		if(name=="banknote.txt"):
			predict = predict.strip();
			#print predict;
			t = Test_set[i][len(Test_set[i])-1].strip();
			#s1 = str(predict);
			if(predict=="0"):
				#print "kj";
				t1=0;
			else:
				#print "ggf";
				t1=1;
			if(t=="0"):
                                t2=0;
			else:
				t2=1;
			confusion_matrix[t1][t2]+=1;
		if(name=="transfusion.data"):
			predict = predict.strip();
			t = Test_set[i][len(Test_set[i])-1].strip();

			if(predict=="0"):
				t1=0;
			else:
				t1=1;
			if(t=="0"):
                                t2=0;
			else:
				t2=1;
			confusion_matrix[t1][t2]+=1;

		#print predict
		if(t1==t2):
				#print "kanika"
				total_number_correct+=1;

	#print knn,total_number_correct, total_number_testcases
	accuracy_knn_3.append(total_number_correct/float(total_number_testcases));
	print "For iteration", m;
	for ken in range(0,lenth):
		for ken1 in range(0,lenth):
			print confusion_matrix[ken][ken1],       
		print "\n"; 

#print accuracy_knn_3
avg = average(accuracy_knn_3);
variance = map(lambda x:(x - avg)**2, accuracy_knn_3);
sd = math.sqrt(average(variance));
print "Random sampling with 3 nearest neighbour" ,avg,sd  

grand_mean = [];
for m in range(0,10):
	k_fold = 5;
	length = len(f);
	length = length/k_fold;
	random.shuffle(f);
	i=0;
	total_number_correct=0;
	total_number_of_testcases=len(f);
	plot_accuracies=[];
	knn = 1

	count = 0;
	each_fold_accuracies=[];
	while i<len(f) and count<k_fold:
		if count==k_fold-1:
			test_set = f[i:len(f)];
			training_set = f[0:i];
		else:
			test_set = f[i:i+length];
			training_set = f[0:i];
			if i+length < len(f):
				training_set.extend(f[i+length:len(f)]);

		#print test_set 
		#print "\n"
		#	print training_set
		#	print "\n"
		i = i + length;
		count = count +1;
		for j in range(0,len(test_set)):
			test_set[j]=test_set[j].split(",");
		#print test_set
		for j in range(0,len(training_set)):
			training_set[j] = training_set[j].split(",");
		#knn = raw_input("Enter the k for k nearest neighbour ");
		#knn = int(knn);
		for knn in range(1,2):
		    total_number_correct=0;
		    total_number_of_testcases=len(test_set);
		    for j in range(0,len(test_set)):
			#print training_set
			training_set_copy = [];
			test_sample = test_set[j];
			#print test_sample
			training_set_copy = copy.deepcopy(training_set);
			for k in range(0,len(training_set_copy)):
				sumi =0;
				for l in range(0,len(test_sample)-1):
				         sumi=sumi+(float(test_sample[l])-float(training_set_copy[k][l]))*(float(test_sample[l])-float(training_set_copy[k][l]));
				training_set_copy[k].append(sumi);
			#print training_set_copy
			training_set_copy = sorted(training_set_copy,key=lambda s: s[len(s)-1]);
			#print training_set_copy
			training_set_copy = training_set_copy[0:knn];
			#print training_set_copy
			hashi={};
			for k in range(0,knn):
				lent=len(training_set_copy[k]);
				item=training_set_copy[k][lent-2];
				if item in hashi.keys():
					hashi[item]+=1;
				else:
					hashi[item]=1;
			maxi = 0;
			for k in hashi.values():
				if maxi<k:
					maxi=k;
			#print maxi
			for k in hashi.keys():
				if hashi[k]==maxi:
					predict=k;
					break;
			#print test_sample
			if(predict==test_sample[len(test_sample)-1]):
				#print "kanika"
				total_number_correct+=1;
		    #print knn,total_number_correct, total_number_of_testcases
		    each_fold_accuracies.append(total_number_correct/float(total_number_of_testcases));

	avg = average(each_fold_accuracies);
	variance = map(lambda x:(x - avg)**2, each_fold_accuracies);
	sd = math.sqrt(average(variance));
	print "Five fold cross validation with 1 nearest neighbour" ,avg,sd 
	grand_mean.append(avg);

print "Grand mean for 1 nearest neighbour" ,average(grand_mean);

grand_mean = [];
for m in range(0,10):
	k_fold = 5;
	length = len(f);
	length = length/k_fold;
	random.shuffle(f);
	i=0;
	total_number_correct=0;
	total_number_of_testcases=len(f);
	plot_accuracies=[];
	knn = 3

	count = 0;
	each_fold_accuracies=[];
	while i<len(f) and count<k_fold:
		if count==k_fold-1:
			test_set = f[i:len(f)];
			training_set = f[0:i];
		else:
			test_set = f[i:i+length];
			training_set = f[0:i];
			if i+length < len(f):
				training_set.extend(f[i+length:len(f)]);

		#print test_set 
		#print "\n"
		#	print training_set
		#	print "\n"
		i = i + length;
		count = count +1;
		for j in range(0,len(test_set)):
			test_set[j]=test_set[j].split(",");
		#print test_set
		for j in range(0,len(training_set)):
			training_set[j] = training_set[j].split(",");
		#knn = raw_input("Enter the k for k nearest neighbour ");
		#knn = int(knn);
		for knn in range(1,2):
		    total_number_correct=0;
		    total_number_of_testcases=len(test_set);
		    for j in range(0,len(test_set)):
			#print training_set
			training_set_copy = [];
			test_sample = test_set[j];
			#print test_sample
			training_set_copy = copy.deepcopy(training_set);
			for k in range(0,len(training_set_copy)):
				sumi =0;
				for l in range(0,len(test_sample)-1):
				         sumi=sumi+(float(test_sample[l])-float(training_set_copy[k][l]))*(float(test_sample[l])-float(training_set_copy[k][l]));
				training_set_copy[k].append(sumi);
			#print training_set_copy
			training_set_copy = sorted(training_set_copy,key=lambda s: s[len(s)-1]);
			#print training_set_copy
			training_set_copy = training_set_copy[0:knn];
			#print training_set_copy
			hashi={};
			for k in range(0,knn):
				lent=len(training_set_copy[k]);
				item=training_set_copy[k][lent-2];
				if item in hashi.keys():
					hashi[item]+=1;
				else:
					hashi[item]=1;
			maxi = 0;
			for k in hashi.values():
				if maxi<k:
					maxi=k;
			#print maxi
			for k in hashi.keys():
				if hashi[k]==maxi:
					predict=k;
					break;
			#print test_sample
			if(predict==test_sample[len(test_sample)-1]):
				#print "kanika"
				total_number_correct+=1;
		    #print knn,total_number_correct, total_number_of_testcases
		    each_fold_accuracies.append(total_number_correct/float(total_number_of_testcases));

	avg = average(each_fold_accuracies);
	variance = map(lambda x:(x - avg)**2, each_fold_accuracies);
	sd = math.sqrt(average(variance));
	print "Five fold cross validation with 3 nearest neighbour" ,avg,sd
	grand_mean.append(avg);   

print "Grand mean for 3 nearest neighbour" ,average(grand_mean);  

def drange(start, stop, step):
     r = start
     while r < stop:
     	yield r
     	r += step


f = open("iris.txt","r");
f =f.readlines();
#print f
for i in range(0,len(f)):
	f[i] = f[i].split(',');

#print f

plt.xlim(0,5);
plt.ylim(0,3.5);
plt.xlabel("Sepal width");
plt.ylabel("Petal width");
'''for i in range(0,len(f)):
	#print f[i][4];
	k = str(f[i][4]);
	#print k
	if(k=="Iris-setosa\n"):
	    #print "a";
	    plt.plot(float(f[i][1]), float(f[i][3]), 'kx');
	if(k=="Iris-versicolor\n"):
	    #print "t";
	    plt.plot(float(f[i][1]),float(f[i][3]),'kx');
	if(k=="Iris-virginica\n"):
	    #print "b";
	    plt.plot(float(f[i][1]),float(f[i][3]),'kx');

#plt.show();'''

for i in drange(0,5,0.1):
	for j in drange(0,3.5,0.1):
		test_set = [i,j];
		Training_set_copy = copy.deepcopy(f);
		#print test_set;
		for k in range(0,len(f)):
			train_set = f[k];
			dist = (float(Training_set_copy[k][1])-i)*(float(Training_set_copy[k][1])-i) + (float(Training_set_copy[k][3])-j)*(float(Training_set_copy[k][3])-j);
			Training_set_copy[k].append(dist);
		Training_set_copy = sorted(Training_set_copy,key=lambda s: s[len(s)-1]);
		ans = Training_set_copy[0];
		k = str(ans[len(ans)-2]);
		if(k=="Iris-setosa\n"):
	   	    #print "a";
	            plt.plot(i, j, 'go');
	        if(k=="Iris-versicolor\n"):
	            #print "t";
	            plt.plot(i,j,'ro');
	        if(k=="Iris-virginica\n"):
	            #print "b";
	            plt.plot(i,j,'bo');
		
for i in range(0,len(f)):
	#print f[i][4];
	k = str(f[i][4]);
	#print k
	if(k=="Iris-setosa\n"):
	    #print "a";
	    plt.plot(float(f[i][1]), float(f[i][3]), 'cx');
	if(k=="Iris-versicolor\n"):
	    #print "t";
	    plt.plot(float(f[i][1]),float(f[i][3]),'mx');
	if(k=="Iris-virginica\n"):
	    #print "b";
	    plt.plot(float(f[i][1]),float(f[i][3]),'kx');

plt.show();

'''n_neighbors = 1;

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()'''

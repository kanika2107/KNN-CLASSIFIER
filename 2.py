import matplotlib.pyplot as plt
import copy

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

for i in drange(0,5,0.01):
	for j in drange(0,3.5,0.01):
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

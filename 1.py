import random
import copy


k = raw_input("Enter the name of the file ");
f = open(str(k));
k_fold = raw_input("Enter the value for k fold cross validation ");

k_fold = int(k_fold);

f = f.readlines();
#print f
length = len(f);
length = length/k_fold;
random.shuffle(f);
i=0;
total_number_correct=0;
total_number_of_testcases=len(f);
plot_accuracies=[];
for knn in range(1,6):
	plot_accuracies.append(0);

count = 0;
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
	for knn in range(1,6):
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
	    print knn,total_number_correct, total_number_of_testcases

            plot_accuracies[knn-1] = plot_accuracies[knn-1]+ (total_number_correct)/float(total_number_of_testcases);

	
for i in range(1,6):
	plot_accuracies[i-1]=(plot_accuracies[i-1]/k_fold)*100;

print plot_accuracies

plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.errorbar(range(1,6),plot_accuracies,xerr=0,yerr=0)
plt.show();


	



from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split as splitter
import pickle
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB,CategoricalNB
from sklearn.neighbors import KNeighborsClassifier								
from sklearn.linear_model import LogisticRegression,SGDClassifier,LogisticRegressionCV
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

data = datasets.load_breast_cancer()
p=data

dataset2_train = pd.read_csv("nba_logreg.csv")


df2 = pd.DataFrame(data=dataset2_train)
df2.fillna(999,inplace=True)

Labels = df2.TARGET_5Yrs
Features = df2.loc[:, df2.columns != 'TARGET_5Yrs']
Features = Features.loc[:,Features.columns !='Name']

X_train2,X_test2,y_train2,y_test2 = splitter(Features,
										 Labels,
										 test_size=0.3,
										 random_state=12)



# print(X_train2)										 
# exit()
										 

# print(data.DESCR)
# print(data.feature_names)
# print(data.data)
# print(data.target)
# print(data.target_names)


add=data.target.reshape(569,1)	
columns = np.append(data.feature_names, 
					data.target_names[0],
					axis=None)
					
# data=np.hstack((data.data,add))					

data = np.append(data.data,
				 add,
				 axis=1)						

df = pd.DataFrame(data=data,columns=columns)


# Comparing any two features , X= feature on x axis and Y = feature on y axis
# Green plots will show cancerious people

# area1 = np.ma.masked_where(df.iloc[:,30] <1,
						   # df.iloc[:,30])
						   
# x=1
# y=5
# plt.xlabel(columns[x])
# plt.ylabel(columns[y])

# plt.scatter(df.iloc[:,x],
			# df.iloc[:,y],
			# c='red')
			
# plt.scatter(df.iloc[:,x],
			# df.iloc[:,y],
			# c='green',
			# s=area1*15,
			# marker='^')

# plt.show()




X_train,X_test,y_train,y_test = splitter(p.data,
										 p.target,
										 test_size=0.3,
										 random_state=12)
 
									 
# Run  only once



LR = LogisticRegression()

gnb1 = GaussianNB()
gnb2=MultinomialNB()
gnb3=ComplementNB()
gnb4=BernoulliNB()
gnb5=CategoricalNB()

neigh = KNeighborsClassifier(n_neighbors=1)
 
kmeans = KMeans(n_clusters=2, random_state=1)
clf = svm.SVC()
forester = RandomForestClassifier(max_depth=5, random_state=1)
line = SGDClassifier()
newLG=LogisticRegressionCV()
ada = AdaBoostClassifier(n_estimators=15, random_state=0)

param_grid = {'C': [0.00001, 0.001,0.01,0.1,1,10,100,1000,1000]}
CV = GridSearchCV(LogisticRegression(),param_grid,verbose=False)

# for  dataset 2 <--------------------------------------

LR2 = LogisticRegression()

gnb12 = GaussianNB()
gnb22=MultinomialNB()
gnb32=ComplementNB()
gnb42=BernoulliNB()
gnb52=CategoricalNB()

neigh2 = KNeighborsClassifier(n_neighbors=1)
 
kmeans2 = KMeans(n_clusters=2, random_state=1)
clf2 = svm.SVC()
forester2 = RandomForestClassifier(max_depth=5, random_state=1)
line2 = SGDClassifier()
newLG2=LogisticRegressionCV()
ada2 = AdaBoostClassifier(n_estimators=15, random_state=0)

param_grid = {'C': [0.00001, 0.001,0.01,0.1,1,10,100,1000,1000]}
CV2 = GridSearchCV(LogisticRegression(),param_grid,verbose=False)


LR.fit(X_train,y_train)

gnb1.fit(X_train, y_train)
gnb2.fit(X_train, y_train)
gnb3.fit(X_train, y_train)
gnb4.fit(X_train, y_train)
gnb5.fit(X_train, y_train)
neigh.fit(X_train, y_train)
kmeans.fit(X_train)
clf.fit(X_train, y_train)
forester.fit(X_train, y_train)
line.fit(X_train, y_train)
newLG.fit(X_train, y_train)
ada.fit(X_train, y_train)
CV.fit(X_train,y_train)

# for dataset 2

LR2.fit(X_train2,y_train2)

gnb12.fit(X_train2, y_train2)
gnb22.fit(X_train2, y_train2)
gnb32.fit(X_train2, y_train2)
gnb42.fit(X_train2, y_train2)
gnb52.fit(X_train2, y_train2)
neigh2.fit(X_train2, y_train2)
kmeans2.fit(X_train2)
clf2.fit(X_train2, y_train2)
forester2.fit(X_train2, y_train2)
line2.fit(X_train2, y_train2)
newLG2.fit(X_train2, y_train2)
ada2.fit(X_train2, y_train2)
CV2.fit(X_train2,y_train2)



pickle.dump(LR, open('logisticRegression', 'wb'))
pickle.dump(gnb1, open('naiveBayes', 'wb'))
pickle.dump(gnb2, open('naiveBayes2', 'wb'))
pickle.dump(gnb3, open('naiveBayes3', 'wb'))
pickle.dump(gnb4, open('naiveBayes4', 'wb'))
pickle.dump(gnb5, open('naiveBayes5', 'wb'))
pickle.dump(neigh, open('neigh', 'wb'))
pickle.dump(kmeans, open('kmeans', 'wb'))
pickle.dump(clf, open('clf', 'wb'))
pickle.dump(forester, open('forester', 'wb'))
pickle.dump(line, open('line', 'wb'))
pickle.dump(newLG, open('newLG', 'wb'))
pickle.dump(ada, open('ada', 'wb'))
pickle.dump(CV, open('CV', 'wb'))

# for dataset 2

pickle.dump(LR2, open('2logisticRegression', 'wb'))
pickle.dump(gnb12, open('2naiveBayes', 'wb'))
pickle.dump(gnb22, open('2naiveBayes2', 'wb'))
pickle.dump(gnb32, open('2naiveBayes3', 'wb'))
pickle.dump(gnb42, open('2naiveBayes4', 'wb'))
pickle.dump(gnb52, open('2naiveBayes5', 'wb'))
pickle.dump(neigh2, open('2neigh', 'wb'))
pickle.dump(kmeans2, open('2kmeans', 'wb'))
pickle.dump(clf2, open('2clf', 'wb'))
pickle.dump(forester2, open('2forester', 'wb'))
pickle.dump(line2, open('2line', 'wb'))
pickle.dump(newLG2, open('2newLG', 'wb'))
pickle.dump(ada2, open('2ada', 'wb'))
pickle.dump(CV2, open('CV2', 'wb'))

''

# now! everytime you need LR comment above three lines, 
# when you've once run them
# and  now use this one to unpickle the object


LR = pickle.load(open('logisticRegression', 'rb'))

gnb1 = pickle.load(open('naiveBayes', 'rb'))
gnb2 = pickle.load(open('naiveBayes2', 'rb'))
gnb3 = pickle.load(open('naiveBayes3', 'rb'))
gnb4 = pickle.load(open('naiveBayes4', 'rb'))
gnb5 = pickle.load(open('naiveBayes5', 'rb'))

neigh = pickle.load(open('neigh', 'rb'))
kmeans = pickle.load(open('kmeans', 'rb'))
clf = pickle.load(open('clf', 'rb'))
forester = pickle.load(open('forester', 'rb'))
line = pickle.load(open('line', 'rb'))
newLG = pickle.load(open('newLG', 'rb'))
ada = pickle.load(open('ada', 'rb'))
CV = pickle.load(open('CV', 'rb'))

# for dartaset 2

LR2 = pickle.load(open('2logisticRegression', 'rb'))

gnb12 = pickle.load(open('2naiveBayes', 'rb'))
gnb22 = pickle.load(open('2naiveBayes2', 'rb'))
gnb32 = pickle.load(open('2naiveBayes3', 'rb'))
gnb42 = pickle.load(open('2naiveBayes4', 'rb'))
gnb52 = pickle.load(open('2naiveBayes5', 'rb'))

neigh2 = pickle.load(open('2neigh', 'rb'))
kmeans2 = pickle.load(open('2kmeans', 'rb'))
clf2 = pickle.load(open('2clf', 'rb'))
forester2 = pickle.load(open('2forester', 'rb'))
line2 = pickle.load(open('2line', 'rb'))
newLG2 = pickle.load(open('2newLG', 'rb'))
ada2 = pickle.load(open('2ada', 'rb'))
CV2 = pickle.load(open('CV2', 'rb'))
 


data = { 

"Algorithm":['logistic_regerssion',"GaussianNB:",
"MultinomialNB:","ComplementNB:","BernoulliNB:",
"KNN:","K means:","SVM:","Random Forest: ",
"SGDClassifier","LogisticR CV:","ada boast:","Cross Validation LR:"
   ],
"Score_1":[

LR.score(X_test,y_test),
gnb1.score(X_test,y_test),
gnb2.score(X_test,y_test),
gnb3.score(X_test,y_test),
gnb4.score(X_test,y_test),
neigh.score(X_test,y_test),
kmeans.score(X_test,y_test),
clf.score(X_test,y_test),
forester.score(X_test,y_test),
line.score(X_test,y_test),
newLG.score(X_test,y_test),
ada.score(X_test,y_test),
CV.score(X_test,y_test)

],


"Score_2":[

LR2.score(X_test2,y_test2),
gnb12.score(X_test2,y_test2),
gnb22.score(X_test2,y_test2),
gnb32.score(X_test2,y_test2),
gnb42.score(X_test2,y_test2),
neigh2.score(X_test2,y_test2),
kmeans2.score(X_test2,y_test2),
clf2.score(X_test2,y_test2),
forester2.score(X_test2,y_test2),
line2.score(X_test2,y_test2),
newLG2.score(X_test2,y_test2),
ada2.score(X_test2,y_test2),
CV2.score(X_test2,y_test2)

]

}
 

ScoreTable = pd.DataFrame(data=data)
print(ScoreTable)


from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
#Statistical Analysis
#What questions could we ask here? Is walking pace the most affected by age? gender? height?
#What feature of us is most telling about the walking pace?
#talk about the equipment we used

#WE CAN SAY THAT CLT ALLOWS IT FOR EACH ACCELERATION'S distribution is aight, so we can do a oneway anova on the frequency of X 
#pvalue = 0 for underflow meaning that we reject the null hypothesis that the mean frequencies are equal.
#this means that the frequency of steps are different in some way. we don't know which way yet, but this gives us the confidence to continue with more tests.
freq = pd.read_csv('anovafreq.csv')
freq = freq[freq.index < 574]
TJ = freq['TJ']
Hana = freq['Hana']
Angus = freq['Angus']
Barry = freq['Barry']
Kevin = freq['Kevin']
Anna = freq['Anna']
Adam = freq['Adam']
anova = stats.f_oneway(TJ,Hana,Angus,Barry,Kevin,Anna,Adam).pvalue
print("ANOVA p-value " + str(anova) + "\n")
#Post Hoc Analysis is not beneficial to our report because we are only interested in how the walking paces differ
#and not so much about whose walking pace is different.

#We can do the mann-whitneyu test because our data satisfies the assumptions that (1) observations are independent and (2) walking pace can be sorted by magnitude

walk = pd.read_csv('walkdata.csv')
#Do young people walk faster than old people?
#p=0.0876
old = walk[walk['age'] > 30]
young = walk[walk['age'] <= 30]

oldpace = old['pace']
youngpace = young['pace']
u_testAge = stats.mannwhitneyu(oldpace,youngpace).pvalue
print("MANN-WHITNEYU TEST FOR AGE: " + str(u_testAge))

#Do males walk faster than females?
#p = 0.04068
male = walk[walk['gender'] == 'male']['pace']
female = walk[walk['gender'] == 'female']['pace']

u_testGender = stats.mannwhitneyu(male,female).pvalue
print("MANN-WHITNEYU TEST FOR GENDER: " + str(u_testGender))

sns.set(color_codes=True)
plt.title('Mann-Whitney U-test for Gender')
plt.xlabel('Pace (steps/second)')
plt.ylabel('Count')
plt.hist(female, label='female')
plt.hist(male,label='male')
plt.legend()
plt.savefig('u_test_gender.png')

#Do tall people walk faster than short people?
#use the average height for cutoff between tall and short. presents a bias because of the small data sample, but its better than an uneducated guess
#p = 0.0558, close but we still can't come to a reasonable conclusion with height that it has some insights on how fast people walk.
cutoff = walk['height'].sum() / walk['height'].size
#print('cutoff for tall people: ' + str(cutoff))    #171.57cm

tall = walk[walk['height']>= cutoff]['pace']
short = walk[walk['height']< cutoff]['pace']

u_testHeight = stats.mannwhitneyu(tall,short).pvalue
print("MANN-WHITNEYU TEST FOR HEIGHT: " + str(u_testHeight))

#MACHINE LEARNING CLASSIFICATION
#setting up the data to be trained and tested on
#0 for male and 1 for female
#classifications were slow for pace <54, medium for 54 < pace <56, and fast for 56 < pace
#we wanted to set up MLs Classifier to see if, based on age, gender, and height, they can predict the correct walking pace

#train_test_split on the data, removing adam and counting him as the unseen data
print('\nMACHINE LEARNING CLASSIFIERS\n')
walk1 = walk.copy()
walk1 = walk1.replace(to_replace='male',value=0)
walk1 = walk1.replace(to_replace='female',value=1)
stop = walk1['pace'].size
for i in range(stop):
    if walk1.iloc[i]['pace'] <= 108.0:
        walk1 = walk1.replace(to_replace=walk1.iloc[i]['pace'],value='slow')
    elif (walk1.iloc[i]['pace'] > 108.0) & (walk1.iloc[i]['pace'] < 120.0):
        walk1 = walk1.replace(to_replace=walk1.iloc[i]['pace'],value='medium')
    else:
        walk1 = walk1.replace(to_replace=walk1.iloc[i]['pace'],value='fast')

#ML WITH GENDER
print("USING GENDER TO CLASSIFY WALKING PACE\n")
xvals = ['gender']
X = walk1[xvals]
y = walk1['pace']

X_train,X_test, y_train,y_test = train_test_split(X,y,train_size = (3/7))

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
print("ACCURACY SCORE FOR DECISION TREE: " + str(tree.score(X_test,y_test)))

nb = GaussianNB()
nb.fit(X_train,y_train)
print("ACCURACY SCORE FOR NAIVE BAYES: " +str(nb.score(X_test,y_test) ))

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
print("ACCURACY SCORE FOR KNN: " + str(knn.score(X_test,y_test)) )

#ML WITH AGE
print("\nUSING AGE TO CLASSIFY WALKING PACE\n")
xvals = ['age']
X = walk1[xvals]
y = walk1['pace']

X_train,X_test, y_train,y_test = train_test_split(X,y,train_size = (3/7))

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
print("ACCURACY SCORE FOR DECISION TREE: " + str(tree.score(X_test,y_test)))

nb = GaussianNB()
nb.fit(X_train,y_train)
print("ACCURACY SCORE FOR NAIVE BAYES: " +str(nb.score(X_test,y_test) ))

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
print("ACCURACY SCORE FOR KNN: " + str(knn.score(X_test,y_test)) )

#ML WITH HEIGHT 
print("\nUSING HEIGHT TO CLASSIFY WALKING PACE\n")
xvals = ['height']
X = walk1[xvals]
y = walk1['pace']

X_train,X_test, y_train,y_test = train_test_split(X,y,train_size = (3/7))

tree = DecisionTreeClassifier()
tree.fit(X_train,y_train)
print("ACCURACY SCORE FOR DECISION TREE: " + str(tree.score(X_test,y_test)))

nb = GaussianNB()
nb.fit(X_train,y_train)
print("ACCURACY SCORE FOR NAIVE BAYES: " +str(nb.score(X_test,y_test) ))

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
print("ACCURACY SCORE FOR KNN: " + str(knn.score(X_test,y_test)) )


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_excel(r'C:/Users/karth/Desktop/isolation/data_new.xlsx')
df.head(10)

df = df.iloc[:, [3, 4]]

model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0)
model.fit(df[['Annual Income (k$)','Spending Score (1-100)']])

df['scores']=model.decision_function(df[['Annual Income (k$)','Spending Score (1-100)']])
df['anomaly']=model.predict(df[['Annual Income (k$)','Spending Score (1-100)']])
df.head(20)

anomaly=df.loc[df['anomaly']==-1]
anomaly_index=list(anomaly.index)
print(anomaly)

# #outlier_class and outlier_score must be array
# fpr,tpr,thresholds_sorted=metrics.roc_curve(outlier_class,scores)
# aucvalue_sorted=metrics.auc(fpr,tpr)
# aucvalue_sorted




##apply an Isolation forest
outlier_detect = IsolationForest(n_estimators=250, max_samples=1000, contamination=.04, max_features=df.shape[1])
outlier_detect.fit(df)
outliers_predicted = outlier_detect.predict(df)

#check the results
df['outlier'] = outliers_predicted
plt.figure(figsize = (20,10))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['outlier'])
plt.show()





import pickle

pickle.dump(model, open("savesss.pckl", "wb"))

















def iqr_bounds(scores,k=1.5):
    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower_bound=(q1 - k * iqr)
    upper_bound=(q3 + k * iqr)
    print("Lower bound:{} \nUpper bound:{}".format(lower_bound,upper_bound))
    return lower_bound,upper_bound
lower_bound,upper_bound=iqr_bounds(df['scores'],k=2)


#check the results
df['outlier'] = anomaly
plt.figure(figsize = (20,10))
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['outlier'])
plt.show()






def plot_outlier(df, outliers, rbf, nu):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	print(df[10])
	ax.scatter(df[:, 0], df[:, 1], color='b', label='normal')
	ax.scatter(outliers[:, 0], outliers[:, 1], color='r', label='outliers')
	ax.set_xlabel('L')
	ax.set_ylabel('F')
	ax.set_zlabel('M')
	ax.legend()
	plt.title('kernel=%s, nu=%.2f' % (rbf, nu))
	plt.show()


df['anomaly']=0
df['anomaly']=(df['scores'] < lower_bound) |(df['scores'] > upper_bound)
df['anomaly']=df['anomaly'].astype(int)
plot_outlier(df,'iqr based')
import pandas as pd
import numpy as np


df=pd.read_csv("https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv", header=None, usecols=[0,1,2])
#print(df.head())

df.columns=['Class label', 'Alcohol', 'Malic acid']

print(df.head())

from sklearn import preprocessing


#Standard Scaler
std_scale=preprocessing.StandardScaler().fit(df[['Alcohol','Malic acid']])
df_std = std_scale.transform(df[['Alcohol', 'Malic acid']])
print('Mean after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(df_std[:,0].mean(), df_std[:,1].mean()))

#MinMax Scaler
minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol', 'Malic acid']])
df_minmax = minmax_scale.transform(df[['Alcohol', 'Malic acid']])
print('\nStandard deviation after standardization:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(df_std[:,0].std(), df_std[:,1].std()))
print('Min-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(df_minmax[:,0].min(), df_minmax[:,1].min()))
print('\nMax-value after min-max scaling:\nAlcohol={:.2f}, Malic acid={:.2f}'.format(df_minmax[:,0].max(), df_minmax[:,1].max()))

print(df_std[0:5])
print(df_minmax[0:5])

from matplotlib import pyplot as plt

def plot():
	plt.figure(figsize=(8,6))

	plt.scatter(df['Alcohol'], df['Malic acid'], color='green', label='input scale', alpha=0.5)
	plt.scatter(df_std[:,0], df_std[:,1], color='red', label='standard scaler scale', alpha=0.3)
	plt.scatter(df_minmax[:,0], df_minmax[:,1], color='yellow', label='MinMax scaler scale', alpha=0.3)
	plt.title('Alcohol and Malic Acid content of the wine dataset')
	plt.xlabel('Alcohol')
	plt.ylabel('Malic Acid')
	plt.legend(loc='upper left')
	plt.grid()
	plt.tight_layout()

plot()
plt.show()



fig, ax = plt.subplots(3, figsize=(6,14))
for a,d,l in zip(range(len(ax)),
				(df[['Alcohol', 'Malic acid']].values, df_std, df_minmax),
				('Input Scale',
				'Standard Scaler',
				'MinMax Scaler')
				):
	for i,c in zip(range(1,4), ('red', 'blue', 'green')):
		ax[a].scatter(d[df['Class label'].values == i, 0], d[df['Class label'].values == i, 1], alpha=0.5, color=c, label='Class %s' %i)
	ax[a].set_title(l)
	ax[a].set_xlabel('Alcohol')
	ax[a].set_ylabel('Malic Acid')
	ax[a].legend(loc='upper left')
	ax[a].grid()
plt.tight_layout()
plt.show()



print("Using standardization and scaling in PCA")
from sklearn.model_selection import train_test_split

X_wine=df.values[:,1:]
y_wine=df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.30, random_state=12345)

from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


from sklearn.decomposition import PCA

# on non-standardized data
pca = PCA(n_components=2).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# om standardized data
pca_std = PCA(n_components=2).fit(X_train_std)
X_train_std = pca_std.transform(X_train_std)
X_test_std = pca_std.transform(X_test_std)


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))

for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
	ax1.scatter(X_train[y_train==l, 0], X_train[y_train==l, 1],
	color=c,
	label='class %s' %l,
	alpha=0.5,
	marker=m
	)

for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
	ax2.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1],
		color=c,
		label='class %s' %l,
		alpha=0.5,
		marker=m
		)

ax1.set_title('Transformed NON-standardized training dataset after PCA')
ax2.set_title('Transformed standardized training dataset after PCA')


for ax in (ax1, ax2):
	ax.set_xlabel('1st principal component')
	ax.set_ylabel('2nd principal component')
	ax.legend(loc='upper right')
	ax.grid()
plt.tight_layout()
plt.show()


print("Guassian Naive Bayes using standardization")

from sklearn.naive_bayes import GaussianNB

# on non-standardized data
gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)

# on standardized data
gnb_std = GaussianNB()
fit_std = gnb_std.fit(X_train_std, y_train)

from sklearn import metrics

pred_train = gnb.predict(X_train)

print("Before standardization")
print('\nPrediction accuracy for the training dataset')
print('{:.2%}'.format(metrics.accuracy_score(y_train, pred_train)))

pred_test = gnb.predict(X_test)
print('\nPrediction accuracy for the test dataset')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))


print("After standardization")
pred_train_std = gnb_std.predict(X_train_std)
print('\nPrediction accuracy for the training dataset')
print('{:.2%}'.format(metrics.accuracy_score(y_train, pred_train_std)))

pred_test_std = gnb_std.predict(X_test_std)
print('\nPrediction accuracy for the test dataset')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_std)))
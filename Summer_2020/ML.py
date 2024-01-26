# Create feature and label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

d1 = pd.read_csv('tsum.txt')
d2 = pd.read_csv('vsum.txt')
d3 = pd.read_csv('ssum.txt')

d1['polar.'] = 2
d2['polar.'] = 1
d3['polar.'] = 0

d = pd.DataFrame()
d['r1 tensor'] = np.hstack([d1['logZ-tensor-R-1'],d2['logZ-tensor-R-1'],d3['logZ-tensor-R-1']])
d['n0 tensor'] = np.hstack([d1['logZ-tensor-N-0'],d2['logZ-tensor-N-0'],d3['logZ-tensor-N-0']])
d['r1 vector'] = np.hstack([d1['logZ-vector-R-1'],d2['logZ-vector-R-1'],d3['logZ-vector-R-1']])
d['n0 vector'] = np.hstack([d1['logZ-tensor-N-0'],d2['logZ-tensor-N-0'],d3['logZ-tensor-N-0']])
d['n0 scalar'] = np.hstack([d1['logZ-scalar-N-0'],d2['logZ-scalar-N-0'],d3['logZ-scalar-N-0']])
d['p1 scalar'] = np.hstack([d1['logZ-scalar-P-1'],d2['logZ-scalar-P-1'],d3['logZ-scalar-P-1']])
d['snr'] = np.hstack([d1['snr'],d2['snr'],d3['snr']])
d['polarization'] = np.hstack([d1['polar.'],d2['polar.'],d3['polar.']])

s = pd.DataFrame()
s['B1'] = d['r1 tensor'] - d['r1 vector']
s['B2'] = d['r1 tensor'] - d['n0 vector']
s['B3'] = d['n0 tensor'] - d['r1 vector']
s['B4'] = d['n0 tensor'] - d['n0 vector']
s['B5'] = d['r1 tensor'] - d['n0 scalar']
s['B6'] = d['r1 tensor'] - d['p1 scalar']
s['B7'] = d['n0 tensor'] - d['n0 scalar']
s['B8'] = d['n0 tensor'] - d['p1 scalar']
s['B9'] = d['r1 vector'] - d['n0 scalar']
s['B10'] = d['r1 vector'] - d['p1 scalar']
s['B11'] = d['n0 vector'] - d['n0 scalar']
s['B12'] = d['n0 vector'] - d['p1 scalar']
s['snr'] = d['snr']
s['polarization'] = d['polarization']

from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Eliminate 'NaN' data
is_NaN = s.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = s[row_has_NaN]
s = s.drop(53)
#s = s[s.snr>=8]

# Do Machine Learning
df = shuffle(s)
x = df.drop(['polarization'],axis=1)
y = df['polarization']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
reg = RandomForestClassifier(n_estimators=500)
model = reg.fit(x_train,y_train)
label = reg.predict(x_test)
acc = accuracy_score(y_test,label)
print('predictions:',label)
print('accuracy score:',acc)

# Confusion matrix and heatmap
import seaborn as sns;sns.set()
m = confusion_matrix(y_test,label)
print(m)
sns.heatmap(m.T,square=True,annot=True,fmt='d',cbar=False)
plt.title('Confusion matrix')
plt.xlabel('Predictions')
plt.ylabel('True value')
plt.savefig('confm')

# Let's play
s = int(input('please enter an index,0 to 60:'))
print('actual:',y[s])
print('predicted:',label[s])

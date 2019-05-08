import pandas as pd
#fever,nausea,lumpain,urine,micpain,burning,inf,neph,
df = pd.read_csv("diagnosis.csv")
df.head()
#inputs = pd.read_csv('diagnosis.txt', index_col=0)


#inouts = drop(['neph', 'inf',], axis=1)
#inputs = inputs.drop('inf',axis='columns')

inputs = df

from sklearn.preprocessing import LabelEncoder
le_fever = LabelEncoder()
le_nausea = LabelEncoder()
le_lumpain = LabelEncoder()
le_urine = LabelEncoder()
le_micpain = LabelEncoder()
le_burning = LabelEncoder()
le_inf = LabelEncoder()
le_neph = LabelEncoder()

inputs['fever_n'] = le_fever.fit_transform(inputs['fever'])
inputs['nausea_n'] = le_nausea.fit_transform(inputs['nausea'])
inputs['lumpain_n'] = le_lumpain.fit_transform(inputs['lumpain'])
inputs['urine_n'] = le_urine.fit_transform(inputs['urine'])
inputs['micpain_n'] = le_micpain.fit_transform(inputs['micpain'])
inputs['burning_n'] = le_burning.fit_transform(inputs['burning'])
#inputs['inf_n'] = le_inf.fit_transform(inputs['inf'])
#inputs['neph_n'] = le_neph.fit_transform(inputs['neph'])

target = inputs['inf']
inputs = df.drop(['neph','inf'],axis='columns')
inputs = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print(inputs)
inputs_n = inputs.drop(['fever','nausea','lumpain','urine','micpain','burning','neph','inf'],axis='columns')
print(inputs_n)
print (target)

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
model.score(inputs_n,target)

print("Do they have it? \n",model.predict([[0,0,0,1,1,1]]))
print(model.predict([[1,1,1,1,1,1]]))
print(model.predict([[0,0,0,0,0,0]]))

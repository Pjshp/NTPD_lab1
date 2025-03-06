import joblib
import pandas as pd
from sklearn.datasets import load_iris

model = joblib.load('model.joblib')

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

random_sample = df.sample(n=1, random_state=42)
sample_record = random_sample.drop(columns=['target']).values
actual_class = random_sample['target'].values[0]

feature_names = iris.feature_names
sample_df = pd.DataFrame(sample_record, columns=feature_names)

prediction = model.predict(sample_df)

class_labels = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}

print(f'Actual class: {class_labels[actual_class]}')
print(f'Predicted class: {class_labels[prediction[0]]}')
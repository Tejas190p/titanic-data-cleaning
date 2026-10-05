import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Load the Titanic Dataset
# Ensure your Kaggle dataset is saved as 'titanic.csv' in the same directory
df = pd.read_csv('titanic.csv')

# 2. Clean missing data
# Fill missing 'Age' with the median and 'Embarked' with the mode
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Dropping 'Cabin' as it typically contains too many missing values for basic imputation
df.drop(columns=['Cabin'], inplace=True, errors='ignore')

# 3. Encode Sex and Embarked columns
# Using LabelEncoder for 'Sex' (binary: male/female)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Using One-Hot Encoding for 'Embarked' (multiple categories)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 4. Visualize age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], kde=True, bins=30, color='teal')
plt.title('Titanic Passengers Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 5. Output cleaned dataset as new CSV
output_filename = 'cleaned_titanic.csv'
df.to_csv(output_filename, index=False)
print(f"Project complete. Cleaned data saved as '{output_filename}'.")
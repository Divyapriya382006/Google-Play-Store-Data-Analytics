import kagglehub
import pandas as pd
import numpy as np


# Download latest version
path = kagglehub.dataset_download("lava18/google-play-store-apps")


print("Path to dataset files:", path)


data=pd.read_csv(path+"/googleplaystore.csv")
print("Dataset loaded!")


print(data.shape)
print(data.head())


print(data["Category"].nunique())
print(data["Category"].unique())


print(data.duplicated().sum())


print(data.isna().sum())


data["Rating"].fillna(data["Rating"].mean(),inplace=True)


print(data.isna().sum())


data.drop_duplicates(inplace=True)
print(data.shape)


data["Price"]=data["Price"].astype(str).str.strip()
data["Price"]=data["Price"].str.replace('$',"",regex=False)
data['Price']=pd.to_numeric(data['Price'],errors='coerce')


data["Installs"]=data["Installs"].astype(str).str.strip()
data["Installs"]=data["Installs"].str.replace('+',"",regex=False)
data["Installs"]=data["Installs"].str.replace(',',"",regex=False)
data["Installs"]=pd.to_numeric(data['Installs'],errors='coerce')


data["Reviews"]=data["Reviews"].astype(str).str.strip()
data["Reviews"]=pd.to_numeric(data['Reviews'],errors='coerce')


data.dropna(subset=['Price','Installs','Reviews'],inplace=True)


print("Data Type Cleaning done!")
data[['App','Price','Installs','Reviews']].head()


print(data.describe())


data=data[data["Rating"]<=5]


data['isPaid']=data['Type'].apply(lambda x:1 if x=='Paid'else 0)
data[['App','Type','isPaid']].head()


data.to_csv('cleaned_playstore.csv',index=False)
print("Cleaned and saved!")


import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv("cleaned_playstore.csv")
sns.set(style="whitegrid")
print(data.shape)
print(data.head())


print(data.info())


print(data.isna().sum())


print(data.describe())


topcats=data["Category"].value_counts().head(10)
#print(topcats)


#plt.figure()
sns.barplot(x=topcats.values,y=topcats.index)
plt.title("Top 10 Categories")
plt.xlabel("Number of Apps")
plt.ylabel("Categories")
plt.show()


avgrating=data.groupby("Category")["Rating"].mean().sort_values(ascending=False).head(10)
print(avgrating)


sns.barplot(x=avgrating.values,y=avgrating.index)
plt.title("Top 10 Average Ratings by Categories")
plt.xlabel("Ratings")
plt.ylabel("Categories")
plt.show()


apptypes=data['Type'].value_counts()
print(apptypes)


sns.barplot(y=apptypes.values,x=apptypes.index)


sns.histplot(data['Rating'],bins=20,kde=True)


sns.scatterplot(x=data['Price'],y=data['Rating'],color='pink')


sns.scatterplot(x=data['Installs'],y=data['Rating'],color='green',hue=data['Type'])


corr=data[['Rating','Reviews','Installs','Price','isPaid']].corr()
print(corr)


sns.heatmap(corr,annot=True)


data.describe().to_csv('eda_summary.csv')
print("EDA Summary Saved as eda_summary.csv !!!")




















import pymongo
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client.miniproject
collection = db.test
# print(collection);

data = pd.DataFrame(list(collection.find()))


# print(data);
# df=data[['total_sqft','bath','balcony','size','price']]
# print("testing");
# X = df.drop(df.columns[4],axis='columns')
# print("fdsdf");
# Y=df[df.columns[4]]
# # print(X);

# from sklearn.model_selection import train_test_split
# print("nana")
# X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,test_size=0.2)
# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# print(X_train.shape)
# print(Y_train.shape)

# Y_train=pd.DataFrame(Y_train)
# print("nana1")
# # print(Y_train)
# print(lr.fit(X_train,Y_train) )

# print("Model Trained")
# l=[[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]]
# # print(l)
# # print(lr.predict(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))
# # l=[[sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]]]
# # print(l)
# print(lr.predict(l)[0][0])





	# total_sqft	bath	balcony	price	bhk
df=data[['total_sqft','bath','balcony','price','bhk']]
X = df.drop(['price'],axis='columns')
# X.head(3)

Y = df.price
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
y_train=pd.DataFrame(y_train)
x_train=pd.DataFrame(X_train)
# print(x_train)

lr_clf.fit(X_train,y_train)
# lr_clf.score(X_test,y_test)

x = np.zeros(len(X.columns))
x[0] = sys.argv[1]
x[1] = sys.argv[2]
x[2] = sys.argv[3]
x[3] = sys.argv[4]
# print(x)
print(round(lr_clf.predict([x])[0][0],2))

# print([sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]])

# print([1,2,3]);


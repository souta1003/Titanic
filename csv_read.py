import pandas as pd
import numpy as np
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train_shape = train.shape
test_shape  = test.shape
pd.set_option("max_columns", 12)

### 欠損値の数を集計
def kesson_table( df ):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len( df )
    kesson_table = pd.concat( [null_val, percent], axis=1 )
    kesson_table_len_collumns = kesson_table.rename( columns = {0: '欠損数', 1: '%'})

    return kesson_table_len_collumns

### 欠損値を穴埋め 中央値 or 任意の値
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

print( kesson_table(train) )
#print( kesson_table(test)  )

### 文字列を数値に変換
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][ train["Embarked"] == "S" ] = 0
train["Embarked"][ train["Embarked"] == "C" ] = 1
train["Embarked"][ train["Embarked"] == "Q" ] = 2
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][ test["Embarked"] == "S" ] = 0
test["Embarked"][ test["Embarked"] == "C" ] = 1
test["Embarked"][ test["Embarked"] == "Q" ] = 2

#print( train.head(10) )
#print( test.head(10) )

target = train["Survived"].values
features_one = train[ ["Pclass", "Sex", "Age", "Fare"] ].values
print(features_one)

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one.fit( features_one, target )

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values
my_prediction = my_tree_one.predict(test_features)
print(f"予想結果:{my_prediction}")



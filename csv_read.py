import pandas as pd
import numpy as np

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
train_shape = train.shape
test_shape  = test.shape
#pd.set_option("max_columns", 12)

# print( test.head())
# print( train.head())
# print(test_shape)
# print( train_shape)
# print( test.describe() )
# print( train.describe() )

def kesson_table( df ):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len( df )
    kesson_table = pd.concat( [null_val, percent], axis=1 )
    kesson_table_len_collumns = kesson_table.rename( columns = {0: '欠損数', 1: '%'})

    return kesson_table_len_collumns



print( kesson_table(train) )
print( kesson_table(test)  )

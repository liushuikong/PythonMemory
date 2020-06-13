##  one

（1）读取csv文件，**nrows**读取部分数据

```PYTHON
import pandas as pd

train = pd.read_csv('train.tsv', names=['device_id', 'sex', 'age'], nrows=10000)
```

（2）函数**agg**、**reset_index**、**sort_values**。agg可以sum、mean、min等

```PYTHON
app_use_time = packtime.groupby(['app'])['period'].agg('sum').reset_index() # 统计各个app总使用时长，注意reset_index
app_use_top100 = app_use_time.sort_values(by='period', ascending=False)[:100]['app'] # 筛选出使用时长前100的APP

device_app_use_time = packtime.groupby(['device_id', 'app'])['period'].agg('sum').reset_index()
use_time_top100_statis = device_app_use_time.set_index('app').loc[list(app_use_top100)].reset_index() #另一个表中前100app的数据
```

（3）**value_counts**、**columns**的应用。value_counts的主要参数有normalize、sort和ascending。**normalize**默认False， 如果为True，则返回归一化的相对频率；**sort** 默认True，返回按**值**排序；**ascending**默认False，如果为True，则按**频率**计数升序排序


```PYTHON
ph_ver = brand['ph_ver'].value_counts(True，ascending=False) # value_counts

app_num = packtime['app'].value_counts().reset_index()
app_num.columns = ['app', 'app_num'] # 改变dateframe的columns
```

（4）**直接修改**dateframe**部分数据**，可以用来处理**长尾分布**数据

```PYTHON
mask = (brand.ph_ver_cnt < 100) # 找出ph_ver_cnt小于100的index
brand.loc[mask, 'ph_ver'] = 'other' # 修改ph_ver_cnt小于100那些行，ph_ver的值
# 另一种方式
df['存活'] = df.Survived.apply(lambda x: '倖存' if x else '死亡')
```

（5）**merge**和**concat**

```PYTHON
train = pd.merge(brand[['device_id', 'ph_ver']], train, on='device_id', how='right')

test['sex'] = -1
test['age'] = -1
test['label'] = -1
data = pd.concat([train, test], ignore_index=True) # axis=0，train和test行拼接
data = pd.concat([data, ph_ver_dummy], axis=1) # axis=1，列拼接
# 把train和test分开
train = data[data.sex != -1]
test = data[data.sex == -1]
```

（6）**sklearn**中的**LabelEncoder**

```PYTHON
from sklearn import preprocessing
# train数据和test数据分开时
ph_ver_le = preprocessing.LabelEncoder()
train['ph_ver'] = ph_ver_le.fit_transform(train['ph_ver']) # train数据用fit_transform学习，并更改
test['ph_ver'] = ph_ver_le.transform(test['ph_ver']) # test数据用transform

train['label'] = train['sex'].astype(str) + '-' + train['age'].astype(str)
# 只有train数据 
label_le = preprocessing.LabelEncoder()
train['label'] = label_le.fit_transform(train['label']) # 用fit_transform
```

（7）**groupby**可以和**apply**一起使用。**agg**只能对**数字**类列操作

```PYTHON
# 统计每台设备的app数量
df_app = packtime[['device_id', 'app']]
apps = df_app.drop_duplicates().groupby(['device_id'])['app'].apply(' '.join).reset_index() # app列不是数字
apps['app_length'] = apps['app'].apply(lambda x:len(x.split(' '))) #apply的使用

df['存活'] = df.Survived.apply(lambda x: '倖存' if x else '死亡')
```

（8）**get_dummies**的one-hot处理

```PYTHON
ph_ver_dummy = pd.get_dummies(data['ph_ver'])
ph_ver_dummy.columns = ['ph_ver_' + str(i) for i in range(ph_ver_dummy.shape[1])]
del data['ph_ver'] # 删除ph_ver列
```

（9）**sklearn** 的 **CountVectorizer**是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该**训练文本中出现的频率**。CountVectorizer会将文本中的词语转换为词频矩阵，**它通过fit_transform函数计算各个词语出现的次数**。**stop_words**即是**停用词**是信息检索中自动过滤掉的字或词

```PYTHON
# 获取每台设备所安装的apps的tfidf
tfidf = CountVectorizer(lowercase=False, min_df=3, stop_words=top100_statis.columns.tolist()[1:7])
apps['app'] = tfidf.fit_transform(apps['app']) # fit_transform训练学习
X_tr_app = tfidf.transform(list(train['app'])) # transform
```

（10）**Word2Vec**特征处理

```PYTHON
from gensim.models import Word2Vec, FastText

embed_size = 128
fastmode = Word2Vec(list(packages['apps']), size=embed_size, window=4, min_count=3, negative=2,
                 sg=1, sample=0.002, hs=1, workers=4)  

embedding_fast = pd.DataFrame([fastmode[word] for word in (fastmode.wv.vocab)])
embedding_fast['app'] = list(fastmodel.wv.vocab)
embedding_fast.columns= ["fdim_%s" % str(i) for i in range(embed_size)]+["app"]
```



（11） **pivot()**方法则是针对**列的值**进行操作，即指定**某列的值**作为**行索引**，指定某列的值作为**列索引**，然后再指定哪些列作为索引对应的值。

```PYTHON
# 原始数据
data = {'date': ['2018-08-01', '2018-08-02', '2018-08-03', '2018-08-01', '2018-08-03', '2018-08-03',
                 '2018-08-01', '2018-08-02'],
        'variable': ['A','A','A','B','B','C','C','C'],
        'value': [3.0 ,4.0 ,6.0 ,2.0 ,8.0 ,4.0 ,10.0 ,1.0 ]}

df = pd.DataFrame(data=data, columns=['date', 'variable', 'value'])
print(df)

#          date variable  value
# 0  2018-08-01        A    3.0
# 1  2018-08-02        A    4.0
# 2  2018-08-03        A    6.0
# 3  2018-08-01        B    2.0
# 4  2018-08-03        B    8.0
# 5  2018-08-03        C    4.0
# 6  2018-08-01        C   10.0
# 7  2018-08-02        C    1.0

# eg1.如果要根据时间统计各variable的值，做法如下
# 让index为date，让variable里的值变为单独的列（pivot）
df1 = df.pivot(index='date', columns='variable', values='value')
print(df1)

# variable      A    B     C
# date
# 2018-08-01  3.0  2.0  10.0
# 2018-08-02  4.0  NaN   1.0
# 2018-08-03  6.0  8.0   4.0

# eg2.如果value有多个情况下，列名会变成Hierarchical columns的结构，即MultiIndex
df['value_other'] = df['value'] * 2
df2 = df.pivot(index='date', columns='variable', values=['value', 'value_other'])
print(df2)

#            value            value_other
# variable       A    B     C           A     B     C
# date
# 2018-08-01   3.0  2.0  10.0         6.0   4.0  20.0
# 2018-08-02   4.0  NaN   1.0         8.0   NaN   2.0
# 2018-08-03   6.0  8.0   4.0        12.0  16.0   8.0

print(df2['value_other'])

# variable       A     B     C
# date
# 2018-08-01   6.0   4.0  20.0
# 2018-08-02   8.0   NaN   2.0
# 2018-08-03  12.0  16.0   8.0
```

（12）对于简单DataFrame来说，pivot_table和pivot类似。但是在那基础上增加了更多的一些功能。主要参数如下：

* data：DataFrame
* index：index列可以**多个**。pivot不可以。
* columns：column列
* values：value列
* aggFuc：聚合函数，可以是多个。index和columns都相同时，确定的值如果**有多个**，则根据这个函数计算除一个值作为这个index和columns的值，默认情况是np.mean。**pivot在这种情况会提示有重复值，不能处理这种情况**。

```python
df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                    "bar", "bar", "bar", "bar"],
              "B": ["one", "one", "one", "two", "two",
                      "one", "one", "two", "two"],
              "C": ["small", "large", "large", "small",
                      "small", "large", "small", "small",
                      "large"],
              "D": [1, 2, 2, 3, 3, 4, 5, 6, 7]})

print(df)

# 原始数据
#      A    B      C  D
# 0  foo  one  small  1
# 1  foo  one  large  2
# 2  foo  one  large  2
# 3  foo  two  small  3
# 4  foo  two  small  3
# 5  bar  one  large  4
# 6  bar  one  small  5
# 7  bar  two  small  6
# 8  bar  two  large  7

print(pd.pivot_table(df, index=['A', 'B'], columns=['C'], values=['D'], aggfunc=[np.mean, np.sum, max]))

#          mean         sum         max      
#             D           D           D      
# C       large small large small large small
# A   B                                      
# bar one   4.0   5.0   4.0   5.0   4.0   5.0
#     two   7.0   6.0   7.0   6.0   7.0   6.0
# foo one   2.0   1.0   4.0   1.0   2.0   1.0
#     two   NaN   3.0   NaN   6.0   NaN   3.0
```

（13）而pivot()方法是针对**列的值**进行操作，而**unstack()**方法是针对**索引**进行操作，即将**列索引**转成最内层的**行索引**。**stack()**则是unstack()的逆操作，将**行索引**（默认是最内层，即最右侧，行索引）转成**列索引**。stack和unstack都是用来操作**MultiIndex**（多重索引）。stack和unstack就是在多重行索引和多重列索引（列名）之间转化的

![img](https://img-blog.csdn.net/20180704191137494?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1Nfb19sX29fbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![img](https://img-blog.csdn.net/20180704191457916?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1Nfb19sX29fbg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

```PYTHON
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8,2), index=index, columns=['A', 'B'])
print(df)

#                      A         B
# first second
# bar   one    -0.332862  0.929766
#       two     0.857515  0.181623
# baz   one    -0.769248  0.200083
#       two     0.907549 -0.781607
# foo   one    -1.683440  0.868688
#       two    -1.556559 -0.591569
# qux   one    -0.399071  0.115823
#       two     1.665903  2.210725

# eg1.stack方法可以来"压缩"一下列索引，这可能产生一个Series（如果本身列所以那就是一重的），或者一个DataFrame（列是多重索引）
# 这里的结果就是产生了一个Series，行索引是三重的。
df1 = df.stack()
print(df1)
print(df1.index)

# first  second
# bar    one     A   -0.332862
#                B    0.929766
#        two     A    0.857515
#                B    0.181623
# baz    one     A   -0.769248
#                B    0.200083
#        two     A    0.907549
#                B   -0.781607
# foo    one     A   -1.683440
#                B    0.868688
#        two     A   -1.556559
#                B   -0.591569
# qux    one     A   -0.399071
#                B    0.115823
#        two     A    1.665903
#                B    2.210725


# eg2.相反的操作是unstack()，即减少一重行索引，增加一重列索引
df2 = df1.unstack()
print(df2)

#                      A         B
# first second
# bar   one    -0.332862  0.929766
#       two     0.857515  0.181623
# baz   one    -0.769248  0.200083
#       two     0.907549 -0.781607
# foo   one    -1.683440  0.868688
#       two    -1.556559 -0.591569
# qux   one    -0.399071  0.115823
#       two     1.665903  2.210725

# eg3.如果索引是多重的，我们可以指定去"压缩"哪一层的索引。对于行索引来说
# - 行索引，从0开始，最左边最小为0
# - 列索引，从0开始，最上边最小为0
# 也可以不用0和1..等，用索引层的名字，比如这里的first和second，但是这样有可能有的索引层没有名字，比如第一次stack后的df1。
# 数字和名字但不能混用，但是可以同时指定多个level值。
#
df3 = df1.unstack(level=0)
print(df3)

# first          bar       baz       foo       qux
# second
# one    A -0.332862 -0.769248 -1.683440 -0.399071
#        B  0.929766  0.200083  0.868688  0.115823
# two    A  0.857515  0.907549 -1.556559  1.665903
#        B  0.181623 -0.781607 -0.591569  2.210725

# eg4.stack和unstack内部都实现了排序，如果如果对一个DataFrame进行了stack在进行unstack，DataFrame会按照行索引排好序.
# 经过试验列索引并不会排好序！
index = pd.MultiIndex.from_product([[2,1], ['a', 'b']])
df4 = pd.DataFrame(np.random.randn(4,1), index=index, columns=[100])
print(all(df4.stack().unstack() == df4.sort_index()))
df5 = pd.DataFrame(np.random.randn(4,3), index=index, columns=[100,99,102])
print(df5)

# True

#           100       99        102
# 2 a  0.094463  1.766611  0.588748
#   b -1.262801  0.737156 -0.450470
# 1 a -0.888983  0.903101 -1.179545
#   b  1.015863 -0.486976 -1.097248

```

（13）lightgbm多分类

```PYTHON
train_df['age'] = train_df['age'] - 1
columns = [col for col in test_df.columns if col not in ['user_id']]
print(columns)

kf = KFold(n_splits=5, shuffle=True, random_state=2020)

def age_model(train_df, test_df, cols, test=False):
    
    y_test = 0
    oof_train = np.zeros((train_df.shape[0], 10))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols],train_df['age'])):
        print("fold_{0}".format(i+1))
        x_train, y_train = train_df.loc[train_index, cols], train_df[train_index,'age']
        x_val, y_val = train_df.loc[val_index, cols], train_df[val_index,'age']

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val,reference=lgb_train)
        params = {
            'boosting_type': 'gbdt',
            'learning_rate' : 0.01, #0.008
            'verbose': -1,
            'num_leaves':128,
            'max_depth':12, #10
            # 'max_bin':10, 
            # 'lambda_l2': 1, 
            'min_child_weight':30,
            "num_class":10,
            'objective':'multiclass', 
            'feature_fraction':0.7, #0.7
            'bagging_fraction':0.9, # 0.9是目前最优的
            'bagging_freq':3,  # 3是目前最优的
            'bagging_seed':10,
            'seed': 2020,
            'metric':{'multi_logloss','multi_error'},
            'nthread': 50,
            'device': 'gpu'
            # 'silent': True,
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        )
        y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        # print(y_pred[:10])
        if test:
            y_test += gbm.predict(test_df[cols], num_iteration=gbm.best_iteration)
        oof_train[val_index] = y_pred
        #break
    auc = accuracy_score(train_df.age.values, np.argmax(oof_train, axis=1))
    y_test /= kf.n_splits
    feature_list = pd.DataFrame()
    feature_list['names'] = cols
    feature_list['imp'] = gbm.feature_importance()
    feature_list = feature_list.sort_values(by='imp', ascending=False)
    print(feature_list)
    print('5 Fold auc:', auc)
    gc.collect()
    return auc, oof_train, y_test, feature_list

age_auc, age_oof_train, age_y_test, age_imp = age_model(train_df, test_df, columns, True)

put_result = pd.DataFrame()
put_result['user_id'] = test_df['user_id']
age_result = np.argmax(age_y_test, axis=1) + 1
put_result['predicted_age'] = age_result
# print(put_result.head())
put_result.to_csv('data/submission.csv', index=False)
```

（14）lightgbm二分类

```PYTHON
train_df['gender'] = train_df['gender'] - 1

def gender_model(train_df, test_df, cols, test=False):
    
    y_test = 0
    oof_train = np.zeros((train_df.shape[0], ))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols],train_df.gender)):
        x_train, y_train = train_df.loc[train_index, cols], train_df.loc[train_index,'gender']
        x_val, y_val = train_df.loc[val_index, cols], train_df.loc[val_index,'gender']

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val)
        params = {
            'boosting_type': 'gbdt',
            'learning_rate' : 0.01, 
            'verbose': -1,
            'num_leaves':256,
            'max_depth':-1, 
            # 'max_bin':10, 
            # 'lambda_l2': 1, 
            # 'min_child_weight':30,
            'objective': 'binary', 
            # 'feature_fraction':0.6,
            'bagging_fraction':0.9, # 0.9是目前最优的
            'bagging_freq':5,  # 3是目前最优的
            'seed': 2020,
            'metric': {'binary_logloss', 'binary_error'},
            'nthread': 50,
            'device': 'gpu'
            # 'silent': True,
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        )
        y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        # print(y_pred[:10])
        if test:
            y_test += gbm.predict(test_df[cols], num_iteration=gbm.best_iteration)
        oof_train[val_index] = y_pred
        # break
    oof_train = list(map(lambda x:1 if x>0.5 else 0,oof_train))
    auc = accuracy_score(train_df.gender.values, oof_train)
    y_test /= kf.n_splits
    y_test = list(map(lambda x:2 if x>0.5 else 1,y_test)) # y_test = [2 if x>0.5 else 1 for x in y_test]
    feature_list = pd.DataFrame()
    feature_list['names'] = cols
    feature_list['imp'] = gbm.feature_importance()
    feature_list = feature_list.sort_values(by='imp', ascending=False)
    print(feature_list)
    print('5 Fold auc:', auc)
    gc.collect()
    return auc, oof_train, y_test, feature_list

gender_auc, gender_oof_train, gender_y_test, gender_imp = gender_model(train_df, test_df, columns, True)

put_result = pd.DataFrame()
put_result['user_id'] = test_df['user_id']

put_result['predicted_gender'] = gender_y_test
# print(put_result.head())
# print(put_result['predicted_gender'].value_counts())
put_result.to_csv('data/submission.csv', index=False)
```

（15）lightgbm回归

```PYTHON
from sklearn.model_selection import KFold

columns = [col for col in test_df.columns if col not in ['user_id']]
print(columns)

kf = KFold(n_splits=5, shuffle=True, random_state=2020)

def age_model(train_df, test_df, cols, test=False):
    
    y_test = 0
    oof_train = np.zeros((train_df.shape[0], 10))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols],train_df['targets'])):
        print("fold_{0}".format(i+1))
        x_train, y_train = train_df.loc[train_index, cols], train_df[train_index,'targets']
        x_val, y_val = train_df.loc[val_index, cols], train_df[val_index,'targets']

        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_val, y_val,reference=lgb_train)
        params = {'num_leaves': 38,
                  'min_data_in_leaf': 50,
                  'objective': 'regression',
                  'max_depth': -1,
                  'learning_rate': 0.02,
                  "min_sum_hessian_in_leaf": 6,
                  "boosting": "gbdt",
                  "feature_fraction": 0.9,
                  "bagging_freq": 1,
                  "bagging_fraction": 0.7,
                  "bagging_seed": 11,
                  "lambda_l1": 0.1,
                  "verbosity": -1,
                  "nthread": 4,
                  'metric': 'mae',
                  "random_state": 2019,
                  # 'device': 'gpu'
                  }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=40000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=100,
                        verbose_eval=100,
                        )
        y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
        # print(y_pred[:10])
        if test:
            y_test += gbm.predict(test_df[cols], num_iteration=gbm.best_iteration)
        oof_train[val_index] = y_pred
        #break

    y_test /= kf.n_splits
    feature_list = pd.DataFrame()
    feature_list['names'] = cols
    feature_list['imp'] = gbm.feature_importance()
    feature_list = feature_list.sort_values(by='imp', ascending=False)
    print(feature_list)

    gc.collect()
    return oof_train, y_test, feature_list

oof_train, y_test, imp = age_model(train_df, test_df, columns, True)

print('mse %.6f' % mean_squared_error(train_df['targets'], oof_train))
print('mae %.6f' % mean_absolute_error(train_df['targets'] oof_train))

put_result = pd.DataFrame()
put_result['user_id'] = test_df['user_id']
put_result['prediction'] = y_test
# print(put_result.head())
put_result.to_csv('data/submission.csv', index=False)


```


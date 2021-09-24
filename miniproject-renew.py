# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:12:28 2021

@author: Rony's PC
"""



"""
0.
Table of Contents:
------------------
    1.  whether small data tabular could be predictable ?
    2.  Libraries to import.
    3.  Managinig the place dic in Cach memory.
    4.  Load and check data.
    5  Inquery by Plotting.
    6.  Feature analysis.
    7.  Feature engineering.
    8.  Modeling.
    9.  Training basic LSTM model.
    10. Addinl LSTM to Dense model deep.
    11. Hyperparameters.
    12. Fitting + plotting 'Loss'.
    13. Prediction.
    14. Pearson.
    15. Checking @ comparing all features to be predictable.
    16. Summery and conclusion.


1.
How small data@problems of sequences arrangement could fit Lstm@dense deep model?
---------------------------------------------------------------------------------
1.1
The LSTM model is a serial model. Following my small data economic 
I am trying to produce series where time is important, 
I am looking to preserve the time. Since that sequences arrangement is
important.I note that I have not been able to preserve the time component
because of the data structure and economic definitions aspects that do not 
allow me to complete missing cells just like that, with no meaning. When this
is the case!! And after all, the problems I check is an economic question - 
the prediction was done in two ways and if each one of them gives me the 
same answer I know I overcame the problem of the schedule. 
To appoint we have 2 problems: 
1) Time schedule and data structure, 
2) Small data? 
    These are the two main problems are inquering and solving in the economic
data structure I have on the hands. And the question is: will we be able to 
get a prediction despite both problems for those I test it in Pearson as well 
as in deep? And after all I get a model that converges, and accuracy ranges are 
from 83% -93%. More of that I check if there is a single variable, a feature 
that could give me a prediction (I did not find), and further check if all the 
variables together being in synergy each one to other, they together can give 
me a prediction, and here I see that together, only together I get a moving 
prediction picture that is Between 83% -93%. It is true that when time misorder
the value of the LSTM is eroded. I hope it's clearer now. Yet in 
the economics worlds, there are quite a few problems and approximations that 
were done to fit problems and this LSTM approximation economically - 
is right to be done.
"""

"""
2. Libraries to import.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import requests



"""
3. Managinig the place dic in Cach memory
""" 

cache = dict()
def get_article_from_server(url):
    print("Fetching article from server...")
    response = requests.get(url)
    return response.text

def get_article(url):
    print("Getting article...")
    if url not in cache:
        cache[url] = get_article_from_server(url)
    return cache[url]
# get_article("https://realpython.com/sorting-algorithms-python/")
# get_article("https://realpython.com/sorting-algorithms-python/")



"""
4. Load and check data

4.1 Load data
"""

x_features = pd.read_excel(r"C:/Users/Rony's PC/OneDrive/ML/miniproject1/Features data set.xlsx") # features 
y_sales = pd.read_csv(r"C:/Users/Rony's PC/OneDrive/ML/miniproject1/sales data-set.csv.zip") # labels
z_stores = pd.read_csv(r"C:/Users/Rony's PC/OneDrive\ML/miniproject1/stores data-set.csv") # stores

"""
4.2 Inquering all Data
"""

print(x_features.head())
print(y_sales.head())
print(z_stores.head())
print('-'*40)
print('\n')

"""
After looking the data we have a problem with missing data
So to fix it we need apply the dates in both columns to same type date.
I choose as best as I can the featurs that fit the economic aspects predict.
Following this vision: 
    1) I didn't take data size stores.
    2) I merge only the full cells data.
"""


"""
4.3 Groupby working
The groupby drop all missing rows
"""  

y_sales['Date'] = pd.to_datetime(y_sales['Date'])
# optimize the sales via store in a unique date with all Deparments to be preper 
y_sales = y_sales.groupby(['Store', 'Date']).sum()
# we have to repeat its again to fit x_features y_sales 
x_features['Date'] = pd.to_datetime(x_features['Date'])
x_features = x_features.groupby(['Store', 'Date']).sum()
print(x_features.dtypes)
print(y_sales.dtypes)
print(z_stores.dtypes)
print('-'*40)
print('\n')

"""
4.4 Merging all Data
As an inner mean - marge just the common Datas.
"""

Combined_table = pd.merge(x_features, y_sales['Weekly_Sales'], how='inner', right_index=True, left_index=True)
Combined_table.isna().sum()
Combined_table.info
Combined_table['Weekly_Sales'].describe()
Combind_graf = Combined_table.copy()

# Very well... It seems that your minimum price is larger than zero. Excellent!

"""
4.5 inquery by: Skewness @ skew
"""
print("Skewness: %f" % Combined_table['Weekly_Sales'].skew())
print("Kurtosis: %f" % Combined_table['Weekly_Sales'].kurt())
"""
4.5.1) Have appreciable positive skewness.
4.5.2) Have appreciable positive skewness.
"""



"""
5 Inquery by Plotting 

5.1) histogram plot
"""
# let see how Weekly_Sales corellative to other features ?
# sns.distplot(Combined_table['Weekly_Sales'])
sns.displot(data = Combined_table, x = 'Weekly_Sales', kde=True)

"""
5.2) Inquery of outlier by box plot store / Weekly_Sales
"""
# first convert the store from index into a column
# Takea a look over each sotre outlayers via "Weekly_Sales'.
Combined_table = Combined_table.reset_index(level=0)
var = 'Store'
data = pd.concat([Combined_table['Weekly_Sales'], Combined_table[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Weekly_Sales", data=data)
fig.axis(ymin = 180000, ymax = 4000000)

"""
5.3) scatterplot - Data Visualization
"""
# Let display the relationship between two numerical variables
# For any combination features. 
sns.set()
cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']
sns_plot = sns.pairplot(Combind_graf[cols].sample(100), height = 2.5) 
plt.show()

"""
5.4) Plotting all features via the label (prediction of sales)
"""
# I try to see if it's any corralation between the features
Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].plot(subplots=True, figsize=(20,15))
plt.show()




"""
6. Feature analysis


6.1) Plotting Confusional Correlation matrix between numerical values
"""
# In the following Matrix we will see how mach the features are confusing?.
# As we see not all features are not corelative. 
g = sns.heatmap(Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()
#At the end we will see that Pearson accurecy is around 90%

"""
6.2) Plotting Confusional Correlation matrix with no features correlative
"""
# Since I have two Prameters with I correlasition, I drop 'MarkDown1', Why ?
# To prevent  Linkage featurs that are corelatived.
g = sns.heatmap(Combind_graf[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Weekly_Sales']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
plt.show()
# Now at the end we will see Pearson accurecy here is around 87.5%
"""
conclusion: Finally The linckage is no so big.
----------
"""
Combined_table = Combined_table.drop(['MarkDown1'], axis = 1)

"""
6.3 Let see the first 10 variable (for example)
"""
print("Dats shape = {}".format(Combined_table.shape))
print()
print("Lets see some feature:")
print(Combined_table[1:10])



"""
7. Feature engineering

To be on the safe side I take correlasition Pearson Metric as another referance

==============================================================================
------------------------------ Economic vision analization -------------------
I analize only two table : features and sales arreies, to predict sales stores 
growing.Union data done with an economic vision and not blind idea.
The Data are not so big. In mean while when i Take Pearson correlation at the
end.The correlation is around 80%-90%
As a model of economic predicts, I can't do any augmantation with no meaning - 
So I lost the order time. Even that is no augmantation Model 
but Pearson + dense deep models give us an excelent predicts and answers.
------------------------------------------------------------------------------
==============================================================================
"""


 
"""
8. Modeling
"""

# Definition of Y (label) and X (inputs)
# Define label for the new merge table: "combined_table"
# The 'Weekly_Sales' is Indexial so it dosn't take as a label
y = Combined_table['Weekly_Sales']
# Define features for the new merge table: "combined_table"
x = Combined_table.drop(['Weekly_Sales'], axis=1)


"""
8.1 preprocessing normalization values between 0 and 1
"""
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
y.shape
y = y.values.reshape(4320, 1)
y_scaled = scaler.fit_transform(y)
x.head()
x.tail()

"""
8.2 splitting the Datas to train @ test : 80%-20% 
"""
# we need to take x_scaled after being notmalized
# In the first step we will split the data in training and remaining dataset
# random_state = 0 to keep order time as best as I can
x_train, x_valid, y_train, y_valid = train_test_split(x_scaled, y, train_size=0.80, random_state=0)

"""
8.3 splitting the test to test @ val : 50%-50%
"""
# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
x_test, x_valid, y_test, y_valid = train_test_split(x_scaled,y, test_size=0.5)

"""
8.4 Finally splitting will be : 80%-10%-10% as train-val-test
"""

"""
8.5 we need to add an additional dimantion to get numpy arrey for keras shape
"""
x_train = x_train.reshape(x_train.shape[0],1,x_train.shape[1]) 
x_valid = x_valid.reshape(x_valid.shape[0],1,x_valid.shape[1])
x_test = x_test.reshape(x_test.shape[0],1,x_test.shape[1]) 

Test_Data = (x_test, y_test)

print('-'*40)
print('\n')
print(x_train.shape), print(y_train.shape)
print(x_valid.shape), print(y_valid.shape)
print(x_test.shape), print(y_test.shape)
print('-'*40)
print('\n')



"""
9. Training basic model LSTM model
"""
# Training basic LSTM model# Initializing the Recurrent Neural Network AS LSTM
inputs = tf.random.normal([32, 50, 9])
model = Sequential()



"""
10. Addinl LSTM to Dense model deep
"""
 
# Adding the first LSTM layer with a sigmoid activation function and some
# Dropout regularization
# Units - dimensionality of the output space
model.add(LSTM(units = 32, return_sequences = False, input_shape =(1,x_train.shape[2])))
# Adding the output layer
model.add(Dense(units = 128))
model.add(Dense(units = 64))
model.add(Dense(units = 1, activation="relu", input_shape=(4,)))
model.summary()



"""
11. Hyperparameters
"""
# learning rate need to be learn !!
# Adam was config as Adaptive Learning Rate Methods
"""
11.1 Optimiztion
"""
opt = tf.keras.optimizers.Adam(learning_rate = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0)
# Best learning_rate was found to be 0.1  .
# Model.compile(loss = 'mean_absolute_error', for regression 
# Metrics=[soft_acc], optimizer = opt
"""
11.2 Compilation
"""
model.compile(loss = 'mse', optimizer = opt)
"""
11.3 Create logger and Tensorboard for analyzing.
"""
# A logger was created for logs the best whieghts of the training to be saved.
my_callbacks = [tf.keras.callbacks.ModelCheckpoint(save_best_only = True, filepath = 'model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir = './logs')]
# Create a TensorBoard logger need to be check (no usefull here since I take
# other ways like plots)
logger = tf.keras.callbacks.TensorBoard(log_dir = 'logs', write_graph = True,
    histogram_freq = 5)



"""
12. Fitting + plotting 'Loss'.

12.1 Fitting
"""
history = model.fit(x_train,y_train,epochs = 1000, batch_size = 16, validation_data = (x_valid,y_valid), callbacks = my_callbacks)
p = history.history['loss']
print('-'*40)
print('\n')
# list all data in history
print(history.history.keys())

"""
12.2 Plotting 'Loss'
"""
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['x_train', 'x_valid'], loc='upper left')
plt.legend(['y_train', 'y_valid'], loc='upper left')
            
"""
12.3 Mini conclusion
===================================================================================
Finally take a look on the Loss,'y_train' became fitting with 'y_valid even
1) All the faurues arn't correlative each one to others.                           
2) All most all the stores are with outlayers.                                    
3) Correlation graph are not a simetric Gaussian.                                 
4) Small Data.                                                                    
5) Very big numbers sales that cause big differances and complex calaulation.
6) We lost the order time      
===================================================================================
++++++++++++++++++++++++++  Conclussiotion  +++++++++++++++++++++++++++++++++++++++
                Model: Dense Deep Learning work very good  !!!!!!                 
===================================================================================
"""


"""
13. Prediction.

13.1 Verification of the prediction.
"""
"""
13.1.1 For Validation.
"""
preds_val = model.predict(x_valid)
preds_val = preds_val.squeeze()
result_val = y_valid - preds_val

"""
13.1.2 for Test
"""
preds_test = model.predict(x_test)
preds_test = preds_test.squeeze()
result_test = y_test - preds_val



"""
14. Pearson.
"""

"""
Now we I check my small data with Pearson a statistic problem.
Forecasting Accuracy = Pearson
Spearman was checked for correlation, but Spearman is better for this problem.
"""
"""
14.1 Checking for the validetion coefficient.
"""
def correlation_coefficient_var(y_valid, preds_coef_val):
    pearson_r_val = tfp.stats.correlation(preds_coef_val, y_valid)
    return(pearson_r_val)
    print(pearson_r_val)

"""
14.2 Checking for the test coefficient.
"""  
def correlation_coefficient_test(y_test, preds_coef_test):
    pearson_r_test = tfp.stats.correlation(preds_coef_test, y_test)
    return(pearson_r_test)
    print(pearson_r_test)
    
print('-'*40)
print('\n')

"""
14.3 prediction for validetion.
"""
preds_val = preds_val.reshape(-1, 1)
y_valid = y_valid.astype('float32')
print(correlation_coefficient_var(y_valid, preds_val))

"""
14.4 prediction for test.
"""
preds_test = preds_test.reshape(-1, 1)
y_test = y_test.astype('float32')
print(correlation_coefficient_test(y_test, preds_test))



"""
15. Check comparing all features to be predictable.
"""

# Plotting the label prediction of sales.
Combind_graf = Combined_table.copy()
Combind_graf[['Weekly_Sales']].plot(subplots=True, figsize=(20,15))
plt.title('model Weekly sales predict')
plt.ylabel('Income')
plt.xlabel('Weekly Time Sales')
plt.show()



"""
16. Summery and conclusion.
"""
"""
16.1 Print pridict of validetion.
"""
(print(preds_val))
corr, p_val = pearsonr(y_valid.squeeze(), preds_val.squeeze())
print('Pearson corr validation')
print(corr)
print('-'*40)
print('\n')

"""
16.2 Print pridict of test.
"""
(print(preds_test))
corr, p_test = pearsonr(y_test.squeeze(), preds_test.squeeze())
print('Pearson corr test')
print(corr)

"""
16.3 Conclusion
"""

"""
==============================================================================
Via Pearson on Validation and Test verificatuin after  lots of epocs.
WE could declair that the model prediction fits the Datas training in 
Aproximitly around 88% to 92% Pearson accurecy.
In both model we couls see same conclusion.
==============================================================================
"""
print('Via Pearson on Validation and Test verification after  1000 epochs.')
print('WE could declare that the model prediction fits the data training in')
print('Approximately around 83% to 92% accuracy even all problems.')
print('Via this data we could see that future predict is no big changes')
print('Via Weekly Sales, meaning that the predict will be same as every week')
    






    



































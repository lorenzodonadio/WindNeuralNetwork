
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,LeakyReLU
from keras.callbacks import EarlyStopping
from keras import metrics

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,median_absolute_error,max_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split

from scipy.interpolate import LinearNDInterpolator

from shapely.geometry import Point

class WindNeuralNetwork:
    """
    Neural network to treat WRF data, transform normalize and lag data
    creates a NeuralNetwork and train it.

    One must initialize the WindNeuralNetwork with the input Dataframes.
    Then run the prep_data() method, and then the create_and_fit_model()

    see the WNN_retrain_tutorial.ipynb for a more extensive tutorial

    ...

    Attributes
    ----------
    df1 and df2 : pandas.Dataframe
        Two dataframes containing the relevant information to feed to the neural network:
        They must contain the same columns, i.e 
            -the features of the model
            -the target variable
            -A time stamp for every row
    var : str
        the name of the target variable, it must be a columns of df1

    features : list
        A list of columns commun to both dataframes that will be used as predictors i.e:
        input data for the neural network

    y_scaler : str
        the scaling method either 'Robust','MinMax','Standard'
        'Standard' is the default and the only supported method so far
        TODO: make robust and minmax work (bug in the inverse_transform method)

    logtrans : bool
        weather to take the log(1+x) of the target variable before training

    look_back: int 
        number of observations to look back in the past for the prediction step,
        this variable will determine the number of neurons and the shape of the input layer
        and the entire architecture of the neural network, default=12
        
    time_jump: int
        Must be smaller than look_back, it is the time step between observations,
        if time_jump=1 (default), then we take into account all the observations 
        before the look_back value, if time_jump=3 then we skip 2 observations so the 
        time interval becomes 30 minutes instead of 10.
        
        This parameter reduces the size of the input layer and thus the complexity of the netwok
        
    include_tanh: bool
        weather to add a forth hidden layer with the activation function tanh,
        this would be the forth hidden layer, it is helpful since the last layer is 
        a linear activation function and the predictions are somewhat bounded, the tanh helps capture that behaviour

    q_weight: float
        weight to give the observations beyond a certain quantile,
        it is so that the loss function of the NN gives more importance to certain variables
        it is used in the create_and_fit_model method

    quantile: float
        quantile of the target variable ditribution after which the q_weight
        will be assigned to the observations

    patience: int
        early stopping patience
        number of times we wait for improvement before stopping the training
    
    loss_function: str
        loss function for the NN training

    model: Keras.Sequential
        Trained neural network for prediction

    Y: np.array
        target variable, already scaled and ready for the neural network
        comes from the prep_y method that runs automatically in __init__


    y_train,y_test : np.array
        splitted Y in training and test set

    X_test ,X_train : np.array
        splitted X in training and test set

    Methods
    -------
    This methods list is not extensive, only the methods for the user are described.
    -------

    prep_y(self)
        prepares the target variable, only to be called once in __init__
        by calling the y_scaler and lagging by day

    prep_data(self,test_size=0.1,random_state=42,features=None)
        Prepares the default or user defined features by:
            1. scaling the data
            2. lagging by day
            3. stacking both dataframes
            4. splitting in test and train set

    create_and_fit_model()
        Must be run after prep_data

        it creates the keras model and fits it to the training data

    Once the model is trained we can use some other functionalities

    predict_and_getmetrics_test()

        Uses the trained model to predict the test set and stores the metrics in self.stats

    predict_and_plot_test()

         Uses the trained model to predict the test set and plots the results

    predict_new_data(df1,df2)
        
        predicts new data, the dataframes must be in a valid format,
        i.e they must contain the features a time stamp and a 'Day_month' column

        It is higly advided that df1 and df2 to be the outputs from prepare_new_data()
        
    re_train_model(self,df1,df2,var='WT_Power')
        
        Fine tuning of the already trained model

        Both dataframes must contain the features a time stamp and a 'Day_month' column,
        and the df1 must contain var as a column (target variable)

        It is higly advided that df1 and df2 to be the outputs from prepare_new_data()
        but now merged with the data from juvent website or other kind of data that contains
        the target variable.
 
    """
    
    def __init__(self,
                 df1,
                 df2,
                 var = 'WT_Power',
                 scaler = 'Standard',
                 features = ['Ws','Wdir','T','P'],
                 look_back = 12,
                 time_jump = 1,
                 q_weight = 2,
                 quantile = 0.6,
                 patience = 3,
                 epochs = 30,
                 dropout = 0.2,
                 include_tanh = True,
                 model_v = 1,
                 logtrans = False):

        self.df1 = df1
        self.df2 = df2
        self.var = var
        self.features = features
        self.y_scaler = self.scaler_choice(scaler)  #for scaling y
        self.scaler = self.scaler_choice(scaler)    #for scaling X
        self.logtrans = logtrans 
        self.look_back = look_back
        self.time_jump = time_jump
        self.q_weight = q_weight
        self.quantile = quantile
        self.patience = patience
        self.include_tanh = include_tanh
        self.model_v = model_v
        self.epochs = epochs
        self.dropout = dropout
        self.prep_y()
        self.loss_function = 'mean_squared_logarithmic_error' if logtrans else 'mean_squared_error'
        #self.loss_function='mean_squared_logarithmic_error' if logtrans else 'mean_absolute_error'
    def scaler_choice(self,scaler):
        if scaler=='Robust':
            return RobustScaler()
        if scaler=='MinMax':
            return MinMaxScaler()
        if scaler=='Standard':
            return StandardScaler()
    
    class LogTransformer:
        '''
        A log transformer for the target variable, with the capabilities 
        of inversing the transform after the predictions are done
        '''
        def __init__(self,Y):
            self.Y=np.array(Y).squeeze()
            self.min=np.min(self.Y)
            self.y_trans=np.log1p(self.Y-self.min)
        def transform(self):
            return self.y_trans
        def inverse_transform(self,y_inv):
            return np.expm1(y_inv)+self.min 
        
    def prep_y(self):
        '''
        Prepares the Y data, this method should be run only by __init__ 
        '''
        y=self.df1[['Date-time',self.var,'Day_Month']].copy()
        
        if self.logtrans:
            self.logtrans=self.LogTransformer(y[self.var])
            y[self.var]=self.logtrans.transform()
        y[self.var]=self.y_scaler.fit_transform(np.array(y[self.var].values).reshape(-1,1))
        Y=[]
        for dm in self.df1.Day_Month.unique():
            Y=np.append(Y,y[y['Day_Month']==dm][self.var].values[:-self.look_back])
        self.Y=Y
    
    def lag_values(self,X):
        '''X must be a numpy-like array
           Lags values in time according to the features selected and the look_back
        '''
        X=np.array(X)
        Xlag=[]
        l,c=X.shape
        for i in range(l-self.look_back):
            Xlag.append(np.ravel([X[i:i+self.look_back+self.time_jump:self.time_jump,ci] for ci in range(c)]))
        return np.stack(Xlag)
    
    def lag_per_day(self,df,features=None):
        '''
        Groups values per day and lags them,
        the last 'look_back' values of each day are thus deleted
        
        returns: np.array of the lagged features in df
        '''
        if features == None: #default values 
            features = self.features

        X=[]
        for dm in df.Day_Month.unique():
            X.append(self.lag_values(df[df['Day_Month']==dm][features]))
        return np.concatenate(X)
    
    def scale_data(self,df,features=None):
        '''
        Scales data according to the specified scaler

        returns: a copy of the scaled dataframe
        '''
        if features == None: #default values 
            features = self.features

        df_aux=df.copy()
        df_aux[features]=self.scaler.fit_transform(df_aux[features])
        return df_aux
    
    def prep_data(self,test_size=0.1,random_state=42,features=None):
        ''' Prepares the data for the NeuralNetwork
            by calling:
            1. scale_data
            2. lag_per_day
            3. stacking both dataframes
            4. splitting data in test and train set

            the following attributes are created:
            self.X, self.X_train, self.X_test, self.y_train, self.y_test
        '''
        
        if features==None: #default values 
            features = self.features

        X1=self.scale_data(self.df1)
        X2=self.scale_data(self.df2)

        self.X=np.hstack((self.lag_per_day(X1),self.lag_per_day(X2)))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.Y,test_size=test_size,random_state=random_state)
       
    def generate_sample_weights(self):
        '''generates sample weiths for the sample_weights parameter in the model.fit method from the Keras model
           It gives more importance to values higher than a certain quantile
           the weighting scheme is a simple step function
        '''
        return np.asarray([1 if x<np.quantile(self.y_train,self.quantile) else self.q_weight for x in self.y_train])
    
    def create_and_fit_model(self,verbose=1):

        '''
        Creates a Keras.Sequential model and fits it to self.X_train and self.y_train

        the trained model is saved to self.model
        '''
        #create model
        if self.model_v == 1:
            model = Sequential()
            n_cols=self.X_train.shape[1]
            #add model layers
            model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))
            model.add(Dense(int(n_cols*2/3), activation='relu'))
            model.add(Dense(int(n_cols*2/3), activation='relu'))
            model.add(Dropout(self.dropout))

            model.add(Dense(int(n_cols/2)))
            model.add(LeakyReLU(alpha=0.05))

            #model.add(Dense(int(n_cols/2), activation='relu'))
            if self.include_tanh: 
                model.add(Dense(int(n_cols/2), activation='tanh'))
            
            model.add(Dense(int(n_cols*0.35), activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss=self.loss_function)
            model.fit(self.X_train, self.y_train,
                      validation_split=0.15,
                      epochs=self.epochs,
                      verbose=verbose,
                      sample_weight = self.generate_sample_weights(),
                      callbacks=[EarlyStopping(patience=self.patience)])
            self.model=model

        if self.model_v == 2:
            model = Sequential()
            n_cols=self.X_train.shape[1]
            #add model layers
            model.add(Dense(n_cols,input_shape=(n_cols,)))
            model.add(LeakyReLU(alpha=0.05))

            model.add(Dense(int(n_cols*2/3)))
            model.add(LeakyReLU(alpha=0.05))

            model.add(Dense(int(n_cols*2/3)))
            model.add(LeakyReLU(alpha=0.05))

            model.add(Dropout(self.dropout))

            model.add(Dense(int(n_cols/2)))
            model.add(LeakyReLU(alpha=0.05))

            if self.include_tanh: 
                model.add(Dense(int(n_cols/2), activation='tanh'))
            
            model.add(Dense(int(n_cols/3), activation = 'relu'))

            model.add(Dense(1))

            model.compile(optimizer='adam', loss=self.loss_function)
            model.fit(self.X_train, self.y_train,
                      validation_split=0.15,
                      epochs=self.epochs,
                      verbose=verbose,
                      sample_weight = self.generate_sample_weights(),
                      callbacks=[EarlyStopping(patience=self.patience)])
            self.model=model

    
    def inverse_transform(self,y):
        '''
        Reverses the scaling of the target variable
        Usefull when predicting data, the output of the neural network is per se scaled,
        so we apply the inverse transform of the scaling to return predictions in original units
        '''
        y=np.array(y).astype(float)
        if self.logtrans==False:
            return self.y_scaler.inverse_transform(y.reshape(-1, 1))
        else:
            return self.logtrans.inverse_transform(self.y_scaler.inverse_transform(y.reshape(-1, 1)))
    
    def predict_and_plot_test(self,n=250):
        '''
        Predicts and plots n points from the test data set
        '''
        if n >self.y_test.shape[0]:
            n=self.y_test.shape[0]
            print('n exceeded the number of samples, set to the maximun possible')
        
        X,y=self.X_test[:n],self.y_test[:n]
        self.y_predictions_rescaled = self.inverse_transform(self.model.predict(X))
        self.y_true_rescaled=self.inverse_transform(y)
        fig,ax=plt.subplots(figsize=(14,6))
        plt.plot(self.y_true_rescaled,'--C0',label='data')
        plt.plot(self.y_predictions_rescaled,'C1',label='prediction')
        R2=r2_score(self.y_true_rescaled,self.y_predictions_rescaled)
        plt.legend()
        plt.title('R2 = '+str(R2))
        plt.show()
    
    def predict_and_getmetrics_test(self):

        '''
        Predicts and gets the metrics from the entire test set,
        the metrics are calculated using the scikit-learn library

        their order is the following:

        [r2_score,mean_absolute_error,mean_squared_error,median_absolute_error,max_error]

        the metrics are stored in self.stats
        '''

        X,y=self.X_test,self.y_test
        self.y_predictions_rescaled = self.inverse_transform(self.model.predict(X))
        self.y_true_rescaled=self.inverse_transform(y)
        
        self.stats=[r2_score(self.y_true_rescaled,self.y_predictions_rescaled),
                    mean_absolute_error(self.y_true_rescaled,self.y_predictions_rescaled),
                    mean_squared_error(self.y_true_rescaled,self.y_predictions_rescaled),
                    median_absolute_error(self.y_true_rescaled,self.y_predictions_rescaled),
                    max_error(self.y_true_rescaled,self.y_predictions_rescaled)]

    def predict_new_data(self,df1_new,df2_new,col_name='Predicted_Power'):
        ''' Predicts new data by following the same instructions as in prep_data()

            and then using self.model.predict

            the predictions from the Keras model are then rescaled into the original 
            units with inverse_transform() and writtend into a new column of df1

            returns: a copy of df1 with the predictions as the column with col_name
        

        '''
        
        X1=self.scale_data(df1_new)
        X2=self.scale_data(df2_new)
        
        X_for_predictions=np.hstack((self.lag_per_day(X1),self.lag_per_day(X2)))
        
        y_pred=self.inverse_transform(self.model.predict(X_for_predictions))
        df_out=pd.DataFrame()
        for dm in df1_new.Day_Month.unique():
                df_out=pd.concat([df_out,df1_new[df1_new.Day_Month==dm][:-self.look_back]])
        df_out[col_name]=y_pred
        
        return df_out
    
    def re_train_model(self,df1,df2,var='WT_Power'):
        '''
        Fine tunes the already trained model,

        df1,df2 must come from prepare_new_data and the target variable (var) must be present as a column of df1

        '''


        X1 = self.scale_data(df1)
        X2 = self.scale_data(df2)
        X = np.hstack((self.lag_per_day(X1),self.lag_per_day(X2)))

        y = df1[['Date-time',var,'Day_Month']].copy()

        if self.logtrans:
            self.logtrans = self.LogTransformer(y[var])
            y[var]=self.logtrans.transform()
        y[var] = self.y_scaler.fit_transform(np.array(y[var].values).reshape(-1,1))
        Y=[]

        for dm in df1.Day_Month.unique():
            Y = np.append(Y,y[y['Day_Month']==dm][var].values[:-self.look_back])

        def local_generate_sample_weights():
            return np.asarray([1 if x<np.quantile(Y,self.quantile) else self.q_weight for x in Y])
        
        self.model.fit(X, Y,
              validation_split=0.15,
              epochs=30,
              sample_weight = local_generate_sample_weights(),
              callbacks=[EarlyStopping(patience = self.patience)])





def add_daymonth_and_sort(df):

    '''
    helper fot prepare_new_data()
    adds column 'Day_month' and 
    
    returns: sorted dataframe by 'Date-time'
    '''
    df['Day_Month']=df['Date-time'].apply(lambda x: str(x.day)+'_'+str(x.month))
    return df.sort_values(by='Date-time',ascending=False)

def get_wt_pos():
    '''
    Gets the position of all  the wind turbines

    '''
    wt_coords = pd.read_csv('wt_coordinates.csv',sep=';')

    # creating a geometry column 
    geometry = [Point(xy) for xy in zip(wt_coords['Y'], wt_coords['X'])]# Coordinate reference system : CH1903
    crs = {'init': 'epsg:21781'}# Creating a Geographic data frame 
    wt_coords = gpd.GeoDataFrame(wt_coords, crs=crs, geometry=geometry)
    wt_coords=wt_coords.to_crs(epsg=4326)

    return wt_coords.geometry.y,wt_coords.geometry.x

def interp_lin_latlong(df,lat,lon,var):
    ip=LinearNDInterpolator((df['lat'].values,df['long'].values), df[var].values)
    I=[ip((lat,lon)) for lat,lon in zip(lat,lon)]
    return I

def interp_wt(df,wt_lat,wt_long):
    '''Returns a dictionary for each wind turbine with the interpolated variables T,P,Ws,Wdir'''
    
    wt_dict={}
    for i in range(1,17): # number of wind turbines
        wt_dict['WTG'+ f'{i:02}'] = pd.DataFrame()

    for i in df['Date-time'].unique():
        df_tmp=df[df['Date-time']==i]

        interp_T=interp_lin_latlong(df_tmp, wt_lat, wt_long, 'T')
        interp_P=interp_lin_latlong(df_tmp, wt_lat, wt_long, 'P')
        interp_Ws=interp_lin_latlong(df_tmp, wt_lat, wt_long, 'Ws')
        interp_Wdir=interp_lin_latlong(df_tmp, wt_lat, wt_long, 'Wdir')

        for j,wt in enumerate(wt_dict):
            a=pd.DataFrame()
            a['Date-time'] = [i]
            a['Ws'] = [interp_Ws[j]]
            a['Wdir'] = [interp_Wdir[j]]
            a['T']=[interp_T[j]]
            a['P']=[interp_P[j]]
            wt_dict[wt]=pd.concat([wt_dict[wt],a])
    return wt_dict


def prepare_new_data(file):
    '''
    Prepares a WRF_output file for the WindNeuralNetwork
    
    returns: two dictionaries wt_95_new,wt_150_new
        the first one with data at 95m above the ground and the second one at 150m

        the keys of the dict are 'WTG01', 'WTG02' ... 'WTG16', i.e the turbine names

        the values of the dict are pandas.Dataframes with the correct format to
        give them as inputs for the WindNeuralNetwork, either as predict_new_data or as __init__
    '''

    wt_lat,wt_long=get_wt_pos()
    
    wrf_df_new=pd.read_csv(file)
    wrf_df_new['Date-time']= pd.to_datetime(wrf_df_new['Date-time'].apply(lambda x: x.replace('_', ' ')))

    # creating a geometry column 
    geometry = [Point(xy) for xy in zip(wrf_df_new['long'], wrf_df_new['lat'])]# Coordinate reference system : WGS84
    crs = {'init': 'epsg:4326'}# Creating a Geographic data frame 
    wrf_df_new = gpd.GeoDataFrame(wrf_df_new, crs=crs, geometry=geometry)
    #extract WRF data for the given height 95,150

    #Interpolate data for all the wind turbines
    wt_95_new=interp_wt(wrf_df_new.loc[wrf_df_new['height']==95].copy().reset_index(0,drop=True),wt_lat,wt_long)
    wt_150_new=interp_wt(wrf_df_new.loc[wrf_df_new['height']==150].copy().reset_index(0,drop=True),wt_lat,wt_long)
    for i in wt_95_new:
        wt_95_new[i]=add_daymonth_and_sort(wt_95_new[i])
    for i in wt_150_new:
        wt_150_new[i]=add_daymonth_and_sort(wt_150_new[i])
    
    return wt_95_new,wt_150_new







    



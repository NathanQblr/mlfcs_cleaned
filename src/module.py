"""Module used for defining the class which train a FCS data analysis machine learnbing model"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from pyptu import PTUParser
from fitter import classical_fitD,classical_fitalpha, return_tau
from signal_tools import definition_set_g,g_convol_semilogx
from open_mycsv import CSVNameParser
from joblib import Parallel, delayed
import multiprocessing
import os
import time


class MlfcsTrainer:
    """Class permitting the training of machine learning method for FCS data analysis
    """
    def __init__(self,data):
        self.size_data = data.shape[0]
        print("Taille dataset", self.size_data)


        self.omega_x_encoder = LabelEncoder().fit(data.omega_x)
        self.omega_encoder = LabelEncoder().fit(data.omega)
        self.model_encoder = LabelEncoder().fit(data.Model)

        self.t_max_windows = LabelEncoder().fit(data['Time window']).classes_

        self.omega_x_encoded = self.omega_x_encoder.transform(self.omega_x_encoder.classes_)
        self.omega_x = self.omega_x_encoder.classes_
        self.omega_encoded = self.omega_encoder.transform(self.omega_encoder.classes_)
        self.omega = self.omega_encoder.classes_
        self.model_encoded = self.model_encoder.transform(self.model_encoder.classes_)
        self.model = self.model_encoder.classes_

        self.models_d = ['MB','FBM','RWF']
        self.models_alpha = ['CTRW', 'FBM','RWF']
        self.models_d_encoded = []
        self.models_alpha_encoded = []
        for m in self.models_d:
            if m in self.model:
                self.models_d_encoded.append(m)
        for m in self.models_alpha:
            if m in self.model:
                self.models_alpha_encoded.append(m)
        if len(self.models_d_encoded)>0:
            self.models_d_encoded = self.model_encoder.transform(self.models_d_encoded)
        if len(self.models_alpha_encoded)>0:
            self.models_alpha_encoded = self.model_encoder.transform(self.models_alpha_encoded)


        self.classifier_model = None

        self.regressors_alphas = pd.DataFrame(columns=['omega_x','omega','model','regressor'])
        self.regressors_ds = pd.DataFrame(columns=['omega_x','omega','model','regressor'])

        self.regressor_total_alpha = None
        self.regressor_total_d = None

        self.data_array,self.alpha,self.d,self.model = self.init_input_outputs(data)
        self.function_on_data_array = np.zeros((self.data_array.shape[0],self.omega_x_encoder.classes_.size,self.omega_encoder.classes_.size,self.model_encoder.classes_.size,2))

        self.predictor = None
        del data

    def load_data(self,data):
        self.size_data = data.shape[0]
        self.data_array,self.alpha,self.d,self.model = self.init_input_outputs(data)
        self.function_on_data_array = np.zeros((self.data_array.shape[0],self.omega_x_encoder.classes_.size,self.omega_encoder.classes_.size,self.model_encoder.classes_.size,2))
        del data

    def train_predictor_total_D(self):
        """Train the predictor that gives the final output from predictions of every alpha , diffusion coef and model predictors
        """
        self.apply_regressors()
        self.function_on_data_array = np.column_stack([self.function_on_data_array.reshape(self.size_data,-1),self.data_array[:,0:3],self.model])
        del self.data_array

        idx = np.isin(self.model,self.models_d_encoded)&(self.alpha>=0.9*np.ones(self.size_data))
        print(f'Size training D = {np.sum(idx)}')
        self.regressor_total_d = HistGradientBoostingRegressor().fit(np.column_stack((self.alpha[idx],self.function_on_data_array[idx])), self.d[idx])
        del self.function_on_data_array

    def train_model(self):
        """Call every methods to train the machine learning model with data
        """
        self.train_classifier_model()
        print("Classifier ok")
        self.train_regressors_alpha()
        print("Alpha ok")
        self.train_regressors_d()
        print("D ok")
        self.train_predictor_total()
        print("The end")

    def train_classifier_model(self):
        """Train classifier model with data
        """
        self.classifier_model = HistGradientBoostingClassifier().fit(self.data_array, self.model)

    def train_regressors_alpha(self):
        """Train regressors for alpha models with data for evey omega and omega_x settings
        """
        #Attention pas prendre omega en entrée
        for wx in self.omega_x:
            for w in self.omega:
                for m in self.models_alpha_encoded:
                    idx = (self.data_array[:,1]==wx*np.ones(self.size_data))&(self.data_array[:,2]==w*np.ones(self.size_data))&(self.model==m*np.ones(self.size_data))
                    if idx.sum()>0:
                        new_row = {'omega_x':wx,'omega':w,'model':m,'regressor':HistGradientBoostingRegressor().fit(self.data_array[idx,:], self.alpha[idx])}
                    else:
                        new_row = {'omega_x':wx,'omega':w,'model':m,'regressor':None}
                    self.regressors_alphas.loc[len(self.regressors_alphas)] = new_row

    def train_regressors_d(self):
        """Train regressors for diffusion coefficient models with data for evey omega and omega_x settings
        """
        #Attention pas prendre omega en entrée
        for wx in self.omega_x:
            for w in self.omega:
                for m in self.models_d_encoded:
                    idx = (self.data_array[:,1]==wx*np.ones(self.size_data))&(self.data_array[:,2]==w*np.ones(self.size_data))&(self.model==m*np.ones(self.size_data))&(self.alpha>=0.9*np.ones(self.size_data))
                    if idx.sum()>0:
                        new_row = {'omega_x':wx,'omega':w,'model':m,'regressor':HistGradientBoostingRegressor().fit(self.data_array[idx], self.d[idx])}
                    else:
                        new_row = {'omega_x':wx,'omega':w,'model':m,'regressor':None}
                        print(f'Wrong m = {m},wx={wx},w={w},number over 0.9 = {np.sum(self.alpha>=0.9*np.ones(self.size_data))}')
                    self.regressors_ds.loc[len(self.regressors_ds)] = new_row

    def train_predictor_total(self):
        """Train the predictor that gives the final output from predictions of every alpha , diffusion coef and model predictors
        """
        self.apply_regressors()
        self.function_on_data_array = np.column_stack([self.function_on_data_array.reshape(self.size_data,-1),self.data_array[:,0:3],self.model])
        del self.data_array

        self.regressor_total_alpha = HistGradientBoostingRegressor().fit(self.function_on_data_array, self.alpha)
        idx = np.isin(self.model,self.models_d_encoded)&(self.alpha>=0.9*np.ones(self.size_data))
        print(f'Size training D = {np.sum(idx)}')
        self.regressor_total_d = HistGradientBoostingRegressor().fit(np.column_stack((self.alpha[idx],self.function_on_data_array[idx])), self.d[idx])
        del self.function_on_data_array

    def apply_regressors(self):
        """Compute on data every trained regressors to have input of predictor total
        """
        for wx in self.omega_x:
            wx_encoded = self.omega_x_encoder.transform([wx])[0]
            for w in self.omega:
                w_encoded = self.omega_encoder.transform([w])[0]
                for m in self.model_encoded:
                    idx = (self.regressors_alphas.omega_x==wx)&(self.regressors_alphas.omega==w)&(self.regressors_alphas.model==m)
                    if idx.sum()>0:
                        self.function_on_data_array[:,wx_encoded,w_encoded,m,0] = self.regressors_alphas.loc[idx,'regressor'].values[0].predict(self.data_array)
                    idx = (self.regressors_ds.omega_x==wx)&(self.regressors_ds.omega==w)&(self.regressors_ds.model==m)
                    if idx.sum()>0:
                        self.function_on_data_array[:,wx_encoded,w_encoded,m,1] = self.regressors_ds.loc[idx,'regressor'].values[0].predict(self.data_array)



    def init_input_outputs(self,data):
        """Initialise numpy array so the sklearns algorithmscan use the data

        Args:
            data: dataframe containing the data

        Returns:
            data of dataframes containes in different np.arrays
        """
        tmax = data['Time window'].to_numpy()
        data.drop(columns=['Time window'])
        omega_x = data['omega_x'].to_numpy()
        data.drop(columns=['omega_x'])
        omega = data['omega'].to_numpy()
        data.drop(columns=['omega'])
        autoco = np.vstack(data['G'].to_numpy())
        data.drop(columns = ['G'],axis = 1)
        autoco = np.divide(autoco,autoco[:,:5].mean(axis=1).reshape(autoco.shape[0],1))
        data_array=np.column_stack((tmax,omega_x,omega,autoco))
        del tmax
        del omega_x
        del omega
        model = self.model_encoder.transform(data['Model'].to_numpy())
        alpha = data['alpha'].to_numpy()
        d = data['D'].to_numpy()
        return data_array,alpha,d,model

    def init_input_outputs_eval(self,data):
        """Initialise numpy array so the sklearns algorithmscan use the data

        Args:
            data: dataframe containing the data

        Returns:
            data of dataframes containes in different np.arrays
        """
        tmax = data['Time window'].to_numpy()
        omega_x = data['omega_x'].to_numpy()
        omega = data['omega'].to_numpy()
        autoco = np.vstack(data['G'].to_numpy())
        autoco = np.divide(autoco,autoco[:,:5].mean(axis=1).reshape(autoco.shape[0],1))
        data_array=np.column_stack((tmax,omega_x,omega,autoco))
        del tmax
        del omega_x
        del omega
        return data_array

    def predict(self,data_array):
        """Construct the total predictor of the method

        Args:
            data_array: data to apply prediction
        """
        size_data_array = data_array.shape[0]
        model_predictions = self.classifier_model.predict(data_array)
        proba_model_prediction = self.classifier_model.predict_proba(data_array)
        inter_predict = np.zeros((data_array.shape[0],self.omega_x_encoder.classes_.size,self.omega_encoder.classes_.size,self.model_encoder.classes_.size,2))
        for wx in self.omega_x:
            wx_encoded = self.omega_x_encoder.transform([wx])[0]
            for w in self.omega:
                w_encoded = self.omega_encoder.transform([w])[0]
                for m in self.model_encoded:
                    idx = (self.regressors_alphas.omega_x==wx)&(self.regressors_alphas.omega==w)&(self.regressors_alphas.model==m)
                    if idx.sum()>0:
                        inter_predict[:,wx_encoded,w_encoded,m,0] = self.regressors_alphas.loc[idx,'regressor'].values[0].predict(data_array)
                    idx = (self.regressors_ds.omega_x==wx)&(self.regressors_ds.omega==w)&(self.regressors_ds.model==m)
                    if idx.sum()>0:
                        inter_predict[:,wx_encoded,w_encoded,m,1] = self.regressors_ds.loc[idx,'regressor'].values[0].predict(data_array)
        inter_predict = np.column_stack([inter_predict.reshape(size_data_array,-1),data_array[:,0:3],model_predictions])
        alpha_predictions = self.regressor_total_alpha.predict(inter_predict)
        beta_predictions = self.regressor_total_d.predict(np.column_stack((alpha_predictions,inter_predict)))
        return(alpha_predictions,beta_predictions,model_predictions,proba_model_prediction)
        #if model = MB => alpha = 1

    def test(self,data_test):
        """Test the predictor on data_test

        Args:
            data_test: test_set

        Returns:
            ps.DataFrame containing predictions and ground truth
        """
        point_number = 1000
        tau = return_tau(point_number)
        data_test['D fit'] = np.zeros(data_test.shape[0])
        data_test['alpha fit'] = np.zeros(data_test.shape[0])

        def func_fit(row):
            diff = classical_fitD(tau,row)
            diffD, alpha = classical_fitalpha(tau,row)
            return [diff,alpha]

        def fitParallel(df):
            df[['D fit', 'alpha fit']] = df.apply(func_fit,axis = 1, result_type='expand')
            return df

        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
            return pd.concat(retLst)
        print('Using ',multiprocessing.cpu_count(),' cores')
        data_test = applyParallel(data_test.groupby(data_test.index), fitParallel)
        data_array = self.init_input_outputs_eval(data_test)
        pred = self.predict(data_array)
        data_test['alpha pred'] = pred[0]
        data_test['D pred'] = pred[1]
        data_test['model pred'] = self.model_encoder.inverse_transform(pred[2])
        return data_test

    def load_ptu(self,path_container_folder,t_max_experiment,stride,omega_x,omega,t_max_window):
        """Load data from every .ptu files in a folder

        Args:
            path_container_folder: folder containg all .ptu files
            t_max_experiment: time window of FCS measurement kept
            stride: sliding time distance between each time window
        """
        omega_x = int(omega_x)
        omega = int(omega)

        colonnes=['window number' ,'file name','omega_x','omega','Time window','stride','G']
        data = pd.DataFrame(columns=colonnes)
        points_number=1000
        dirlist =  os.listdir(path=path_container_folder)
        print('Number of files = ', len(dirlist))
        for file in dirlist:
            if file[-4:]=='.ptu':
                parser = PTUParser(path_container_folder+'/'+file)
                parser.load()
                fcs_signal = parser.photons['Resolved Time Tag'].sort_values().to_numpy()*1e-9
                fcs_signal = fcs_signal[fcs_signal<t_max_experiment]
                evaluation_points = definition_set_g(t_max_window,points_number)
                autocorrelograms_estim = g_convol_semilogx(np.sort(fcs_signal[fcs_signal<999][1:]),
                                                            evaluation_points, t_max_window, points_number, stride)
                for i in range(np.shape(autocorrelograms_estim)[0]):
                    new_row = {'window number' : i,'file name':file,'omega_x':omega_x,'omega':omega,
                                'Time window':t_max_window,'stride' : stride, 'G': autocorrelograms_estim[i][:]}
                    data.loc[len(data)] = new_row

        point_number = 1000
        tau = return_tau(point_number)
        data['D fit'] = np.zeros(data.shape[0])
        data['alpha fit'] = np.zeros(data.shape[0])

        def func_fit(row):
            diff = classical_fitD(tau,row)
            diffD, alpha = classical_fitalpha(tau,row)
            return [diff,alpha]

        def fitParallel(df):
            df[['D fit', 'alpha fit']] = df.apply(func_fit,axis = 1, result_type='expand')
            return df

        def applyParallel(dfGrouped, func):
            retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
            return pd.concat(retLst)

        data = applyParallel(data.groupby(data.index), fitParallel)
        data_array = self.init_input_outputs_eval(data)
        pred = self.predict(data_array)
        col = ['alpha pred','D pred','Model pred n#']
        for model in self.model_encoder.classes_:
            col.append('Proba '+model)
        data[col] = pd.DataFrame(
            np.column_stack(pred), index=data.index)
        data['Model pred'] = self.model_encoder.inverse_transform(pred[2])
        return data

    def predict_dataframe(self,dataframe):
        """Load data from a dataframe

        Args:
            pd.DataFrame: dats with col containing at least  ['omega_x','omega','Time window','G']
        """
        data_array = self.init_input_outputs_eval(dataframe)
        pred = self.predict(data_array)
        col = ['alpha pred','D pred','Model pred']
        for model in self.model_encoder.classes_:
            col.append('Proba '+model)
        dataframe[col] = pd.DataFrame(
            np.column_stack(pred), index=dataframe.index)
        dataframe['Model pred'] = self.model_encoder.inverse_transform(pred[2])
        return dataframe

    def load_csv(self,path_container_folder,t_max_experiment,path_res,subprocess):
        """Load data from every csv files in a folder

        Args:
            path_container_folder: folder containg all csv files
            t_max_experiment: time window of FCS measurement kept
            stride: sliding time distance between each time window
            path_res: path of the result folder
            subprocess: number of output files it will create (the less memory you have the more you need subprocesses)
        """


        colonnes=['window number' ,'file name','omega_x','omega','Time window','G','alpha','D','Illumination','Illu Params','Model']
        points_number=1000
        dirlist =  os.listdir(path=path_container_folder)



        #Equilibrate numbers of data from 3 models
        """countF = 0
        countM = 0
        countC = 0

        for file in dirlist:
            if file[4]=='F':
                countF+=1
        for file in dirlist:
            if file[4]=='C':
                if countC >countF:
                    dirlist.remove(file)
                countC+=1
            if file[4]=='M':
                if countM >countF:
                    dirlist.remove(file)
                countM+=1"""

        print('Number of files = ', len(dirlist))


        jobs = multiprocessing.cpu_count()

        list_dirlist = []

        for i in range(jobs*subprocess):
            list_dirlist.append(dirlist[int(i*np.floor(len(dirlist)/(jobs*subprocess))):int((1+i)*np.floor(len(dirlist)/(jobs*subprocess)))])
            #dataframes.append(pd.DataFrame(columns=colonnes))

        def open_csv(thread):
            number_emptyfiles = 0
            dataframe = pd.DataFrame(columns=colonnes)
            for file in list_dirlist[thread]:
                if file[-4:]=='.csv':
                    parser = CSVNameParser(file)
                    fcs_signal = np.loadtxt(str(path_container_folder+'/'+file), delimiter=";", dtype = str)
                    if np.size(np.shape(fcs_signal))>1.5:
                        number_emptyfiles +=1
                    else:
                        fcs_signal = fcs_signal[:-1]
                        fcs_signal = np.array([i for i in fcs_signal if i != ''], dtype= float)
                        alpha,D = fcs_signal[1],fcs_signal[0]
                        fcs_signal = np.sort(fcs_signal[3:])
                        fcs_signal = fcs_signal[fcs_signal<t_max_experiment]
                        if fcs_signal.size >0:
                            photonbysec = fcs_signal.size/(fcs_signal[-1]-fcs_signal[0])
                            for time in self.t_max_windows:
                                evaluation_points = definition_set_g(time,points_number)
                                autocorrelograms_estim = g_convol_semilogx(fcs_signal,
                                                                            evaluation_points, time, points_number, time)
                                for i in range(np.shape(autocorrelograms_estim)[0]):
                                    new_row = {'window number' : i,'file name':file,'omega_x':parser.wxy,'omega':parser.w,
                                                'Time window':time, 'G': autocorrelograms_estim[i][:],'alpha':alpha,'D':D,'Illumination':photonbysec,'Illu Params':parser.illu,'Model':parser.motion}
                                    dataframe.loc[len(dataframe)] = new_row
                        else:
                            number_emptyfiles +=1
            print("Number of empty files : ",number_emptyfiles)
            return dataframe

        start = time.time()
        for k in range(subprocess):
            print('Starting subprocess n°'+str(k+1)+' sur '+str(subprocess))
            print('At time (in hour) : ',(time.time()-start)/3600)
            dataframes = Parallel(n_jobs=jobs)(delayed(open_csv)(number) for number in range(k*jobs,(k+1)*jobs))
            self.predict_dataframe(pd.concat(dataframes)).to_pickle(path_res+'/prediction'+str(k)+'.pk')
            del dataframes

    def load_csv_evo(self,path_container_folder,t_max_experiment,path_res,subprocess):
        """Load data from every csv files in a folder

        Args:
            path_container_folder: folder containg all csv files
            t_max_experiment: time window of FCS measurement kept
            stride: sliding time distance between each time window
            path_res: path of the result folder
            subprocess: number of output files it will create (the less memeory you have the more you need subprocesses)
        """


        colonnes=['window number' ,'file name','omega_x','omega','Time window','G','val','Switch time','stride' ,'Start','End']
        points_number=1000
        dirlist =  os.listdir(path=path_container_folder)

        print('Number of files = ', len(dirlist))


        jobs = multiprocessing.cpu_count()

        list_dirlist = []

        for i in range(jobs*subprocess):
            list_dirlist.append(dirlist[int(i*np.floor(len(dirlist)/(jobs*subprocess))):int((1+i)*np.floor(len(dirlist)/(jobs*subprocess)))])
            #dataframes.append(pd.DataFrame(columns=colonnes))
        def open_csv_evo(thread):
            time = 0.5
            stride = 0.05
            number_emptyfiles = 0
            dataframe = pd.DataFrame(columns=colonnes)
            for file in list_dirlist[thread]:
                if file[-4:]=='.csv':
                    parser = CSVNameParser(file)
                    fcs_signal = np.loadtxt(str(path_container_folder+'/'+file), delimiter=";", dtype = str)
                    if np.size(np.shape(fcs_signal))>1.5:
                        number_emptyfiles +=1
                    else:
                        params = fcs_signal[1:19].reshape(-1,2).T
                        val, switch_time = np.array(np.hstack((fcs_signal[0],params[1,:])),dtype=float),np.array(params[0,:],dtype=float)
                        fcs_signal = np.array(fcs_signal[19:],dtype=float)
                        fcs_signal= np.sort(fcs_signal)
                        start = fcs_signal[0]
                        end = fcs_signal[-1]

                        #fcs_signal = np.array([i for i in fcs_signal if i != ''], dtype= float)
                        #alpha,D = fcs_signal[1],fcs_signal[0]
                        #fcs_signal = np.sort(fcs_signal[3:])
                        fcs_signal = fcs_signal[fcs_signal<t_max_experiment]
                        if fcs_signal.size >0:
                            photonbysec = fcs_signal.size/(fcs_signal[-1]-fcs_signal[0])
                            evaluation_points = definition_set_g(time,points_number)
                            autocorrelograms_estim = g_convol_semilogx(fcs_signal,
                                                                        evaluation_points, time, points_number, stride)
                            for i in range(np.shape(autocorrelograms_estim)[0]):
                                new_row = {'window number' : i,'file name':file,'omega_x':parser.w,'omega':parser.illu,
                                            'Time window':time, 'G': autocorrelograms_estim[i][:],'val':val,'Switch time':switch_time,'stride' : stride,'Start':start,'End':end}
                                dataframe.loc[len(dataframe)] = new_row
                        else:
                            number_emptyfiles +=1
            print("Number of empty files : ",number_emptyfiles)
            return dataframe

        start = time.time()
        for k in range(subprocess):
            print('Starting subprocess n°'+str(k+1)+' sur '+str(subprocess))
            print('At time (in hour) : ',(time.time()-start)/3600)
            dataframes = Parallel(n_jobs=jobs)(delayed(open_csv_evo)(number) for number in range(k*jobs,(k+1)*jobs))
            self.predict_dataframe(pd.concat(dataframes)).to_pickle(path_res+'/prediction'+str(k)+'.pk')
            del dataframes






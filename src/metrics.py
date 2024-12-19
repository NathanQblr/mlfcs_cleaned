"""Librairy used for compute diffreent metrics on the test set"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.metrics import mean_absolute_error, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import plotly.figure_factory as ff
import umap





def return_mae(Data, omegas,omegas_x,time_list,models):
    """Compute mae for a given Data on different sets of parameters

    Args:
        Data: predictions and groung truth (pd.DataFrame)
        omegas: list of parameters (python list of float)
        omegas_x: list of parameters (python list of float)
        time_list: list of parameters (python list of float)
        models: list of parameters (python list of string)

    Returns:
        mae (pd.DataFrame)
    """
    #Create new DataFrame for the figure
    fig = pd.DataFrame(columns=['Time window','$\omega_x$','$\omega$','Misclassified','MAE alpha','MAE D MB','MAE D Non Anom','Model','Valid'])

    for time in time_list:
        for omega_x in omegas_x:
            for omega in omegas:
                for model in models:
                    valid = 1
                    #Select sample with the good combination of parameters
                    idx = (Data['Model']==model)&(Data['Time window']==time)&(Data['omega_x']==omega_x)&(Data['omega']==omega)
                    if (idx.sum()>0)&(model=='MB'):
                        valid = 0
                        mae_ml_D = mean_absolute_error(Data.loc[idx,'D pred'],Data.loc[idx,'D'])
                    else:
                        mae_ml_D = None
                    if idx.sum()>0:
                        valid = 0
                        mae_ml_alpha = mean_absolute_error(Data.loc[idx,'alpha pred'],Data.loc[idx,'alpha'])
                    idx = (Data['Model']==model)&(Data['Time window']==time)&(Data['omega_x']==omega_x)&(Data['omega']==omega)&(Data['alpha']>=0.9)
                    if (idx.sum()>0):
                        if model =='RWF':
                            valid = 0
                            mae_ml_D_non_anom = mean_absolute_error(Data.loc[idx,'D pred'],Data.loc[idx,'D'])
                        else:
                            valid = 0
                            mae_ml_D_non_anom = mean_absolute_error(Data.loc[idx,'D pred'],Data.loc[idx,'D'])

                    row = {'Time window':time,'$\omega_x$':omega_x,'$\omega$':omega,'Misclassified':0,'MAE D MB':mae_ml_D,'MAE D Non Anom':mae_ml_D_non_anom,'MAE alpha':mae_ml_alpha,'Model':model,'Valid':valid}
                    fig.loc[len(fig)] = row
    return fig


def return_f1(Data, omegas,omegas_x,time_list,models):
    """Compute F1 Score for a given Data on different sets of parameters

    Args:
        Data: predictions and groung truth (pd.DataFrame)
        omegas: list of parameters (python list of float)
        omegas_x: list of parameters (python list of float)
        time_list: list of parameters (python list of float)
        models: list of parameters (python list of string)

    Returns:
        f1 Score (pd.DataFrame)
    """
    #Create a dataframe for the F1-Score Results
    fig = pd.DataFrame(columns=['Time window','$\omega_x$','$\omega$','F1 Score'])
    for time in time_list:
        for omega_x in omegas_x:
            for omega in omegas:
                valid = 1
                #Select every combination of parameters
                idx = (Data['Time window']==time)&(Data['omega_x']==omega_x)&(Data['omega']==omega)
                if idx.sum()>0:
                    valid = 0
                    f1 = f1_score(Data.loc[idx,'Model'],Data.loc[idx,'model pred'],average='micro')
                row = {'Time window':time,'$\omega_x$':omega_x,'$\omega$':omega,'F1 Score':f1}
                fig.loc[len(fig)] = row
    return fig

def return_mae_fit_ml(Data, omegas,omegas_x,time_list,models):
    """Compute F1 Score and MAE for a given Data on different sets of parameters

    Args:
        Data: predictions and groung truth (pd.DataFrame)
        omegas: list of parameters (python list of float)
        omegas_x: list of parameters (python list of float)
        time_list: list of parameters (python list of float)
        models: list of parameters (python list of string)

    Returns:
        F1 Score and MAE (pd.DataFrame)
    """
    fig = pd.DataFrame(columns=['Time window','omega_x','omega','Misclassified','MAE alpha','MAE D','Method'])
    i = 0
    for time in time_list:
        for omega_x in omegas_x:
            for omega in omegas:
                i+=1
                #Select every combination of parameters (the method only predict D if it predict Bm)
                idxD = (Data['Model']=='MB')&(Data['Time window']==time)&(Data['omega_x']==omega_x)&(Data['omega']==omega)&(Data['D fit'].notna())
                if idxD.sum()>0:
                    mae_fit_D = mean_absolute_error(Data.loc[idxD,'D fit'],Data.loc[idxD,'D'])
                    mae_ml_D = mean_absolute_error(Data.loc[idxD,'D pred'],Data.loc[idxD,'D'])
                #Select every combination of parameters
                idxa = (Data['Time window']==time)&(Data['omega_x']==omega_x)&(Data['omega']==omega)&(Data['alpha fit'].notna())
                if idxa.sum()>0:
                    mae_fit_alpha = mean_absolute_error(Data.loc[idxa,'alpha fit'],Data.loc[idxa,'alpha'])
                    mae_ml_alpha = mean_absolute_error(Data.loc[idxa,'alpha pred'],Data.loc[idxa,'alpha'])
                if idxD.sum()>0 and idxa.sum()>0 :
                    row = {'Time window':time,'omega_x':omega_x,'omega':omega,'Misclassified':0,'MAE D':mae_ml_D,'MAE alpha':mae_ml_alpha,'Method':'ML'}
                    fig.loc[len(fig)] = row
                    row = {'Time window':time,'omega_x':omega_x,'omega':omega,'Misclassified':0,'MAE D':mae_fit_D,'MAE alpha':mae_fit_alpha,'Method':'Fit'}
                    fig.loc[len(fig)] = row
    return fig


def create_confusion_matrices(Data, omegas, omegas_x, time_list, models):
    """Create confusion matrices for alpha and D for different models.

    Args:
        Data: DataFrame containing predictions and ground truth (pd.DataFrame)
        omegas: list of omega parameter values (list of float)
        omegas_x: list of omega_x parameter values (list of float)
        time_list: list of time window parameter values (list of float)
        models: list of model names (list of string)

    Returns:
        confusion_matrices_alpha: Dictionary of confusion matrices for alpha for each model
        confusion_matrices_D: Dictionary of confusion matrices for D for each model
    """
    confusion_matrices_alpha = {}
    confusion_matrices_D = {}

    for model in models:
        # Filter data by model
        model_data = Data[Data['Model'] == model]

        if not model_data.empty:
            # Extract true and predicted values for alpha and D
            y_true_alpha = model_data['alpha'].values
            y_pred_alpha = model_data['alpha pred'].values
            y_true_D = model_data['D'].values
            y_pred_D = model_data['D pred'].values

            # Create confusion matrix for alpha
            cm_alpha = confusion_matrix(y_true_alpha, y_pred_alpha)
            confusion_matrices_alpha[model] = cm_alpha

            # Create confusion matrix for D
            cm_D = confusion_matrix(y_true_D, y_pred_D)
            confusion_matrices_D[model] = cm_D

    return confusion_matrices_alpha, confusion_matrices_D


def normalize_data4plot(Res):
    """Create a dataframe with fit and ml metrics on different lines so they'll appear on the same plot

    Args:
        Res: pd.DataFrame

    Returns:
        pd.DataFrame
    """
    res_ml = Res.drop(columns=['D fit','alpha fit'],inplace=False,axis = 1)
    res_ml['Method'] = 'ML'
    res_ml = res_ml.rename(columns={'D pred':'D','alpha pred':'alpha'})

    res_fit = Res.drop(['D pred'],inplace=False,axis = 1)
    res_fit['Method'] = 'Fit'
    res_fit = res_fit.rename(columns={'D fit':'D','alpha fit':'alpha'})

    res_compare = pd.concat((res_ml,res_fit))
    del res_ml
    del res_fit

    Data = res_compare

    #A way to normalize file names
    name_encoder = LabelEncoder() #initializing an object of class LabelEncoder
    name_encoder.fit(Data['file name'])
    Data['file name'] = name_encoder.transform(Data['file name'])

    Data['alpha'].fillna(value = 2,inplace=True)#if the fitting method of alpha doesn't converge the classical method predict alpha = NA
    Data['D'].fillna(value = 1000,inplace=True)#if the fitting method of D doesn't converge the classical method predict D = NA
    Data = Data.loc[Data['D']!=1000,:]#delete value when D doesn't converge
    return Data

def create_recap_table(Data,files,Dtheo,glycerol):
    """Create a table that recap means alpha and D by methods and theoritical one

    Args:
        Data: pd.DataFrame
        files: list string
        Dtheo: list float
        glycerol: list float

    Returns:
        fig : figure
    """
    res_list = []
    for i,expe in enumerate(glycerol):
        d_theo = Dtheo[i]
        idx = (Data['Glycerol']==expe)&(Data['Method']=='ML')
        D_ml = np.round(Data.loc[idx,'D'].mean(),3)
        idx = (Data['Glycerol']==expe)&(Data['Method']=='ML')
        alpha_ml = np.round(Data.loc[idx,'alpha'].mean(),3)
        idx = (Data['Glycerol']==expe)&(Data['Method']=='Fit')
        D_fit = np.round(Data.loc[idx,'D'].mean(),3)
        idx = (Data['Glycerol']==expe)&(Data['Method']=='Fit')&(Data['alpha']!=2)
        alpha_fit = np.round(Data.loc[idx,'alpha'].mean(),3)
        line = {'Expe':expe,'D theoric':d_theo,'Mean D ML':D_ml,'Mean D fit':D_fit,'Mean alpha ML':alpha_ml,'Mean alpha fit':alpha_fit}
        res_list.append(line)
    res = pd.DataFrame(res_list)

    fig = ff.create_table(res)
    return fig


def plot_ternary(data_beads):
    """Plot data from classification in a ternary representation when we have 3 candidates models

    Args:
        data_beads: pd.DataFrame with prediction of models

    Returns:
        plt.figure
    """
    #Load predicted probability for every models
    x = data_beads['Proba MB']
    y = data_beads['Proba RWF']
    z = data_beads['Proba CTRW'] + data_beads['Proba FBM']
    # Convert data in ternary representation
    data_beads['nx'] = 0.5*(2*y+z)
    data_beads['ny'] = np.sqrt(3)/2*z

    glycerol = np.unique(data_beads['Glycerol'])

    for gly in glycerol:
        name_encoder = LabelEncoder() #initializing an object of class LabelEncoder
        name_encoder.fit(data_beads.loc[(data_beads['Glycerol']== gly),'file name'])
        data_beads.loc[(data_beads['Glycerol']== gly),'file name'] = name_encoder.transform(data_beads.loc[(data_beads['Glycerol']== gly),'file name'])

    Expe =1
    triangle = np.array([[0,0],[1,0],[0.5,np.sqrt(1.25)]])
    palette3 = ['#ffffff','#bc8977','#8e7cc3','#719b78']#'#719b78','#bc8977']#'#6a73a4'
    #Blanc,  Green = Bm, Violet = CTRW + fBM ,Brown = RWf

    #Creation a colormap corrzsponding to the 3 zones where the probability of each model is maximale
    x,y = np.meshgrid(np.linspace(0, 1, 1000),np.linspace(0, 1, 1000))
    idx = (0<x)&(0<y)&(x<1)&(y<1)&(1-x-y<1)&(0<1-x-y)
    u = 1*((x>=y)&(x>=1-x-y)&idx)+2*((y>x)&idx&(y>=1-x-y))+3*((x<1-x-y)&idx&(y<1-x-y))
    u = u[:-1, :-1]
    u_min, u_max = -np.abs(z).max(), np.abs(z).max()
    nx = 0.5*(2*x+y)
    ny= np.sqrt(3)/2*y


    cm = 1/2.54  # centimeters in inches
    sns.set_style ("whitegrid")
    plt.rcParams['font.size'] = 8

    k = 0

    fig, axs = plt.subplots(2, int(np.ceil(len(glycerol)/2)),figsize=(15*cm, 15*cm))

    for line in axs:
        for col in line:
            if k<len(glycerol):
                col.pcolormesh(nx, ny, u,cmap=ListedColormap(palette3))
                sns.lineplot(data=data_beads.loc[(data_beads['Glycerol']== glycerol[k])&(data_beads['file name']< 5)], x="nx", y="ny",style_order="window number",hue = "file name",palette='dark',ax = col,legend=False,size=1,marker='o', markersize = 3)
                col.axis([0,1,0,1])
                col.set_aspect('equal', 'box')
                col.tick_params(bottom=False,top=False,labelbottom=False)
                col.axis('off')
                col.set(title = glycerol[k],frame_on = True)
                k+=1
    fig.tight_layout()
    return fig


def plot_umap(data_beads):
    """Plot data from classification in a UMAP 2D representation when we have 4 candidates models

    Args:
        data_beads: pd.DataFrame with prediction of models

    Returns:
        plt.figure
    """
    # Load predicted probabilities for each model
    prob_data = data_beads[['Proba MB', 'Proba FBM', 'Proba CTRW', 'Proba RWF']]

    # Apply UMAP for dimensionality reduction
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_model.fit_transform(prob_data)

    # Add UMAP results to the DataFrame
    data_beads['UMAP1'] = embedding[:, 0]
    data_beads['UMAP2'] = embedding[:, 1]

    glycerol = np.unique(data_beads['Glycerol'])

    # Normalize 'file name' for color coding
    for gly in glycerol:
        name_encoder = LabelEncoder()
        name_encoder.fit(data_beads.loc[(data_beads['Glycerol'] == gly), 'file name'])
        data_beads.loc[(data_beads['Glycerol'] == gly), 'file name'] = name_encoder.transform(
            data_beads.loc[(data_beads['Glycerol'] == gly), 'file name'])

    # Create the UMAP plot
    cm = 1 / 2.54  # centimeters in inches
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 8

    fig, axs = plt.subplots(2, int(np.ceil(len(glycerol) / 2)), figsize=(15 * cm, 15 * cm))

    k = 0
    for line in axs:
        for col in line:
            if k < len(glycerol):
                sns.scatterplot(
                    data=data_beads.loc[(data_beads['Glycerol'] == glycerol[k])],
                    x="UMAP1", y="UMAP2", hue="file name", palette='dark',
                    ax=col, legend=False, marker='o', s=15
                )
                col.set_aspect('equal', 'box')
                col.set(title=glycerol[k], frame_on=True)
                k += 1

    fig.tight_layout()
    return fig



def compute_dynamic_mae(data_D,data_alpha,time_list,omegas,omegas_x):
    """Compute MAE on data with a dynamic diffusion coefficient (data_D) and data with a dynamic anomalous exponent (data_alpha)

    Args:
        data_D: pd.DataFrame
        data_alpha: pd.DataFrame
        time_list: list float
        omegas: list float
        omegas_x: list float

    Returns:
        pd.DataFrame
    """
    res = pd.DataFrame(columns = ['Time window', 'window number','MAE','False negative','Number','$\omega_x$','$\omega$','Dynamic'])
    for omega_x in omegas_x:
        for omega in omegas:
            for tmax in time_list:
                    for times in np.unique(data_D['window number']):
                        lines = (data_D['Time window']==tmax)&(data_D['window number']==times)&(data_D['omega_x']==omega_x)&(data_D['omega']==omega)&(data_D['Dynamic']=='D')
                        if lines.sum()>0:
                            val0 = data_D.loc[lines,'val_0'].to_numpy()
                            val1 = data_D.loc[lines,'val_1'].to_numpy()
                            D_pred = data_D.loc[lines,'D pred'].to_numpy()
                            if times<15:
                                mae = mean_absolute_error(val0,D_pred)
                            else:
                                mae = mean_absolute_error(val1,D_pred)
                            false_negative = (data_D.loc[lines,'Model pred'] != 'MB').to_numpy().mean()

                            new_row = {'Time window' : tmax, 'window number':times,'MAE':mae,'False negative':false_negative,'Number' : lines.to_numpy().sum(),'$\omega_x$':omega_x,'$\omega$':omega,'Dynamic':'D'}
                            res.loc[len(res)] = new_row
                        lines = (data_alpha['Time window']==tmax)&(data_alpha['window number']==times)&(data_alpha['omega_x']==omega_x)&(data_alpha['omega']==omega)&(data_alpha['Dynamic']=='A')
                        if lines.sum()>0:
                            val0 = data_alpha.loc[lines,'val_0'].to_numpy()
                            val1 = data_alpha.loc[lines,'val_1'].to_numpy()
                            alpha_pred = data_alpha.loc[lines,'alpha pred'].to_numpy()
                            if times<15:
                                mae = mean_absolute_error(val0,alpha_pred)
                            else:
                                mae = mean_absolute_error(val1,alpha_pred)
                            false_negative = (data_alpha.loc[lines,'Model pred'] != 'CTRW').to_numpy().mean()

                            new_row = {'Time window' : tmax, 'window number':times,'MAE':mae,'False negative':false_negative,'Number' : lines.to_numpy().sum(),'$\omega_x$':omega_x,'$\omega$':omega,'Dynamic':'A'}
                            res.loc[len(res)] = new_row
    return res

def metrics_fn_illu(datasets,time_list):
    """Compute F1 Score and MAE for a given Data on different sets of illumination parameters

    Args:
        datasets: pd.DataFrame
        time_list: list float

    Returns:
        pd.DataFrame
    """
    res = pd.DataFrame(columns = ['Time window', 'MAE','Number','omega_x','omega','Illumination','Illu Params','F1'])
    illu_param = 0
    illu = 0
    omega = 0
    omega_x = 0

    #for omega_x in omegas_x:
    #    for omega in omegas:
    for tmax in time_list:
        #for illu in range(5):
        for illu_param in range(1,6):
            lines = (datasets['Time window']==tmax)&(datasets['Illu Params']>=2*illu_param+2)&(datasets['Illu Params']<=2*illu_param+3)
            if lines.sum()>0 :
                alpha_real = datasets.loc[lines,'alpha'].to_numpy()
                alpha_pred = datasets.loc[lines,'alpha pred'].to_numpy()
                mae = mean_absolute_error(alpha_real,alpha_pred)
                model_real = datasets.loc[lines,'Model pred'].to_numpy()
                model_pred = datasets.loc[lines,'Model'].to_numpy()
                f1 = f1_score(model_real,model_pred,average='weighted')
                new_row = {'Time window' : tmax, 'Illumination':2.5e4*illu,'Illu Params':'[6e3*'+str(2*illu_param+2)+', 6e3*'+str(2*illu_param+4)+'[','MAE':mae,'Number' : lines.to_numpy().sum(),'omega_x':omega_x,'omega':omega,'F1':f1}
                res.loc[len(res)] = new_row
    return res


def create_scatter_plots(Data, omegas, omegas_x, time_list, models):
    """Create scatter plots for alpha and D for different models.

    Args:
        Data: DataFrame containing predictions and ground truth (pd.DataFrame)
        omegas: list of omega parameter values (list of float)
        omegas_x: list of omega_x parameter values (list of float)
        time_list: list of time window parameter values (list of float)
        models: list of model names (list of string)

    Returns:
        None (displays scatter plots)
    """
    fig, axes = plt.subplots(nrows=len(models), ncols=2, figsize=(12, 6 * len(models)))
    fig.suptitle('Scatter Plots of Predictions vs Real Values', fontsize=16)

    for i, model in enumerate(models):
        # Filter data by model
        model_data = Data[Data['Model'] == model]

        if not model_data.empty:
            # Extract true and predicted values for alpha and D
            y_true_alpha = model_data['alpha'].values
            y_pred_alpha = model_data['alpha pred'].values
            y_true_D = model_data['D'].values
            y_pred_D = model_data['D pred'].values

            # Plot for alpha
            ax_alpha = axes[i, 0] if len(models) > 1 else axes[0]
            ax_alpha.scatter(y_true_alpha, y_pred_alpha, alpha=0.05, edgecolors='k')
            ax_alpha.plot([y_true_alpha.min(), y_true_alpha.max()], [y_true_alpha.min(), y_true_alpha.max()], 'r--', lw=2)
            ax_alpha.set_title(f'Model: {model} - Alpha')
            ax_alpha.set_xlabel('True Alpha')
            ax_alpha.set_ylabel('Predicted Alpha')

            # Plot for D
            ax_D = axes[i, 1] if len(models) > 1 else axes[1]
            ax_D.scatter(y_true_D, y_pred_D, alpha=0.1, edgecolors='k')
            ax_D.plot([y_true_D.min(), y_true_D.max()], [y_true_D.min(), y_true_D.max()], 'r--', lw=2)
            ax_D.set_title(f'Model: {model} - D')
            ax_D.set_xlabel('True D')
            ax_D.set_ylabel('Predicted D')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def create_conf_mat(Data, models):
    alpha_ranges = [(i, i + 0.2) for i in np.arange(0, 1, 0.2)]
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Confusion Matrices for Different Alpha Ranges', fontsize=20)

    for idx, (alpha_min, alpha_max) in enumerate(alpha_ranges):
        range_data = Data[(Data['alpha'] > alpha_min) & (Data['alpha'] <= alpha_max)]
        if not range_data.empty and 'model pred' in range_data.columns:
            cm = confusion_matrix(range_data['Model'], range_data['model pred'], labels=models, normalize='true')
            row, col = divmod(idx, 2)
            sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=models, yticklabels=models, ax=axes[row, col])
            axes[row, col].set_title(f'Alpha {alpha_min:.1f}-{alpha_max:.1f}')
            axes[row, col].set_xlabel('Predicted Model')
            axes[row, col].set_ylabel('True Model')

    # Hide any unused subplots
    for idx in range(len(alpha_ranges), 6):
        row, col = divmod(idx, 2)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def create_conf_matD(Data, models):
    D_ranges = [(0,2),(2,10)]
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Confusion Matrices for Different D Ranges', fontsize=20)

    for idx, (D_min, D_max) in enumerate(D_ranges):
        range_data = Data[(Data['D'] > D_min) & (Data['D'] <= D_max)]
        if not range_data.empty and 'model pred' in range_data.columns:
            cm = confusion_matrix(range_data['Model'], range_data['model pred'], labels=models, normalize='true')
            row, col = divmod(idx, 2)
            sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=models, yticklabels=models, ax=axes[row, col])
            axes[row, col].set_title(f'D {D_min:.1f}-{D_max:.1f}')
            axes[row, col].set_xlabel('Predicted Model')
            axes[row, col].set_ylabel('True Model')

    # Hide any unused subplots
    for idx in range(len(D_ranges), 6):
        row, col = divmod(idx, 2)
        fig.delaxes(axes[row, col])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



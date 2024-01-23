from pathlib import Path
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import dill
import pickle
import copy
import itertools
from keras.layers import Input

# Import Data Science Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import time

# import birds_utils.BIRDS
from sklearn.model_selection import train_test_split

# Tensorflow Libraries
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint, ReduceLROnPlateau,History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import plot_model
from sklearn.utils.class_weight import compute_class_weight

# System libraries
from pathlib import Path
import os.path
import random
import pickle
# Visualization Libraries
import matplotlib.cm as cm
import cv2
import seaborn as sns
import birds_utils as birds
sns.set_style('darkgrid')

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import dill
import copy

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import ImageFont
import visualkeras




from sklearn.metrics import accuracy_score




# load the images into a data frame
def load_data(project_dir):
  image_dir = Path(project_dir + '/data')
  filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
  labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

  filepaths = pd.Series(filepaths, name='Filepath').astype(str)
  labels = pd.Series(labels, name='label')

  # Concatenate filepaths and labels
  image_df = pd.concat([filepaths, labels], axis=1)
  return image_df


def get_label_idx(image_df,label):
  idx = list(image_df[image_df['label'].isin([label])].index)
  if (len(idx)==0):
    print(f'{label} does not exist in the df')
  return idx

def get_label_data_set_size(iamge_df,label):
  idx = get_label_idx(label)
  return len(idx)

def get_labels(image_df):
  labels = list(image_df['label'].unique())
  return (labels)


# plot images from image_df,according to user desires
# inputs:
# image_df - image data frame with columns (FilePath amd label)
# label (optional) - which label to plot (default None)
# fig_width (optional) - fig width (default 20)
# n_cols (optional) - number of image columns (default 8)
# N (optional) - Number of augemnated images to plot (default 32)
# idx (optional) - Index of images to plot (default None). This option is used when label=None 
def plot_label_images(image_df,label=None,N=None,idx=None,fig_width=20,n_cols=8):
  font_size=10*fig_width/10*4/n_cols
  if (label != None):
    idx = get_label_idx(image_df,label)
    if (N == None):
      N = image_df[image_df['label']==label].shape[0]
      idx = idx[0:N+1]

  elif (idx != None):
      N = len(idx)

  N_image_in_fig = N

  n_rows = int(np.ceil(N_image_in_fig/n_cols))
  fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_width*n_rows/n_cols),
                      subplot_kw={'xticks': [], 'yticks': []})

  for label_ind, ax in enumerate(axes.flat):
    if (label_ind<N):
      ax.imshow(plt.imread(image_df.loc[idx[label_ind]].Filepath))
      ax.set_title(f'{image_df.loc[idx[label_ind]].label} {idx[label_ind]}',fontsize=font_size)

  # plt.subplots_adjust(wspace=0)
  plt.tight_layout(pad=0.5)
  plt.show()

def plot_images(df,label=None,N=None,idx=None,fig_width=25,n_cols=8,font_size=None):
  if (df.shape[0]==0):
    print('df is empty')
    return

  if (n_cols==1):
      fig_width=10

  if (font_size is None):
      font_size=fig_width*3/n_cols

  if (label != None):
    idx = get_label_idx(df,label)
    if (N is None):
      N = image_df[image_df['label']==label].shape[0]
      idx = idx[0:N+1]

  elif (idx is not None):
      N = len(idx)

  N_image_in_fig = N

  n_rows = int(np.ceil(N_image_in_fig/n_cols))
  fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_width*n_rows/n_cols+2),
                      subplot_kw={'xticks': [], 'yticks': []})

  if (N==1):
     axes = np.array([axes])

  for ind, ax in enumerate(axes.flat):
    if (ind<N):
      ax.imshow(plt.imread(df.loc[idx[ind]].Filepath))
      if 'predicted_label' not in df.columns:
        ax.set_title(f'{df.loc[idx[ind]].label} {idx[ind]}',fontsize=font_size)
      else:
        if df['status'].loc[idx[ind]]:
          color = "green"
        else:
          color = "red"

        ax.set_title(f"index:{idx[ind]}\nTrue: {df.label.loc[idx[ind]]}\nPredicted: {df.predicted_label.loc[idx[ind]]}", color=color,fontsize=font_size)
        # ax.set_title(f"index:{df.index[ind]}\nTrue: {df.label.iloc[ind]}\nPredicted: {df.predicted_label.iloc[ind]}", color=color,fontsize=font_size)

  # plt.subplots_adjust(wspace=0)
  plt.tight_layout(pad=0.5)
  plt.show()

# plot an agumented set of images 
# inputs:
# aug_img - an augmentation object (output of ImageDataGenerator.flow)
# sample_image (optional) - original image () default is None
# fig_width (optional) - fig width (default 20)
# n_cols (optional) - number of image columns (default 8)
# N (optional) - Number of augemnated images to plot (default 32)
def plot_augumented_images(aug_img_obj,image_title=None,sample_image=None,fig_width=20,n_cols=8,N=32):
  font_size=10*fig_width/10*4/n_cols
  
  if (sample_image is not None):
    plt.figure(figsize=(3, 3))
    plt.title("Original Image")
    plt.imshow(sample_image)
    plt.axis("off")
    plt.show()
    

  n_rows = int(np.ceil(N/n_cols))
  fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_width*n_rows/n_cols),
                      subplot_kw={'xticks': [], 'yticks': []})


  for ind, ax in enumerate(axes.flat):
    if (ind<N):
      ax.imshow(aug_img_obj.next().astype("uint8")[0])

  # plt.subplots_adjust(wspace=0)
  plt.tight_layout(pad=0.5)

  suptitle = fig.suptitle('Figure Title', fontsize=25)
  suptitle.set_position((0.5, 1.3))

  # plt.show()



#  filter the df according to various options:
#  labels (list of strings) - The filter will output all the samples of the desired labels. 
#  N_samples_per_label (integer) - The filter will output N_samples_per_label samples from each label
#                                  (or all samples of the label if there are less than N_samples_per_label).
#                                  N_samples_per_label = 'all' will return the input df unchanged
     
def filter_df(df,labels=None,N_samples_per_label=None,status=None,predicted_labels=None):
  if isinstance(labels, str):
    labels = [labels]

  if isinstance(predicted_labels, str):
    predicted_labels = [predicted_labels]


  df_filt = df
  if (labels is not None): 
    if (labels=='all'):
      df_filt = df
    else:  
      df_filt = df[df['label'].isin(labels)]
  elif (predicted_labels is not None):
    if (predicted_labels=='all'):
      df_filt = df
    else:        
      df_filt = df[df['predicted_label'].isin(predicted_labels)]



  elif (N_samples_per_label is not None):
    if (N_samples_per_label=='all'):
      df_filt = df
    else:
      df_filt = pd.DataFrame()
      labels = get_labels(df)
      for label in labels:
        df_tmp = filter_df(df,labels=list([label]))
        df_tmp = df_tmp.iloc[0:min(N_samples_per_label,df_tmp.shape[0])]
        df_filt = pd.concat([df_filt,df_tmp])

  if (status is not None):
    df_filt = df_filt[df_filt['status']==status]
  
  return df_filt



# plot the a histogram of the 1'st N_labels top. if  N_labels is empty it is taken as teh number of all lables that exist
def plot_labels_count(image_df, N_labels=None):
    if N_labels is None:
        N_labels = len(image_df['label'].unique())

    label_counts = image_df['label'].value_counts()[:N_labels]

    plt.figure(figsize=(15, 10))
    fontsize = 15

    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='dark:salmon_r')
    plt.title(f'Distribution of Top {N_labels} Labels in Image Dataset', fontsize=30)
    plt.xlabel('label', fontsize=fontsize)
    plt.ylabel('Count', fontsize=fontsize)
    
    # Limit X-ticks to 10
    plt.xticks(rotation=45, fontsize=10)
    plt.xticks(range(0, len(label_counts.index), max(len(label_counts.index) // 10, 1)), label_counts.index[::max(len(label_counts.index) // 10, 1)])
    
    # Add mean +/- std dashed lines
    mean_line = plt.axhline(label_counts.mean(), color='black', linestyle='dashed', linewidth=2, label='Mean')
    upper_std_line = plt.axhline(label_counts.mean() + label_counts.std(), color='red', linestyle='dashed', linewidth=2, label='Mean + Std')
    lower_std_line = plt.axhline(label_counts.mean() - label_counts.std(), color='blue', linestyle='dashed', linewidth=2, label='Mean - Std')
    
    # Add min and max lines
    min_line = plt.axhline(label_counts.min(), color='green', linestyle='dashed', linewidth=2, label='Min')
    max_line = plt.axhline(label_counts.max(), color='purple', linestyle='dashed', linewidth=2, label='Max')
    
    # Add y ticks for the dashed lines
    y_ticks = [label_counts.mean(), label_counts.mean() + label_counts.std(), label_counts.mean() - label_counts.std(), label_counts.min(), label_counts.max()]
    plt.yticks(y_ticks, fontsize=10)
    
    # Add text above the red dashed line with normalized standard deviation
    std_norm = label_counts.std() / label_counts.mean() * 100
    text = f'Normalized STD = {std_norm:.2f}%'
    plt.text(len(label_counts) // 2, label_counts.mean() + label_counts.std() + 2, text, ha='center', va='bottom', fontsize=fontsize, color='red')
    
    # Add legend
    plt.legend(handles=[mean_line, upper_std_line, lower_std_line, min_line, max_line], loc='upper right', fontsize=fontsize)
    
    plt.show()

# Example usage:
# Assuming 'im
# 'get_image' gets an image from the image_df
# inputs:
#   image_df - image data frame
#   idx - image index        
#   df_index (optional) - whether to take idx according to data frame true index or just a row number(default True)          
def get_image(image_df,idx,df_index=True):
  if (df_index==True):
      try:
          return(plt.imread(image_df.loc[idx].Filepath))
      except:
          print(f'idx {idx} does not exist in data frame')
          return None
  else:
      return(plt.imread(image_df.iloc[idx].Filepath))

# 'create_lables_dic' creates lables dictionary from the train_images_obj (used by apply_model) 
# inputs: 
#   train_images_obj - the output of an ImageDataGenerator.flow_from_dataframe loaded with the train_df
# outpus:
#    labels_dic - dictioary with the labels
def create_lables_dic(train_images_obj):
    # Map the label
    labels = (train_images_obj.class_indices)
    labels_dic = dict((v,k) for k,v in labels.items())
    return labels_dic
    

def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report



def calculate_accuracy_per_label(df, label_col='label', predicted_col='predicted_label'):
    """
    Calculate accuracy for each unique label in a DataFrame.

    Parameters:
    - df: DataFrame
        The DataFrame containing 'label' and 'predicted_label' columns.
    - label_col: str, default='label'
        The column name for the true labels.
    - predicted_col: str, default='predicted_label'
        The column name for the predicted labels.

    Returns:
    - accuracy_per_label: dict
        A dictionary containing accuracy for each unique label.
    """
    accuracy_per_label = {}

    # Get unique labels
    labels = df[label_col].unique()

    for label in labels:
        mask = df[label_col] == label
        accuracy = accuracy_score(df.loc[mask, label_col], df.loc[mask, predicted_col])
        accuracy_per_label[label] = accuracy

    accuracy_df = pd.DataFrame(list(accuracy_per_label.items()), columns=['label', 'accuracy'])
    accuracy_df.set_index('label', inplace=True)

    return accuracy_df


# apply_model applies a model on the test_images_obj and returns the test_df with the additional 'predict_label' 
# and 'status' indictiating if the prediction succeeded
# inputs:
#   model - a trained keras model
#   labels_dic - labels dictionary
#   obj_obj - the output of an ImageDataGenerator.flow_from_dataframe loaded with an image_df
# outpus:
#   obj_obj - updated image_df
def apply_model(model,labels_dic,obj_dic,plot_report=True):
# apply the model    
    pred = model.predict(obj_dic['images_obj'])
    results = model.evaluate(obj_dic['images_obj'], verbose=0)

    pred = np.argmax(pred,axis=1)
    pred = [labels_dic[k] for k in pred]
    
    obj_dic['df']['predicted_label'] = pred
    obj_dic['df']['status'] = obj_dic['df']['predicted_label']==obj_dic['df']['label']

    print ('\n')    
    print ('--------------------------------')
    print (f"    results for {obj_dic['name']}")
    print ('--------------------------------')
    print(f"{obj_dic['name']} Loss: {results[0]:.5f}")
    print(f"{obj_dic['name']} Accuracy: {(results[1] * 100):.2f}%")

    # print(f"{obj_dic['name']}:")
    obj_dic['classification_report'] = get_classification_report(obj_dic['df']['label'], obj_dic['df']['predicted_label'])

    # add accuracy
    accuracy_df = calculate_accuracy_per_label(obj_dic['df'], label_col='label', predicted_col='predicted_label')

    obj_dic['classification_report'] = obj_dic['classification_report'].merge(accuracy_df, left_index=True, right_index=True)


    # plot if desired
    if (plot_report):
        plot_columns = list(obj_dic['classification_report'].columns)
        plot_columns.remove('support')
        name = obj_dic['name']
        obj_dic['classification_report'][plot_columns].plot(rot=45,title=f'{name}:classification report')


    print(obj_dic['classification_report'] )

    return obj_dic

def save_obj_dic_stack(obj_dic_stack,obj_dic_stack_path):
    for key in list(obj_dic_stack.keys()):

# remove images_obj as it cannot be saved      
        if ('images_obj' in obj_dic_stack[key]):
            obj_dic_stack[key].pop('images_obj')

    with open(obj_dic_stack_path, 'wb') as file:
        dill.dump(obj_dic_stack, file)


def get_obj_dic_stack(model,train_obj_dic,val_obj_dic,test_obj_dic,obj_dic_stack_path):
    if os.path.exists(obj_dic_stack_path):
        print(f'loding obj_dic_stack from {obj_dic_stack_path}')
        with open(obj_dic_stack_path, 'rb') as file:
            obj_dic_stack = pickle.load(file)
    else:
        labels_dic = create_lables_dic(train_obj_dic['images_obj'])
        test_obj_dic = apply_model(model,labels_dic,test_obj_dic)
        train_obj_dic = apply_model(model,labels_dic,train_obj_dic)
        val_obj_dic = apply_model(model,labels_dic,val_obj_dic)


        # def analyze_classifaction_reports(train_obj_dic,val_obj_dic,test_obj_dic):
        obj_dic_stack = {'train':train_obj_dic,'val':val_obj_dic,'test':test_obj_dic}

        save_obj_dic_stack (obj_dic_stack,obj_dic_stack_path)
# add accuracy
    for key in obj_dic_stack.keys():
        if 'accuracy' not in obj_dic_stack[key]['df']:
            accuracy_df = calculate_accuracy_per_label(obj_dic_stack[key]['df'], label_col='label', predicted_col='predicted_label')
            obj_dic_stack[key]['classification_report'] = obj_dic_stack[key]['classification_report'].merge(accuracy_df, left_index=True, right_index=True)

    return obj_dic_stack
    

def plot_obj_dic_stack_score(obj_dic_stack, score='f1', base_df_type='test'):
    df = pd.DataFrame()

    for key in obj_dic_stack.keys():
        df_pre = obj_dic_stack[key]['classification_report']
        df_pre = df_pre.add_suffix(f'_{key}')
        df = pd.concat([df, df_pre], axis=1)

    # Sort by the specified score for the base_df_type
    df = df.sort_values(f'{score}_{base_df_type}', ascending=True)

    # Plot lines for each key
    ax = df.filter(like=score, axis=1).plot(rot=45, linestyle='-')

    # Plot average lines with corresponding colors
    for line, key in zip(ax.get_lines(), obj_dic_stack.keys()):
        avg_score = df[f'{score}_{key}'].mean()
        line_color = line.get_color()
        ax.axhline(avg_score, linestyle='--', color=line_color)

        # Add y ticks on the right y-axis
        ax2 = ax.twinx()
        ax2.set_yticks([avg_score])
        ax2.set_yticklabels([f'{avg_score:.2f}'], color=line_color)
        ax2.set_ylim(ax.get_ylim())  # Match the y-limits with the left y-axis

    plt.show()


def plot_label_false_and_true(obj_dic_stack,ana_label=None,ana_label_ind=0,n_cols=5,N=5,false_ind=0,false_label = None):
    df = obj_dic_stack['train']['classification_report'].sort_values('f1-score')

    if (ana_label is None):
        ana_label = df.index[ana_label_ind]

    # get the data_frame of the false detection 
    false_df = filter_df(obj_dic_stack['train']['df'],labels=ana_label,status=False)

    # get the data_frame of the true detection 
    true_df = filter_df(obj_dic_stack['train']['df'],labels=ana_label,status=True)

    # plot the distribution of the false detection
    false_label_count_df = false_df.groupby('predicted_label').count().sort_values('status',ascending=False)
    ax = false_label_count_df['status'].plot(kind='bar', title=f'{ana_label}:histogram of false label counts',rot=45)
    ax.set_xticks(range(len(false_label_count_df)))
    ax.set_xticklabels(false_label_count_df.index)

    if (false_label is None):
        false_label = false_label_count_df.index[false_ind]

# filter the false_df according to the false_label    
    false_df = filter_df(false_df,predicted_labels=false_label)
    plot_images(false_df,idx=list(false_df.index[0:N]),n_cols=n_cols)
    plot_label_images(obj_dic_stack['train']['df'],N=N,label=false_label,n_cols=n_cols)



def get_other_images(df):
    df = df[df['Filepath'].str.contains('other', case=False, na=False)]
    return df

def remove_other_images(df):
    other_indexes = df[df['Filepath'].str.contains('other', case=False, na=False)].index
    df = df.drop(other_indexes)
    return df

def save_var(var,file_name):
    status = True
    try:
        with open(file_name, 'wb') as file:
            pickle.dump(var, file)
    except:
        status = False
        print(f'could not open {file_name} for writing')
    return status

def load_var(file_anme):
    var = None
    try:
        with open(file_anme, 'rb') as file:
            var = pickle.load(file)
    except:
        print(f'could not open {file_name} for reading')
    return var


def create_run_path_name(base_path,params):
    
    if (base_path[-1] == '/'):
        run_path_name = base_path[:-1]
    else:
        run_path_name = base_path

    for key in params.keys():
        run_path_name = f'{run_path_name}_{params[key]}'
    
    run_path_name = run_path_name + '/'
    return run_path_name


def get_params_permutations(params):
    # Get all keys and their associated value lists
    keys, value_lists = zip(*params.items())

    # Get all permutations of values associated with each key
    permutations_list = list(itertools.product(*value_lists))

    # Create a list of dictionaries, each representing a combination of parameter values
    param_permutations = [dict(zip(keys, values)) for values in permutations_list]

    return param_permutations



def create_model(pretrained_model,params={},visualize_model = False,AUGMENTATON = False):
    print(params.keys())
    if ('dense1_size' not in params.keys()):
        params['dense1_size'] = 128

    if ('dense2_size' not in params.keys()):
        params['dense2_size'] = 256

    if ('N_labels' not in params.keys()):
        params['N_labels'] = 2
    
    if (visualize_model):
        input_shape=(224, 224, 3)
        inputs = Input(shape=input_shape)
        font_size = 10
        scale_xy=0.8
    else:
        inputs = pretrained_model.input
        inputs.__dict__['_type_spec']
        inputs = pretrained_model.input
        font_size = 100
        scale_xy=3

    if (AUGMENTATON):
        x = augment(inputs)
        x = pretrained_model(x)
        x = Dense([params['dense1_size']], activation='relu')(x)
        x = Dropout(0.45)(x)
        x = Dense(params['dense2_size'], activation='relu')(x)
        x = Dropout(0.45)(x)

    else:
        if (visualize_model):
            # x = augment(inputs)
            # x = Dense(params['dense1_size'], activation='relu')(pretrained_model.output)
            x = inputs
            x = pretrained_model(x)
            x = Dense(params['dense1_size'], activation='relu')(x)
            x = Dropout(0.45)(x)
            x = Dense(params['dense2_size'], activation='relu')(x)
            x = Dropout(0.45)(x)
        else:
            inputs = pretrained_model.input
            # x = augment(inputs)
            x = Dense(params['dense1_size'], activation='relu')(pretrained_model.output)
            x = Dropout(0.45)(x)
            x = Dense(params['dense2_size'], activation='relu')(x)
            x = Dropout(0.45)(x)
        


    outputs = Dense(params['N_labels'], activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )



    # Adjust the font size
    font = ImageFont.truetype("arial.ttf", font_size)

    
    if (visualize_model):
        # Save the model image to a file with specific colors for each layer
        visualkeras.layered_view(model, legend=True, font=font, to_file='model.png', scale_xy=scale_xy, color_map=create_color_map())
        plot_model(model, show_shapes=True, show_layer_names=True)
    return model 




def train_model (model,models_path,train_obj_dic,val_obj_dic,params,run_path=None):
    if (run_path is None):
      run_path = create_run_path_name (models_path,params)
      
    print(f'run_path = {run_path}')
    if ('N_epochs_patitence' not in params.keys()):
        params['N_epochs_patitence'] = 128

    if ('N_epochs') not in params.keys():
        params['N_epochs'] = 100

    if ('N_labels') not in params.keys():
        params['N_labels'] = 2
    

    RUN_NAME = os.path.basename(os.path.normpath(run_path))

    # Create checkpoint callback
    checkpoint_path = f'{run_path}/{RUN_NAME}_check_point.h5'
    print(checkpoint_path)
    checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                        save_weights_only=False,
                                        monitor="val_accuracy",
                                        save_best_only=True)

    # Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
    early_stopping = EarlyStopping(monitor = "val_loss", # watch the val loss metric
                                patience = params['N_epochs_patitence'],
                                restore_best_weights = True) # if val loss decreases for 3 epochs in a row, stop training

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


    model_file_path = f'{run_path}/{RUN_NAME}_model.keras'
    history_file_path = f'{run_path}/{RUN_NAME}_history.pkl'

    if os.path.exists(model_file_path):
    # load model    
        print(f'loading {RUN_NAME} and related history')
        model = keras.models.load_model(model_file_path)
        # Later, you can load the history object
    # load history   
        with open(history_file_path, 'rb') as file:
            history = pickle.load(file)
            history = pd.DataFrame({'history':history})

    else:
        if os.path.exists(checkpoint_path):
            print (f'loading check point from {checkpoint_path}')
            model = keras.models.load_model(checkpoint_path)
        
        # Calculate class weights
        labels = train_obj_dic['df']['label'].tolist() # Make sure this is your training data labels
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights_dict = dict(enumerate(class_weights))

        start_time = time.time()

        history = model.fit(
            train_obj_dic['images_obj'],
            steps_per_epoch=len(train_obj_dic['images_obj']),
            validation_data=val_obj_dic['images_obj'],
            validation_steps=len(val_obj_dic['images_obj']),
            epochs=params['N_epochs'],
            class_weight=class_weights_dict,
            callbacks=[
                early_stopping,
                # birds.create_tensorboard_callback("training_logs",
                #                             "bird_classification"),
                checkpoint_callback,
                reduce_lr
            ]
        )

        end_time = time.time()

# Calculate the elapsed time
        elapsed_time = end_time - start_time        
        birds.save_var(elapsed_time,f'{run_path}/elapsed_time.keras')

        model.save(model_file_path)
        with open(history_file_path, 'wb') as file:
            pickle.dump(history.history, file)

    return model,history


def create_color_map():
    color_map = defaultdict(dict)
    # customize the colours
    color_map[layers.Conv2D]['fill'] = '#00f5d4'
    color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
    color_map[layers.Dropout]['fill'] = '#03045e'
    color_map[layers.Dense]['fill'] = '#fb5607'
    color_map[layers.Flatten]['fill'] = '#ffbe0b'
    color_map[layers.Dropout]['fill'] = '#03045e'
    return color_map


def get_filtered_subfolders(base_dir, filter_str):
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        print(f"The base directory '{base_dir}' does not exist.")
        return []

    # Get the list of subfolders
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    # Filter subfolders based on the provided filter string
    filtered_subfolders = [folder+'/' for folder in subfolders if filter_str in folder]

    return filtered_subfolders

import os

def get_filtered_files(folder_path, filtering_str,full_path = 'True'):
    filtered_files = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return filtered_files

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the filtering string is present in the file name
        # print(file_name)
        if filtering_str in file_name:
            if (not full_path):
                # Add the file to the list if it matches the criteria
                filtered_files.append(file_name)
            else:
                filtered_files.append(folder_path+file_name)

    return filtered_files


def plot_training_history(files):
    if not isinstance(files, list):
        files = [files]

    # Create subplots for each metric
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes = axes.flatten()

    legend_list = []
    for file_index, file in enumerate(files):
        # Load training history from the file using birds.load_var
        history = birds.load_var(file)

        # Create a DataFrame from the history
        history_df = pd.DataFrame(history)

        # Extract training and validation metrics
        accuracy = history_df['accuracy']
        val_accuracy = history_df['val_accuracy']
        loss = history_df['loss']
        val_loss = history_df['val_loss']

        # create the legends
        substrings = file.split('_')
        mat_size = substrings[-6:-4]
        str = mat_size[0]+'_'+mat_size[1]
        legend_list.append(str)

        if (str=='128_256'):
            LineWidth = 3
        else:
            LineWidth = 1
            
        # Plotting on each subplot
        axes[0].plot(accuracy, label=f'File {file_index + 1}',linewidth=LineWidth)
        axes[1].plot(val_accuracy, label=f'File {file_index + 1}',linewidth=LineWidth)
        axes[2].plot(loss, label=f'File {file_index + 1}',linewidth=LineWidth)
        axes[3].plot(val_loss, label=f'File {file_index + 1}',linewidth=LineWidth)



    # Set titles and labels
    axes[0].set_title('Training Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[2].set_title('Training Loss')
    axes[3].set_title('Validation Loss')


    for ax in axes:
        ax.set_xlabel('Epochs')
        ax.legend(legend_list)

    plt.tight_layout()
    plt.show()


def plot_history_single_run(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.xlabel('epocs')
    plt.ylabel('validate')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('epocs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


from PIL import Image
def resize_images(df):
    for i in range(1):
        img = Image.open(df['Filepath'].iloc[i])

        # Resize the image
        resized_img = img.resize(TARGET_SIZE)
        plt.imshow(resized_img)
        resized_img.save(df['Filepath'].iloc[i])    
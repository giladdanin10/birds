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

def plot_images(df,label=None,N=None,idx=None,fig_width=25,n_cols=8):
  if (df.shape[0]==0):
    print('df is empty')
    return

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
      labels = birds.get_labels(df)
      for label in labels:
        df_tmp = filter_df(df,labels=list([label]))
        df_tmp = df_tmp.iloc[0:min(N_samples_per_label,df_tmp.shape[0])]
        df_filt = pd.concat([df_filt,df_tmp])

  if (status is not None):
    df_filt = df_filt[df_filt['status']==status]
  
  return df_filt



# plot the a histogram of the 1'st N_labels top. if  N_labels is empty it is taken as teh number of all lables that exist
def plot_labels_count (image_df,N_labels=None):
    if N_labels==None:
        N_labels = len(image_df['label'].unique())

    label_counts = image_df['label'].value_counts()[:N_labels]
   
    plt.figure(figsize=(20, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='dark:salmon_r')
    plt.title(f'Distribution of Top {N_labels} Labels in Image Dataset', fontsize=16)
    plt.xlabel('label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

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
    

# apply_model applies a model on the test_images_obj and returns the test_df with the additional 'predict_label' 
# and 'status' indictiating if the prediction succeeded
# inputs:
#   model - a trained keras model
#   labels_dic - labels dictionary
#   obj_obj - the output of an ImageDataGenerator.flow_from_dataframe loaded with an image_df
# outpus:
#   obj_obj - updated image_df


def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = metrics.classification_report(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
    return df_classification_report



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
    obj_dic['classification_report'] = birds.get_classification_report(obj_dic['df']['label'], obj_dic['df']['predicted_label'])

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
    return obj_dic_stack


def plot_obj_dic_stack_score(obj_dic_stack,score='f1'):
    df = pd.DataFrame()
    for key in obj_dic_stack.keys():
        df_pre = (obj_dic_stack[key]['classification_report'])
        df_pre = df_pre.add_suffix(f'_{key}')
        # if (df.shape[1]==0):
        #     df = df_pre
        # else:
        df = pd.concat([df, df_pre], axis=1)
    df.filter(like=score, axis=1).plot(rot=45)

def plot_label_false_and_true(obj_dic_stack,ana_label=None,ana_label_ind=0,n_cols=5,N=5,false_ind=0,false_label = None):
    df = obj_dic_stack['train']['classification_report'].sort_values('f1-score')

    if (ana_label is None):
        ana_label = df.index[ana_label_ind]

    # get the data_frame of the false detection 
    false_df = filter_df(obj_dic_stack['train']['df'],labels=ana_label,status=False)

    # get the data_frame of the true detection 
    true_df = filter_df(obj_dic_stack['train']['df'],labels=ana_label,status=True)

    # plot the distribution of the false dedctection
    false_label_count_df = false_df.groupby('predicted_label').count().sort_values('status',ascending=False)
    # ax = false_label_count_df['status'].plot(kind='bar', title=f'{ana_label}:histogram of false label counts',rot=45)
    # ax.set_xticks(range(len(false_label_count_df)))
    # ax.set_xticklabels(false_label_count_df.index)

    if (false_label is None):
        false_label = false_label_count_df.index[false_ind]

# filter the false_df according to the false_label    
    false_df = filter_df(false_df,predicted_labels=false_label)
    plot_images(false_df,idx=list(false_df.index[0:N]),n_cols=n_cols)
    plot_label_images(obj_dic_stack['train']['df'],N=N,label=false_label,n_cols=n_cols)

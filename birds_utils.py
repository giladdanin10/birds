from pathlib import Path
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# class BIRDS :
#   def __init__(self):
#     self.image_df = None
  
#   # load the images into a data frame
#   def load_data(self,project_dir):
#     image_dir = Path(project_dir + '/data')
#     filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
#     labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

#     filepaths = pd.Series(filepaths, name='Filepath').astype(str)
#     labels = pd.Series(labels, name='label')

#     # Concatenate filepaths and labels
#     self.image_df = pd.concat([filepaths, labels], axis=1)


#   def get_label_idx(self,label):
#     idx = list(self.image_df[self.image_df['label'].isin([label])].index)
#     if (len(idx)==0):
#       print(f'{label} does not exist in the df')
#     return idx

#   def get_label_data_set_size(self,label):
#     idx = self.get_label_idx(label)
#     return len(idx)

#   def get_labels(self):
#     labels = self.image_df['label'].unique()
#     return (labels)

#   def plot_label_images(self,label=None,N=None,idx=None,fig_width=20,n_cols=8):
#     font_size=10*fig_width/10*4/n_cols
#     if (label != None):
#       idx = self.get_label_idx(label)
#       if (N == None):
#         N = self.image_df[self.image_df['label']==label].shape[0]
#         idx = idx[0:N+1]

#     elif (idx != None):
#         N = len(idx)

#     N_image_in_fig = N

#     n_rows = int(np.ceil(N_image_in_fig/n_cols))
#     fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_width*n_rows/n_cols),
#                         subplot_kw={'xticks': [], 'yticks': []})

#     for label_ind, ax in enumerate(axes.flat):
#       if (label_ind<N):
#         ax.imshow(plt.imread(self.image_df.loc[idx[label_ind]].Filepath))
#         ax.set_title(f'{self.image_df.loc[idx[label_ind]].label} {idx[label_ind]}',fontsize=font_size)

#     # plt.subplots_adjust(wspace=0)
#     plt.tight_layout(pad=0.5)
#     plt.show()


# #  filter the df according to various options:
# #  labels (list of strings) - a list of t he desired labels 
#     def filter_df(self,df=None,labels=None):
#       if (df==None):
#         df = self.image_df

#       if isinstance(labels, str):
#         labels = [labels]

#       df_filt = pd.DataFrame()
#       if (labels != None):        
#         df_filt = df[df['label'].isin(labels)]

#       return df_filt

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
     
def filter_df(df,labels=None,N_samples_per_label=None):
  if isinstance(labels, str):
    labels = [labels]

  df_filt = pd.DataFrame()
  if (labels != None): 
    if (labels=='all'):
      df_filt = df
    else:  
      df_filt = df[df['label'].isin(labels)]

  elif (N_samples_per_label !=None):
    if (N_samples_per_label=='all'):
      df_filt = df
    else:
      labels = get_labels(df)
      for label in labels:
        df_tmp = filter_df(df,labels=list([label]))
        df_tmp = df_tmp.iloc[0:min(N_samples_per_label,df_tmp.shape[0])]
        df_filt = pd.concat([df_filt,df_tmp])
    


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

def apply_model(model,labels_dic,obj_dic):
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
    print(classification_report(obj_dic['df']['label'], obj_dic['df']['predicted_label']))

    return obj_dic
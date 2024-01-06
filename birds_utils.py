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
#     labels = pd.Series(labels, name='Label')

#     # Concatenate filepaths and labels
#     self.image_df = pd.concat([filepaths, labels], axis=1)


#   def get_label_idx(self,label):
#     idx = list(self.image_df[self.image_df['Label'].isin([label])].index)
#     if (len(idx)==0):
#       print(f'{label} does not exist in the df')
#     return idx

#   def get_label_data_set_size(self,label):
#     idx = self.get_label_idx(label)
#     return len(idx)

#   def get_labels(self):
#     labels = self.image_df['Label'].unique()
#     return (labels)

#   def plot_label_images(self,label=None,N=None,idx=None,fig_width=20,n_cols=8):
#     font_size=10*fig_width/10*4/n_cols
#     if (label != None):
#       idx = self.get_label_idx(label)
#       if (N == None):
#         N = self.image_df[self.image_df['Label']==label].shape[0]
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
#         ax.set_title(f'{self.image_df.loc[idx[label_ind]].Label} {idx[label_ind]}',fontsize=font_size)

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
#         df_filt = df[df['Label'].isin(labels)]

#       return df_filt

# load the images into a data frame
def load_data(project_dir):
  image_dir = Path(project_dir + '/data')
  filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
  labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

  filepaths = pd.Series(filepaths, name='Filepath').astype(str)
  labels = pd.Series(labels, name='Label')

  # Concatenate filepaths and labels
  image_df = pd.concat([filepaths, labels], axis=1)
  return image_df


def get_label_idx(image_df,label):
  idx = list(image_df[image_df['Label'].isin([label])].index)
  if (len(idx)==0):
    print(f'{label} does not exist in the df')
  return idx

def get_label_data_set_size(iamge_df,label):
  idx = get_label_idx(label)
  return len(idx)

def get_labels(image_df):
  labels = list(image_df['Label'].unique())
  return (labels)


# plot the images according to user desires
# parameters:
# 
def plot_label_images(image_df,label=None,N=None,idx=None,fig_width=20,n_cols=8):
  font_size=10*fig_width/10*4/n_cols
  if (label != None):
    idx = get_label_idx(image_df,label)
    if (N == None):
      N = image_df[image_df['Label']==label].shape[0]
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
      ax.set_title(f'{image_df.loc[idx[label_ind]].Label} {idx[label_ind]}',fontsize=font_size)

  # plt.subplots_adjust(wspace=0)
  plt.tight_layout(pad=0.5)
  plt.show()



#  filter the df according to various options:
#  labels (list of strings) - a list of t he desired labels 
def filter_df(df,labels=None):
  if isinstance(labels, str):
    labels = [labels]

  df_filt = pd.DataFrame()
  if (labels != None):        
    df_filt = df[df['Label'].isin(labels)]

  return df_filt

# plot the a histogram of the 1'st N_labels top. if  N_labels is empty it is taken as teh number of all lables that exist
def plot_labels_count (image_df,N_labels=None):
    if N_labels==None:
        N_labels = len(image_df['Label'].unique())

    label_counts = image_df['Label'].value_counts()[:N_labels]

    plt.figure(figsize=(20, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='dark:salmon_r')
    plt.title(f'Distribution of Top {N_labels} Labels in Image Dataset', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()





from pathlib import Path
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BIRDS :
  def __init__(self):
    self.image_df = None
  
  # load the images into a data frame
  def load_data(self,project_dir):
    image_dir = Path(project_dir + '/data')
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    self.image_df = pd.concat([filepaths, labels], axis=1)


  def get_label_idx(self,label):
    idx = list(self.image_df[self.image_df['Label'].isin([label])].index)
    if (len(idx)==0):
      print(f'{label} does not exist in the df')
    return idx

  def get_label_data_set_size(self,label):
    idx = self.get_label_idx(label)
    return len(idx)

  def get_labels(self):
    labels = self.image_df['Label'].unique()
    return (labels)

  def plot_label_images(self,label=None,N=None,idx=None,fig_width=20,n_cols=8):
    font_size=10*fig_width/10*4/n_cols
    if (label != None):
      idx = self.get_label_idx(label)
      if (N == None):
        N = self.image_df[self.image_df['Label']==label].shape[0]
        idx = idx[0:N+1]

    elif (idx != None):
        N = len(idx)

    N_image_in_fig = N

    n_rows = int(np.ceil(N_image_in_fig/n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_width*n_rows/n_cols),
                        subplot_kw={'xticks': [], 'yticks': []})

    for label_ind, ax in enumerate(axes.flat):
      if (label_ind<N):
        ax.imshow(plt.imread(self.image_df.loc[idx[label_ind]].Filepath))
        ax.set_title(f'{self.image_df.loc[idx[label_ind]].Label} {idx[label_ind]}',fontsize=font_size)

    # plt.subplots_adjust(wspace=0)
    plt.tight_layout(pad=0.5)
    plt.show()



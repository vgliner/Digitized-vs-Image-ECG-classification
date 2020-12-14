from torch.utils.data import Dataset
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
from  Perspective_transformation import *
from  Realtime_ECG_drawing import *
import time
import random
import cv2




class ECG_Rendered_Multilead_Dataset(Dataset):
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir=None, transform=None, partial_upload=False,new_format=True, apply_perspective_transformation=False, realtime_rendering=True):
        super().__init__()
        self.data = []
        self.data_info = []
        self.transform = transform
        self.realtime_rendering=realtime_rendering
        self.last_chunk_uploaded_to_memory = 1
        self.partial_upload = partial_upload
        self.new_format=new_format
        self.batch_size_in_file_new_format=650
        self.classification_data=[]
        self.root_dir=root_dir
        self.apply_perspective_transformation=apply_perspective_transformation

        if root_dir is None:
            self.dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Database')
        else:
            self.dataset_path = root_dir
        
        if new_format==False:
            for indx, file in enumerate(glob.glob(self.dataset_path + "*.pkl")):
                unpickled_data = self.unpickle_ECG_data(file=file)
                list_shape = np.shape(unpickled_data)
                self.samples = unpickled_data
                if indx == 0:  # (partial_upload) and
                    break
        else:
            classification_data=[]  
            image_data=[]
            self.last_chunk_uploaded_to_memory=0
            for cntr in range(3):
                f=h5py.File(os.path.join(self.dataset_path,'diagnosis_digitized'+str(cntr)+'.hdf5'), 'r')
                f_keys=f.keys()
                for key in f_keys:
                    n1 = f.get(key)
                    classification_data.append(np.array(n1))

            for batch_cntr in range(len(classification_data)):
                for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
                    self.classification_data.append(bool(classification_data[batch_cntr][record_in_batch_cntr]))
    
        if self.realtime_rendering==True:
            self.canvas= cv2.imread(os.path.join(self.dataset_path,'ECG_paper.jpg'),cv2.IMREAD_COLOR )
            self.canvas = cv2.resize(self.canvas,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
            self.ECG_Data_init = ECG_Multilead_Dataset(root_dir=self.dataset_path,transform=self.transform, partial_upload=self.partial_upload)


    def __len__(self):
        if self.new_format:
            return len(self.classification_data)
        else:
            if self.partial_upload:
                return len(self.samples)
            else:
                return 41830

    def __getitem__(self, idx):
        if self.realtime_rendering==False:
            if self.new_format==False:
                chunk_number = idx // 1000 + 1
                if chunk_number == self.last_chunk_uploaded_to_memory:
                    if self.transform:
                        sample = self.transform(self.samples[idx % 1000])
                    else:
                        sample = self.samples[idx % 1000]
                    return sample
                else:
                    file = self.dataset_path + 'Rendered_data' + str(max(1, idx // 1000 + 1)) + '.pkl'
                    unpickled_data = self.unpickle_ECG_data(file=file)
                    self.last_chunk_uploaded_to_memory = max(1, idx // 1000 + 1)
                    self.samples = unpickled_data
                    if self.transform:
                        sample = self.transform(self.samples[idx % 1000])
                    else:
                        sample = self.samples[idx % 1000]
                    return sample
            else:
                with h5py.File(os.path.join(self.dataset_path,"Unified_rendered_db.hdf5"), "r") as f:
                    n1=f.get(str(idx))
                    image_data=np.array(n1)
                    if self.apply_perspective_transformation:
                        image_data=Perspective_transformation_application(image_data,database_path=self.dataset_path)
              

                sample=(image_data,self.classification_data[idx])
                return sample
        else:
            ECG_Data=self.ECG_Data_init[idx]
            image_data=draw_ECG_multilead_vanilla_ver2(ECG_Data[0], self.canvas,to_plot=False)
            # print(f'Size of canvas is : {np.shape(image_data)}')
            if self.apply_perspective_transformation:
                image_size_before_rendering=np.shape(image_data)
                image_data=Perspective_transformation_application(image_data,database_path=self.dataset_path,realtime_rendering=True)
                image_size_after_rendering=np.shape(image_data)    
                size_diff=(np.asarray(image_size_after_rendering)-np.asarray(image_size_before_rendering))
                if self.new_format==True:
                    #image_data=image_data[size_diff[0]//2:-size_diff[0]//2,size_diff[1]//2:-size_diff[1]//2,[2,1,0]]
                    image_data=cv2.resize(image_data,None,fx=image_size_before_rendering[0]/image_size_after_rendering[0], fy=image_size_before_rendering[1]/image_size_after_rendering[1], interpolation = cv2.INTER_AREA)
            image_data=image_data[:,:,[2,1,0]]
            # print(f'Size of canvas is : {np.shape(image_data)}')

            sample=(image_data,self.classification_data[idx])
            return sample



    def unpickle_ECG_data(self, file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data

    def plot(self, idx):
        # TODO : Implement plot
        item_to_show = self.__getitem__(idx)
        plt.imshow(item_to_show[0])
        plt.show()
        return


if __name__ == "__main__":
    # New database directory
    target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
    ECG_test = ECG_Rendered_Multilead_Dataset(root_dir=target_path, transform=None, partial_upload=False,apply_perspective_transformation=False,realtime_rendering=True)  # For KNN demo
    testing_array=list(range(0,10000))
    start = time.time()
    for indx in testing_array:
        rand_indx=random.randint(10,41830)
        K = ECG_test[rand_indx]
        if (indx%20==0) and indx>0:
            print(f'Currently processing index: {indx}')
            end = time.time()
            print((end - start)/indx)
        # plt.imshow(K[0])
        # figManager = plt.get_current_fig_manager()
        # plt.show()
        print(f'Record: {indx} ,Is AFIB: {K[1]}')
    end = time.time()
    print(end - start)



from torch.utils.data import Dataset
import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

PRINT_FLAG = False
class ECG_Multilead_Dataset(Dataset):
    def __init__(self, root_dir=None, transform=None, partial_upload=False,new_format=True, multiclass=False,multiclass_to_binary=False, multiclass_to_binary_type=1):
        super().__init__()
        self.data = []
        self.data_debug =[]
        self.data_info = []
        self.transform = transform
        self.new_format=new_format

        if root_dir is None:
            self.dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Database',"Digitized_emergency.p")
        else:
            self.dataset_path = root_dir

        if new_format==False:
            for file in glob.glob(self.dataset_path+"*.pkl"):
                unpickled_data = self.unpickle_ECG_data(file=file)
                list_shape = np.shape(unpickled_data)
                if len(list_shape) > 1:
                    self.data = self.data+unpickled_data
                else:
                    self.data_info.append(unpickled_data)
                
                if partial_upload:
                    break

        else:
            print('Uploading new format')
            short_leads_data=[]
            long_lead_data=[]
            classification_data=[]
            classification_data_debug=[]

            # for cntr in range(3):
            #     f=h5py.File(root_dir+'short_leads_digitized'+str(cntr)+'.hdf5', 'r')
            #     f_keys=f.keys()
            #     for key in f_keys:
            #         n1 = f.get(key)
            #         short_leads_data.append(np.array(n1))

            #     f=h5py.File(root_dir+'long_lead_data_digitized'+str(cntr)+'.hdf5', 'r')
            #     f_keys=f.keys()
            #     for key in f_keys:
            #         n1 = f.get(key)
            #         long_lead_data.append(np.array(n1))

            #     if True: #if multiclass==False:
            #         f=h5py.File(root_dir+'diagnosis_digitized'+str(cntr)+'.hdf5', 'r')
            #         f_keys=f.keys()
            #         for key in f_keys:
            #             n1 = f.get(key)
            #             classification_data.append(np.array(n1))

            # if True:#multiclass:
            #     f=h5py.File(root_dir+'multiclass_classification'+'.hdf5', 'r')
            #     f_keys=f.keys()
            #     for key in f_keys:
            #         n1 = f.get(key)
            #         if multiclass_to_binary==False:
            #             classification_data_debug.append(np.array(n1))  
            #         else:
            #             k=n1.shape
            #             classification_data_debug.append((k[multiclass_to_binary_type]))  #bool

            # print('Here')                  
                    # for batch_cntr in range(len(classification_data)):
                    #     for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
                    #         self.data.append(((short_leads_data[batch_cntr][record_in_batch_cntr],long_lead_data[batch_cntr][record_in_batch_cntr]),bool(classification_data[batch_cntr][record_in_batch_cntr])))

########################
        pickled_data=pickle.load( open(self.dataset_path, "rb" ) )
        short_leads_data_debug=[]
        long_lead_data_debug=[]
        Data_pickled=pickled_data['Data']
        classification_data_debug=pickled_data['Class']
        for d in Data_pickled:
            short_leads_data_debug.append(d[0][0])
            long_lead_data_debug.append(d[0][1])


########################
        # if multiclass==False:
        #     for batch_cntr in range(len(classification_data)):
        #         for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
        #             self.data.append(((short_leads_data[batch_cntr][record_in_batch_cntr],long_lead_data[batch_cntr][record_in_batch_cntr]),bool(classification_data[batch_cntr][record_in_batch_cntr])))
        # else:
        #     for batch_cntr in range(len(classification_data)):
        #         for record_in_batch_cntr in range(len(classification_data[batch_cntr])):
        #             self.data.append(((short_leads_data[batch_cntr][record_in_batch_cntr],long_lead_data[batch_cntr][record_in_batch_cntr]),classification_data[batch_cntr][record_in_batch_cntr]))

        # classification_data_debug=np.squeeze(classification_data_debug)
        # for batch_cntr in range(len(short_leads_data)):
        #     for record_in_batch_cntr in range(len(short_leads_data[batch_cntr])):
        #         self.data_debug.append(((short_leads_data[batch_cntr][record_in_batch_cntr],long_lead_data[batch_cntr][record_in_batch_cntr]),bool(classification_data_debug[batch_cntr*1000+record_in_batch_cntr][multiclass_to_binary_type])))
        if multiclass==False:
            for record_in_batch_cntr in range(len(classification_data_debug)):
                self.data_debug.append(((short_leads_data_debug[record_in_batch_cntr],long_lead_data_debug[record_in_batch_cntr]),bool(classification_data_debug[record_in_batch_cntr][multiclass_to_binary_type])))
        else:
            for record_in_batch_cntr in range(len(classification_data_debug)):
                self.data_debug.append(((short_leads_data_debug[record_in_batch_cntr],long_lead_data_debug[record_in_batch_cntr]),classification_data_debug[record_in_batch_cntr]))



        # with open('self_data.txt', 'w') as f:
        #     for item_num,item in enumerate(self.data):
        #         f.write(f'Item num: {item_num}: \n')
        #         f.write(f'{item[0][0]} {item[0][1]} \n {item[1]} \n')

        # with open('self_data_debug.txt', 'w') as f:
        #     for item_num,item in enumerate(self.data_debug):
        #         f.write(f'Item num: {item_num}: \n')
        #         f.write(f'{item[0][0]} {item[0][1]} \n {item[1]} \n')

        self.samples = self.data_debug
        # data = [d[0] for d in self.data]
        # self.data = data
        # self.target = [d[1] for d in self.data]
        if PRINT_FLAG:
            print(f'Uploaded data, size of {np.shape(self.data)}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx])
        return self.samples[idx]

    def plot(self, idx):
        Leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        Long_lead_type = 'Lead2'
        item_to_plot = self.samples[idx]
        fig, axes = plt.subplots(nrows=6, ncols=2)
        fig.suptitle(f'Record number {idx}, Is AFIB: {item_to_plot[1]}')
        titles = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        B = item_to_plot[0][0]
        for ax, cntr in zip(axes.flatten(), range(12)):
            ax.plot(B[cntr, :], linewidth=1.0)
            ax.set(title=titles[cntr])
        plt.plot()
        plt.show()
        return

    def unpickle_ECG_data(self, file='ECG_data.pkl'):
        with open(file, 'rb') as fo:
            pickled_data = pickle.load(fo, encoding='bytes')
        if PRINT_FLAG:
            print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data  


if __name__=="__main__":    
    print(f'Argument List: {str(sys.argv)}')
    target_path=os.path.join(os.getcwd(),'Data','')
    Sum=np.zeros(9)
    ECG_test=ECG_Multilead_Dataset(target_path, multiclass=True,multiclass_to_binary=False, multiclass_to_binary_type=1)
    for cntr in range(len(ECG_test)):
        for ext_cntr in range(9):
            B=ECG_test[cntr]
            if B[1][ext_cntr]:
                Sum[ext_cntr]+=1
    print(f'Finished. The statistics is : {Sum}')



#%% Create multiclass model
# This script uses as an input the Chinese database and creates multiclasses database of 12 lead ECG for further rendering and 
#usage for deep nets.

# %% Create united database from all data that I have
import matplotlib.pyplot as plt
import glob
import os
import scipy.io as sio
import numpy as np
import pandas as pd
import os
import pickle
from ECG_pickling import *
import h5py


# %% Chineese challenge
def Upload_db_records(DB_path, plot_flag=True):
    db_splitted_records = []
    dB_classes_list=[]
    records_per_file = 4000
    for file in glob.glob("*.mat"):
        print(file)
        mat_contents = sio.loadmat(DB_path + file)
        b = mat_contents['ECG']['data'].item()
        db_ref_path = DB_path + 'REFERENCE.csv'

        classification = upload_classification(db_ref_path, file)
        # Plotting, if necessary
        if plot_flag:
            fig, axes = plt.subplots(nrows=6, ncols=2)
            fig.suptitle(f'Record number {file}, Is AFIB: {classification}')
            for ax, cntr in zip(axes.flatten(), range(12)):
                ax.plot(b[cntr, :], linewidth=1.0)
                ax.set(title=titles[cntr])
            plt.plot()
            plt.show()
        splitted_records = split_records(b)
        if splitted_records == -1:
            continue
        # Splitting to a standard records
        for splitted_record in splitted_records:
            db_splitted_records.append((splitted_record, classification))
            dB_classes_list.append(classification)
    # for file_num in range(len(db_splitted_records) // records_per_file + 1):
    #     max_storage_value = min([len(db_splitted_records), (file_num + 1) * records_per_file])
    #     filename = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data\Chinese_db' + str(file_num) + '.pkl'
    #     pickle_ECG_data(db_splitted_records[file_num * records_per_file:max_storage_value], file=filename)
    #     print(f'Pickled: file Chinese_db_{file_num}.pkl')

        print(f'Created {len(db_splitted_records)} records')
    return (db_splitted_records,dB_classes_list)


# %% Upload DB classification
def upload_classification(DB_ref_path, required_entry):
    # print(DB_ref_path)
    data = pd.read_csv(DB_ref_path)
    data.head()
    _entries = data.Recording.to_list()
    _entry_number_in_list = _entries.index(required_entry[0:5])
    _values = data.values[_entry_number_in_list, :]
    classification= np.zeros(9)
    for val in range(1,4):
        if _values[val]<10:
            classification[int(_values[val])-1]=1
    return classification


def split_records(ECG_raw):
    # Definitions
    fs = 500  # Hz
    bit_encoding = 8  # bit
    leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    long_lead_type = 'Lead2'
    long_lead_index = leads.index(long_lead_type)
    scale_type = 1  # 0- No scaling, 1- Normalized between 0-1, 2- Normalized between 0- bit_encoding
    results_list = []

    # Calculations
    num_of_seconds_in_record = len(ECG_raw[0, :]) / fs
    if num_of_seconds_in_record < 10:
        return -1
    number_of_output_records = int(np.floor(num_of_seconds_in_record / 2.5))
    for record_cntr in range(number_of_output_records):
        # Scale and quantize the records
        relevant_data = ECG_raw[:, int(2.5 * record_cntr * fs):int(2.5 * (record_cntr + 1) * fs)]
        if record_cntr < number_of_output_records - 4:  # Taking according to the first 2.5 sec.
            long_lead_recording = \
                ECG_raw[long_lead_index, int(2.5 * record_cntr * fs):int(2.5 * (record_cntr + 4) * fs)]
        else:
            long_lead_recording = \
                ECG_raw[long_lead_index, int(2.5 * (number_of_output_records - 4) * fs):int(
                                      2.5 * number_of_output_records * fs)]

        scaled_data = relevant_data
        scaled_data_long_lead = long_lead_recording

        if (long_lead_recording.max() - long_lead_recording.min()) <= 0:
            return -1

        if (relevant_data.max() - relevant_data.min()) <= 0:
            return -1

        if scale_type > 0:
            scaled_data = (relevant_data - relevant_data.min()) / (relevant_data.max() - relevant_data.min())
            scaled_data_long_lead = (long_lead_recording - long_lead_recording.min()) / (
                    long_lead_recording.max() - long_lead_recording.min())

        if scale_type > 1:
            scaled_data = (scaled_data * (2 ** bit_encoding - 1)).astype(int)
            scaled_data_long_lead = (scaled_data_long_lead * (2 ** bit_encoding - 1)).astype(int)

        results_list.append((scaled_data, np.expand_dims(scaled_data_long_lead, axis=0)))
        # print(f'Record number {record_cntr} out of  {number_of_output_records} , long record length {len(Long_lead_recording)}, total : {len(ECG_raw[0,:])}')
    # Return tuple of (2.5 sec X 12 lead matrix + one strip of 10 records)
    return results_list

def store_uploaded_records(target_path, records_to_store,dB_classes_list):
    print('Storage...')
    with h5py.File(target_path+"multiclass_classification"+".hdf5","w") as f:
        dset=f.create_dataset('Multiclassification',data=dB_classes_list)
    

def store_uploaded_records_new(target_path, records_to_store,dB_classes_list):
    print('New Storage...')
    with h5py.File(target_path+"multiclass_classification_new"+".hdf5","w") as f:
        for db_num in range(len(records_to_store)):
            dset=f.create_dataset(str(db_num),data=dB_classes_list[db_num])    
        # for x,y in enumerate(records_to_store):
        #     dset=f.create_dataset(str(x),y[1])
            # print(f"Processed : {x}")

def test_created_db(target_path):
    f=h5py.File(target_path+'multiclass_classification_new'+'.hdf5', 'r')
    f_keys=f.keys()
    uploaded_data=[]
    for key in range(len(f_keys)):
        n1 = f.get(str(key))
        uploaded_data.append(np.array(n1))


    with open(target_path+'self_data_new_new.txt', 'w') as f:
        for item_num,item in enumerate(uploaded_data):
            f.write(f'Item num: {item_num}: \n')
            f.write(f'{bool(uploaded_data[item_num][1])} \n')
      
    with open(target_path+'self_data_new_new2.txt', 'w') as f1:
        classification_data_sorted=[]
        classification_data=[]
        order=[]
        classification_reordered=[]
        for cntr in range(3):
            f=h5py.File(target_path+'diagnosis_digitized'+str(cntr)+'.hdf5', 'r')
            f_keys=f.keys()
            for key in f_keys:
                n1 = f.get(key)
                classification_data.append(np.array(n1))
                order.append(int(key))
        v=np.arange(42)
        v=v[order]
        classification_data_sorted = [classification_data[i] for i,j in enumerate(v)]

        for j in classification_data:
            for k in j:
                classification_reordered.append(k)
        for cntr in range(len(classification_reordered)):
            f1.write(f'Item num: {cntr}: \n')
            f1.write(f'{bool(classification_reordered[cntr])} \n')

# %% Main loop
# returned_dict=unpickle_CIFAR_dataset(r'data_batch_1')
# Pickle test
# pickle_ECG_data('Vadim')
# unpickle_ECG_data()
if __name__=='__main__':
    create=False
    DB_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\SW\Chinese Challenge\Data - Original'
    cwd = os.getcwd()
    # DB_path = cwd + r'\Data\Original\Chineese' + '\\'
    DB_path = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\SW\Chinese Challenge\Data - Original' + '\\'
    os.chdir(DB_path)
    target_path=os.path.join(cwd,'Data','')
    # store_uploaded_records(target_path, Created_records,dB_classes_list)
    if create:
        Created_records,dB_classes_list=Upload_db_records(DB_path, plot_flag=False)        
        store_uploaded_records_new(target_path, Created_records,dB_classes_list)
    else:
        test_created_db(target_path)
    print('Finished')



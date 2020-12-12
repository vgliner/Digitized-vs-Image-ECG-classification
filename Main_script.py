# %% Running my mutual net with Noam
"""
Net that was made as a final project at "Deep learning course" 
- 236605
"""
# %% Imports
from __future__ import print_function
import torch
import torch.optim as optim
import torchvision.transforms as tvtf
import transforms as tf
from transforms import *
import torchvision
import timeit
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from ECG_rendered_multilead_dataloader import *
from ECG_multi_lead_dataloader import *


import models

# %% Definitions
# root_dir = r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
# root_dir = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', 'Data'))+'//'
# root_dir = os.path.abspath(os.path.join(
#     os.path.dirname(__file__), '..', 'Data'))

# root_dir = os.path.join(root_dir, '')

# %% The execution loop
# %%  Settings


def RunNoamsNetDigitizedToClass():
    torch.multiprocessing.freeze_support()
    # Define the transforms that should be applied to each ECG record before returning it
    tf_ds = tvtf.Compose([
        tf.ECG_tuple_transform(-1)  # Reshape to 1D Tensor
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 1
    print('Using device:', device)
    #ds = ECG_Multilead_Dataset(root_dir=root_dir)
    ds = ECG_Multilead_Dataset(root_dir=root_dir, multiclass=False,
                               multiclass_to_binary=True, multiclass_to_binary_type=4)
# %%  Prepare the dataloaders
    # Define how much data to load
    # for real training:
    num_train = 35000
    # for small set overfit experiments:
    # num_train = 3500
    num_val = 1000
    num_test = 5000
    batch_size = 256  # 512
    # Training dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size)
    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size)
# %% Model creation
    # CNNs structure:
    # num of channels and kernel length in each layer of each branch, note that list lengths must correspond
    short_hidden_channels = [16, 32, 64, 128, 256, 512]
    long_hidden_channels = [4, 8, 16, 32, 64, 128, 256, 512]
    short_kernel_lengths = [5]*6
    long_kernel_lengths = [5]*8
    # which tricks to use: dropout, stride, batch normalization and dilation
    short_dropout = 0.5
    long_dropout = 0.5
    short_stride = 2
    long_stride = 2
    short_dilation = 1
    long_dilation = 1
    short_batch_norm = True
    long_batch_norm = True
    # enter input length here
    short_input_length = 1250
    long_input_length = 5000
    # FC net structure:
    # num of hidden units in every FC layer
    fc_hidden_dims = [128]
    # num of output classess
    num_of_classes = 2
    model = models.Ecg12LeadNet(short_hidden_channels, long_hidden_channels,
                                short_kernel_lengths, long_kernel_lengths,
                                fc_hidden_dims,
                                short_dropout, long_dropout,
                                short_stride, long_stride,
                                short_dilation, long_dilation,
                                short_batch_norm, long_batch_norm,
                                short_input_length, long_input_length,
                                num_of_classes).to(device)
    print(model)
# %%  Dimensions Check
    x, y = iter(dl_train).next()
    x1, x2 = x
    print('Long lead data of shape: ', x2.shape)
    print('Short lead data of shape: ', x1.shape)
    print('Labels of shape: ', y.shape)
    x_try = (x1.to(device, dtype=torch.float),
             x2.to(device, dtype=torch.float))
    y_pred = model(x_try)
    print('Output batch size is:',
          y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')
    num_correct = torch.sum((y_pred > 0).flatten() == (
        y.to(device, dtype=torch.long) == 1))
    print(100*num_correct.item()/len(y),
          '% Accuracy... maybe we should consider training the model')
# %% Training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadNetTrainerBinary
    # for reproducibility
    torch.manual_seed(42)
    lr = 0.001
    num_epochs = 50
    torch.cuda.empty_cache()   # entirely clear all allocated memory
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadNetTrainerBinary(model, loss_fn, optimizer, device)
    fitResult = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=r'checkpoints/Ecg12LeadNetDigitizedToClass',
                            early_stopping=10, print_every=1)
    lr = 0.0001
    num_epochs = 50
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadNetTrainerBinary(model, loss_fn, optimizer, device)
    fitResult2 = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=r'checkpoints/Ecg12LeadNetDigitizedToClass',
                             early_stopping=10, print_every=5)


def RunNetDigitizedToMultiClassBinary(class_type=0, kernel_size=17, train_set_size=35000,test_only=False,classification_threshold=None):
    torch.multiprocessing.freeze_support()
    # Define the transforms that should be applied to each ECG record before returning it
    tf_ds = tvtf.Compose([
        tf.ECG_tuple_transform(-1)  # Reshape to 1D Tensor
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    #ds = ECG_Multilead_Dataset(root_dir=root_dir)
    ds = ECG_Multilead_Dataset(multiclass=False,multiclass_to_binary=True, multiclass_to_binary_type=class_type)
    checkpoints_str = r'checkpoints/Ecg12LeadNetDigitizedToClass__' + \
        f'{class_type}'
# %%  Prepare the dataloaders
    # Define how much data to load
    # for real training:
    num_train = train_set_size
    # for small set overfit experiments:
    # num_train = 3500
    num_val = 1000
    num_test = 5830
    batch_size = 1024  # 512
    # Training dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=False)
    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size)
    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size)



# %% Model creation
    # CNNs structure:
    # num of channels and kernel length in each layer of each branch, note that list lengths must correspond
    short_hidden_channels = [16, 32, 64, 128, 256, 512]
    long_hidden_channels = [4, 8, 16, 32, 64, 128, 256, 512]
    short_kernel_lengths = [kernel_size]*6  # 5
    long_kernel_lengths = [kernel_size]*8  # 5
    # which tricks to use: dropout, stride, batch normalization and dilation
    short_dropout = 0.5
    long_dropout = 0.5
    short_stride = 2
    long_stride = 2
    short_dilation = 1
    long_dilation = 1
    short_batch_norm = True
    long_batch_norm = True
    # enter input length here
    short_input_length = 1250
    long_input_length = 5000
    # FC net structure:
    # num of hidden units in every FC layer
    fc_hidden_dims = [128]
    # num of output classess
    num_of_classes = 2
    model = models.Ecg12LeadNet(short_hidden_channels, long_hidden_channels,
                                short_kernel_lengths, long_kernel_lengths,
                                fc_hidden_dims,
                                short_dropout, long_dropout,
                                short_stride, long_stride,
                                short_dilation, long_dilation,
                                short_batch_norm, long_batch_norm,
                                short_input_length, long_input_length,
                                num_of_classes).to(device)
    print(model)
# %%  Dimensions Check
    # x, y = iter(dl_train).next()
    # x1, x2 = x
    # print('Long lead data of shape: ', x2.shape)
    # print('Short lead data of shape: ', x1.shape)
    # print('Labels of shape: ', y.shape)
    # x_try = (x1.to(device, dtype=torch.float),
    #          x2.to(device, dtype=torch.float))
    # y_pred = model(x_try)
    # print('Output batch size is:',
    #       y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')
    # num_correct = torch.sum((y_pred > 0).flatten() == (
    #     y.to(device, dtype=torch.long) == 1))
    # print(100*num_correct.item()/len(y),
    #       '% Accuracy... maybe we should consider training the model')
# %% Training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadNetTrainerBinary
    # for reproducibility
    torch.manual_seed(42)
    loss_fn = nn.BCEWithLogitsLoss()
    lrs = [0.01, 0.0001]  # , 0.01, 0.00001
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    torch.cuda.empty_cache()   # entirely clear all allocated memory
    trainer = Ecg12LeadNetTrainerBinary(model, loss_fn, optimizer, device,classification_threshold=classification_threshold)
    if not test_only:
        for lr in lrs:
            num_epochs = 100
            torch.cuda.empty_cache()   # entirely clear all allocated memory
            optimizer = optim.Adam(model.parameters(), lr=lr)  
            trainer = Ecg12LeadNetTrainerBinary(model, loss_fn, optimizer, device)            
            fitResult = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=checkpoints_str,  # dl_val
                                    early_stopping=20, print_every=1)
    else:
        fitResult = trainer.fit(dl_train, dl_val, 0, checkpoints=checkpoints_str,  # dl_val
                                early_stopping=20, print_every=100)
 


    #####################   ROC #################################################
    thresholds=np.arange(0,1,0.01,dtype=float)
    for th in thresholds:      
        trainer.classification_threshold=th
        test_result = trainer.test_epoch(dl_test, verbose=True)                    
        with open(f'ROC_Digital_{class_type}.txt', "a") as myfile:
            myfile.write(f'{th}\t{test_result.num_TP}\t{test_result.num_TN}\t{test_result.num_FP}\t{test_result.num_FN}\t{test_result.accuracy}\n')
    trainer.classification_threshold=None
    #####################   END OF ROC  #################################################

        # ###################  DATA FOR CONFUSION MATRIX   ###########################
        # ds_full = ECG_Multilead_Dataset(root_dir=root_dir, multiclass=True,
        #                            multiclass_to_binary=False, multiclass_to_binary_type=class_type)
        # ds_test_full = tf.SubsetDataset(ds_full, num_test, offset=num_train + num_val)
        # joined_list=[]
        # for row_in_list in test_result.out:
        #     for el in row_in_list:
        #         joined_list.append(el.cpu().item())
        # with open(f'LOG_Digital_{class_type}.txt', "a") as myfile:
        #     for i in range(num_test):
        #         myfile.write(f'{i}\t{int(ds_test[i][1])}\t{joined_list[i]}\t{ds_test_full[i][1][0]}\t{ds_test_full[i][1][1]}\t{ds_test_full[i][1][2]}\t{ds_test_full[i][1][3]}\t{ds_test_full[i][1][4]}\t{ds_test_full[i][1][5]}\t{ds_test_full[i][1][6]}\t{ds_test_full[i][1][7]}\t{ds_test_full[i][1][8]}\n')   
        # ####################END OF  DATA FOR CONFUSION MATRIX   #############################
        # 
        #                                           
        # zipped = zip(flat_list_out, flat_list_y)
        # it = iter(dl_test_full)       
        # for item in zipped:
        #     with open(f'Classification_Digital_Output{class_type}.txt', "a") as myfile:
        #         x_, y_= it.next()
        #         y__= (y_.data).cpu().numpy()     
        #         myfile.write(f'{item[0].item()}\t{item[1].item()}\t')
        #         for y___ in y__:
        #             myfile.write(f'{int(y___)}\t')
        #         myfile.write(f'\n')

    test_result = trainer.test_epoch(dl_test, verbose=True)
    with open(f'Test_Accuracy_Log.txt', "a") as myfile:
        myfile.write(f'Class type: {class_type},Accuracy:{ test_result[1]}\n')
    print('Test accuracy is: ', test_result[1], '%')
    return test_result                                


def RunNoamsECG_ImageClassification(perspective_transform=False, realtime_rendering=False, Is_classifier=False, Image_to_classify=None,classification_threshold=None,GPU_num=0):
    import torch
    import models
    import transforms as tf
    import matplotlib.pyplot as plt
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device=GPU_num)
    print('Using device:', device)    
    checkpoints_name = r'checkpoints/Ecg12LeadImageNetPerspective' + \
        str(perspective_transform)+r'Rendering' + str(realtime_rendering)

    apply_perspective_transformation = perspective_transform
    ds = ECG_Rendered_Multilead_Dataset(realtime_rendering=realtime_rendering,apply_perspective_transformation=apply_perspective_transformation)

    # ds = ECG_Multilead_Dataset(root_dir=root_dir,transform=None, partial_upload=False)
    # Define how much data to load (only use a subset for speed)

    # for real training:
    num_train = 35000
    num_val = 1000
    num_test = 5830
    # for small set overfit experiment:
    # num_train = 4
    # num_val = 4
    # num_test = 4
    if apply_perspective_transformation:
        batch_size = 30
    else:
        batch_size = 100

    # Training dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)  # (train=True, transform=tf_ds)
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    x, y = next(iter(dl_train))

    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size, num_workers=2, pin_memory=True)

    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size, num_workers=2, pin_memory=True)

# %% Let's see what we uploaded
    import matplotlib.pyplot as plt

    x, y = iter(dl_train).next()

    in_h = x.shape[1]
    in_w = x.shape[2]
    in_channels = x.shape[3]
    batch_memory = x.element_size() * x.nelement() // 1024**2

    print('Images of shape: ', x.shape)
    print('Labels of shape: ', y.shape)
    print('Size of a batch in the memory is: ~', batch_memory, 'MB')

    print('\nLet us see the first sample:\n')

    # plt.figure(figsize = (20,15))
    # plt.imshow(x[0,:,:,:])

    x = x.transpose(1, 2).transpose(1, 3)
    # plt.show()

# %% Architecture definition
    # num of channels and kernel length in each layer, note that list lengths must correspond
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7

    # which tricks to use: dropout, stride, batch normalization and dilation
    dropout = 0.2
    stride = 2
    dilation = 1
    batch_norm = True

    # FC net structure:

    # num of hidden units in every FC layer
    fc_hidden_dims = [128]

    # num of output classess
    num_of_classes = 2

    model = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2).to(device)

    # print(model)

# %% Test the dimentionality
    x_try = x.to(device, dtype=torch.float)
    y_pred = model(x_try)
    # print('Output batch size is:',
    #       y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')

    num_correct = torch.sum((y_pred > 0).flatten() == (
        y.to(device, dtype=torch.long) == 1))
    # print(100*num_correct.item()/len(y),
    #       '% Accuracy... maybe we should consider training the model')

    del x, y, x_try, y_pred

# %% Let's start training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadImageNetTrainerBinary

    torch.manual_seed(42)

    # lr = 0.01
    # num_epochs = 1

    # loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # trainer = Ecg12LeadImageNetTrainerBinary(model, loss_fn, optimizer, device)

    # fitResult = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=r'checkpoints/Ecg12LeadImageNetDemonstration',
    #                         early_stopping=5, print_every=1)

    lr = 0.001
    checkpoint_filename = f'{checkpoints_name}.pt'
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    if os.path.isfile(path+'//'+checkpoint_filename):
        num_epochs = 0
    else:
        num_epochs = 10

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadImageNetTrainerBinary(model, loss_fn, optimizer, device,classification_threshold=classification_threshold)

    fitResult2 = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=checkpoints_name,   #dl_val
                             early_stopping=5, print_every=1)

# %% Test results
    # if Is_classifier:
    #     K = Image_to_classify.permute(0, 3, 1, 2).to(device)
    #     out = model(K)
    #     print(f'Out: {out}')
    # else:
    #####################   ROC #################################################
    thresholds=np.arange(0,1,0.01,dtype=float)
    for th in thresholds:      
        trainer.classification_threshold=th
        test_result = trainer.test_epoch(dl_test, verbose=True)                    
        with open(f'ROC_Image_Pers_{perspective_transform}.txt', "a") as myfile:
            myfile.write(f'{th}\t{test_result.num_TP}\t{test_result.num_TN}\t{test_result.num_FP}\t{test_result.num_FN}\t{test_result.accuracy}\n')
    trainer.classification_threshold=None
    #####################   END OF ROC  #################################################

    test_result = trainer.test_epoch(dl_test, verbose=True)
    print('Test accuracy is: ', test_result[1], '%')
    return test_result




def RunVadimsNetDigitizedToMultiClass():
    torch.multiprocessing.freeze_support()
    # Define the transforms that should be applied to each ECG record before returning it
    tf_ds = tvtf.Compose([
        tf.ECG_tuple_transform(-1)  # Reshape to 1D Tensor
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    ds = ECG_Multilead_Dataset(root_dir=root_dir, multiclass=True)
# %%  Prepare the dataloaders
    # Define how much data to load
    # for real training:
    num_train = 35000
    # for small set overfit experiments:
    # num_train = 3500
    num_val = 1000
    num_test = 5000
    batch_size = 256  # 512
    # Training dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size)
    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size)
# %% Model creation
    # CNNs structure:
    # num of channels and kernel length in each layer of each branch, note that list lengths must correspond
    short_hidden_channels = [16, 32, 64, 128, 256, 512]
    long_hidden_channels = [4, 8, 16, 32, 64, 128, 256, 512]
    short_kernel_lengths = [5]*6
    long_kernel_lengths = [5]*8
    # which tricks to use: dropout, stride, batch normalization and dilation
    short_dropout = 0.5
    long_dropout = 0.5
    short_stride = 2
    long_stride = 2
    short_dilation = 1
    long_dilation = 1
    short_batch_norm = True
    long_batch_norm = True
    # enter input length here
    short_input_length = 1250
    long_input_length = 5000
    # FC net structure:
    # num of hidden units in every FC layer
    fc_hidden_dims = [128]
    # num of output classes
    num_of_classes = 9
    model = models.Ecg12LeadMultiClassNet(short_hidden_channels, long_hidden_channels,
                                          short_kernel_lengths, long_kernel_lengths,
                                          fc_hidden_dims,
                                          short_dropout, long_dropout,
                                          short_stride, long_stride,
                                          short_dilation, long_dilation,
                                          short_batch_norm, long_batch_norm,
                                          short_input_length, long_input_length,
                                          num_of_classes).to(device)
    print(model)
# %%  Dimensions Check
    x, y = iter(dl_train).next()
    x1, x2 = x
    print('Long lead data of shape: ', x2.shape)
    print('Short lead data of shape: ', x1.shape)
    print('Labels of shape: ', y.shape)
    x_try = (x1.to(device, dtype=torch.float),
             x2.to(device, dtype=torch.float))
    y_pred = model(x_try)
    print('Output batch size is:',
          y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')
    num_correct = torch.mean(1-abs(torch.sub(y_pred, y.to(device))))
    print(100*num_correct.item()/len(y),
          '% Accuracy... maybe we should consider training the model')
# %% Training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadNetTrainerMulticlass
    # for reproducibility
    torch.manual_seed(42)
    lr = 0.001
    num_epochs = 50
    torch.cuda.empty_cache()   # entirely clear all allocated memory
    loss_fn = nn.MSELoss()
    # loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = Ecg12LeadNetTrainerMulticlass(model, loss_fn, optimizer, device)
    fitResult = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=r'checkpoints/Ecg12LeadNetDigitizedToMultiClass',
                            early_stopping=10, print_every=1)
    # lr = 0.0001
    # num_epochs = 50
    # loss_fn = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # trainer = Ecg12LeadNetTrainerMulticlass(model, loss_fn, optimizer, device)
    # fitResult2 = trainer.fit(dl_train, dl_val, num_epochs, checkpoints=r'checkpoints/Ecg12LeadNetDigitizedToMultiClass',
    #                         early_stopping=10, print_every=5)


def RunNetImageToMultiClassBinary(class_type=0, perspective_transform=False, run_tag=''):
    import torch
    import models
    import transforms as tf
    import matplotlib.pyplot as plt
    torch.multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device=0)
    print('Using device:', device)

    checkpoints_name = r'checkpoints/Ecg12LeadImageNet' + \
        str(run_tag)+'Perspective' + str(perspective_transform) + \
        r'ClassType'+str(class_type)

    apply_perspective_transformation = perspective_transform
    realtime_rendering = True
    ds = ECG_Rendered_Multilead_Dataset(root_dir=root_dir, realtime_rendering=realtime_rendering,
                                        apply_perspective_transformation=apply_perspective_transformation)

    # for real training:
    num_train = 35000
    num_val = 1000
    num_test = 5830
    # for small set overfit experiment:
    # num_train = 4
    # num_val = 4
    # num_test = 4
    batch_size = 100

    # Training dataset & loader
    ds_train = tf.SubsetDataset(ds, num_train)  # (train=True, transform=tf_ds)
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    x, y = next(iter(dl_train))

    # Validation dataset & loader
    ds_val = tf.SubsetDataset(ds, num_val, offset=num_train)
    dl_val = torch.utils.data.DataLoader(
        ds_val, batch_size, num_workers=2, pin_memory=True)

    # Test dataset & loader
    ds_test = tf.SubsetDataset(ds, num_test, offset=num_train + num_val)
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size, num_workers=2, pin_memory=True)

    in_h = x.shape[1]
    in_w = x.shape[2]
    in_channels = x.shape[3]
    batch_memory = x.element_size() * x.nelement() // 1024**2

    print('Images of shape: ', x.shape)
    print('Labels of shape: ', y.shape)
    print('Size of a batch in the memory is: ~', batch_memory, 'MB')

    print('\nLet us see the first sample:\n')

    # plt.figure(figsize = (20,15))
    # plt.imshow(x[0,:,:,:])

    x = x.transpose(1, 2).transpose(1, 3)
    # plt.show()

# %% Architecture definition
    # num of channels and kernel length in each layer, note that list lengths must correspond
    hidden_channels = [8, 16, 32, 64, 128, 256, 512]
    kernel_sizes = [5] * 7

    # which tricks to use: dropout, stride, batch normalization and dilation
    dropout = 0.2
    stride = 2
    dilation = 1
    batch_norm = True

    # FC net structure:

    # num of hidden units in every FC layer
    fc_hidden_dims = [128]

    # num of output classess
    num_of_classes = 2

    model = models.Ecg12ImageNet(in_channels, hidden_channels, kernel_sizes, in_h, in_w,
                                 fc_hidden_dims, dropout=dropout, stride=stride,
                                 dilation=dilation, batch_norm=batch_norm, num_of_classes=2).to(device)

    print(model)

# %% Test the dimentionality
    x_try = x.to(device, dtype=torch.float)
    y_pred = model(x_try)
    print('Output batch size is:',
          y_pred.shape[0], ', and number of class scores:', y_pred.shape[1], '\n')

    num_correct = torch.sum((y_pred > 0).flatten() == (
        y.to(device, dtype=torch.long) == 1))
    print(100*num_correct.item()/len(y),
          '% Accuracy... maybe we should consider training the model')

    del x, y, x_try, y_pred

# %% Let's start training
    import torch.nn as nn
    import torch.optim as optim
    from training import Ecg12LeadImageNetTrainerBinary

    torch.manual_seed(42)

    lr = 0.0001
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    lr = 0.001
    checkpoint_filename = f'{checkpoints_name}.pt'
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    if os.path.isfile(path+'//'+checkpoint_filename):
        num_epochs = 0
    else:
        num_epochs = 30

    loss_fn = nn.BCEWithLogitsLoss()
    for lr in lrs:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        trainer = Ecg12LeadImageNetTrainerBinary(
            model, loss_fn, optimizer, device)

        fitResult2 = trainer.fit(dl_train, dl_test, num_epochs, checkpoints=checkpoints_name,
                                 early_stopping=100, print_every=1)
        with open(f"Execution_dump_{checkpoints_name}.txt", "a") as myfile:
            myfile.write(
                "Fit result:\n  train accuracy:  train loss:    test accuracy: test loss:  \n")
            for i, j in enumerate(fitResult2.test_acc):
                myfile.write(
                    f'{fitResult2.train_acc[i]}  {fitResult2.train_loss[i]}  {fitResult2.test_acc[i]} {fitResult2.test_loss[i]}\n')

# %% Test results
    test_result = trainer.test_epoch(dl_test, verbose=True)
    print('Test accuracy is: ', test_result[1], '%')


# %% Execution of the main loop
if __name__ == "__main__":
    print('Start training')
    print('Train Digitized to class')
    for class_type in range(9):
        with open("Execution_dump.txt", "a") as myfile:
            myfile.write(f'Executing class number: {class_type}\n')
            RunNetDigitizedToMultiClassBinary(class_type=class_type)    
    print('Train Image to class without perspective transformation')
    perspective_transform=False
    realtime_rendering=False
    for class_type in range(9):
        test_results=RunNoamsECG_ImageClassification(perspective_transform=perspective_transform, realtime_rendering=realtime_rendering,classification_threshold=None,GPU_num=0)
    print('Train Image to class with perspective transformation')
    perspective_transform=True
    for class_type in range(9):
        test_results=RunNoamsECG_ImageClassification(perspective_transform=perspective_transform, realtime_rendering=realtime_rendering,classification_threshold=None,GPU_num=0)



###############################    ##########################################
    # kernels= [3 , 5 , 7, 9, 11 , 13, 15, 17 , 19, 21]
    # for class_type in range(9):
    #     for kernel_size in kernels:
    #         with open("Execution_dump.txt", "a") as myfile:
    #             myfile.write(f'Executing class number: {class_type}, {kernel_size} \n')
    #             RunNetDigitizedToMultiClassBinary(class_type=class_type,kernel_size=kernel_size)

###############################    ##########################################
    # for class_type in range(9):
    # with open("Execution_dump_new.txt", "a") as myfile:
    #     class_type = 8
    #     myfile.write(f'Executing class type:  {class_type} \n')
    #     RunNetDigitizedToMultiClassBinary(class_type=class_type,kernel_size=17,train_set_size=int(35000))

#################################   ##########################################    
    # perspective_transform=False
    # classification_thresholds=np.linspace(0,1,100)
    # for c_th in classification_thresholds:
    #     test_results=RunNoamsECG_ImageClassification(perspective_transform=True, realtime_rendering=False,classification_threshold=c_th)
    #     with open("ROC.txt", "a") as myfile:
    #         myfile.write(f'{c_th}\t{test_results.num_TP}\t{test_results.num_TN}\t{test_results.num_FP}\t{test_results.num_FN}\n')
###############################    ##########################################
    # perspective_transform=False
    # test_results=RunNoamsECG_ImageClassification(perspective_transform=perspective_transform, realtime_rendering=False,classification_threshold=None)
###############################    ##########################################


############################### DEC 23rd   ##########################################
    # classes_list=[0,1,2,3,4,5,6,7,8]
    # classes_list=[1]
    # for class_type in classes_list:
    #     test_results=RunNetDigitizedToMultiClassBinary(class_type=class_type,kernel_size=17,train_set_size=int(35000),test_only=False,classification_threshold=None)
###############################    ##########################################


############################### DEC 23rd    ##########################################
    # perspective_transform=True
    # test_results=RunNoamsECG_ImageClassification(perspective_transform=perspective_transform, realtime_rendering=False,classification_threshold=None,GPU_num=1)


###############################    ##########################################


# %%  Example how to train and just estimate - > apply_perspective_transformation=False,realtime_rendering=False
    #  Image (675, 1450, 3)
    # target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Work\Twelve_lead_ECG_Classification_app_gen2\Data'+'\\'
    # ECG_test = ECG_Rendered_Multilead_Dataset(root_dir=target_path, transform=None, partial_upload=False,apply_perspective_transformation=False,realtime_rendering=False)  # For KNN demo
    # Test_image=ECG_test[38000]
# %% Try on real image
    # from PIL import Image
    # A=Image.open('C:\ST.jpg')
    # img = A.resize((1450,675), Image.ANTIALIAS)
    # Image_to_test=torch.from_numpy(Test_image[0]).float().unsqueeze(0)
    # Image_to_test=torch.from_numpy(np.array(img)).float().unsqueeze(0)
    # ,Is_classifier=True,Image_to_classify=Image_to_test



    # print(f'Real out: {Test_image[1]}')
    print('Finished execution')


# %%

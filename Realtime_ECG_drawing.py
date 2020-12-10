# import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ECG_multi_lead_dataloader import *
from scipy import signal
from scipy.interpolate import interp1d
import time
import random


def draw_calibration_signal_on_canvas(canvas,pivots, line_width,pixels_to_mv,pixels_to_sec ):
    for pivot in pivots:
        canvas[pivot-10*pixels_to_mv:pivot,0:line_width]=0  # 10 mV height
        canvas[pivot-10*pixels_to_mv-line_width:pivot-10*pixels_to_mv,0:int(pixels_to_sec*0.2)]=0 # Roof line
        canvas[pivot-10*pixels_to_mv-line_width:pivot,int(pixels_to_sec*0.2):int(pixels_to_sec*0.2)+line_width]=0  # 10 mV height
    return canvas

def draw_long_lead(canvas, long_lead, pivot, line_width,pixels_to_mv,pixels_to_sec):
    scaling_factor=10
    downsample_factor=int(pixels_to_sec*10)
    f = signal.resample(long_lead[0], downsample_factor)    
    K=(f*pixels_to_mv*scaling_factor+pivot[-1]-10*pixels_to_mv+pixels_to_mv).astype(int)
    x=np.linspace(int(pixels_to_sec*0.2)+line_width,int(pixels_to_sec*0.2)+line_width+len(K)-1, len(K))
    x=x.astype(int)
    K=K+(pivot[-1]-K[0])
    for pixel in range(line_width):
        canvas[K+pixel,x]=0

    for indx, pixel in enumerate(K):
        if indx==len(K)-1:
            break
        for pxl in range(line_width):
            canvas[K[indx]+pxl:K[indx+1]+pxl,indx+int(pixels_to_sec*0.2)+line_width]=0
            canvas[K[indx+1]+pxl:K[indx]+pxl,indx+int(pixels_to_sec*0.2)+line_width]=0

    return canvas


def draw_short_leads(canvas, short_leads, pivot_arr, line_width,pixels_to_mv,pixels_to_sec,is_scaling_applied=False):
    scaling_factor=10
    downsample_factor=int(pixels_to_sec*2.5)
    f=[]
    # downsampling:
    # TODO: Temporary scaling- to debug and remove
    if is_scaling_applied:
        for lead_num,lead in enumerate(short_leads):
            if (max(lead)-min(lead))<0.05:
                continue
            offs=lead[0]
            short_leads[lead_num]/=(max(lead)-min(lead))
            short_leads[lead_num]-=short_leads[lead_num][0]-offs
    for lead_num in range(12):
        f.append((signal.resample(short_leads[lead_num]*pixels_to_mv*scaling_factor, downsample_factor)+pivot_arr[lead_num//4]-pixels_to_mv*5).astype(int))        
        if (lead_num%4==0):
            offset=(f[lead_num][0]-pivot_arr[lead_num//4 ])        
        f[lead_num]=f[lead_num]+offset
    # Marking the line and taking care of thickness
    for f_indx,fs in enumerate(f): # running on each lead data
        K=fs
        x=np.linspace(int(pixels_to_sec*0.2)+line_width+int((f_indx%4)*downsample_factor) ,int(pixels_to_sec*0.2)+line_width+len(K)-1+int((f_indx%4)*downsample_factor) , len(K))
        x=x.astype(int)
        for pixel in range(line_width): # Making the line thicker
            canvas[K+pixel,x]=0        
        for indx, pixel in enumerate(K):
            if indx==len(K)-1:
                break
            for pxl in range(line_width):
                canvas[K[indx]+pxl:K[indx+1]+pxl,int((f_indx%4)*downsample_factor) + indx+int(pixels_to_sec*0.2)+line_width]=0
                canvas[K[indx+1]+pxl:K[indx]+pxl,int((f_indx%4)*downsample_factor) + indx+int(pixels_to_sec*0.2)+line_width]=0

    return canvas    

def draw_ECG_multilead_vanilla_ver2(ECG_Data, canvas, to_plot=False):
    Leads = ['Lead1', 'Lead2', 'Lead3', 'aVR', 'aVL',
             'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    new_canvas=canvas[:,300:].copy()
    canvas_dims=np.shape(new_canvas)
    is_scaling_applied=True



    pixels_to_mv = 60 // 10  # 60 pixels /10mV
    pixels_to_sec = 90 // (0.2 * 3)  #
    image_pixels_ratio = [canvas_dims[1], canvas_dims[0]]  # (X,Y)
    line_width=3

    ## Draw calibration signal
    calibration_signal_pivots=[180,417,655,893]
    new_canvas=draw_calibration_signal_on_canvas(new_canvas,calibration_signal_pivots, line_width,pixels_to_mv,pixels_to_sec )
    # draw_canvas(new_canvas)
    new_canvas=draw_long_lead(new_canvas, ECG_Data[1], calibration_signal_pivots, line_width,pixels_to_mv,pixels_to_sec)
    # draw_canvas(new_canvas)
    new_canvas=draw_short_leads(new_canvas, ECG_Data[0], calibration_signal_pivots, line_width,pixels_to_mv,pixels_to_sec,is_scaling_applied)
    # draw_canvas(new_canvas)

    if to_plot:
        draw_canvas(new_canvas)

    return new_canvas

# def draw_canvas(canvas):
#     def click_event(event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             print(f'{x}, {y}')
    
#     cv2.imshow('image',canvas)
#     cv2.setMouseCallback("image", click_event)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__=="__main__":
    # img = cv2.imread('ECG_paper.jpg',cv2.IMREAD_COLOR )
    # img_shape=np.shape(img)
    # print(img_shape)
    # res = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
    # background= img#img[int(img_shape[0]*0.02):int(img_shape[0]*0.90), int(img_shape[1]*0.1):int(img_shape[1]*0.9)]
    # # cv2.imshow('image',background)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # cv2.imshow('image',res)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # img_shape=np.shape(res)
    # print(img_shape)
    # # res[:,100:105]=0
    # # cv2.imshow('image',res)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()


    # res_matplotlib=res/255
    # plt.imshow(res_matplotlib[:,:,[2,1,0]])
    # plt.show()

    
    target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'
    # canvas = cv2.imread(target_path+'ECG_paper.jpg',cv2.IMREAD_COLOR )
    # img_shape=np.shape(canvas)
    # print(f' Image size before interpolation: {img_shape[0]}  x   {img_shape[1]}')
    # canvas = cv2.resize(canvas,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
    # img_shape=np.shape(canvas)
    # print(f' Image size after interpolation: {img_shape[0]}  x   {img_shape[1]}')    
    # ECG_test = ECG_Multilead_Dataset(root_dir=target_path,transform=None, partial_upload=False)
    # start = time.time()
    # for cntr in range(100):
    #     rand_indx=random.randint(10,41830)
    #     rand_indx=cntr
    #     ECG_Data=ECG_test[rand_indx]
    #     # Merged_ECG=draw_ECG_on_paper_canvas(ECG_Data[0], res)
    #     OUT=draw_ECG_multilead_vanilla_ver2(ECG_Data[0], canvas,to_plot=False)
    #     if (cntr% 100)==0:
    #         print(f'cntr: {cntr}')
    # end = time.time()
    # print(end - start)
    # print('Finished')

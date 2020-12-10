import random
from PIL import Image
import torchvision.transforms.functional as F
import torchvision
from ECG_rendered_to_matrix_DB_dataloader import *
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from ECG_multi_lead_dataloader import *
# import cv2
import numpy as np





class RandomPerspective(object):
    """Performs Perspective transformation of the given PIL Image randomly with a given probability.
    Args:
        interpolation : Default- Image.BICUBIC
        p (float): probability of the image being perspectively transformed. Default value is 0.5
        distortion_scale(float): it controls the degree of distortion and ranges from 0 to 1. Default value is 0.5.
    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=Image.BICUBIC):
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be Perspectively transformed.
        Returns:
            PIL Image: Random perspectivley transformed image.
        """
        if not F._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        if random.random() < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return F.perspective(img, startpoints, endpoints, self.interpolation)
        return img

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.
        Args:
            width : width of the image.
            height : height of the image.
        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

def Perspective_transformation_application(image,database_path='',realtime_rendering=False):
    k=str(random.randint(1,10))
    with h5py.File(database_path+'backgrounds_db.hdf5', 'r') as f:
        bgrnd= np.array(f[k])    
    merged_im=bgrnd
    K=image
    starting_point=((bgrnd.shape[0]-K.shape[0])//2,(bgrnd.shape[1]-K.shape[1])//2)
    merged_im[starting_point[0]:starting_point[0]+K.shape[0],starting_point[1]:starting_point[1]+K.shape[1]]=K[:,:,[2,1,0]]
    T=torchvision.transforms.RandomPerspective(distortion_scale=0.15, p=0.9, interpolation=3)
    Output=np.array(T(Image.fromarray(merged_im)))
    if realtime_rendering==False:
        sub_image=Output[250:-250,400:-400,:]
    else:
        sub_image=Output[141:-141,222:-222,:]
    return sub_image[:,:,[2,1,0]]




PHASE =1

if __name__=="__main__":
    target_path=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Data_new_format'+'\\'

    print('Evaluating')
    ECG_test = ECG_Rendered_to_matrix_Dataset(root_dir=target_path, transform=None, partial_upload=False)  
    K=ECG_test[2]
    print(f'ECG shape is : {np.shape(K[0])} ')
    load_file='Backgrounds\\'
    ECG_shape=np.shape(K[0])
    sizes_log=[]
    cv2.imshow("ECG",K[0][:,:,[2,1,0]])
    cv2.waitKey(0)
    # #####################   Creating background database ####################
    # if PHASE==1:
    #     print('Executing phase 1')
    #     for image_cntr in range(1,11):
    #         load_file='Backgrounds\\'
    #         load_file=load_file+str(image_cntr)+'.jpg'
    #         image = cv2.imread(load_file)
    #         # cv2.imshow("original", image)
    #         # cv2.waitKey(0)
    #         # face = misc.imread(load_file)
    #         # plt.imshow(face)
    #         # plt.show()            
    #         background_shape=np.shape(image)
    #         ratio=[aItem/bItem for aItem, bItem in zip(ECG_shape, background_shape)]
    #         minimal_ratio=max(ratio[0:2])
    #         zoom_ratio=np.ceil(2/(1/minimal_ratio))
    #         print(f'Zoom ratio is going to be : {zoom_ratio}')
    #         dim = (int(image.shape[1] * zoom_ratio),int(image.shape[0] * zoom_ratio))
             
    #         # perform the actual resizing of the image and show it
    #         resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)

    #         sizes_log.append(np.shape(resized))


    #         # backgroundImage=Image.open(load_file)            
    #     minimas=np.min(sizes_log,axis=0)
    #     print('Finished phase 1')
    #     with h5py.File("backgrounds_db.hdf5", "w") as f: 
    #         for image_cntr in range(1,11):
    #             load_file='Backgrounds\\'
    #             load_file=load_file+str(image_cntr)+'.jpg'
    #             image = cv2.imread(load_file)        
    #             background_shape=np.shape(image)
    #             ratio=[aItem/bItem for aItem, bItem in zip(ECG_shape, background_shape)]
    #             minimal_ratio=max(ratio[0:2])
    #             zoom_ratio=np.ceil(2/(1/minimal_ratio))
    #             print(f'Zoom ratio is going to be : {zoom_ratio}')
    #             dim = (int(image.shape[1] * zoom_ratio),int(image.shape[0] * zoom_ratio))
                
    #             # perform the actual resizing of the image and show it
    #             resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)
    #             resized= resized[:minimas[0],:minimas[1]]
    #             dset = f.create_dataset(str(image_cntr), data=resized)
    # #####################  END OF Creating background database ####################

        # Convert PIL to numpy  -> pix = numpy.array(pic)
        # Convert numpy to PIL -> im = Image.fromarray(np.uint8(ECG_image*255))

#TODO: Save backgrounds as numpy array

        # ECG_image_PLL=Image.fromarray(ECG_image)
        # ECG_image_PLL.show()
        # load_file=load_file+str(random.randint(1,10))+'.jpg'
        # backgroundImage=Image.open(load_file)
        # P=RandomPerspective(distortion_scale=0.4, p=0.5, interpolation=3)
        # new_img = Image.blend(backgroundImage, Im, 0.5)
        # Im2=P(Im)
        # Im2.show()
    
    with h5py.File(target_path+'backgrounds_db.hdf5', 'r') as f:
        for cntr in range(50):
            k=str(random.randint(1,10))
            bgrnd= np.array(f[k])
            # cv2.imshow("background",bgrnd)
            # cv2.waitKey(0)
            merged_im=bgrnd
            starting_point=((bgrnd.shape[0]-K[0].shape[0])//2,(bgrnd.shape[1]-K[0].shape[1])//2)
            merged_im[starting_point[0]:starting_point[0]+K[0].shape[0],starting_point[1]:starting_point[1]+K[0].shape[1]]=K[0][:,:,[2,1,0]]
            # cv2.imshow("Merged",merged_im)        
            # cv2.waitKey(0)
            # for i in range(50):
            T=torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=0.9, interpolation=3)
            Output=np.array(T(Image.fromarray(merged_im)))
            sub_image=Output[450:-450,700:-700,:]
            print(f'ECG shape is {np.shape(K[0])}, sub_image shape is: {np.shape(sub_image)}')

            # cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
            # cv2.imshow("Output",sub_image)        
            # cv2.waitKey(0)
            plt.imshow(sub_image[:,:,[2,1,0]])
            plt.show()

    print('Finished')


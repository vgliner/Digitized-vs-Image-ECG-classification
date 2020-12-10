import pickle


# Flags
PRINT_FLAG = False



def pickle_ECG_data(ECG_data, file=r'C:\Users\vgliner\OneDrive - JNJ\Desktop\ECG_data.pkl'):
    with open(file, 'wb') as fo:
        pickle.dump(ECG_data, fo, -1)  # Pickling with the highest protocol available


def unpickle_ECG_data(file='ECG_data.pkl'):
    with open(file, 'rb') as fo:
        pickled_data = pickle.load(fo, encoding='bytes')
        if PRINT_FLAG:
            print(f'Loaded data with type of: {type(pickled_data)}')
        return pickled_data    


def unpickle_CIFAR_dataset(file):
    """ Upolading CIFAR hust to see 2the convention
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
    
    """    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
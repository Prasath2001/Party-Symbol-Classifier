import pathlib
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Running this .py file will store all the augmented images to respective class of images in the same directory.


data_dir = pathlib.Path('./Original Dataset') # Directory of the original dataset.

ext = 'jpg' # specify the extension for the augmented images
prefix = 'aug' #set the prefix for the augmented images
batch_size = 32 # set the batch size
passes = 10  # set the number of time to cycle the generator
datagen = ImageDataGenerator( rotation_range = 60,
                                horizontal_flip = True, fill_mode = 'nearest', vertical_flip=True)
                                
data = datagen.flow_from_directory(
    directory = data_dir, batch_size = batch_size,  target_size = (180, 180),
    shuffle=True
)

for i in range (passes):
    
    images, labels = next(data)
    class_dict = data.class_indices
    new_dict = {}
    # make a new dictionary with keys and values reversed
    for key, value in class_dict.items(): # dictionary is now {numeric class label: string of class_name}
        new_dict[value] = key    

    for j in range (len(labels)):                
        class_name = new_dict[np.argmax(labels[j])]         
        dir_path = os.path.join(data_dir,class_name)         
        new_file = prefix + '-' + str(i*batch_size +j) + '.'  + ext       
        img_path = os.path.join(dir_path, new_file)
        img = cv2.cvtColor(images[j], cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, img)
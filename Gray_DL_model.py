"""
    writing script to convert the model to version
    that accepts the grayscale image only
    This reduces the parameters count of the model also.

    Compressai environment is required for this as the original model
    will be imported from the compressai library

    How to use this is that the run this script
    -script downloads the org BMSHJ model
    -then script converts the model to the Grayscale version
    -draws the inference from the grayscale version of the model
"""
import time
import torch 
import numpy as np
import torch.nn as nn
from torchvision import transforms
from compressai.zoo import models

import os 
from PIL import Image

device="cuda" if torch.cuda.is_available() else "cpu"
#device="cpu"
print("Device is:",device)


#load the model 
model_name="bmshj2018-factorized"
quality=4
metric="ms-ssim"

#net=models[model_name](quality=quality,metric=metric,pretrained=True).eval().to(device)
#print("Model is Loaded")

#print("Model architecture is like")
#print(net)



#modify the first layer of the encoder (g_a) to accept only single channel
def modify_model_gray(model):
    #extract the first conv layer 
    first_conv=model.g_a[0]
    
    #create a new conv2d layer with single input channel
    new_conv=nn.Conv2d(1,first_conv.out_channels,kernel_size=first_conv.kernel_size,stride=first_conv.stride,padding=first_conv.padding)
    
    #Initialize new weights for the grayscale input
    with torch.no_grad():
        #Average the original weights across the input channels to match the 1 channel input
        new_conv.weight[:]=first_conv.weight.mean(dim=1,keepdim=True)
        new_conv.bias[:]=first_conv.bias
        
    #replace the old conv layer with the new one
    model.g_a[0]=new_conv
    
    #return the new model
    return model            

def image_process(image_path):
    image=Image.open(image_path)
    #resize image to HD resolution  
    #image=image.resize((1280,1280))
    image=image.convert("L") #convert to grayscale
    return np.array(image)

def gray_3_channel(image_path):
    
    image=Image.open(image_path).convert("L")
    #resize image to HD resolution
    #image=image.resize((1280,1280))
    image_np=np.array(image).transpose() #converted the image to numpy 
    #print("Image shape in numpy:",image_np.shape)
    
    #make a blank numpy array of image shape
    image_gray=np.zeros((3,image_np.shape[0],image_np.shape[1]))
    #print("Image shape in numpy:",image_gray.shape)
    
    #set the first channel of image_gray as the image_np
    image_gray[0]=image_np
    #normalize the image
    image_gray=image_gray/255.0
    #convert to np. float data type
    image_gray=image_gray.astype(np.float32)
    image_gray=image_gray
    #write the transpose command to convert (3,3280,2464) to (3280,2464,3)
    image_gray=image_gray.transpose(1,2,0)
    print("image final shape is",image_gray.shape)
    #now return this as the image
    return image_gray
    
    

def inference(model,image):
    """ Expects the image as the numpy array

    Args:
        model (_type_): _description_
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    #resize and normalize the image
    image=transforms.ToTensor()(image).unsqueeze(0).to(device)
    print("Image shape is:",image.shape)
    
    #run inference on the image
    model=model.to(device)  #move the model to the device
    
    #run the inference
    start=time.time()
    with torch.no_grad():
        out_net=model.forward(image)
    out_net['x_hat'].clamp_(0,1)
    end=time.time()
    print("Compression time:",round(end-start,2))  
    #convert this out_net to the PIL image and then display the image
    output=out_net['x_hat'].squeeze(0).cpu().detach().numpy()
    output=output.transpose(1,2,0)*255
    output=output.astype(np.uint8)
    #print("shape is:",output.shape)
    #print(output)
    image=Image.fromarray(output)
    image.save("output.png")
    #image.show()
    return out_net

def org_model():
    model_name="bmshj2018-factorized"
    quality=4
    metric="ms-ssim"
    model=models[model_name](quality=quality,metric=metric,pretrained=True).eval().to(device)
    return model

def main(image_path):
    
    model_org=org_model()
    image=gray_3_channel(image_path)
    print("Original model compression time is:")
    inference(model_org,image)
    
    print()
    print()
    model_modified=modify_model_gray(model_org)
    image=image_process(image_path)
    print("Modified model compression time is:")
    inference(model_modified,image)

if __name__=="__main__":
    image_path=r"C:\Swapnil\Narrowband_DRONE\Image_compression_code\Final_DL_compession_September_24\DL_MODEL_FINETUNE\image.png"
    main(image_path)
    #gray_3_channel(image_path)
    

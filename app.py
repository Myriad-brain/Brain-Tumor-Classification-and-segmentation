import tensorflow
from tensorflow import keras
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

im_height=256
im_width=256
smooth=100

def dice_coef(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)
    And=K.sum(y_truef* y_predf)
    return((2* And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']
@st.cache(allow_output_mutation=True)
def load_model_class():
  model=tf.keras.models.load_model('./effnet.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model_class()

model_seg = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})
st.write("""
         # Brain Tumor Classification
         """
         )

file = st.file_uploader("Upload the image to be classified \U0001F447", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def upload_predict(image, model):
    
        size = (150,150)    
        # image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(150, 150),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
    
        # prediction = model.predict(img_reshape)
        # pred_class=decode_predictions(prediction,top=1)
        pred = model.predict(img_reshape)
        pred = np.argmax(pred)
        
        return pred
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    image2 = Image.open(file)

    image2 = np.asarray(image2)
    img = cv2.resize(image2 ,(im_height, im_width))
    img = img / 255
    img = img[np.newaxis, :, :, :]
    pred=model_seg.predict(img)
    color = np.array([255,255,0], dtype='uint8')
    pred=np.squeeze(pred)>.5
    masked_img = np.where(pred[...,None], color, img)
    masked_img = np.squeeze(masked_img)

    # image2 = cv2.resize(masked_img,(150,150))

#   st.image(image, use_column_width=True)
    image = np.asarray(image)
    image = cv2.resize(image ,dsize=(256,256))
    print(image.shape)
    print(masked_img.shape)
    st.image([image,masked_img],clamp=True) 

    predictions = labels[upload_predict(image,model)]
    st.write("The image is classified as :    ",predictions)
    print("The image is classified as :    ",predictions)
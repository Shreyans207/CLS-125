# Controller
# Model is the database, View is for fetching from the user and returning back the values {here i.e. API}, Controller does the computing
import pandas
import numpy as np  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Python Image Library (not  Public Interest Litigation)
from PIL import Image
import PIL.ImageOps
import os 

# Secure Soket Layer {used in websites to secure it from hackers}
import ssl

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context' , None)) : 
    # context is a setting used for creating a secured connection with the website.
    ssl._create_default_https_context = ssl._create_unverified_context

X,y = fetch_openml('mnist_784' , version = 1 , return_X_y = True)

X_train,X_test,y_train,y_test = train_test_split(X, y , random_state = 9 , train_size = 7500 , test_size = 2500)

X_train = X_train/255
X_test = X_test/255

clf = LogisticRegression(solver = 'saga' , multi_class = 'multinomial').fit(X_train , y_train)

def getPrediction(Image) : 
    im_pil = Image.open(Image)

    # To convert it into a gray scale ; L for gray scale
    image_bw = im_pil.convert('L') 

    # Resizing and making the image smoother with Image.ANTIALIAS
    image_bw_resized = image_bw.resize((28,28) , Image.ANTIALIAS)

    # To specify the lower range 
    pixel_filter = 20 

    # Removing the lighter pixel {which are below 20 percentile from the max_pixel} from the image captured roi
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter) 
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255) 

    # Determining the brightess pixel
    max_pixel = np.max(image_bw_resized_inverted) 

    # Scaling ; Dividing the remaining the individual values by the max to determine 0 and 1 for images

    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel 
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)

    # Data obtained from the area of interest and reshaped it into 784 grids of pixels with 0 and 1 value 
    # We converted it in the format as of that we trained the classifier by reshaping it.

    prediction = clf.predict(test_sample)
    return(prediction[0])
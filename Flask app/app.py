from flask import Flask, render_template, request
#from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import math
import torch
import torchvision.transforms as transforms
#import cv2
from ResNet9 import ResNet9
from ResNet18 import ResNet18

app = Flask(__name__)
#model_gender = load_model('model_gender.h5')
#model_age = load_model('model_age.h5')
model_ge = ResNet9(1,10)
model_ge.load_state_dict(torch.load('models/weights.pth', map_location=torch.device('cpu')))
model_age = ResNet18(1)
model_age.load_state_dict(torch.load('models/weights21.pth', map_location=torch.device('cpu')))
#model_ethnicity = load_model('model_ethnicity.h5')

def predict_GE(image_path):
    
    image = Image.open(image_path).convert('L')  # Open the image and convert it to grayscale

    resized_image = image.resize((48, 48)) 
    normalized_image = np.array(resized_image) / 255.0
    img = normalized_image.astype(np.float32)
    transform = transforms.ToTensor()
    img = transform(img)
    xb = img.unsqueeze(0)
    yb = model_ge(xb)
    _, preds = torch.max(yb, dim = 1)
    return [preds[0].item()]

def predict_age2(image_path):
    image = Image.open(image_path).convert('L')  # Open the image and convert it to grayscale

    resized_image = image.resize((48, 48)) 
    normalized_image = np.array(resized_image) / 255.0
    img = normalized_image.astype(np.float32)
    transform = transforms.ToTensor()
    img = transform(img)
    xb = img.unsqueeze(0)
    yb = model_age(xb)
    return(yb)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['file']

        if file:
            # Save the uploaded file to a temporary location
            file_path = 'temp/' + file.filename
            file.save(file_path)

            # Get the predicted gender
           # gender = predict_gender(file_path)
            #age = predict_age(file_path)
            #race = predict_race(file_path)
            GE = predict_GE(file_path)
            GE = GE[0]
            age = predict_age2(file_path)
            age = age.detach()
            age = round(age.item())
            if GE < 5:
                gender = 'Male'
                e = GE
            else:
                gender = 'Female'
                e = GE - 5
            if e == 0:
                race = 'White'
            elif e == 1:
                race = 'Black'
            elif e == 2:
                race = 'Asian'
            elif e == 3:
                race = 'Indian'
            else:
                race = 'Other'
            # Render the result template with the predicted gender
            return render_template('result.html', gender = gender, race = race, age=age)

    # Render the upload template for GET requests
    return render_template('upload.html')

if __name__ == '__main__':
    app.debug = True
    app.run()

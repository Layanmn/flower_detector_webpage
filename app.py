from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired

app = Flask(__name__)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'  


model = tf.keras.models.load_model(r'my_model.h5')


Flowers_name = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Create the upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the UploadFileForm
class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/')
def home():
    return render_template('index.html', form=UploadFileForm())

@app.route('/predict', methods=['POST'])
def predict():
    form = UploadFileForm()
    filename = None  
    ensemble_class = None  

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  

      
        image = load_img(file_path, target_size=(224, 224))  # Resize image to 224x224
        image_array = img_to_array(image)
        rescaled_image = image_array / 255.0
        rescaled_image = rescaled_image.reshape((1,) + rescaled_image.shape)

        # Make predictions
        resnet_predictions = model.predict(rescaled_image)

        if np.max(resnet_predictions) >= 0.7:
            predicted_class_index = np.argmax(resnet_predictions)
            ensemble_class = Flowers_name[predicted_class_index]
        else:
            ensemble_class = "Unknown"

       

    result = ensemble_class if ensemble_class else "No prediction"
    return render_template('index.html', form=form, prediction=result, image_url=filename)

if __name__ == '__main__':
    app.run(debug=True)

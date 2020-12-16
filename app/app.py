import os
import time
import hashlib

from flask import Flask, render_template, redirect, url_for, request, abort
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from wtforms import SubmitField

from evaluate.evaluate_model import ModelEvaluator

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

# evaluator = ModelEvaluator('model.h5', (1, 40, 32, 3), 'rgb')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if request.method == 'POST':
        request.files['file'].save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], request.files['file'].filename))
        return redirect(url_for('upload_file'))

    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    print(files_list)
    file_urls = [photos.url(filename) for filename in files_list]
    # evaluator.predict(f'uploads/{name}.jpg')
    predictions = ['Vegetation' for filename in files_list]

    return render_template('upload.html',
                           form=form,
                           files=zip(files_list, file_urls, predictions))


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('upload_file'))


if __name__ == '__main__':
    app.run(debug=True)

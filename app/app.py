import os
import time
import hashlib

from flask import Flask, render_template, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

from evaluate.evaluate_model import ModelEvaluator

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')
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
    if form.validate_on_submit():
        for filename in request.files.getlist('photo'):
            name = hashlib.md5(str('admin' + str(time.time())).encode('utf-8')).hexdigest()[:15]
            photos.save(filename, name=name + '.')
            prediction = 'Anomaly'
            # prediction = evaluator.predict(f'uploads/{name}.jpg')
            # print(prediction)
        success = True
    else:
        prediction = 'Unknown'
        success = False
    return render_template('upload.html', form=form, success=success, prediction=prediction)


@app.route('/manage')
def manage_file():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    return render_template('manage.html', files_list=files_list)


@app.route('/open/<filename>')
def open_file(filename):
    file_url = photos.url(filename)
    return render_template('browser.html', file_url=file_url)


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file'))


if __name__ == '__main__':
    app.run(debug=True)

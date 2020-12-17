import os

from flask import Flask, render_template, redirect, url_for, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from wtforms import SubmitField

# from evaluate.evaluate_model import ModelEvaluator

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']

# evaluator = ModelEvaluator('../app/model.h5', (1, 40, 32, 3), 'rgb')

CLASSES = [('No-Anomaly',
            ' Nominal solar module',
            'static/no-anomaly.png'),
           ('Cell',
            'Hot spot occurring with square geometry in single cell',
            'static/cell.png'),
           ('Cell-Multi',
            'Hot spots occurring with square geometry in multiple cells',
            'static/cell-multi.png'),
           ('Cracking',
            'Module anomaly caused by cracking on module surface',
            'static/cracking.png'),
           ('Hot-Spot',
            'Hot spot on a thin film module',
            'static/hot-spot.png'),
           ('Hot-Spot-Multi',
            'Multiple hot spots on a thin film module',
            'static/hot-spot-multi.png'),
           ('Shadowing',
            'Sunlight obstructed by vegetation, man-made structures, or adjacent rows',
            'static/shadowing.png'),
           ('Diode',
            'Activated bypass diode,'
            ' typically 1/3 of module',
            'static/diode.png'),
           ('Diode-Multi',
            'Multiple activated bypass diodes, typically affecting 2/3 of module',
            'static/diode-multi.png'),
           ('Vegetation',
            'Panels blocked by vegetation',
            'static/vegetation.png'),
           ('Soiling',
            'Dirt, dust, or other debris on surface of module',
            'static/soiling.png'),
           ('Offline-Module',
            'Entire module is heated',
            'static/offline-module.png')]

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = UploadForm()
    if request.method == 'POST':
        request.files['file'].save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], request.files['file'].filename))
        return redirect(url_for('upload_file'))

    files_list = os.listdir(os.path.join(basedir, 'uploads'))
    # files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    file_urls = [photos.url(filename) for filename in files_list]

    predictions = [file_name for file_name in files_list]
    # predictions = [evaluator.predict_from_file(f'../app/uploads/{file_name}') for file_name in files_list]
    # predictions = [evaluator.predict(img_to_array(load_img(photos.path(file_name),
    #                                                        color_mode='rgb')).reshape(1, 40, 32, 3))
    #                for file_name in files_list]

    return render_template('index.html',
                           classes=CLASSES,
                           form=form,
                           files=zip(files_list, file_urls, predictions))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)

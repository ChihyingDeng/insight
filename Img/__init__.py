#!/usr/bin/python
import os, fnmatch, datetime
import re
import torch
import numpy as np
import matplotlib.image
from flask import Flask, render_template, flash, request, redirect, url_for, session
from flask_bootstrap import Bootstrap
from flask_wtf import Form, RecaptchaField, FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import BooleanField, SubmitField, ValidationError, validators
from wtforms.validators import Required
from werkzeug.utils import secure_filename
from flask_nav import Nav
from flask_nav.elements import Navbar, View
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_paginate import Pagination, get_page_parameter, get_per_page_parameter, get_page_args
from Img.runmodel import RunModel
from Img.util import Grad_CAM


WTF_CSRF_ENABLED = False
sorted_data = []

class ExampleForm(FlaskForm):
    submit_button = SubmitField(' DEMO ')
    show = BooleanField('Show possibility', default=True)

def create_app(configfile=None):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'devkey'
    app.config['RECAPTCHA_PUBLIC_KEY'] = \
        '6Lfol9cSAAAAADAkodaYl9wvQCwBMr3qGR_PPHcw'   
    app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
    app.config['UPLOAD_FOLDER'] = 'static/'
    app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
    app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
    app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
    app.config['DROPZONE_REDIRECT_VIEW'] = 'result_list'
    
    Bootstrap(app)

    dropzone = Dropzone(app)

    nav = Nav()
    nav.register_element('top', Navbar(
    View('Home', '.index'),
    View('Upload', '.index'),
    View('List', '.result_list'),
    View('Image', '.result_img')
    ))
    nav.init_app(app)

    photos = UploadSet('photos', IMAGES)
    configure_uploads(app, photos)
    patch_request_class(app)

    # ---- Index ---- 
    @app.route('/', methods=('GET', 'POST'))
    def index():
        form = ExampleForm()
        show = True
        file_urls = []
        
        # upload images from dropzone
        if request.method == 'POST':
            for file in request.files:
                f = request.files.get(file)
                filename = photos.save(f, name=f.filename)
                file_urls.append(photos.url(filename))  
            sorted_data = prediction(file_urls, 'upload')
        
        # demo
        if form.validate_on_submit():
            show = form.show.data
            listOfFiles = os.listdir('uploads/')  
            pattern = "*.jpg"  
            file = []
            for entry in listOfFiles:  
                if fnmatch.fnmatch(entry, pattern):
                    file.append('uploads/' + entry)
            sorted_data = prediction(file, 'demo')
            return render_template("list.html", file_urls=sorted_data, show=show)

        return render_template('index.html', form=form)

    # ---- List ---- 
    @app.route('/result_list')
    def result_list():
        # redirect to home if no images to display
        if not sorted_data:
            return redirect(url_for('index'))

        return render_template('list.html', file_urls=sorted_data, show=True)
        
    # ---- Image ---- 
    @app.route('/result_img')
    def result_img():
        # redirect to home if no images to display
        if not sorted_data:
                return redirect(url_for('index'))
        
        # pagination
        page = request.args.get(get_page_parameter(), type=int, default=1)
        per_page = request.args.get(get_per_page_parameter(), type=int, default=1)
        page, per_page, offset = get_page_args(page_parameter='page',
                                              per_page_parameter='per_page')
        per_page = int(per_page/10)
        offset = int(offset/10)

        img = get_img(offset=offset, per_page=per_page)
        pagination = Pagination(page=page, total=len(sorted_data), search=False, 
                                per_page=per_page, css_framework='bootstrap3')
        
        # original image & Grad-cam image if exist
        img = [(img[0], os.path.exists('Img/static/'+img[0].split('/')[1]))]

        return render_template('img.html', pagination=pagination, page=page, per_page=per_page, img=img)
    
    
    def get_img(offset=0, per_page=1):
        for file_urls, date, pred in sorted_data[offset: offset + per_page]:
            return [file_urls]
    
    def prediction(file_urls, mode):
        global sorted_data
        date = []
        imgtransCrop = 224
        
        if mode == 'demo':
            img = file_urls
        else:
            img = ['uploads/'+x.split('/')[-1] for x in file_urls] 
            
        # get image creation time
        for f in img:
            img_time = os.path.getmtime(f)
            img_time = datetime.datetime.fromtimestamp(img_time)
            date.append(img_time.strftime("%m/%d/%Y, %H:%M:%S"))
        
        # get prediction
        pred = RunModel.run(img)
        pred = torch.sigmoid(pred).numpy().flatten()
        
        # generate grad-cam image when possibility > 0.5 
        for i in range(len(pred)):
            if pred[i]>0.5:
                pathIn = img[i]
                pathOut = 'Img/static/' + pathIn.split('/')[1]
                h = Grad_CAM("Img/best.pth.tar")
                cam = h.generate(pathIn)
                matplotlib.image.imsave(pathOut, cam)
               
        sorted_data = list(zip(img, date, np.round(pred*100,2)))
        sorted_data = sorted(sorted_data, key = lambda x : (-x[2], x[1]))
        
        return sorted_data

    return app

if __name__ == '__main__':
    create_app().run(debug=True)


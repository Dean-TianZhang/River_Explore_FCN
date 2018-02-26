import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import scipy.misc as scimc
import scipy.io as sio
from fastRCNN import fastrcnn
from pred import predImg
from waterGaugeDetection import scaleRead



REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = './images'
LABEL_FOLDER = './predictions'
SEG_FOLDER = './segments'
LAB_FOLDER = './labels'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif', 'JPG', 'JPEG','PNG'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

@app.route('/gauge')
def gauge():
    return flask.render_template('water_gauge.html', has_result=False)

@app.route('/area')
def area():
    return flask.render_template('water_area.html', has_result=False)


# Water Area Detection
@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        # image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'water_area.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    origin, prediction, segmentation,label, time = predImg(string_buffer,"./model/deploy.prototxt","./model/River_726_iter_4000.caffemodel")
    return flask.render_template(
        'water_area.html', has_result=True, time = time, originsrc=embed_image_html(origin),
        predictionsrc=embed_image_html_annotation(prediction),segmentationsrc=embed_image_html(segmentation),labelsrc=embed_image_html_annotation(label))


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
	print(filename)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'water_area.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    origin, prediction, segmentation,label, time = predImg(filename,"./model/deploy.prototxt","./model/River_726_iter_4000.caffemodel")
    return flask.render_template(
        'water_area.html', has_result=True, time = time, originsrc=embed_image_html(origin),
        predictionsrc=embed_image_html_annotation(prediction),segmentationsrc=embed_image_html(segmentation),labelsrc=embed_image_html_annotation(label))


#Water Gauge Detection and Measurement
@app.route('/detect_url', methods=['GET'])
def detect_url():
    imageurl_detect = flask.request.args.get('imageurl_detect', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl_detect).read())
        # image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'water_gauge.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl_detect)
    detection, water_gauge, wh_ratio, time = fastrcnn(string_buffer)
    detect = scimc.imread(detection)
    origin_detect = scimc.imread(string_buffer)
    water_gauge_detect = scimc.imread(water_gauge)
    return flask.render_template(
        'water_gauge.html', has_result=True, time = time, originsrc_detect=embed_image_html(origin_detect),
        detectionsrc=embed_image_html(detect),watergaugesrc=embed_image_html(water_gauge_detect),frameratio = wh_ratio)


@app.route('/detect_upload', methods=['POST'])
def detect_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile_detect']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'water_gauge.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    detection, water_gauge, wh_ratio, time = fastrcnn(filename)
    detect = scimc.imread(detection)
    origin_detect = scimc.imread(filename)
    water_gauge_detect = scimc.imread(water_gauge)
    measure_result,resultPath = scaleRead(water_gauge_detect,water_gauge)
    scale_detect = scimc.imread(resultPath)
    s_h,s_w = scale_detect.shape[0:2]
    return flask.render_template(
        'water_gauge.html', has_result=True, time = time,measure_result=measure_result,frameratio = wh_ratio, originsrc_detect=embed_image_html(origin_detect),detectionsrc=embed_image_html(detect),watergaugesrc=embed_image_html(water_gauge_detect),scalesrc=embed_image_html(scale_detect),scale_w = s_w,scale_h = s_h )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def embed_image_html_annotation(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255*image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5000)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)

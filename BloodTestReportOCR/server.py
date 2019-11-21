#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from io import StringIO
import base64
import bson
import cv2
import flask
import numpy
from PIL import Image
from bson.json_util import dumps
from flask import Flask, request, Response, jsonify, redirect, json
from pymongo import MongoClient
from werkzeug.utils import secure_filename

import tf_predict
from imageFilter import ImageFilter

app = Flask(__name__, static_url_path="")

# 读取配置文件
app.config.from_object('config')

# 连接数据库，并获取数据库对象
db = MongoClient(app.config['DB_HOST'], app.config['DB_PORT']).test


# 将矫正后图片与图片识别结果（JSON）存入数据库
def save_file(file_str, f, report_data):
    content = StringIO(file_str)

    try:
        mime = Image.open(content).format.lower()
        print('content of mime is：', mime)
        if mime not in app.config['ALLOWED_EXTENSIONS']:
            raise IOError()
    except IOError:
        abort(400)
    c = dict(report_data=report_data, content=bson.binary.Binary(content.getvalue()), filename=secure_filename(f.name),
             mime=mime)
    db.files.save(c)
    return c['_id'], c['filename']


@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            flash('No file part')
            return jsonify({"error": "No file part"})
        imgfile = request.files['imagefile']
        if imgfile.filename == '':
            flash('No selected file')
            return jsonify({"error": "No selected file"})
        if imgfile:
            # pil = StringIO(imgfile)
            # pil = Image.open(pil)
            # print 'imgfile:', imgfile
            img = cv2.imdecode(numpy.frombuffer(imgfile.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
            report_data = ImageFilter(image=img).ocr(22)
            if report_data == None:
                data = {
                    "error": 1,
                }
                return jsonify(data)

            with open('temp_pics/region.jpg', 'rb') as f:
                if f is None:
                    print('Error! f is None!')
                else:

                    '''
                        定义file_str存储矫正后的图片文件f的内容（str格式）,方便之后对图片做二次透视以及将图片内容存储至数据库中
                    '''
                    file_str = f.read().decode(errors="ignore")
                    '''
                        使用矫正后的图片，将矫正后图片与识别结果（JSON数据）一并存入mongoDB，
                        这样前台点击生成报告时将直接从数据库中取出JSON数据，而不需要再进行图像透视，缩短生成报告的响应时间
                    '''
                    #img_region = cv2.imdecode(numpy.fromstring(file_str, numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
                    #report_data = ImageFilter(image=img).ocr(22)
                    print('report_data===',report_data)
                    fid, filename = save_file(file_str, f, report_data)
            print ('fid:', fid)
            if fid is not None:
                templates = "<div><img id=\'filtered-report\' src=\'/file/%s\' class=\'file-preview-image\' width=\'100%%\' height=\'512\'></div>" % (
                    fid)
                data = {
                    "templates": templates,
                }
            return jsonify(data)
            # return render_template("result.html", filename=filename, fileid=fid)
    # return render_template("error.html", errormessage="No POST methods")
    return jsonify({"error": "No POST methods"})

@app.route('/imageUpload',  methods=['POST'])
def image_upload():
    base64image = request.values['image']
    fileName = request.values['name']
    
    imgfile = open(fileName, 'wb')
    imgfile.write(base64.decodestring(base64image.encode()))
    imgfile.close()
    
    img_read = open(fileName,'rb')

    #print('{0},{1}'.format(base64image,fileName))

    img = cv2.imdecode(numpy.fromstring(img_read.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)
    report_data = ImageFilter(image=img).ocr(22)
    print(report_data)
    if report_data == None:
        data = {
            "error": 1,
        }
        return jsonify(data)
    else:
        return json.dumps(report_data, ensure_ascii=False).encode('utf8')
'''
    根据图像oid，在mongodb中查询，并返回Binary对象
'''
@app.route('/file/<fid>')
def find_file(fid):
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()
        return Response(file['content'], mimetype='image/' + file['mime'])
    except bson.errors.InvalidId:
        flask.abort(404)


'''
    直接从数据库中取出之前识别好的JSON数据，并且用bson.json_util.dumps将其从BSON转换为JSON格式的str类型
'''


@app.route('/report/<fid>')
def get_report(fid):
    # print 'get_report(fid):', fid
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()

        print ('type before transform:\n', type(file['report_data']))

        report_data = bson.json_util.dumps(file['report_data'])

        print ('type after transform:\n', type(report_data))
        if report_data is None:
            print ('report_data is NONE! Error!!!!')
            return jsonify({"error": "can't ocr'"})
        return jsonify(report_data)
    except bson.errors.InvalidId:
        flask.abort(404)

@app.route("/predict", methods=['POST'])
def predict():
    print ("predict now!")

    data = json.loads(request.form.get('data'))
    ss = data['value']
    arr = numpy.array(ss)
    arr = numpy.reshape(arr, [1, 22])

    sex, age = tf_predict.predict(arr)

    result = {
        "sex":sex,
        "age":int(age)
    }

    return json.dumps(result)



if __name__ == '__main__':

    app.run(host=app.config['SERVER_HOST'], port=app.config['SERVER_PORT'])
    

from flask import Flask, request, render_template
from flask_restful import Resource, Api
from tagv2 import TagProduct
from flask_api import FlaskAPI
import logging
import sys
import os
import json
import re
from hyperparameter import Hyper

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
api = Api(app)

# Thiết kế api
@app.route("/", methods = ['GET'])
def home():
    return 'Hello World!'

@app.route("/api/predictAutoTags", methods = ['GET', 'POST'])
def predictTags():
    if request.method == 'GET':
        # return render_template('home.html')
        return 'Hello World!'
    elif request.method == 'POST':
        form = request.json
        tagProduct = TagProduct(modelVersion=form['modelVersion'], apiType='oldProduct', productCode=form['productCode']).tagProduct
        try:
            tagList = tagProduct._create_dataset()['tags'].iloc[0]
        except:
            return json.dumps("not exist productCode!")
        tagList = [tag for tag in tagList]
        response = {"productCode":tagProduct.productCode,
                    "productTags":tagList,
                    "modelVersion": tagProduct.modelVersion}
        return json.dumps(response, ensure_ascii=False)

@app.route("/api/predictAutoTagsNewProduct", methods = ['GET', 'POST'])
def predictTagsNewProduct():
    if request.method == 'GET':
        return 'Hello World!'
    elif request.method == 'POST':
        form = request.json
        tagProduct = TagProduct(modelVersion=form['modelVersion'], apiType='newProduct', productName=form['productName'],
        cateName=form['cateName'], branchName=form['branchName'], description=form['description']).tagProduct
        logging.info('tagProduct: {}'.format(tagProduct.modelVersion))
        tagList = tagProduct._create_dataset()['tags'].iloc[0]
        tagList = [tag for tag in tagList]
        # logging.info("New version!")
        response = {"productName": tagProduct.productName,
                    "productTags": tagList,
                    "modelVersion": tagProduct.modelVersion}
        return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    with open(Hyper.CONFIGFILE, 'rb') as fp:
        config = json.loads(fp.read())
    app.run(debug=True, host=config['ip'])

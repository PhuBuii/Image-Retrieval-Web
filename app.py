import numpy as np
from PIL import Image
from feature_extractor import VGG16_FE,Xception_FE,EfficientNetV2L_FE
from datetime import datetime
from flask import Flask, request, render_template
from scipy.spatial.distance import cosine
from pathlib import Path

app = Flask(__name__)

# Read image features
fe_vgg16 = VGG16_FE()
features_vgg16 = []
img_paths = []
for feature_path in Path("./static/feature/vgg16").glob("*.npy"):
    features_vgg16.append(np.load(feature_path))
    img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
features_vgg16 = np.array(features_vgg16)

fe_xception = Xception_FE()
features_Xception = []
for feature_path in Path("./static/feature/xception").glob("*.npy"):
    features_Xception.append(np.load(feature_path))
features_Xception = np.array(features_Xception)

fe_efficientNetV2L = EfficientNetV2L_FE()
features_efficientNetV2L = []
for feature_path in Path("./static/feature/efficientNetV2L").glob("*.npy"):
    features_efficientNetV2L.append(np.load(feature_path))
features_efficientNetV2L = np.array(features_efficientNetV2L)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        file = request.files['file']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        img = img.convert('RGB')
        uploaded_img_path = "static/uploaded/query.jpg" 
        img.save(uploaded_img_path, format=None)
        return render_template('index.html')
    else:
        return render_template('index.html')

@app.route('/result', methods=['POST'])
def result_page():
    img_path='static/uploaded/query.jpg'
    img=Image.open(img_path)
    method = request.form.get("method")

    if method =="VGG16":
        print("Vgg16")
        # Run search
        query = fe_vgg16.extract(img)
        dist = []

        for feature in features_vgg16:
            dist.append(1-cosine(query,feature))
        dists=np.array(dist)
        ids = np.argsort(-dists)[:80]  # Top 80 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        return render_template('result.html',
                                query_path = img_path,
                                scores = scores)
    elif method =="Xception":
        print("Xception")
         # Run search
        query = fe_xception.extract(img)
        dist = []

        for feature in features_Xception:
            dist.append(1-cosine(query,feature))
        dists=np.array(dist)
        ids = np.argsort(-dists)[:80]  # Top 80 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        return render_template('result.html',
                                query_path = img_path,
                                scores = scores)
    elif method =="EfficientNetV2L":
        print("EfficientNetV2L")
        # Run search
        query = fe_efficientNetV2L.extract(img)
        dist = []

        for feature in features_efficientNetV2L:
            dist.append(1-cosine(query,feature))
        dists=np.array(dist)
        ids = np.argsort(-dists)[:80]  # Top 80 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        return render_template('result.html',
                                query_path = img_path,
                                scores = scores)        

if __name__ == "__main__":
    app.run("0.0.0.0")

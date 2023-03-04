import time
import cv2 as cv
import numpy as np
import feature_extractor as fe

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pickle import dump, load
from scipy.spatial.distance import cosine

class evaluate:
    def __init__(self, method='VGG16') -> None:
        # Select feature extractor based on corresponding method
        if method == 'VGG16':
            self.feature_extractor = fe.VGG16_FE()
        elif method == 'Xception':
            self.feature_extractor = fe.Xception_FE()
        elif method == 'EfficientNetV2L':
            self.feature_extractor = fe.EfficientNetV2L_FE()

        # Save method name
        self.method = method
        
    def retrieve_image(self, query_image, K=25) -> tuple:
        # Start counting time
        start = time.time()
        method = self.method
        
        if isinstance(query_image, str):  # If query image is a path (string type)
            query_image = Image.open(query_image)  # Open query image at image path
       
        if method =="VGG16":
            fe_vgg16 = fe.VGG16_FE()
            features_vgg16 = []
            img_paths = []
            for feature_path in Path("./static/feature/vgg16").glob("*.npy"):
                features_vgg16.append(np.load(feature_path))
                img_paths.append(feature_path.stem)
            features_vgg16 = np.array(features_vgg16)
            print("Vgg16")
            # Run search
            query = fe_vgg16.extract(query_image)
            dist = []

            for feature in features_vgg16:
                dist.append(1-cosine(query,feature))
            dists=np.array(dist)
            ids = np.argsort(-dists)[:K]  # Top 80 results
            scores = [(dists[id], img_paths[id]) for id in ids]
                    # End counting time
            end = time.time()

            # Return top K relevant images and query time
            return scores, end - start

        elif method =="Xception":
            fe_xception = fe.Xception_FE()
            features_Xception = []
            img_paths = []
            for feature_path in Path("./static/feature/xception").glob("*.npy"):
                features_Xception.append(np.load(feature_path))
                img_paths.append(feature_path.stem)
            features_Xception = np.array(features_Xception)
            print("Xception")
            # Run search
            query = fe_xception.extract(query_image)
            dist = []

            for feature in features_Xception:
                dist.append(1-cosine(query,feature))
            dists=np.array(dist)
            ids = np.argsort(-dists)[:K]  # Top 80 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            end = time.time()

            # Return top K relevant images and query time
            return scores, end - start

        elif method =="EfficientNetV2L":
            fe_efficientNetV2L = fe.EfficientNetV2L_FE()
            features_efficientNetV2L = []
            img_paths = []
            for feature_path in Path("./static/feature/efficientNetV2L").glob("*.npy"):
                features_efficientNetV2L.append(np.load(feature_path))
                img_paths.append(feature_path.stem)
            features_efficientNetV2L = np.array(features_efficientNetV2L)

            print("EfficientNetV2L")
            # Run search
            query = fe_efficientNetV2L.extract(query_image)
            dist = []

            for feature in features_efficientNetV2L:
                dist.append(1-cosine(query,feature))
            dists=np.array(dist)
            ids = np.argsort(-dists)[:K]  # Top 80 results
            scores = [(dists[id], img_paths[id]) for id in ids]
            end = time.time()

            # Return top K relevant images and query time
            return scores, end - start

    def AP(self, predict, groundtruth, interpolated=False):
        # Create precision, recall list
        p = []
        r = []
        correct = 0  # Number of correct images

        # Browse each image in predict list
        for id,(score,image) in enumerate(predict):
            # If image in groundtruth
            if image in groundtruth:
                correct += 1  # Increase number of correct images
                p.append(correct / (id + 1))  # Add precision at this position
                if interpolated:  # If interpolated AP
                    r.append(correct / len(groundtruth))  # Add recall at this position

        if interpolated:  # If call interpolated AP
            trec = []  # Calculate precision at 11 point of TREC
            for R in range(11):  # Browse 11 point of recall from 0 to 1 with step 0.1
                pm = []  # Create precision list to find max precision
                for id, pr in enumerate(p):  # Browse each precision above
                    if r[id] >= R / 10:  # If corresponding recall is greater than or equal to this point
                        pm.append(pr)  # Add precision to precision list to find max
                trec.append(max(pm) if len(pm) else 0)  # Add max precision at this point to trec
            return np.mean(np.array(trec)) if len(trec) else 0  # Return interpolated AP

        return np.mean(np.array(p)) if len(p) else 0  # Return non - interpolated AP

    def evaluating(self):
        # Create results folder path --> contains result file of evaluating
        # If not exist, create a new folder
        results_folder_path = Path("./static/evaluation")
        images_folder_path =Path("./static/img")
        # Create result file path --> save result of evaluating
        # If exist, remove file
        result_file_path = results_folder_path / f'evaluate_{self.method}.txt'
        if result_file_path.exists():
            result_file_path.unlink()

        groundtruth_folder_path = Path("./static/groundtruth")
        # Open result file at result file path to write
        result_file = open(result_file_path, 'a')

        # Write header line
        result_file.write('-' * 20 + 'START EVALUATING' + '-' * 20 + '\n\n')

        # Start counting time of evaluating
        start = time.time()

        # Get query files from groundtruth folder path
        queries_file = sorted(groundtruth_folder_path.glob('*_query.txt'))

        # Create list to save non - interpolated and interpolated APs
        nAPs = []
        iAPs = []

        # Browse each query file in queries file
        for id, query_file in enumerate(queries_file):
            # Create groundtruth list
            groundtruth = []

            # Get groundtruth from file with 'good' result
            with open(str(query_file).replace('query', 'good'), 'r') as groundtruth_file:
                groundtruth.extend([line.strip() for line in groundtruth_file.readlines()])

            # Get groundtruth from file with 'ok' result            
            with open(str(query_file).replace('query', 'ok'), 'r') as groundtruth_file:
                groundtruth.extend([line.strip() for line in groundtruth_file.readlines()])

            # Get length of groundtruth
            G = len(groundtruth)

            # Open query file to read
            with open(query_file, 'r') as query:
                # Get content of query file
                content = query.readline().strip().split()

                # Get image path and coordinates of bounding box
                image_name = content[0].replace('oxc1_', '') + '.jpg'
                image_path = images_folder_path / image_name
                bounding_box = tuple(float(coor) for coor in content[1:])

                # Open query image and crop image based on bounding box
                query_image = Image.open(image_path)
                query_image = query_image.crop(bounding_box)

                # Retrieve image and get relevant images, query time
                rel_imgs, query_time = self.retrieve_image(query_image, G)

                # Calculate non - interpolated and interpolated AP
                nAP = self.AP(rel_imgs, groundtruth, interpolated=False)
                iAP = self.AP(rel_imgs, groundtruth, interpolated=True)

                # Add the AP values to the corresponding AP list
                nAPs.append(nAP)
                iAPs.append(iAP)

            # Write id and name of query file
            result_file.write(f'+ Query {(id + 1):2d}: {Path(query_file).stem}.txt\n')

            # Write non - interpolated and interpolated AP
            result_file.write(' ' * 12 + f'Non - Interpolated Average Precision = {nAP:.2f}\n')
            result_file.write(' ' * 12 + f'Interpolated Average Precision = {iAP:.2f}\n')

            # Write query time
            result_file.write(' ' * 12 + f'Query Time = {query_time:2.2f}s\n')

        # End counting time of evaluating
        end = time.time()

        # Write footer line
        result_file.write('\n' + '-' * 19 + 'FINISH EVALUATING' + '-' * 20 + '\n\n')

        # Calculate non - interpolated and interpolated MAP
        nMAP = np.mean(np.array(nAPs))
        iMAP = np.mean(np.array(iAPs))

        # Write total number of queries
        result_file.write(f'Total number of queries = {len(queries_file)}\n')

        # Write non - interpolated and interpolated MAP
        result_file.write(f'Non - Interpolated Mean Average Precision = {nMAP:.2f}\n')
        result_file.write(f'Interpolated Mean Average Precision = {iMAP:.2f}\n')

        # Write evaluating time
        result_file.write(f'Evaluating Time = {(end - start):2.2f}s')

        # Close result file
        result_file.close()
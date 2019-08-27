

import os

import cv2
import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

from mxnet import gluon
import mxnet as mx
from mxnet import nd

def load_img(path):
    img = cv2.imread(path)
    return img


class NearestNeighbor:
    def __init__(self, threshold=.6, norm=False, verbose=False, metric='cosine'):
        self.threshold = threshold
        self.x = None
        self.y = None
        self.norm = norm
        self.verbose = verbose
        
        if metric == 'euclidean':
            self.metric = euclidean_distances
        elif metric == 'cosine':
            self.metric = cosine_distances
    
    def fit(self, x, y):
        if self.norm:
            self.x = normalize(x)
        else:
            self.x = x
        self.y = y
    
    def predict(self, x_):
        if self.norm:
            dists = self.metric(normalize(x_), self.x)
        else:
            dists = self.metric(x_, self.x)
        
        idx = dists.argmin(axis=1)
        min_dist = dists[np.arange(dists.shape[0]), idx]

        preds = []
        for i in range(dists.shape[0]):
            if min_dist[i] <= self.threshold:
                preds.append(self.y[idx[i]])
            else:
                preds.append('Unknown')
        
        if self.verbose:
            print(list(zip(preds, min_dist)))

        return np.array(preds), min_dist


class Arcface:
    def __init__(self, model_name='mobilenet', batch_size=32):
        filepath, _ = os.path.split(os.path.realpath(__file__))
        model_path = os.path.join(filepath, 'models/arc_%s' % model_name.lower())

        self.model_name = model_name
        self.batch_size = batch_size
        self.ctx = mx.gpu()
        self.model = gluon.nn.SymbolBlock.imports(os.path.join(model_path, "model-symbol.json"), ['data'], os.path.join(model_path, "model-0000.params"), ctx=self.ctx)
        self.model.forward(nd.zeros((batch_size, 3, 112, 112), ctx=self.ctx))

    def predict(self, imgs):

        if imgs.ndim == 3:
            imgs = imgs[np.newaxis,:,:,:]
        imgs = imgs.transpose([0,3,1,2])
        
        feats = []
        for start in range(0, len(imgs), self.batch_size):
            mximg = nd.array(imgs[start:start+self.batch_size], ctx=self.ctx)
            feat = self.model.forward(mximg)
            feats.append(feat.asnumpy())
            
        feats = np.vstack(feats)
        return feats

    def get_size(self):
        return (112, 112)

    def get_name(self):
        return "ArcFace " + self.model_name


class FaceRecognition:
    def __init__(self, dataset=None, labels=None, dist_threshold=.6, cpudet=True, det_threshold=[0.6, 0.7, 0.8], det_factor=0.709, det_minsize=20, detection_backend='mxnet'):
        # Choose which implementation
        if detection_backend == 'mxnet':
            from .mxnet_detector import MTCNNDetector
        else:
            from .tf_detector import MTCNNDetector
        
        size = 112
        self.encoder = Arcface('mobilenet')
        self.detector = MTCNNDetector(shape=size, cpu=cpudet, threshold=det_threshold, factor=det_factor, minsize=det_minsize)

        self.known_encodings = []
        self.labels = []

        if dataset is None:
            filepath, _ = os.path.split(os.path.realpath(__file__))
            dataset = os.path.join(filepath, 'data/rec')
    
        if isinstance(dataset, str):
            self.read_folder(dataset)
        if isinstance(dataset, np.ndarray):
            self.known_encodings = dataset
            if labels is None:
                raise ValueError("If dataset is an array, labels should be supplied.")
            
            if not isinstance(labels, (np.ndarray, list)):
                raise ValueError("Labels should be a numpy array or a list.")

            self.labels = np.array(labels)
        if isinstance(dataset, dict):
            data = []
            labels = []
            for key in dataset.keys():
                feats = dataset[key]
                data.append(feats)
                labels.append(np.array([key] * feats.rows))
            
            self.known_encodings = data
            self.labels = labels

        
        self.clf = NearestNeighbor(verbose=False, norm=True, threshold=dist_threshold, metric='cosine')
        self.clf.fit(self.known_encodings, self.labels)
    
    def read_folder(self, folder):
        img_names = os.listdir(folder)
        # Read images
        imgs = np.array( [load_img(os.path.join(folder, name)) for name in img_names] )
        # Get labels
        self.labels = np.array( [ name.split('_')[0] for name in img_names ] )

        known_encodings = []

        for i, img in enumerate(imgs):
            bbs, faces = self.detector.detect(img)[0] # 1 image passed
            bb, face = bbs[0], faces[0] # Should return 1 face. If returns more, ignore the rest

            if face.ndim == 3:
                face = face[np.newaxis, ...]
            
            known_encodings.append(self.encoder.predict(face))
        
        self.known_encodings = np.vstack(known_encodings)

    def evaluate(self, img):
        bbs, faces = self.detector.detect(img)[0] # 1 image passed

        if faces:
            faces = np.array(faces)
            encodings = self.encoder.predict(faces)
            names, dists = self.clf.predict(encodings)

            return names, dists, bbs
        else:
            return None
    
    def encode(self, img):
        # Assumes one person per image
        bbs, faces = self.detector.detect(img)[0] # 1 image passed

        if faces:
            faces = np.array(faces)
            encodings = self.encoder.predict(faces)
            
            return bbs, faces, encodings
        else:
            return None




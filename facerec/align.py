#!/usr/bin/python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from skimage import transform


def align(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2

    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
            [0.34191608, 0.4615741 ],
            [0.6565339 , 0.45983392],
            [0.500225  , 0.6405054 ],
            [0.3709759 , 0.82469195],
            [0.63151693, 0.8232509 ]], dtype=np.float32)
        src[:, 0] *= image_size[1]
        src[:, 1] *= image_size[0]
        # src *= alpha
        
        dst = landmark.astype(np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        #M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None: #use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret 
    else: #do align using landmark
        assert len(image_size)==2

        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)

        return warped



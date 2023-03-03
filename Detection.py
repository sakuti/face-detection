# -*- coding: utf-8 -*-
# Simple face detection and recognition wrapper,
# which uses dlib to detect faces and recognize faces.
# Saku, 2022.

import PIL.Image         
import dlib                 
import numpy as np       

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



class Detector:
    def __init__(self, modelsPath='Models/', datasetPath='Dataset/'):
        self.modelsPath = modelsPath
        self.datasetPath = datasetPath
    
        self.faceDetector = dlib.get_frontal_face_detector()
        self.posePredictor_68_Model = self.modelsPath + 'shapePredictor_68.dat'
        self.posePredictor_5_Model = self.modelsPath + 'shapePredictor_5.dat'
        self.faceDetector_2_Model = self.modelsPath + 'mmod_face_v1.dat'
        self.faceRecognition_Model = self.modelsPath + 'dlib_resnet_v1.dat'
        
        self.posePredictor_68 = dlib.shape_predictor(self.posePredictor_68_Model)
        self.posePredictor_5 = dlib.shape_predictor(self.posePredictor_5_Model)
        self.faceDetector_2 = dlib.cnn_face_detection_model_v1(self.faceDetector_2_Model)
        self.faceRecognition = dlib.face_recognition_model_v1(self.faceRecognition_Model)



    def rectToCss(self, rect):
        return rect.top(), rect.right(), rect.bottom(), rect.left()


    def cssToRect(self, css):
        return dlib.rectangle(css[3], css[0], css[1], css[2])

    
    def trimCssBoundaries(self, css, image):
        return max(css[0], 0), min(css[1], image[1]), min(css[2], image[0]), max(css[3], 0)

    
    def getFaceDistance(self, encodings, face):
        if len(encodings) == 0: return np.empty((0))
        return np.linalg.norm(encodings - face, axis=1)


    def loadImageFile(self, file, mode='RGB'):
        Image = PIL.Image.open(file)
        if mode: Image = Image.convert(mode)
        return np.array(Image)


    def getRawFaceLocations(self, image, upsample=1, model="hog"):
        if model == "cnn": return self.faceDetector_2(image, upsample)
        return self.faceDetector(image, upsample)


    def getFaceLocations(self, image, upsample=1, model="hog"):
        if model == "cnn": return [self.trimCssBoundaries(self.rectToCss(face.rect), image.shape) for face in self.getRawFaceLocations(image, upsample, "cnn")]
        return [self.trimCssBoundaries(self.rectToCss(face), image.shape) for face in self.getRawFaceLocations(image, upsample, model)]


    def getRawFaceLocationsBatched(self, images, upsample=1, batchSize=128):
        return faceDetector_2(images, upsample, batch_size=batchSize)


    def getBatchFaceLocations(self, images, upsample=1, batchSize=128):
        def convertDetectionsToCss(detections):
            return [self.trimCssBoundaries(self.rectToCss(face.rect), images[0].shape) for face in detections]

        rawDetections = self.getRawFaceLocationsBatched(images, upsample, batchSize)
        return list(map(convertDetectionsToCss, rawDetections))

    
    def getRawFaceLandmarks(self, image, locations=None, model="large"):
        if locations is None:
            locations = self.getRawFaceLocations(image)
        else:
            locations = [self.cssToRect(location) for location in locations]

        posePredictor = self.posePredictor_68
        if model == "small": posePredictor = self.posePredictor_5
        return [posePredictor(image, location) for location in locations]

    
    def getFaceLandmarks(self, image, locations=None, model="large"):
        landmarks = self.getRawFaceLandmarks(image, locations, model)
        landmarksAsTuples = [[(p.x, p.y) for p in i.parts()] for i in landmarks]

        if model == 'large':
            return [{
                "chin": i[0:17],
                "leftEyebrow": i[17:22],
                "rightEyebrow": i[22:27],
                "noseBridge": i[27:31],
                "noseTip": i[31:36],
                "leftEye": i[36:42],
                "rightEye": i[42:48],
                "topLip": i[48:55] + [i[64]] + [i[63]] + [i[62]] + [i[61]] + [i[60]],
                "bottomLip": i[54:60] + [i[48]] + [i[60]] + [i[67]] + [i[66]] + [i[65]] + [i[64]]
            } for i in landmarksAsTuples]
        elif model == 'small':
            return [{
                "nose_tip": [i[4]],
                "left_eye": i[2:4],
                "right_eye": i[0:2],
            } for i in landmarksAsTuples]
        else:
            raise ValueError("Invalid landmark model type!")


    def getFaceEncodings(self, image, locations=None, jitters=1, model="small"):
        rawLandmarks = self.getRawFaceLandmarks(image, locations, model)
        return [np.array(self.faceRecognition.compute_face_descriptor(image, i, jitters)) for i in rawLandmarks]
    

    def compareFaces(self, knownEncodings, encoding, tolerance=0.6):
        return list(self.getFaceDistance(knownEncodings, encoding) <= tolerance)
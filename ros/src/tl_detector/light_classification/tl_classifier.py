from styx_msgs.msg import TrafficLight

import cv2
import rospy
import rospkg
import numpy as np

import tensorflow as tf
from keras.models import load_model



class TLClassifier(object):
    def __init__(self):
        
        r = rospkg.RosPack()
        model_path = r.get_path('tl_detector')
        self.tl_model = load_model(model_path + '/light_classification_model.h5')

        self.tl_model._make_predict_function()
        self.graph = tf.get_default_graph()


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img = cv2.resize(image, (400,400))
        img = img.astype(float)
        img = img/255.0

        img = img[np.newaxis,:,:,:]
        with self.graph.as_default():
            preds = self.tl_model.predict(img)
            
        # [[9.99999881e-01, 1.06137037e-07]]
        # [[green_prob, red_prob]]
        prediction = np.argmax(preds, axis=1)
        light_id = prediction[0]
        rospy.loginfo("traffic light id (0-green, 1-red): " + str(light_id))

        if (light_id == 1):
            return TrafficLight.RED

        return TrafficLight.UNKNOWN

#
# This file is partly based on code that carries the following license:
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from styx_msgs.msg import TrafficLight
from collections import deque, Counter
import cv2
import numpy as np
import tensorflow as tf
import rospy
import os

class TLClassifier(object):
    def __init__(self):
        self.ct = -1
        # Use queue to hold 3 most recent guesses
        self.queue = deque([TrafficLight.UNKNOWN]*3, 3)
        self.graph = tf.Graph()
        self.graph_def = tf.GraphDef()

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pb"), "rb") as f:
            self.graph_def.ParseFromString(f.read())
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def)

        self.label = []
        proto_as_ascii_lines = tf.gfile.GFile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels.txt")).readlines()
        for l in proto_as_ascii_lines:
            self.label.append(l.rstrip())

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        self.ct += 1
        # only classify 1/10 of all camera images. use only when GPU is not available
        if self.ct % 10 != 0:
            return Counter(self.queue).most_common(1)[0][0]
        image = cv2.resize(image, dsize=(224,224))
        image = cv2.normalize(image, None, 0., 1., cv2.NORM_MINMAX)
        np_image = np.asarray(image)
        np_final = np.expand_dims(np_image, axis=0)

        input_name = "import/input"
        output_name = "import/final_result"
        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: np_final
        })
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        res = self.label[top_k[0]]
        rospy.logwarn(res)

        if res.endswith("0"):
            state = TrafficLight.RED
        elif res.endswith("1"):
            state = TrafficLight.YELLOW
        elif res.endswith("2"):
            state = TrafficLight.GREEN
        else:
            state = TrafficLight.UNKNOWN

        self.queue.append(state)
        # Majority vote from last 3 guesses
        return Counter(self.queue).most_common(1)[0][0]

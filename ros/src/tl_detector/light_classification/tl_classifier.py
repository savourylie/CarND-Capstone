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

LABEL_RED = 0
LABEL_YELLOW = 1
LABEL_GREEN = 2
LABEL_UNKNOWN = 4

# labels for the traffic light classes for simulator images
# are offset by this much from the labels for the real images
SIMULATOR_LABELS_OFFSET = 10

class TLClassifier(object):
    def __init__(self):
        # Use queue to hold 10 most recent guesses
        self.queue = deque([TrafficLight.UNKNOWN]*10, 10)
        
        classifier_data_dir = ("../../../../"
                               "traffic-light-mobilenetv2-classifier/")
        self.graph = load_graph(
            CLASSIFIER_DATA_DIR"real_and_sim_disjoint_output_graph.pb")
        self.labels = load_labels(
            CLASSIFIER_DATA_DIR"real_and_sim_disjoint_output_labels.txt")
        
        input_name = "import/input"
        output_name = "import/final_result"
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        
    def load_graph(model_file):
        graph = tf.Graph()
        
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph
    
    def read_tensor_from_image(image_tensor, input_height=299, input_width=299,
                               input_mean=0, input_std=255):
#    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
#                                        name='jpeg_reader')
        float_caster = tf.cast(image_tensor, tf.float32)
        
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)
        return result

    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        input_height = 224
        input_width = 224
        input_mean = 0
        input_std = 255

#        # Our classifier is trained on RGB JPEGs, so convert from BGR
#        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image_jpeg = cv2.imencode('.jpg', image)
        image_tensor = tf.convert_to_tensor(image_jpeg)
        
        t = read_tensor_from_image(
            image_tensor,
            input_height=input_height,
            input_width=input_width,
            input_mean=input_mean,
            input_std=input_std)

        with tf.Session(graph=self.graph) as sess:
            results = sess.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
            })
        results = np.squeeze(results)

        winning_label = self.labels[results.argsort()[-2:][::-1][0]]
        # collapse the classes for the real and simulated images
        winner = int(winning_label) % SIMULATOR_LABELS_OFFSET
        if winner == LABEL_RED:
            state = TrafficLight.RED
        elif winner == LABEL_YELLOW:
            state = TrafficLight.YELLOW
        elif winner == LABEL_GREEN:
            state = TrafficLight.GREEN
        else:
            state = TrafficLight.UNKNOWN

        self.queue.append(state)
        # Majority vote from last 10 guesses
        return Counter(self.queue).most_common(1)[0][0]

    def get_classification_by_pixel_color(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Detection based on color of pixels in image, only works in simulator
        red_min = np.array([0, 0, 200], np.uint8)
        red_max = np.array([50, 50, 255], np.uint8)

        reds = cv2.inRange(image, red_min, red_max)
        no_red = cv2.countNonZero(reds)

        yellow_min = np.array([0, 200, 200], np.uint8)
        yellow_max = np.array([50, 255, 255], np.uint8)

        yellows = cv2.inRange(image, yellow_min, yellow_max)
        no_yellow = cv2.countNonZero(yellows)

        green_min = np.array([0, 200, 0], np.uint8)
        green_max = np.array([50, 255, 50], np.uint8)

        greens = cv2.inRange(image, green_min, green_max)
        no_green = cv2.countNonZero(greens)

        # Override sequence: red > yellow > green
        threshold = 15
        if no_red > threshold:
            state = TrafficLight.RED
        elif no_yellow > threshold:
            state = TrafficLight.YELLOW
        elif no_green > threshold:
            state = TrafficLight.GREEN
        else:
            state = TrafficLight.UNKNOWN

        self.queue.append(state)
        # Majority vote from last 10 guesses
        return Counter(self.queue).most_common(1)[0][0]

from styx_msgs.msg import TrafficLight
from collections import deque, Counter
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        # Use queue to hold 10 most recent guesses
        self.queue = deque([TrafficLight.UNKNOWN]*10, 10)

    def get_classification(self, image):
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

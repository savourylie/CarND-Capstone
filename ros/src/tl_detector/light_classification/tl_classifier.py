from styx_msgs.msg import TrafficLight
from collections import deque, Counter
import cv2
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):
        self.queue = deque([TrafficLight.UNKNOWN]*10, 10)
        self.model = load_model('model_inceptionv3_sim.h5')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        im = cv2.resize(image, (299, 299), interpolation=cv2.INTER_CUBIC)
        b,g,r = cv2.split(im)
        rgb_img = cv2.merge([r,g,b])
        image_array = np.asarray(rgb_img)
        state = self.model.predict(image_array[None, :, :, :], batch_size=1)
        self.queue.append(state)
        return Counter(self.queue).most_common(1)[0][0]

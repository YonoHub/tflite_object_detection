from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from yonoarc_utils.image import to_ndarray, from_ndarray
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import tensorflow as tf


def read_labels(PATH_TO_LABELS):
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    if labels[0] == '???':
        del(labels[0])
    return labels


class TFLiteDetection:
    def on_start(self):
        ''' [Optional] Called at the beginning when the block starts. '''
        self.min_conf_threshold = self.get_property("min_conf_threshold")
        self.input_mean = self.get_property("input_mean")
        self.input_std = self.get_property("input_std")
        # self.get_property("model_path")
        self.model_path = "/home/yonohub/tflite/detect.tflite"
        self.labels_path = "/home/yonohub/tflite/labelmap.txt"
        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.labels = read_labels(self.labels_path)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        self.graph = tf.compat.v1.get_default_graph()

    def on_new_messages(self, messages):
        ''' [Optional] Called according to the execution mode of the block.

        Parameters
        ----------
        messages : dict
            A dictionary of the port keys and the values of the incoming messages.

        '''
        with self.graph.as_default():
            image = to_ndarray(messages['image'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            if self.floating_model:
                input_data = (np.float32(input_data) -
                              self.input_mean) / self.input_std

            self.interpreter.set_tensor(
                self.input_details[0]['index'], input_data)

            start_time = time.time()
            self.interpreter.invoke()
            stop_time = time.time()

            boxes = self.interpreter.get_tensor(
                self.output_details[0]['index'])[0]
            classes = self.interpreter.get_tensor(
                self.output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(
                self.output_details[2]['index'])[0]

            for i in range(len(scores)):
                if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))

                    cv2.rectangle(image, (xmin, ymin),
                                  (xmax, ymax), (10, 255, 0), 2)

                    # Draw label
                    # Look up object name from "labels" array using class index
                    object_name = self.labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(
                        scores[i]*100))  # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    # Make sure not to draw label too close to top of window
                    label_ymin = max(ymin, labelSize[1] + 10)
                    # Draw white box to put label text in
                    cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (
                        xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(image, label, (xmin, label_ymin-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text
            img_msg = from_ndarray(image, messages['image'].header)
            self.publish("out_img", img_msg)
            print("Detection Took {}".format(stop_time-start_time))

    def on_properties_changed(self, affected_properties):
        ''' [Optional] Called when one or more block properties are changed in live mode. 

        Parameters
        ----------
        affected_properties : dict
            A dictionary of the keys of the affected properties and their new values.

        '''
        self.on_start()

# we import the Twilio client from the dependency we just installed
from twilio.rest import Client

# the following line needs your Twilio Account SID and Auth Token
client = Client("AC77e42ee0664303a3ade83c79075fd85a", "66bb5de3dfd90be61865fd4f7274c0f2")

# change the "from_" number to your Twilio number and the "to" number
# to the phone number you signed up for Twilio with, or upgrade your
# account to send SMS to any phone number
# client.messages.create(to="+84376542326", 
#                        from_="+14847026286", 
#                        body="Hello from Python!")
import cv2
import imutils
from _collections import deque
import torch 
import torch.nn as nn
from s1_preprocess import FeatureGenerator
import numpy as np
from skeleton import SkeletonDetector
import time
import os
import warnings
import numpy as np
import time
import glob
import threading
import queue
import multiprocessing
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

MAP_LABEL = {
    0: 'walk',
    1: 'fall', 
    2: 'stand',
    3: 'lie',
    4: 'sit',
}

class LSTM(nn.Module):
    
    def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
        super(LSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim,output_dim)
        self.bn = nn.BatchNorm1d(16)
        
    def forward(self,inputs):
        x = self.bn(inputs)
        lstm_out,(hn,cn) = self.lstm(x)
        out = self.fc(lstm_out[:,-1,:])
        return out

class SaveImage(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.img_deque = deque()

    def add_image_for_video(self, image):
        self.img_deque.append(image)
        self.maintain_deque_size()

    def save_img2video(self, path_save, fps=25):
        (W, H) = (None, None)
        writer = None
        for index_frame in range(len(self.img_deque)):
            # print(self.img_deque[index_frame])
            frame = self.img_deque[index_frame]
            if frame.shape[1] > 400:
                frame = imutils.resize(frame, width=400)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
        #     check if the video writer is None
            if writer is None:
                # initialize our video writer
                # print("iiiiiiiiiiii")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(path_save, fourcc, fps,
                    (frame.shape[1], frame.shape[0]), True)
        #     write the output frame to disk
            writer.write(frame)
        # release the file pointers
        # print("[INFO] cleaning up...")
        writer.release()
        # vs.release()

    def maintain_deque_size(self):
        if len(self.img_deque) > self.window_size:
            self.img_deque.popleft()

class ClassifierOnTest(object):
    ''' Classifier for online inference.. 
    '''

    def __init__(self, window_size, path_weight='app/weights/lstm_size_16_bn.pkl'):
        # -- Settings
        # self.action_labels = action_labels
        self.THRESHOLD_SCORE_FOR_DISP = 0.5
        # -- Time serials storage
        self.feature_generator = FeatureGenerator(window_size)
        self.reset()
        #load model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = self.build_model(path_weight)
        self.model.to(self.device)
        # rnn.load_state_dict(torch.load(path_weight))

    def build_model(self, path_weight,n_hidden=128, n_joints=26, n_categories=5, n_layer=3):
        rnn = LSTM(n_joints,n_hidden,n_categories,n_layer)
        rnn.load_state_dict(torch.load(path_weight))
        return rnn

    def reset(self):
        self.feature_generator.reset()
        self.scores_hist = deque()
        self.scores = None

    def predict(self, skeleton):
        ''' Predict the class (string) of the input raw skeleton '''
        LABEL_UNKNOWN = ""
        is_features_good, features = self.feature_generator.add_cur_skeleton(
            skeleton)
        if is_features_good:
            with torch.no_grad():
                inputs = torch.FloatTensor([features])
                inputs = inputs.to(self.device)
                output = self.model(inputs)
                top_n, top_i = output.topk(1)
                category_i = top_i[0].item()
                return MAP_LABEL[category_i]
        #     curr_scores = self.model.predict(features)[0]
        #     self.scores = self.smooth_scores(curr_scores)

        #     if self.scores.max() < self.THRESHOLD_SCORE_FOR_DISP:  # If lower than threshold, bad
        #         prediced_label = LABEL_UNKNOWN
        #     else:
        #         predicted_idx = self.scores.argmax()
        #         prediced_label = self.action_labels[predicted_idx]
        # else:
        #     prediced_label = LABEL_UNKNOWN
        # return prediced_label
"""
    def smooth_scores(self, curr_scores):
        ''' Smooth the current prediction score
            by taking the average with previous scores
        '''
        self.scores_hist.append(curr_scores)
        DEQUE_MAX_SIZE = 3
        if len(self.scores_hist) > DEQUE_MAX_SIZE:
            self.scores_hist.popleft()

        # Use sum
        score_sums = np.zeros((len(self.action_labels),))
        for score in self.scores_hist:
            score_sums += score
        score_sums /= len(self.scores_hist)
        print("\nMean score:\n", score_sums)
        return score_sums
"""
class ReadFromWebcam(object):
    def __init__(self, max_framerate=30.0, webcam_idx=0):
        ''' Read images from web camera.
        Argument:
            max_framerate {float}: the real framerate will be reduced below this value.
            webcam_idx {int}: index of the web camera on your laptop. It should be 0 by default.
        '''
        # Settings
        self._max_framerate = max_framerate
        queue_size = 3

        # Initialize video reader
        self._video = cv2.VideoCapture(webcam_idx)
        self._is_stoped = False

        # Use a thread to keep on reading images from web camera
        self._imgs_queue = queue.Queue(maxsize=queue_size)
        self._is_thread_alive = multiprocessing.Value('i', 1)
        self._thread = threading.Thread(
            target=self._thread_reading_webcam_images)
        self._thread.start()

        # Manually control the framerate of the webcam by sleeping
        self._min_dt = 1.0 / self._max_framerate
        self._prev_t = time.time() - 1.0 / max_framerate

    def read_image(self):
        dt = time.time() - self._prev_t
        if dt <= self._min_dt:
            time.sleep(self._min_dt - dt)
        self._prev_t = time.time()
        image = self._imgs_queue.get(timeout=10.0)
        return image

    def has_image(self):
        return True  # The web camera always has new image

    def stop(self):
        self._is_thread_alive.value = False
        self._video.release()
        self._is_stoped = True

    def __del__(self):
        if not self._is_stoped:
            self.stop()

    def _thread_reading_webcam_images(self):
        while self._is_thread_alive.value:
            ret, image = self._video.read()
            if self._imgs_queue.full():  # if queue is full, pop one
                img_to_discard = self._imgs_queue.get(timeout=0.001)
            self._imgs_queue.put(image, timeout=0.001)  # push to queue
        print("Web camera thread is dead.")

def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = ReadFromWebcam(
            25, webcam_idx)
    return images_loader

class DetectFall():
    def __init__(self, label_save, window_size=32, time_save=180):
        self.window_size = window_size
        self.time_save = time_save
        self.time = -1
        self.label_save = label_save
        self.save_video = SaveImage(window_size=self.window_size*3)
        self.classifi = ClassifierOnTest(window_size=self.window_size)
        self.skeleton = SkeletonDetector()
    
    def detect(self, link_camera):
        # vs = cv2.VideoCapture(link_camera)
        while images_loader.has_image():
            frame = images_loader.read_image()
            # (grabbed, frame) = vs.read()
            # if not grabbed:
                # break
            cv2.imshow("demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            sk = self.skeleton.detect(frame)
            self.save_video.add_image_for_video(frame)
            predict = self.classifi.predict(np.array(sk))
            print(predict)
            # if predict==self.label_save:
            #     if self.time == -1:
            #         self.save_video.save_img2video('data/test.avi')
            #         self.time = int(time.time())
            #     elif int(time.time()) - self.time > self.time_save:
            #         self.save_video.save_img2video('data/test.avi')
            #         self.time = int(time.time())
            

if __name__ == "__main__":
    # save_video = SaveImage(64)
    # classi = ClassifierOnTest(window_size=32)
    # skeleton = SkeletonDetector()
    detect = DetectFall('sit', window_size=16)
    path_video = 'rtsp://admin:D9ng2612@192.168.1.180:554/cam/realmonitor?channel=1&subtype=1'
    # frame_provider = ImageReader(images)
    images_loader = select_images_loader('webcam', path_video)
    # test_demo(net, 256, None, 1, 1, images_loader)
    detect.detect(images_loader)
        # save_video.add_image_for_video(frame)
    # save_video.save_img2video('test.avi')
    
        

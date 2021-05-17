
import cv2
import numpy as np
import time
import os
import numpy as np
import time
import threading
import queue
import multiprocessing
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

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

# class DetectFall():
#     def __init__(self, label_save, window_size=32, time_save=180):
#         self.window_size = window_size
#         self.time_save = time_save
#         self.time = -1
#         self.label_save = label_save
#         self.save_video = SaveImage(window_size=self.window_size*3)
#         self.classifi = ClassifierOnTest(window_size=self.window_size)
#         self.skeleton = SkeletonDetector()
    
def detect(images_loader):
        # vs = cv2.VideoCapture(link_camera)
    while images_loader.has_image():
        frame = images_loader.read_image()
            # (grabbed, frame) = vs.read()
            # if not grabbed:
                # break
        cv2.imshow("demo",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # sk = self.skeleton.detect(frame)
        # self.save_video.add_image_for_video(frame)
        # predict = self.classifi.predict(np.array(sk))
        # print(predict)
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
    path_video = 'rtsp://admin:D9ng2612@192.168.1.180:554/cam/realmonitor?channel=1&subtype=1'
    # frame_provider = ImageReader(images)
    images_loader = select_images_loader('webcam', path_video)
    # test_demo(net, 256, None, 1, 1, images_loader)
    detect(images_loader)
        # save_video.add_image_for_video(frame)
    # save_video.save_img2video('test.avi')
    
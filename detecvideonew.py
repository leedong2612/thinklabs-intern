import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# lấy struct_time
named_tuple = time.localtime() 
time_string = time.strftime("%d%m%Y-%Hh_%Mm_%Ss", named_tuple)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/truck.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './detections/output_{}.avi'.format(time_string), 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', True, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', True, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

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

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # get video name by using split method
    # Lấy tên video bằng phương pháp tách
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    # bắt đầu quay video
    path_video = 'rtsp://admin:D9ng2612@192.168.1.180:554/cam/realmonitor?channel=1&subtype=1'

    # frame_provider = ImageReader(images)
    images_loader = select_images_loader('webcam', path_video)

    vid = detect(images_loader)  
    # vid = cv2.VideoCapture(str(video_path))
    # vid = cv2.VideoCapture(0)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        # theo mặc định VideoCapture trả về float thay vì int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0

    while images_loader.has_image():
        frame = images_loader.read_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_num += 1
        image = Image.fromarray(frame)

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        # định dạng bounding boxes từ ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        # đọc trong tất cả các class name từ cấu hình
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # mặc định chấp nhận tất cả các classes
        allowed_classes = list(class_names.values())
        counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)

        # if crop flag is enabled, crop each detection and save it as new image
        # nếu cờ cắt được bật, hãy cắt từng phát hiện và lưu nó dưới dạng hình ảnh mới
        if FLAGS.crop:
            named_tuple = time.localtime() 
            time_string = time.strftime("%m%d%Y-%H%M%S", named_tuple)
            crop_rate = 200 # capture images every so many frames (ex. crop photos every 150 frames)
            crop_path = os.path.join(os.getcwd(), 'detections', 'crop', video_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            if frame_num % crop_rate == 0:
                # final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
                
                # try:
                #     os.mkdir(final_path)
                # except FileExistsError:
                #     pass          
                crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes, time_string)

            else:
                pass
        
        image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dont_show:
            cv2.imshow("result", result)
        
        if FLAGS.output:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
    cv2.destroyAllWindows()
    print("Done!!!")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

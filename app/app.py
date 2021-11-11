from flask import Flask, flash, request, redirect, url_for,session
import struct
import glob
import os
import pathlib
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.layers as layers
from sklearn.preprocessing import normalize
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import base64
from io import BytesIO

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

PATH_TO_MODEL_DIR = 'models'
PATH_TO_LABELS = 'label_map.pbtxt'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
UPLOAD_FOLDER = 'outputs'

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  return np.array(Image.open(path))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"
sess = session()

@app.route("/")
def hello_world():
  return "<p>Hello, World!</p>"

@app.route("/upload", methods=['POST'])
def upload_file():
  print('test')
  if request.method == 'POST':
    if 'file' not in request.files:
        flash('No file part')
        return 'invalid file'
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return 'invalid file'
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], '{}'.format(time.time()))
    print('Running inference for {}... '.format(image_path), end='')
    file.save(image_path)
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
        for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=.50,
      agnostic_mode=False)
    plt.figure()
    im = Image.fromarray(image_np_with_detections)
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    image_string = base64.b64encode(buffered.getvalue())
    return "data:image/jpeg;base64," + image_string.decode('utf-8')

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=False)
  sess.init_app(app)
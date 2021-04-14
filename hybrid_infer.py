import os
import pathlib
import time
from math import sqrt
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import cv2
import extcolors
import numpy as np
import tensorflow as tf
from PIL import Image
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def display(img):
    # plt.rcParams["figure.figsize"] = (100, 100)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def load_model():
    model_dir = pathlib.Path(
        "checkpoint/content/table_detect/table_detection/exported-model/mobile-model") / "saved_model"

    model = tf.saved_model.load(str(model_dir))

    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


########################################################################################################################

def _square(x):
    return x * x


def cie94(L1_a1_b1, L2_a2_b2):
    # """Calculate color difference by using CIE94 formulae
    #
    # See http://en.wikipedia.org/wiki/Color_difference or
    # http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html.
    #
    # cie94(rgb2lab((255, 255, 255)), rgb2lab((0, 0, 0)))
    # >>> 58.0
    # cie94(rgb2lab(rgb(0xff0000)), rgb2lab(rgb('#ff0000')))
    # >>> 0.0
    # """

    L1, a1, b1 = L1_a1_b1
    L2, a2, b2 = L2_a2_b2

    C1 = sqrt(_square(a1) + _square(b1))
    C2 = sqrt(_square(a2) + _square(b2))
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_square = _square(delta_a) + _square(delta_b) - _square(delta_C)
    return (sqrt(_square(delta_L)
                 + _square(delta_C) / _square(1.0 + 0.045 * C1)
                 + delta_H_square / _square(1.0 + 0.015 * C1)))


def get_traingle(pts, img):
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y + h, x:x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst
    # display(dst2)
    return dst2


def compute_left_right_trgs_from_table(box_corners, img):
    # Left
    count = 0
    out_triangles_left = []
    prev = box_corners[0][0]
    for i, j in zip(range(box_corners[0][0], box_corners[0][0] + int((box_corners[2][0] - box_corners[0][0]) / 3), 20),
                    [box_corners[1][1]] * 20):
        pts = np.array([box_corners[0], (i, j), [prev, j]], np.int32).reshape((-1, 1, 2))
        dst2 = get_traingle(pts, img)
        out_triangles_left.append({"src": dst2, "c2d": (prev, j)})
        prev = i

    out_triangles_left.pop(0)

    # Right
    count = 0
    out_triangles_right = []
    prev = box_corners[3][0]

    for i, j in zip(range(box_corners[3][0] - int((box_corners[2][0] - box_corners[0][0]) / 3), box_corners[3][0], 20),
                    [box_corners[2][1]] * 20):
        pts = np.array([box_corners[3], (i, j), [prev, j]], np.int32).reshape((-1, 1, 2))
        dst2 = get_traingle(pts, img)

        out_triangles_right.append({"src": dst2, "c2d": (prev, j)})
        prev = i
    out_triangles_right.pop(0)
    out_triangles_right = out_triangles_right[::-1]

    return out_triangles_left, out_triangles_right


def extract_unq_colors(img):
    colors, pixel_count = extcolors.extract_from_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    return colors


def compare_two_triangles(x, y):
    # res = list(set(extract_unq_colors(trg1))^set(extract_unq_colors(trg2)))
    x = extract_unq_colors(x)
    y = extract_unq_colors(y)
    x = list(zip(*x))[0]
    y = list(zip(*y))[0]
    x1 = [i for i in x if i not in y]
    y1 = [i for i in y if i not in x]
    sum_ = []
    for i, j in zip(x1, y1):
        # print(i, j)
        sum_.append(int(cie94(cv2.cvtColor(np.uint8([[i]]), cv2.COLOR_BGR2Lab)[0][0],
                              cv2.cvtColor(np.uint8([[j]]), cv2.COLOR_BGR2Lab)[0][0])))
    return x1, y1, sum(sum_)


def all_traingles(out_traingles=None):
    trgs = []

    for i in range(0, len(out_traingles) - 1):
        try:
            l1, l2, s = compare_two_triangles(out_traingles[i]["src"], out_traingles[i + 1]["src"])
            trgs.append((l1, l2, s, (i, i + 1), (out_traingles[i]["c2d"], out_traingles[i + 1]["c2d"])))
            # display(out_triangles[i])
            # display(out_triangles[i + 1])
            # print("********************************" + str(i) + "--" + str(i + 1) + "********************************")
        except:
            pass
    s_max = sorted(zip(list(zip(*trgs))[2], list(zip(*trgs))[4]), reverse=True)[:3]
    return trgs, s_max


# def get_actual_bb(box_corners, img):
#     out_triangles_left, out_triangles_right = compute_left_right_trgs_from_table(box_corners, img)
#     left_trgs, s_max_left = all_traingles(out_triangles_left)
#     right_trgs, s_max_right = all_traingles(out_triangles_right)
#
#     # Draw Rect
#     bl = box_corners[0]
#     br = box_corners[3]
#     tl = s_max_left[1][1][0]  # Can change this if needed
#     tr = s_max_right[1][1][0]  # Can change this if needed
#
#     # To draw
#     pts = np.array([bl, tl, tr, br], np.int32)
#     pts = pts.reshape((-1, 1, 2))
#     display(cv2.polylines(img, [pts], True, (0, 0, 0)))
#     return [tl, tr, br, bl]


def get_actual_bb(box_corners, img):
    out_triangles_left, out_triangles_right = compute_left_right_trgs_from_table(box_corners, img)
    left_trgs, s_max_left = all_traingles(out_triangles_left)
    right_trgs, s_max_right = all_traingles(out_triangles_right)

    # Draw Rect
    bl = box_corners[0]
    br = box_corners[3]
    tl = s_max_left[2][1][0]  # Can change this if needed
    tr = s_max_right[1][1][0]  # Can change this if needed

    # Distance check
    dist_x = []
    max_x = 0
    max_i = None

    for i in s_max_left:
        for j in s_max_right:
            dist_x.append({(j[1][0][0] - i[1][0][0]): (i[1][0][0], j[1][0][0])})

            if (j[1][0][0] - i[1][0][0]) > max_x:
                max_x = (j[1][0][0] - i[1][0][0])
                max_i = (i[1][0][0], j[1][0][0])

    # for p in dist_x:
    #     tl = (list(p.values())[0][0], box_corners[1][1])
    #     tr = (list(p.values())[0][1], box_corners[2][1])
    #     # To draw
    #     pts = np.array([bl, tl, tr, br], np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     cv2.polylines(img, [pts], True, (0, 0, 0))
    # display(img)
    tl = (max_i[0], box_corners[1][1])
    tr = (max_i[1], box_corners[2][1])

    # To draw
    pts = np.array([bl, tl, tr, br], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (255, 255, 0),thickness=2)
    # display(cv2.polylines(img, [pts], True, (0, 0, 0)))
    return [bl, tl, tr, br]


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'training/object-detection.pbxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with
# images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('images/validation')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
print(TEST_IMAGE_PATHS)

detection_model = load_model()

print(detection_model.signatures['serving_default'].inputs)
print(detection_model.signatures['serving_default'].output_dtypes)
print(detection_model.signatures['serving_default'].output_shapes)
for image_path in TEST_IMAGE_PATHS:
    try:
        st = time.time()
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        img = np.array(Image.open(image_path))
        # Actual detection.
        output_dict = run_inference_for_single_image(detection_model, img)

        # print(output_dict['detection_boxes'])
        rect_box = output_dict['detection_boxes'][0] * np.array(
            [img.shape[0], img.shape[1], img.shape[0], img.shape[1]])
        ymin, xmin, ymax, xmax = rect_box[0], rect_box[1], rect_box[2], rect_box[3]
        box_corners = [(int(xmin), int(ymax)), (int(xmin), int(ymin)), (int(xmax), int(ymin)),
                       (int(xmax), int(ymax))]  # bl, tl, tr, br
        initial_box = [(int(xmin), int(ymin)), (int(xmax), int(ymin)),
                       (int(xmax), int(ymax)), (int(xmin), int(ymax))]  # tl, tr, br,bl
        act_box_corners = get_actual_bb(box_corners, img)

        cv2.imwrite('images/results/' + str(image_path).split("/")[-1], cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("Initial  = ", initial_box)

        print("Corniates = ", act_box_corners)
        print(
            "############################################################################################################")
    except Exception as e:
        pass

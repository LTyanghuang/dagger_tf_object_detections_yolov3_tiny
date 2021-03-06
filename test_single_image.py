####

# edicted by Huangdebo
# test the model using ckpt file

# ***

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model.yolov3 import yolov3
from model.yolov3_tiny import yolov3_tiny

net_name = 'yolov3_tiny'
body_name = 'darknet19'
data_name = 'widerface'
ckpt_name = 'model-step_40000_loss_0.013842_lr_0.0002035062'


parser = argparse.ArgumentParser(description="%s test single image test procedure."%net_name)
parser.add_argument("--input_image", type=str, default="/home/ubuntu/huangyang/t4_spacework/lt_code_sdk/solution/Ultra-Light-Fast-Generic-Face-Detector-1MB/imgs/19.jpg",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/%s_%s_anchors.txt"%(net_name,data_name),
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[224, 224],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/%s.names"%data_name,
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./checkpoint/%s_%s/%s"%(net_name,data_name,ckpt_name),
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
img_ori = cv2.imread(args.input_image)
height_ori, width_ori = img_ori.shape[:2]
img = cv2.resize(img_ori, tuple(args.new_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
#    yolo_model = yolov3(args.num_class, args.anchors)
    yolo_model = yolov3_tiny(args.num_class, args.anchors)
    with tf.variable_scope(net_name):
        pred_feature_maps = yolo_model.forward(input_data, False)
        ##########saved model start############
        # saver = tf.train.Saver()
        # saver.restore(sess, args.restore_path)
        # outs = {'pred_feature_maps0':pred_feature_maps[0], 'pred_feature_maps1':pred_feature_maps[1]}

        # tf.saved_model.simple_save(sess, "./yolov3_tiny_savedmodel",inputs={'input_image':input_data}, 
        #                             outputs={'pred_feature_maps0':outs['pred_feature_maps0'], 
        #                                     'pred_feature_maps1':outs['pred_feature_maps1']}) 
        # print("saved model end ........")
        ##########saved model end##############
        
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.4, iou_thresh=0.5)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
    boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
    boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

    print("box coords:")
    print(boxes_)
    print('*' * 30)
    print("scores:")
    print(scores_)
    print('*' * 30)
    print("labels:")
    print(labels_)

    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
    # cv2.imshow('Detection result', img_ori)
    cv2.imwrite('detection_result.jpg', img_ori)
    # cv2.waitKey(0)

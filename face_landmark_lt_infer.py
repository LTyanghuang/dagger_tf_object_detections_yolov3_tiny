from numpy.lib.arraysetops import isin
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import lt_sdk as lt
from lt_sdk.proto import hardware_configs_pb2, graph_types_pb2
from lt_sdk.graph.transform_graph import utils as lt_utils
# import tools
import cv2
import os


config = lt.get_default_config(hw_cfg=hardware_configs_pb2.DAGGER,graph_type=graph_types_pb2.TFSavedModel)
input_name = 'input'
batch_size = 1

# read picture
def read_img_input(img_ori, width, height):
    input_name = 'input'
    if img_ori is None:
        return None, None
    img = cv2.resize(img_ori, (width, height))
    img = img.astype(np.float32)
    img = img / 255.0
    # [416, 416, 3] => [1, 416, 416, 3]
    img = np.expand_dims(img, 0)
    # print("img shape = ", img.shape)
    named_tensor = lt.data.named_tensor.NamedTensorSet([input_name], [img])
    batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)
    return batch_input

def Coordinate_landmarks_map(landmarks, bboxs):
    x0, y0, x1, y1 = bboxs
    w = x1 - x0
    h = y1 - y0
    landmark_map_res = []
    for i in range(int(landmarks.shape[1]/2)):
        point = [landmarks[0][i*2+0]*112, landmarks[0][i*2+1]*112]
        point1 = [point[0]*w/112, point[1]*h/112]
        point2 = [point1[0]+x0, point1[1]+y0]
        landmark_map_res.append(point2[0])
        landmark_map_res.append(point2[1])
    res = np.array(landmark_map_res)
    res = np.expand_dims(res, 0)
    return res



def pfld_infer_process(light_graph, calibration_data, config):
    outputs = lt.run_functional_simulation(light_graph, calibration_data, config)
    landmarks_res = []
    theta_res = []
    for inf_out in outputs.batches:
        for named_ten in inf_out.results:
            if named_ten.edge_info.name.startswith("PFLD_Netework/MS-FC/landmark_3"):   #输出节点名0
                landmarks = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                landmarks_res.append(landmarks.numpy())
            if named_ten.edge_info.name.startswith("PFLD_Netework/Fc2/pre_theta"):   #输出节点名0
                theta = tf.convert_to_tensor(lt_utils.tensor_pb_to_array(named_ten.data,np.float32))
                theta_res.append(theta.numpy())
    return landmarks_res, theta_res, light_graph
    
if __name__ == '__main__':
    lt_graph_save_path = 'lt_graph_path/pfld_lgf_end2.pb' 
    input_graph = lt.import_graph(lt_graph_save_path, config)

    count = 0
    for i in range(5000):
        img_name = '../images_gt/'+str(i+3001)+'.jpg'
        width = 112
        height = 112
        img, nw, nh, img_ori, show_img = read_img(img_name, width, height)
        named_tensor = lt.data.named_tensor.NamedTensorSet([input_name], [img])
        batch_input = lt.data.batch.batch_inputs(named_tensor, batch_size)

        outs = pfld_infer_process(input_graph, batch_input, config)
        landmarks = outs[0][0]
        theatas = outs[1][0]
        print("landmarks shape = ", landmarks.shape)
        print("theatas shape = ", theatas.shape)
        count += 1
        # print("************* finished ************ : {}/{}".format(count, 2000))
        # img_ori = tools.point_img(img_ori, landmarks, theatas)
        # cv2.imwrite(os.path.join('results', str(i+3001)+'.jpg'), img_ori)   
        
        
# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""
import cv2
import colorsys
import os
from timeit import default_timer as timer
from tools.utils import *

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class BoundingBox:
    def __init__(self, predicted_class, score, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score
        self.predicted_class = predicted_class
        self._2DpointsId = []
        
        self.dict_2Dpoints_FOV_Box = {}

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    import cv2
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2,  
                   (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)        

        
def get_2DpointsId_in_box(box, dict_2Dpoints_FOV):
    for inx, iid in enumerate(dict_2Dpoints_FOV):
            p = dict_2Dpoints_FOV[iid][0] # [[x, y]], p = [x, y]
            #print(p)
            # print(type(p))
            if point_in_bbox_or_not(p, box):
                    box._2DpointsId.append(iid)
    return box

class YOLO(object):
    _defaults = {
        "model_path": '/home/kk/Downloads/yolo.h5', #'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    
    def detect_image_v2(self, frame, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints):
        """
        output: infomation of bboxes and labels
        """
        image = Image.fromarray(frame)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
           
        return out_boxes, out_scores, out_classes
    
    def origin_image_add_lidar_and_bbox(self, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints, c_, frame, out_boxes, out_scores, out_classes):
        image_lidar = print_projection_plt(points=arr_2Dpoints_xyzri_, color=c_, image=frame)
        image = Image.fromarray(frame)
        image_lidar = Image.fromarray(image_lidar)
        # Format of label
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        
        # loop for bbox
        divid_bins = 2 
        n_points_threshold = 500
        global dist_size
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
              
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
             
                
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            # new
            bbox = BoundingBox(predicted_class, score, left, top, right-left, bottom-top)
           
            
            box = test_get(bbox, dict_2Dpoints_FOV)
            ### print(label, (left, top), (right, bottom))

            dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV = get_dict(dict_3Dpoints, arr_2Dpoints_xyzri_)
            
            arr_2Dpoints_Box, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box = get_point_in_box(box, dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV)
            
            

            
            arr_top = get_uv_dist_from_box(box, dict_3Dpoints, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box, divid_bins, n_points_threshold)
            
            #uc , vc, xmid, ymid = estimate_shape_center(box, dist, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box)
            
            text = 'dist'
            #text = text.encode('utf-8')
            
            ### print('len(arr_top): ', len(arr_top))
            if len(arr_top) == 0:
                dist_label = '{} {}'.format(text, 'NA') 
            else:      
                if type(arr_top[0]) == type('string') and len(arr_top)!= 0:
                    dist_label = '{} {}'.format(text, arr_top[0])    
                else:
                    dist_label = '{} {:.2f}'.format(text, arr_top[0])
                    
            draw = ImageDraw.Draw(image_lidar)
            dist_label_size = draw.textsize(dist_label, font)
        
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            text_origin2 =  np.array([left, top])   
                
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size + (0, 14))],
                fill=self.colors[c])
            
            
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            
         
            draw.text(text_origin2, dist_label, fill=(0, 0, 0), font=font)
            del draw
        
        return image_lidar
    
    
    
    def detect_image(self, image, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        ### print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        

        divid_bins = 2 
        n_points_threshold = 500
        global dist_size
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
              
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
             
                
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            #print(label, (left, top), (right, bottom))

            # new
            bbox = BoundingBox(predicted_class, score, left, top, right-left, bottom-top)
           
            
            box = test_get(bbox, dict_2Dpoints_FOV)
            ### print(label, (left, top), (right, bottom))

            dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV = get_dict(dict_3Dpoints, arr_2Dpoints_xyzri_)
            
            arr_2Dpoints_Box, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box = get_point_in_box(box, dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV)
            
            

            
            arr_top = get_uv_dist_from_box(box, dict_3Dpoints, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box, divid_bins, n_points_threshold)
            
            #uc , vc, xmid, ymid = estimate_shape_center(box, dist, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box)
            
            text = 'dist'
            #text = text.encode('utf-8')
            
            ### print('len(arr_top): ', len(arr_top))
            if len(arr_top) == 0:
                dist_label = '{} {}'.format(text, 'NA') 
            else:      
                if type(arr_top[0]) == type('string') and len(arr_top)!= 0:
                    dist_label = '{} {}'.format(text, arr_top[0])    
                else:
                    dist_label = '{} {:.2f}'.format(text, arr_top[0])
                    
            draw = ImageDraw.Draw(image)
            dist_label_size = draw.textsize(dist_label, font)
            
#             if len(arr_top) == 0:
#                print('arr_top vanished.')
#             else:    
#                 if arr_top[0] is not 0.0 :
#                     print('distance: ', arr_top[0])
#                 else:
#                     print('The number of 3D points is not enough.')
            
            ###

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            text_origin2 =  np.array([left, top])   
                
            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size + (0, 14))],
                fill=self.colors[c])
            
            
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            
         
            draw.text(text_origin2, dist_label, fill=(0, 0, 0), font=font)
            del draw
            
            

        end = timer()
        print(end - start)
        return image

    #def detect_distance(self, image):
	#pass
        #return image

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    print('FPS: ', video_fps)
    slow_video_fps = 15
    print('Slow down video display, slow_video_fps: ', slow_video_fps)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, slow_video_fps, video_size) # slow_video_fps = 3 here, to slow down the speed of video.
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
    image_type = 'color' # 'grayscale' or 'color' image
    mode = '00' if image_type == 'grayscale' else '02'
    v2c_filepath = '/home/kk/Downloads/2011_09_26_calib/2011_09_26/calib_velo_to_cam.txt'
    c2c_filepath = '/home/kk/Downloads/2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt'
    dict_3Dpoints = {}
    cnt = 0
    while True:
        return_value, frame = vid.read()
        frame_copy = frame
        
        if not return_value:
              break
        #image = Image.fromarray(frame)
        
        ###
        name = '0000000'+ "{:03}".format(cnt)
        name = name + '.bin'
        path =  '/home/kk/Downloads/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/'

        velo_points = load_from_bin(path + 'velodyne_points/data/' + name)
        velo_points_with_ref = load_from_bin_with_reflect(path + 'velodyne_points/data/' + name) 

        velo_points_with_ref_id = add_velo_points_array_with_id(velo_points_with_ref)

        
        
        arr_2Dpoints_xyz_, c_ , arr_2Dpoints_xyzri_= velo3d_2_camera2d_points(velo_points_with_ref_id, v_fov=(-24.9, 2.0), h_fov=(-45,45), vc_path=v2c_filepath, cc_path=c2c_filepath, mode=mode)

        
#         ans, c_, ans_ri= velo3d_2_camera2d_points(velo_points_with_ref_id, v_fov=(-24.9, 2.0), h_fov=(-45,45), \
#                                vc_path=v2c_filepath, cc_path=c2c_filepath, mode=mode)
    
        #image = print_projection_plt(points=arr_2Dpoints_xyzri_, color=c_, image=frame)
        image = Image.fromarray(frame)
        dict_3Dpoints = getDict3DPoints(velo_points_with_ref_id)
        dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV = get_dict(dict_3Dpoints, arr_2Dpoints_xyzri_) 
        ###
        
        
#         image = yolo.detect_image(image, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints)
#         image_arr = np.asarray(image)
#         image = print_projection_plt(points=arr_2Dpoints_xyzri_, color=c_, image=image_arr)

        out_boxes, out_scores, out_classes = yolo.detect_image_v2(frame, dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints)
        
        image = yolo.origin_image_add_lidar_and_bbox(dict_2Dpoints_FOV, arr_2Dpoints_xyzri_, dict_3Dpoints, c_ , frame, out_boxes, out_scores, out_classes)
    
    
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        cv2.waitKey(2000)
        if isOutput:
            out.write(result)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        cnt += 1
    yolo.close_session()


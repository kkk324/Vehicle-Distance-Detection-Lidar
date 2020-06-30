import numpy as np
import matplotlib.pyplot as plt

def get_dict(dict_3Dpoints, arr_2Dpoints_xyzri_):
    dict_r_FOV = {}
    dict_dist_FOV = {}
    dict_2Dpoints_FOV = {} 
    id_list = []
    id_list = arr_2Dpoints_xyzri_[5]
    for idx, iid in enumerate(id_list):
        dict_2Dpoints_FOV[iid] = list()
        dict_dist_FOV[iid] = list()
        dict_r_FOV[iid] = list()
        dict_2Dpoints_FOV[iid].append([arr_2Dpoints_xyzri_[0, idx], arr_2Dpoints_xyzri_[1, idx]])

        dict_dist_FOV[iid].append([dict_3Dpoints[iid][4]])
        #print([dict_3Dpoints[iid][3]])
        dict_r_FOV[iid].append([dict_3Dpoints[iid][3]])
    
    return dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV


def test_get(box, dict_2Dpoints_FOV):
    #print('get_2DpointsId_in_box')
    for inx, iid in enumerate(dict_2Dpoints_FOV):
        p = dict_2Dpoints_FOV[iid][0] # [[x, y]], p = [x, y]
#         print('p:   ', p)
#         print('box.x, box.y, box.w, box.h', box.x, box.y, box.w, box.h)
        #print(type(p))
        #print('point_in_bbox_or_not(p, box): ', point_in_bbox_or_not(p, box))
        if point_in_bbox_or_not(p, box):
                box._2DpointsId.append(iid)
    return box

def get_2DpointsId_in_box(box, dict_2Dpoints_FOV):
    #print('get_2DpointsId_in_box')
#     for inx, iid in enumerate(dict_2Dpoints_FOV):
#             p = dict_2Dpoints_FOV[iid][0] # [[x, y]], p = [x, y]
#             print(p)
#             print(type(p))
#             print('point_in_bbox_or_not(p, box): ', point_in_bbox_or_not(p, box))
#             if point_in_bbox_or_not(p, box):
#                     box._2DpointsId.append(iid)
    return box

def get_point_in_box(box, dict_2Dpoints_FOV, dict_dist_FOV, dict_r_FOV):
    # dict_2Dpoints_FOV_Box  key:id , value:(u, v) 
    
#     print('--- Inside get point in box ---')
#     print('len(box._2DpointsId)', len(box._2DpointsId))
#     print('--- Inside get point in box ---')
    
    cnt = 0
    dict_2Dpoints_FOV_Box = {}
    for iid in box._2DpointsId:
        #print(iid)
        cnt = cnt + 1
        dict_2Dpoints_FOV_Box[int(iid)] = dict_2Dpoints_FOV[iid]
    #print((dict_2Dpoints_FOV_Box))

    #dict_dist_FOV_Box
    dict_dist_FOV_Box = {}
    for iid in box._2DpointsId:
        dict_dist_FOV_Box[iid] = dict_dist_FOV[iid]
    #print('len(dict_dist_FOV_Box): ', len(dict_dist_FOV_Box))

    #dict_r_FOV_Box
    dict_r_FOV_Box = {}
    for iid in box._2DpointsId:
        dict_r_FOV_Box[iid] = dict_r_FOV[iid]
    #print('len(dict_r_FOV_Box):', len(dict_r_FOV_Box))

    list_2Dpoints_Box = []
    list_dist_Box     = []
    list_r_Box        = []
    for iid in box._2DpointsId:
        list_2Dpoints_Box.append(dict_2Dpoints_FOV_Box[iid])
        list_dist_Box.append(dict_dist_FOV_Box[iid])
        list_r_Box.append(dict_r_FOV_Box[iid])

    arr_2Dpoints_Box = np.array(list_2Dpoints_Box)
    arr_dist_Box     = np.array(list_dist_Box)
    arr_r_Box        = np.array(list_r_Box) 
    #len(arr_2Dpoints_Box)

    n_points = len(arr_2Dpoints_Box)
    #print('n_points: ', n_points)
    
    return arr_2Dpoints_Box, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box

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

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]

def load_from_bin_with_reflect(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # Do NOT ignore reflectivity info
    return obj[:,:4]

def add_velo_points_array_with_id(arr_points):
    list_v = list(arr_points)
    i = 0
    list_v_id=[]
    for v in list_v:
        v=np.append(v, i)
        list_v_id.append(v)
        i=i+1
    arr_points_id = np.array(list_v_id).astype(np.float32)
    return arr_points_id


# We can get ANY things of 3D points (x, y, z, r, dist) if you give me id of 2D point
# dict_bbox[id] = (x, y, z, r, dist)
# dict_bboxes[object_id] = dict_bbox
def getDict3DPoints(arr_velo_points_r_id):
    dict_3Dpoints = {}
    for i, v in np.ndenumerate(arr_velo_points_r_id):
        if i[1] == 4 :
            arr = []
            for n in range(5):
                if   n == 0:
                    x = arr_velo_points_r_id[(i[0] , n)]
                    arr.append(x)
                elif n == 1:
                    y = arr_velo_points_r_id[(i[0] , n)]
                    arr.append(y)
                elif n == 2:
                    z = arr_velo_points_r_id[(i[0] , n)]
                    arr.append(z)
                elif n == 3:
                    r = arr_velo_points_r_id[(i[0] , n)]
                    arr.append(r) 
                else:
                    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    arr.append(dist)
            dict_3Dpoints[arr_velo_points_r_id[i]] = arr
    return dict_3Dpoints

def depth_color(val, min_d=0, max_d=120):
    """ 
    print Color(HSV's H value) corresponding to distance(m) 
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val) # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8) 

def in_h_range_points(points, m, n, fov):
    """ extract horizontal in-range points """
    return np.logical_and(np.arctan2(n,m) > (-fov[1] * np.pi / 180), \
                          np.arctan2(n,m) < (-fov[0] * np.pi / 180))

def in_v_range_points(points, m, n, fov):
    """ extract vertical in-range points """
    return np.logical_and(np.arctan2(n,m) < (fov[1] * np.pi / 180), \
                          np.arctan2(n,m) > (fov[0] * np.pi / 180))

def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """ filter points based on h,v FOV  """
    
    if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points
    
    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:        
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
    return points[np.logical_and(h_points, v_points)]

def in_range_points(points, size):
    """ extract in-range points """
    return np.logical_and(points > 0, points < size)    

def velo_points_filter(points, v_fov, h_fov):
    """ extract points corresponding to FOV setting """
    
    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
    
    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]
    print('x_lim', x_lim)
    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat),axis = 0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    color = depth_color(dist_lim, 0, 70)
    
    return xyz_, color

def velo_points_filter_id(points_id, v_fov, h_fov):
    """ extract points corresponding to FOV setting """
    
    # Projecting to 2D
    x = points_id[:, 0]
    y = points_id[:, 1]
    z = points_id[:, 2]
    r = points_id[:, 3]
    ids = points_id[:, 4]

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)
    
    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:,None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:,None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:,None]
    r_lim = fov_setting(r, x, y, z, dist, h_fov, v_fov)[:,None]
    ids_lim = fov_setting(ids, x, y, z, dist, h_fov, v_fov)[:,None]
    
    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T
    
    xyzri_ = np.hstack((x_lim, y_lim, z_lim))
    xyzri_ = xyzri_.T
    
    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    
    xyz_   = np.concatenate((xyz_, one_mat)  ,axis = 0)
    xyzri_ = np.concatenate((xyzri_, one_mat),axis = 0)
    xyzri_ = np.concatenate((xyzri_, r_lim.T),axis = 0)  
    xyzri_ = np.concatenate((xyzri_, ids_lim.T),axis = 0)
    
    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    #print('dist_lim', dist_lim)
    color = depth_color(dist_lim, 0, 70)
    
    return xyz_, color, xyzri_ 

def calib_velo2cam(filepath):
    """ 
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info 
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()    
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(filepath, mode):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)
        
    in this code, I'll get P matrix since I'm using rectified image
    """
    with open(filepath, "r") as f:
        file = f.readlines()
        
        for line in file:
            (key, val) = line.split(':',1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_
 
def velo3d_2_camera2d_points(points, v_fov, h_fov, vc_path, cc_path, mode='02'):
    """ print velodyne 3D points corresponding to camera 2D image """
    
    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = calib_velo2cam(vc_path)
    
    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = calib_cam2cam(cc_path, mode)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)
    
             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """  
    #xyz_v, c_ = velo_points_filter(points, v_fov, h_fov)
    xyz_v, c_, xyzri_v = velo_points_filter_id(points, v_fov, h_fov)
    #print(xyzri_v.shape)

    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )
    
            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]   
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc),axis = 1)
    
    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c) 
    for i in range(xyz_v.shape[1]):
        xyz_v[:3,i]    = np.matmul(RT_, xyz_v[:4,i])
        xyzri_v[:3, i] = np.matmul(RT_, xyzri_v[:4,i])
    #print(xyzri_v.shape)
    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]   
             [z_1 , z_2 , .. ]
    """ 
    xyz_c = np.delete(xyz_v, 3, axis=0)
     
    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y) 
    for i in range(xyz_c.shape[1]):
        xyz_c[:,i]    = np.matmul(P_, xyz_c[:,i])    
        xyzri_v[:3,i] = np.matmul(P_, xyzri_v[:3,i])
    #print(xyzri_v.shape)
    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]  
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::]/xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)
    
    xyri_i = xyzri_v[:2]/xyzri_v[::][2]
    xyzri_v[:2] = xyri_i
    ans_ri = xyzri_v 
    #print(ans_ri.shape)

    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """
    
    return ans, c_, ans_ri

def point_in_bbox_or_not(point, bbox):
#     print('bbox.x: ', bbox.x)
#     print('point[0]: ', point[0])
    
    if bbox.x < point[0] < bbox.x + bbox.w and \
       bbox.y < point[1] < bbox.y + bbox.h:
           return True  
    else:
           return False        
        
def check_X_std(arr_d, uvd_list):
    X_list = []
    for u,v,d in uvd_list:
        if abs(d - arr_d) < 0.1 :
                X_list.append(u)       
    X = np.array(X_list) 
    return X.std()
        
def get_topk_dist(num, n, patches, uvd_list):
    
    topk = n.argsort()[-num:][::-1]
    arr_top = np.zeros(shape=(1))
    num_small_std = 0
    for i in range(num):
        p = patches[topk[i]].xy[0]
        a = np.array([p])
        
        # issue here
        if check_X_std(a, uvd_list) < 14:
#             print('check_X_std(a, uvd_list)', check_X_std(a, uvd_list)) 
#             print('a', a)
            arr_top = np.append(arr_top, a)
            num_small_std = num_small_std + 1
   
    arr_top = np.delete(arr_top, 0)    
    return arr_top, num_small_std

def check_shape(box, arr, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box):
    is_target = False
    list_xtop = []
    list_ytop = []
    for iid in box._2DpointsId:
                f = dict_dist_FOV_Box[iid][0][0]
                print('arr', arr)
                print('f', f)
                print(abs(f - arr))
                if abs(f - arr) < 2.0:
                    xtop = np.int32(dict_2Dpoints_FOV_Box[iid][0][0])
                    ytop = np.int32(dict_2Dpoints_FOV_Box[iid][0][1])
                    list_xtop.append(xtop)
                    list_ytop.append(ytop)
    
    arr_xtop = np.array(list_xtop)
    arr_ytop = np.array(list_ytop)
    xmax = np.max(arr_xtop) 
    xmin = np.min(arr_xtop)
    ymax = np.max(arr_ytop)
    ymin = np.min(arr_ytop)
    
    del_x = xmax - xmin
    del_y = ymax - ymin
#     print('---')
#     print(del_x)
#     print(del_y)
    if del_x < box.w/3 and del_y > box.h/3:
        is_target = True 
    
    return is_target

def estimate_shape_center(box, d, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box):
    list_xtop = []
    list_ytop = []
    for iid in box._2DpointsId:
        f = dict_dist_FOV_Box[iid][0][0]
        if abs(f - d) < 2.0:
                xtop = np.int32(dict_2Dpoints_FOV_Box[iid][0][0])
                ytop = np.int32(dict_2Dpoints_FOV_Box[iid][0][1])
                list_xtop.append(xtop)
                list_ytop.append(ytop)

    arr_xtop = np.array(list_xtop)
    arr_ytop = np.array(list_ytop)
    xmax = np.max(arr_xtop) 
    xmin = np.min(arr_xtop)
    ymax = np.max(arr_ytop)
    ymin = np.min(arr_ytop)

    xmid = 0.5 * (xmax - xmin)
    ymid = 0.5 * (ymax - ymin)
#     print('xmid', xmid )
#     print('ymid', ymid )
    uc = xmin + xmid
    vc = ymin + ymid
    print('uc', uc )
    print('vc', vc )
    print('d', d)
    return uc, vc, xmid, ymid

# Function get_dist_from_box
# input   : box, dict_3Dpoints # (x, y, z, r, dist)
# output  : distance estimated
# uvd_list (u, v, dist)
# uvd_dict key: dist, value: (u, v)
def get_uv_dist_from_box(box, dict_3Dpoints, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box, divid_bins, n_points_threshold):
    
    uv_list   = []
    uvd_list  = []
    dist_list = []
    r_list    = []
    
    n_points = len(dict_2Dpoints_FOV_Box)
    #print('n_points', n_points)
    ### check n_points is enough 
    global arr_top 
    global label_size
    arr_top = np.zeros(shape=(1))
    
    if n_points > n_points_threshold:
            #print('n_points > n_points_threshold')
            for iid in box._2DpointsId:
                temp_uvd = []
                #print((dict_2Dpoints_FOV_Box[iid][0]))
                #print(dict_3Dpoints[iid][4])
                temp_uvd = [dict_2Dpoints_FOV_Box[iid][0][0], dict_2Dpoints_FOV_Box[iid][0][1],dict_3Dpoints[iid][4]]
                #print(temp_uvd)
                uvd_list.append(temp_uvd)
                dist_list.append(dict_3Dpoints[iid][4])
                r_list.append(dict_3Dpoints[iid][3])

            dist_arr = np.array(dist_list)
            uvd_arr = np.array(uvd_list)
            n, bins, patches = plt.hist(x=dist_arr, bins=int(n_points/divid_bins), color='#0504aa', alpha=0.7, rwidth=0.85)
            plt.xlabel('Distance (meter)')
            plt.ylabel('Frequency')
            plt.title('Lidar Histogram')
            #print(patches[np.argmax(n)].xy[0])
            #topk = n.argsort()[-5:][::-1]
            #top1_dist = patches[topk[0]].xy[0]
            #top2_dist = patches[topk[1]].xy[0]
            #top3_dist = patches[topk[2]].xy[0]
            #top4_dist = patches[topk[3]].xy[0]
            #top5_dist = patches[topk[4]].xy[0]
            #arr_top = np.array([top1_dist, top2_dist, top3_dist, top4_dist, top5_dist])
            
            ### check check_X_std
            num = 100
           
            arr_top, num_small_std = get_topk_dist(num, n, patches, uvd_list)
            return arr_top
    else:
            
            
#             import sys
#             sys.exit()
#             print('FUCK')
            
#             print('check_shape', check_shape(box, arr_top, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box))
#             print('arr_top', arr_top)
#             import sys
#             sys.exit()
#             print('FUCK')
#             #print('num_small_std', num_small_std)
#             #print('arr_top', arr_top)
#             #print('len(arr_top)', len(arr_top))
#             ### check shape 
#             print('---check shape---')
#             list_top_new = []
#             for i, a in enumerate(arr_top):
#                 print('a', a)
#                 if check_shape(box, a, dict_2Dpoints_FOV_Box, dict_dist_FOV_Box):
#                     #print('True')
#                     list_top_new.append(a)
#                 else:
#                     #print('Delete')
#                     pass
                
#             arr_top_new = np.asarray(list_top_new)
#             print('arr_top_new', arr_top_new)
#             print('len(arr_top_new)', len(arr_top_new))
            
# #             for i in range(num_small_std):
# #                  print('top'+str(i+1)+'_dist', arr_top[i])
            
#             for i in range(len(arr_top_new)):
#                  print('top'+str(i+1)+'_dist', arr_top_new[i])
            
            
# #             print('top1_dist', (top1_dist))
# #             print('top2_dist', (top2_dist))
# #             print('top3_dist', (top3_dist))
# #             print('top4_dist', (top4_dist))
# #             print('top5_dist', (top5_dist))
#             #dist = patches[np.argmax(n)].xy[0]
#             dist = arr_top_new[0]
    
            return ['NA'] #arr_top, arr_top[0], uvd_list#arr_top, dist, uvd_list       

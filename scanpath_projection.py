from math import pi
import numpy as np
import json

def read_json_file(filename):
    data = []
    with open(filename, 'r') as f:
        data = [json.loads(_.replace('}]}"},', '}]}"}')) for _ in f.readlines()]
    return data

def get_scanpath(dict_path: str, img_path: str, scanpath_id=1):
    img = img_path.split('\\')[-1]  # Extracting file name (for windows file system)
    img_list = read_json_file(dict_path) 
    for img_sps_list in img_list:
        for img_sps in img_sps_list:
            for img_sp in img_sps:
                # print("IMAGE PATH", img_sp['file'], img)
                # print("CRITERIA 1", img_sp['file'] == img)
                # print("CRITERIA 2", img_sp['scanpath_id'] == scanpath_id)
                # print("RULE", img_sp['file'] == img and img_sp['scanpath_id'] == scanpath_id)
                if img_sp['file'] == img and img_sp['scanpath_id'] == scanpath_id: 
                    posx = img_sp['theta']
                    posy = img_sp['phi']
                    # print('x', img_sp['x'])
                    # print('phi', img_sp['phi'])
                    return posx, posy


def draw_NFOV_edges(image, sc_grid, label_color= [0,0,255], save=False):
    import cv2
    '''
    Get edges in clockwise order 
    Reference: https://stackoverflow.com/questions/41200719/how-to-get-all-array-edges
    Requisito: extrair os elementos de fronteira do grid em formato np.int32(N,2)
    '''
    THICKNESS = 5 
    ISCLOSED = False
    equi_height = image.shape[0]
    equi_width = image.shape[1]

    #Get edges from grid in clock-wise order and with border pixels repetition
    sph_coord_row1 = sc_grid[0,::-1][:]
    sph_coord_row2 = sc_grid[-1,::-1][:]
    sph_coord_col1 = sc_grid[::-1,0][:]
    sph_coord_col2 = sc_grid[::-1,-1][:]

    sph_coord_row1 = np.array([sph_coord_row1[:,0]*equi_width,
                sph_coord_row1[:,1]*equi_height]).astype(np.int32)
    
    sph_coord_row2 = np.array([sph_coord_row2[:,0]*equi_width,
                sph_coord_row2[:,1]*equi_height]).astype(np.int32)

    sph_coord_col1 = np.array([sph_coord_col1[:,0]*equi_width,
                sph_coord_col1[:,1]*equi_height]).astype(np.int32)

    sph_coord_col2 = np.array([sph_coord_col2[:,0]*equi_width,
                sph_coord_col2[:,1]*equi_height]).astype(np.int32)

    #check the fov
    sph_coord_row1_r = []
    sph_coord_row1_l = []
    sph_coord_row1_n = []
    sph_coord_row2_r = []
    sph_coord_row2_l = []
    sph_coord_row2_n = []
    sph_coord_col1_r = []
    sph_coord_col1_l = []
    sph_coord_col1_n = []
    sph_coord_col2_r = []
    sph_coord_col2_l = []
    sph_coord_col2_n = []

    for aux in sph_coord_row1.T:
        if(aux[0]> equi_width ):
            sph_coord_row1_r.append([aux[0] - equi_width , aux [1]])
        elif (aux[0]< 0):
            sph_coord_row1_l.append([aux[0] + equi_width , aux [1]])
        else :
            sph_coord_row1_n.append([aux[0] , aux [1]])
            
    for aux in sph_coord_row2.T:
        if(aux[0] > equi_width ):
            sph_coord_row2_r.append([aux[0] - equi_width , aux [1]])
        elif (aux[0] < 0):
            sph_coord_row2_l.append([aux[0] + equi_width , aux [1]])
        else:
            sph_coord_row2_n.append([aux[0] , aux [1]])

    for aux in sph_coord_col1.T:
        if(aux[0] > equi_width):
            sph_coord_col1_r.append([aux[0] - equi_width , aux [1]])
        elif (aux[0] < 0):
            sph_coord_col1_l.append([aux[0] + equi_width , aux [1]])
        else :
            sph_coord_col1_n.append([aux[0] , aux [1]])

    for aux in sph_coord_col2.T:
        if(aux[0] > equi_width):
            sph_coord_col2_r.append([aux[0] - equi_width , aux [1]])
        elif (aux[0] < 0):
            sph_coord_col2_l.append([aux[0] + equi_width , aux [1]])
        else :
            sph_coord_col2_n.append([aux[0] , aux [1]])


    if sph_coord_row1_r:
        cv2.polylines(image,
                      [np.array(sph_coord_row1_r)],
                      isClosed=ISCLOSED,
                      color=label_color,thickness=THICKNESS)
    if sph_coord_row1_l:
        cv2.polylines(image,[np.array(sph_coord_row1_l)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_row1_n:
        cv2.polylines(image,[np.array(sph_coord_row1_n)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    
    if sph_coord_row2_r:
        cv2.polylines(image,[np.array(sph_coord_row2_r)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_row2_l:
        cv2.polylines(image,[np.array(sph_coord_row2_l)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_row2_n:
        cv2.polylines(image,[np.array(sph_coord_row2_n)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)

    if sph_coord_col1_r:
        cv2.polylines(image,[np.array(sph_coord_col1_r)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_col1_l:
        cv2.polylines(image,[np.array(sph_coord_col1_l)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_col1_n:
        cv2.polylines(image,[np.array(sph_coord_col1_n)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)

    if sph_coord_col2_r:
        cv2.polylines(image,[np.array(sph_coord_col2_r)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_col2_l:
        cv2.polylines(image,[np.array(sph_coord_col2_l)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    if sph_coord_col2_n:
        cv2.polylines(image,[np.array(sph_coord_col2_n)],isClosed=ISCLOSED,color=label_color,thickness=THICKNESS)
    
    image = cv2.circle(image, (int(sc_grid.shape[0]/2), int(sc_grid.shape[1]/2)), radius=10, color=(0, 255, 0), thickness=-1)

    if save:
        save_image('test/equi_drawn.jpg',image)
    return image

def get_coord_rad(nfov_width, nfov_height):
    fov_scale = [0.30,0.60]
    screen_points = get_nfov_grid(nfov_width, nfov_height)
    return (screen_points * 2 - 1) * np.array([pi, pi*0.5]) * (np.ones(screen_points.shape) * fov_scale) 

def get_nfov_grid(nfov_width,nfov_height):
    xx, yy = np.meshgrid(np.linspace(0, 1, nfov_width), np.linspace(0, 1, nfov_height))
    return np.array([xx.ravel(), yy.ravel()]).T

def get_cp_rad(center_point):
    # Operation [phi_rad,theta_rad] -> [0,1]
    cp = (center_point * 2 - 1) * np.array([pi, pi*0.5])
    return cp

def calcSphericaltoGnomonic_pos(pos, cp):
    x = pos[0]
    y = pos[1]

    rou = np.sqrt(x ** 2 + y ** 2)
    c = np.arctan(rou)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    phi = np.arcsin(cos_c * np.sin(cp[1]) + (y * sin_c * np.cos(cp[1])) / rou)
    theta = cp[0] + np.arctan2(x * sin_c, rou * np.cos(cp[1]) * cos_c - y * np.sin(cp[1]) * sin_c)

    phi = (phi / (pi*0.5) + 1.) * 0.5
    theta = (theta / pi + 1.) * 0.5

    return np.array([theta, phi]).T

def calcSphericaltoGnomonic(convertedScreenCoord, cp):
    x = convertedScreenCoord.T[0]
    y = convertedScreenCoord.T[1]

    rou = np.sqrt(x ** 2 + y ** 2)
    c = np.arctan(rou)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    phi = np.arcsin(cos_c * np.sin(cp[1]) + (y * sin_c * np.cos(cp[1])) / rou)
    theta = cp[0] + np.arctan2(x * sin_c, rou * np.cos(cp[1]) * cos_c - y * np.sin(cp[1]) * sin_c)

    phi = (phi / (pi*0.5) + 1.) * 0.5
    theta = (theta / pi + 1.) * 0.5

    return np.array([theta, phi]).T

def prep_img(image_path, size_equi, BGR=False, save=False):
    import cv2
    import numpy as np
    # Load image
    
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32) / 255.0
    # Resize image
    image = cv2.resize(image, (size_equi[0], size_equi[1]), interpolation=cv2.INTER_AREA)
    if BGR: # img is in BGR format if the underlying image is a color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if save:
        save_image('test/image_test.jpg',image)
    return image

def save_image(out_path, img):
    import cv2
    cv2.imwrite(out_path,img)

def render_projections(x, y, equi_img, size_pers, size_equi, save=False):
    import cv2

    #if len(x) != len(y):
    #    raise AssertionError("Scanpath position vectors should have the same length")

    nfov_width = size_pers[0]
    nfov_height = size_pers[1] 
    center_point = np.array([x,y])
    cp = get_cp_rad(center_point)

    #equi_img_copy = equi_img.copy()

    # Compute NFOV grid coordinates in radians
    convertedScreenCoord = get_coord_rad(nfov_width, nfov_height)
    sphericalCoord = calcSphericaltoGnomonic(convertedScreenCoord, cp)

    sc_grid = sphericalCoord.reshape(nfov_height, nfov_width, 2).astype(np.float32)
    # Remap accordingly to the computed grid
    img_NFOV = cv2.remap(equi_img, \
               sc_grid[..., 0]*size_equi[0], sc_grid[..., 1]*size_equi[1], \
               interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)

    convertedScreenCoord_equi = get_coord_rad(size_equi[0], size_equi[1])
    sphericalCoord_equi = calcSphericaltoGnomonic(convertedScreenCoord_equi, cp)
    equi_grid = sphericalCoord_equi.reshape(size_equi[0], size_equi[1], 2).astype(np.float32)
    print(sphericalCoord_equi.shape, sphericalCoord_equi)
    print("CENTER", int(equi_grid.shape[0]/2),
          int(equi_grid.shape[1]/2))
    equi_img_draw = cv2.circle(equi_img, 
                               (int(equi_grid.shape[0]/2), 
                                int(equi_grid.shape[1]/2)),
                               radius=10,
                               color=(0, 255, 0),
                               thickness=-1)

    # Workaround: applying a correction to rotate NFOV
    #img_NFOV = cv2.rotate(img_NFOV, cv2.ROTATE_180)
    if save:
        save_image('test/nfov.jpg',img_NFOV)
    # equi_img_draw = draw_NFOV_edges(equi_img,sc_grid)
    return img_NFOV, equi_img_draw

def show_result(NFOV, EQUI, equi_img_draw, file_name):
    import cv2
    cv2.imshow(f"Example - Equirretangular Frames", equi_img_draw)
    cv2.imshow(f"Example - Equirretangular Frames - {file_name}", EQUI)
    cv2.imshow(f"Example - NFOV -", NFOV)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image

if __name__ == '__main__':
    import os
    size_equi = (1000,500)
    size_pers = (500,250)
    project_path = os.getcwd()
    img_paths = [project_path + r'\data\img83.jpg', project_path + r'\data\img100.png']
    dict_path = project_path + r'\microtest1\sp_test.json'
    #sc_len = 
    # a = read_json_file(dict_path)
    # posx, posy = get_scanpath(dict_path, img_paths,scanpath_id=1)
    # print(posx, posy)
    # a = import_json_file(dict_path)

    for img_path in img_paths:
        equi_img = prep_img(img_path, size_equi)
        posx, posy = get_scanpath(dict_path, img_path,scanpath_id=1)
        # print("POSX",posx , len(posx))
        img_NFOV, equi_img_draw = render_projections(posx[9], posy[9], equi_img, size_pers, size_equi)
        
        img_file = img_path.split('\\')[-1]
        show_result(img_NFOV, equi_img, equi_img_draw, file_name=img_file)
        break
    # print(a)
    
    
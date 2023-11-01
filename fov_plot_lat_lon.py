import matplotlib.pyplot as plt
import numpy as np
import os

def gnomonic2lat_lon(x_y_coords, fov_vert_hor, center_lat_lon):
    '''
    Converts gnomonoic (x, y) coordinates to (latitude, longitude) coordinates.
    
    x_y_coords: numpy array of floats of shape (num_coords, 2) 
    fov_vert_hor: tuple of vertical, horizontal field of view in degree
    center_lat_lon: The (lat, lon) coordinates of the center of the viewport that the x_y_coords are referencing.
    '''
    sphere_radius_lon = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[1] / 2.0)))
    sphere_radius_lat = 1. / (2.0 * np.tan(np.radians(fov_vert_hor[0] / 2.0)))

    x, y = x_y_coords[:,0], x_y_coords[:,1]

    x_y_hom = np.column_stack([x.ravel(), y.ravel(), np.ones(len(x.ravel()))])

    K_inv = np.zeros((3, 3))
    K_inv[0, 0] = 1.0/sphere_radius_lon
    K_inv[1, 1] = 1.0/sphere_radius_lat
    K_inv[0, 2] = -1./(2.0*sphere_radius_lon)
    K_inv[1, 2] = -1./(2.0*sphere_radius_lat)
    K_inv[2, 2] = 1.0

    R_lat = np.zeros((3,3))
    R_lat[0,0] = 1.0
    R_lat[1,1] = np.cos(np.radians(-center_lat_lon[0]))
    R_lat[2,2] = R_lat[1,1]
    R_lat[1,2] = -1.0 * np.sin(np.radians(-center_lat_lon[0]))
    R_lat[2,1] = -1.0 * R_lat[1,2]

    R_lon = np.zeros((3,3))
    R_lon[2,2] = 1.0
    R_lon[0,0] = np.cos(np.radians(-center_lat_lon[1]))
    R_lon[1,1] = R_lon[0,0]
    R_lon[0,1] = - np.sin(np.radians(-center_lat_lon[1]))
    R_lon[1,0] = - R_lon[0,1]

    R_full = np.matmul(R_lon, R_lat)

    dot_prod = np.sum(np.matmul(R_full, K_inv).reshape(1,3,3) * x_y_hom.reshape(-1, 1, 3), axis=2)

    sphere_points = dot_prod/np.linalg.norm(dot_prod, axis=1, keepdims=True)

    lat = np.degrees(np.arccos(sphere_points[:, 2]))
    lon = np.degrees(np.arctan2(sphere_points[:, 0], sphere_points[:, 1]))

    lat_lon = np.column_stack([lat, lon])
    lat_lon = np.mod(lat_lon, np.array([180.0, 360.0]))

    return lat_lon

def angle2img(lat_lon_array, img_height_width):
    '''
    Convertes an array of latitude, longitude coordinates to image coordinates with range (0, height) x (0, width)
    '''
    return lat_lon_array / np.array([180., 360.]).reshape(1,2) * np.array(img_height_width).reshape(1,2)


def plot_fov(center_lat_lon, ax, color, fov_vert_hor, height_width):
    '''
    Plots the correctly warped FOV at a given center_lat_lon.
    center_lat_lon: Float tuple of latitude, longitude. Position where FOV is centered
    ax: The matplotlib axis object that should used for plotting.
    color: Color of the FOV box.
    height_width: Height and width of the image.
    '''
    # Coordinates for a rectangle.
    coords = []
    coords.append([np.linspace(0.0, 1.0, 100), [1.]*100])
    coords.append([[1.]*100, np.linspace(0.0, 1.0, 100)])
    coords.append([np.linspace(0.0, 1.0, 100), [0.]*100])
    coords.append([[0.]*100, np.linspace(0.0, 1.0, 100)])    

    lines = []
    for coord in coords:
        lat_lon_array = gnomonic2lat_lon(np.column_stack(coord), fov_vert_hor=fov_vert_hor, 
                                         center_lat_lon=center_lat_lon)
        img_coord_array = angle2img(lat_lon_array, height_width)
        lines.append(img_coord_array)
        
    split_lines = []
    for line in lines:
        diff = np.diff(line, axis=0)
        wrap_idcs = np.where(np.abs(diff)>np.amin(height_width))[0]
        
        if not len(wrap_idcs):
            split_lines.append(line)
        else:
            split_lines.append(line[:wrap_idcs[0]+1])
            split_lines.append(line[wrap_idcs[0]+1:])

    for line in split_lines:
        ax.plot(line[:,1], line[:,0], color=color, linewidth=1.2, alpha=0.5)
        

def render_exploration_frames(scene_df, 
                              img_path,
                              target_dir='frames',
                              duration=30.,
                              fps=30.):
    '''
    Renders single frames of a video that visualizes scanpaths and FOV of users in the VR scene img_path.
    Frames can subsequently be stitched together with ffmpeg.
    
    scene_df: pandas Dataframe with all the runs for a specific scene.
    img_path: Path to the equirectangular scene file.
    duration: Duration of the animation in seconds.
    fps: Target fps.
    '''
    num_frames = fps * duration
    img_height, img_width = 2048, 4096
    image = cv2.resize(cv2.imread(img_path), (img_width, img_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    frame_no = 0
        
    for t in np.linspace(0., duration, num_frames):
        plt.close('all')

        fig, ax = plt.subplots(frameon=False, figsize=(16,9))
        
        ax.grid(b=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

        ax.imshow(image)
        ax.axis('tight')
        ax.set_xlim([0,img_width])
        ax.set_ylim([img_height, 0])
        
        time_template = 'time %.1fs'
        time_text = ax.text(0.05, 0.9, '', size='large', transform=ax.transAxes)
        time_text.set_text(time_template % t)

        fov_vert = 106.188
        aspect_ratio = 0.82034051
        fov_hor = fov_vert * aspect_ratio
        fov_vert_hor = np.array([fov_vert, fov_hor])

        colors = cm.rainbow(np.linspace(0, 1, len(scene_df)))
                
        for color, log_df in zip(colors, scene_df['data']):
            time_filtered = np.where(log_df['timestamp']<=t)[0]
            if len(time_filtered):
                frame_idx = np.amax(time_filtered)
                
                if frame_idx >= len(log_df['timestamp']): 
                    continue
            else:
                continue
            gaze_coords = angle2img(log_df['gaze_lat_lon'][frame_idx, :], (img_height, img_width))
            head_lat_lon = log_df['headLatLon'][frame_idx, :]
            ax.plot(gaze_coords[1], gaze_coords[0], marker='o', markersize=12., color=color, alpha=.8)
            
            plot_fov(head_lat_lon, ax, color, fov_vert_hor, height_width=np.array([img_height, img_width]))
        
        fig.savefig(os.path.join(target_dir, "%06d.png"%frame_no), bbox_inches='tight', pad_inches=0, dpi=160)
        frame_no += 1
        fig.clf()


def plot_with_viewport(img, viewport_coords, out_path):
    '''
    Gerar animações com a superposição do vídeo com o FoV de cada participante
    '''
    viewport_coords_resh = viewport_coords.reshape(800,800,2)
    upper_line = viewport_coords_resh[0,:,:]
    lower_line = viewport_coords_resh[-1, :,:]
    right_line = viewport_coords_resh[:,-1, :]
    left_line = viewport_coords_resh[:,0,:]

    lines = [upper_line, lower_line, right_line, left_line]

    split_lines = []
    for line in lines:
        diff = np.diff(line, axis=0)
        wrap_idcs = np.where(np.abs(diff)>10)[0]
        if not len(wrap_idcs):
            split_lines.append(line)
        else:
            split_lines.append(line[:wrap_idcs[0]+1])
            split_lines.append(line[wrap_idcs[0]+1:])

    fig, ax = plt.subplots(frameon=False)
    fig.set_size_inches(48,24)
    ax.imshow(img)

    for line in split_lines:
        ax.plot(line[:,1], line[:,0], color='black', linewidth=10)

    for line in split_lines:
        ax.plot(line[:,1], line[:,0], color='white',linewidth=8)

    ax.axis('tight')
    ax.axis('off')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    fig.clf()
    plt.close()


if __name__ == '__main__':
    
    animation = render_exploration_frames(runs_files[0], os.path.join(settings.IMG_PATH, 'cubemap_0000.png'), fps=24)
    stitch2video('scene_0.mp4', fps=24, frame_dir='frames')
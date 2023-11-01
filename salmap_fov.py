from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


#Saliency functions
def salmap_from_norm_coords(norm_coords, sigma, height_width,video_name,frame,save_img=False):
    '''
    Base function to compute general saliency maps
    given the normalized (from 0 to 1) fixation coordinates [dtype = np.array], 
    the sigma of the gaussian blur, and the height and width of the saliency map in pixels.
    '''
    #test saliency maps with different pixel definition
    height_width = np.array(height_width)
    height_width_list = [height_width, (height_width/4).astype(int), (height_width/8).astype(int), (height_width/16).astype(int)]
    for size,hw in enumerate(height_width_list):
        #img_coords = np.mod(np.round(norm_coords * np.array(height_width)), np.array(height_width)-1.0).astype(int)
        #gaze_counts = np.zeros((height_width[0], height_width[1]))
        # Computa as coordenadas de latitude e longitude para 
        # um shape específico da imagem usando as coordenadas normalizadas
        img_coords = np.mod(np.round(norm_coords * np.array(hw)), np.array(hw)-1.0).astype(int)
        gaze_counts = np.zeros((hw[0], hw[1]))
        for coord in img_coords:
            gaze_counts[coord[0], coord[1]] += 1.0
        gaze_counts[0, 0] = 0.0
        sigma_y = sigma
        salmap = ndimage.filters.gaussian_filter1d(gaze_counts, sigma=sigma_y, mode='wrap', axis=0)
        # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
        for row in range(salmap.shape[0]):
            angle = (row/float(salmap.shape[0]) - 0.5) * np.pi
            sigma_x = sigma_y / (np.cos(angle) + 1e-3)
            salmap[row,:] = ndimage.filters.gaussian_filter1d(salmap[row,:], sigma=sigma_x, mode='wrap')
        salmap /= float(np.sum(salmap))
        # Visualization of salmap different sizes
        if save_img == True:
            sal_out_folder = os.path.join(GT_PATH,''.format(video_name),'resultados')
            #create a directory if it doesn't exist already.
            if not os.path.exists(sal_out_folder):
                os.mkdir(sal_out_folder)
            plt.imsave(os.path.join(sal_out_folder,'salmap_{}_{}.jpg'.format(size,frame)),salmap)
    return salmap

def salmap(norm_coords, sigma, height_width,video_name,frame,save_img=False):
    hw = (np.array(height_width)/4).astype(int)
    img_coords = np.mod(np.round(norm_coords * np.array(hw)), np.array(hw)-1.0).astype(int)
    gaze_counts = np.zeros((hw[0], hw[1]))
    for coord in img_coords:
        gaze_counts[coord[0], coord[1]] += 1.0
    gaze_counts[0, 0] = 0.0
    sigma_y = sigma
    salmap = ndimage.filters.gaussian_filter1d(gaze_counts, sigma=sigma_y, mode='wrap', axis=0)
    # In x-direction, we scale the radius of the gaussian kernel the closer we get to the pole
    for row in range(salmap.shape[0]):
        angle = (row/float(salmap.shape[0]) - 0.5) * np.pi
        sigma_x = sigma_y / (np.cos(angle) + 1e-3)
        salmap[row,:] = ndimage.filters.gaussian_filter1d(salmap[row,:], sigma=sigma_x, mode='wrap')
    salmap /= float(np.sum(salmap))
    # Visualization of salmap different sizes
    if save_img == True:
        sal_out_folder = os.path.join(GT_PATH,video_name,'resultados')
        #create a directory if it doesn't exist already.
        if not os.path.exists(sal_out_folder):
            os.mkdir(sal_out_folder)
        plt.imsave(os.path.join(sal_out_folder,'salmap_{}.jpg'.format(frame)),salmap)
    return salmap


def rand_subsampling(log_df, num_gaze_samples, frames_count, Seed=10):
    """
    Foi realizada uma subamostragem dos logs de fixação para o número de 
    frames de cada vídeo. Uma vez que o relatório não indica quais processos
    de filtragem foram realizados nos logs de fixação. É comum realizar
    filtragem de eventos oculares que não são importantes para a atenção visual
    como as saccades e eventuais erros de medida dos sensores do dispositivo
    de cabeça.
    """
    drop_samples = num_gaze_samples - frames_count
    np.random.seed(Seed)
    rng = np.random.default_rng()
    drop_samples_num = rng.choice(num_gaze_samples, drop_samples, replace=False)
    log_df_subsample = log_df.drop(drop_samples_num)
    return log_df_subsample

def get_gaze_salmap(gaze_coord, sigma_deg, height_width):
    '''
    Computes gaze saliency maps.
    gaze_coord: 
    '''
    fixation_coords = []
    if len(gaze_coord.shape) > 1:
        _, unique_idcs = np.unique(gaze_coord, return_index=True)
        all_fixations_coord = gaze_coord[unique_idcs]
    print("fixations shape ",all_fixations_coord.shape)
    print("fixations:",all_fixations_coord)
    return salmap_from_norm_coords(all_fixations_coord, sigma_deg * height_width[1]/360.0, height_width)  

def plot_with_viewport(img, viewport_coords, out_path):
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
    # To run this code first download and unzip the data from
    # https://drive.google.com/drive/folders/13Bn_WQgGCh0ahyqoVG_UBoqNxP4qDLCQ?usp=drive_link
    # into the /data/salmap_dataset folder 
    
    root_path = os.getcwd()

    dataset_path = root_path + r'/data/salmap_dataset' + r'/datasets/Data'
    db0_path =  dataset_path + r'/DirectorCut_Vsense/SubjectiveData'
    db1_path = dataset_path + r'/Data/MotionScenes_Afshin/Traces'
    GT_PATH = r'/Results/db0/


    dir_paths = [f.path for f in os.scandir(db0_path) if f.is_dir()]
    video_log_dirs = [sorted(glob(os.path.join(dir, '*.txt'))) for dir in dir_paths]
    
    # Load V-sense Fixation Data Functions
    frames_num = np.array([7947,7144,11310,4350,5140])
    fps = np.array([30,30,30,30,25])
    video_name = np.array(['luther','DB','360partnership','vaude','war'])

    #Main
    for idx,video in enumerate(video_name):
        print('Processing video: {}'.format(video_name[idx]))
        video_log_dirs = sorted(glob(os.path.join(dir_paths[idx], '*.txt')))
        #Saving height and width
        log_df = pd.read_csv(video_log_dirs[0], delimiter="\t", header=None, names=["w", "h", "lon","lat"])
        height_width = [log_df['h'][1],log_df['w'][0]]
        #Iterate through participant logs to build 
        #All fixation head coordinate  per 
        all_lon_arr = np.zeros((frames_num[idx],1))
        all_lat_arr = np.zeros((frames_num[idx],1))
        #Initialize the random generator
        for log in tqdm.tqdm(video_log_dirs):
            log_df = pd.read_csv(log, delimiter="\t", header=None, names=["w", "h", "lon","lat"])
            #Gaze Fixation subsampling to match fps
            log_df_subsample = rand_subsampling(log_df, log_df.shape[0], frames_num[idx])
            #Build an array of longitudes
            #Build an array of latitudes
            #Load and append all fixations coordinates in the following structure [frames, partcipant_log]
            lat_arr = log_df_subsample['lat'].to_numpy()
            lon_arr = log_df_subsample['lon'].to_numpy()
            all_lat_arr = np.append(all_lat_arr,lat_arr.reshape(frames_num[idx],1),axis=1)
            all_lon_arr = np.append(all_lon_arr,lon_arr.reshape(frames_num[idx],1),axis=1)
            #For each frame (each row of the all_fix_matrix)
            #frame_fix_coord = np.zeros((1,2)) # Initialize frame fixation coordinates array
            #all_lat_arr = all_lat_arr[:,1:]
        #Exclude the first initialization column
        all_lon_arr = all_lon_arr[:,1:]
        all_lat_arr = all_lat_arr[:,1:]
        seq = range(frames_num[idx]) # frame sequence
        #For frame in tqdm.tqdm(range(frames_num[video])):
        for frame in tqdm.tqdm(seq):    
            print('Generating saliency map for frame:',frame)
            #Build frame fixation matrix (vstack)
            frame_fix_coord = np.append(all_lon_arr[frame,:].reshape(20,1),all_lat_arr[frame,:].reshape(20,1),axis=1)
            #frame_fix_coord = np.append(all_lon_arr.reshape()[:,frame],all_lat_arr.T[:,frame],axis=1)
            #Generate saliency map per frame
            #head_salmap = salmap_from_norm_coords(frame_fix_coord,5,height_width,video_name[idx],frame,save_img=True)
            head_salmap = salmap(frame_fix_coord,6,height_width,video_name[idx],frame,save_img=False)
            #Save saliency map per frame in the specific folder
            npy_outpath = os.path.join(GT_PATH, video_name[idx],'GT_sig5')
            if not os.path.exists(npy_outpath):
                    os.mkdir(npy_outpath)
            np.save(os.path.join(npy_outpath,'{:05}.npy'.format(frame)),head_salmap)
# scanpath-fov-generator
Code to simply extract normal field of view (NFOV) from a list of given a scanpath dataset

In this repository you find three approaches to visualize NFOV from original Equirrectangular frames. Each approach has its own dependencies and configuration steps

## Saliency maps with FoV
- Code: `salmap_fov.py`
- Data: To run this code, first download and unzip the data from
[Datasets](https://drive.google.com/drive/folders/13Bn_WQgGCh0ahyqoVG_UBoqNxP4qDLCQ?usp=drive_link) into the `/data/salmap_dataset` folder 
- Tested: No
- Running: Yes

## Running this approach
> Work in progress

## Rendering FoV from a scanpath from spherical coordinates
- Code: `scanpath_projection.py`
- Data: `\data`
- Tested: Yes
- Running: Yes

## Running this approach
1. Install dependencies: `$ pip install -r requirements.txt`
2. Running the code: `$ python scanpath_projection.py`

## Rendering FoV from scanpath from latitude longitude coordinates
- Code: `fov_plot_lat_lon.py`
- Data: No data
- Tested: No
- Running: No
- Reference: https://github.com/vsitzmann/vr-saliency/blob/master/src/Analysis.ipynb

## Running this approach
> Work in progress

## Reference projects: 

* [Director`s Cut - V-sense]()
  * [Dataset](https://drive.google.com/drive/folders/1QhUGA07pAxW2CCOmj8GHMEoIErPocbXk)
  * [Paper 0 - Visual Attention Analysis in 
Cinematic VR Content](https://v-sense.scss.tcd.ie/wp-content/uploads/2018/09/CVMP2018_DirectorsCut_public-1.pdf)
  * [Paper 1 - Cut and transitions](https://v-sense.scss.tcd.ie/wp-content/uploads/2018/12/2018_IC3D_DirectorCut_AttentionStoryTelling.pdf)
  * [Paper2 - Interactive Storytelling](https://v-sense.scss.tcd.ie/wp-content/uploads/2018/09/Directors-Cut-analysis-of-aspects-of-interactive-storytelling.pdf)

* [Motion Scences Dataset - Afshin](https://github.com/acmmmsys/2019-360dataset)
    * [Dataset](https://github.com/acmmmsys/2019-360dataset)
    * [Paper](https://arxiv.org/abs/1905.03823)
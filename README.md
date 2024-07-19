This is the code for my bachelor thesis: 
Biologically plausible spatial navigation. 
A big part of the code was written by Amir El Sewisy

Installation:
 * Required software:
   - Python 3 (Python 2 may work as well)
   - NumPy (pip install numpy)
   - SciPy (pip install scipy)
   - matplotlib (pip install matplotlib)
   - PyBullet (pip install pybullet)
   - tqdm (pip install tqdm)
 * Recommended software:
   - PyCuda for GPU acceleration (https://wiki.tiker.net/PyCuda/Installation/) (if a GPU with CUDA support is available)
   - ffmpeg for producing the video (apt-get install ffmpeg)
 * Set the backend (NumPy or CUDA) in network.py

Scripts:
 - controller.py: run the robot simulation. Environment / dataset and simulation parameters can be set on the top of that file
 - controller_circle.py: run test with constant angular velocity
 - controller_kitti_raw.py: run the hdc Network with raw KITTI data as input. Use import_kitti_raw.py to generate the required pickle file from the KITTI dataset
 - generate_avs_from_sim.py: run pybullet and save all the changes in angle for every timestep in a file readable by controller.py
 - import_kitti.py: generate the change in angle for every timestep from the kitti (http://www.cvlibs.net/datasets/kitti/) odometry dataset, angular velocities are calculated from the orientation ground truth data. 
 - import_kitti_raw.py: generate a pickle file from the raw KITTI datasets (http://www.cvlibs.net/datasets/kitti/raw_data.php) to be used by controller_kitti_raw.py
 - plotting.py: generate most of the figures in the thesis, the ones belonging to the simulation in Chapter 3 are generated with controller.py
 - video_import_ros.py: store ROS messages from IMU and camera in a pickle file to be used by video_create.py
   - 1. set outfile in video_import_ros.py
   - 2. start roscore (roscore)
   - 3. start video_import_ros.py 
   - 4. play bagfile (rosbag play bagfile.bag)
   - 5. once bagfile is done, stop roscore
 - video_create.py: run simulation and produce video, 
   - trajectory video in video_frames/input/trajectory.mp4
   - pickle file made with video_import_ros.py is required, set as infile in video_create.py
   - output file: video_frames/output.mp4

Functionality:
HDC network Amir el Sewisy:
 - hdc_template.py: generate HDC network according to parameters defined in params.py
 - hdcAttractorConnectivity: generate connections inside the HDC attractor network given the parameter lambda
 - hdcNetwork.py: generate the HDC network given weight functions, also contains the function for initializing the HDC network
 - hdcOptimizeAttractor.py: contains the procedure used to find the value for lambda yielding the best weight function
 - hdcOptimizeShiftLayers.py: contains the procedure used to generate the plots for shifting (Figures 2.11, 2.12) as well as finding the factor for the angular velocity -> stimulus function (Equation 2.11)
 - helper.py: helper functions (decode attractor network, distance between angles, ...)
 - network.py: proxy for selecting one of the network backends:
   - network_cuda.py
   - network_numpy.py
 - neuron.py: neuron model parameters
 - params.py: parameters for the network (number of neurons, lambda)
 - polarPlotter.py: live visualization of the HDC network, shown in Figure 3.2
 - pybullet_environment.py: pybullet environment class, braitenberg controller
 - stresstest.py: performance testing

BC-Model (Camillo Heye):
 - BCActivity.py: calculates boundary coding neurons activities during training and simulation 
 - HDCActivity.py: calculate HDC activity during training phase for BC-Model (MakeWeights.py)   
 - BCSimulation.py: gets egocentric activity profile and uses weights previously calculated to generate TR and BVC activity
 - MakeWeights.py: procedure to generate all the weights for the BC-Model 
 - parametersBC: all parameters used in the BC-Model
 - polarBCplotter.py: plot eBC and BVC activities during simulation 
 - polarBCplotterAllLayers: plot eBC, BVC and all TR layers during simulation (set plotAllLAyers TRUE in controller.py)

The weights file for the BC-Model is to big to upload on git. 
To download use this weTransfer link: 
https://wetransfer.com/downloads/1314542a18bc17ca2382059119834f4220211013180549/5d966614e8a40da3491cbe07bffdad4720211013180604/deaec4

Just add the add the weights folder to the project and the code will run

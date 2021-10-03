# POWER_Dips-AzureLSTM
### EcEcursion: Andrew Chen, Shu-Yan Cheng, Yun-Hsuan Tsai, Hao Kuan, Yi-Hsuan Lai, Ming-Yi Wei
Code for training in Azure Machine Learning Studio.
Trained an LSTM Model with TensorFlow and Keras to predict future trends of Photosynthetically Active Radiation, written for 2021 NASA Space Apps Challenge: You Are My Sunshine.
## Packages Used
Keras, Tensorflow, NumPy, Pandas, PlotLy

The following should be run in the folder od code to ensure successful build:
(Note, this build is for MacOS/Unix Based Systems:
<pre><code>python3 -m venv AzureLSTM-env
source tutorial-env/bin/activate
sudo pip install tensorflow
sudo pip install numpy
sudo pip install keras
sudo pip install plotly
python Sunshine_LSTM.py
</code></pre>

## How To Run
Should be able to run directly after ensuring all packages were downloaded in virtual-env of selection
Creates .h5 files after training with given files manually downloaded from [NASA POWER Project](https://power.larc.nasa.gov/), then moved to [AzureServer](https://github.com/NASA2021-EcEcursion/POWER_Dips-AzureServer) for further use in the react-native app.

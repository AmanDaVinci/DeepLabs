echo"
#####################################
#   Setup a Deep Learning Machine   #
#   	[Ubuntu 16.04 LTS]          #
#####################################
"

# Go to Home
cd ~

# ensure system is updated and has basic build tools
echo "\n\nSystem Update & Upgrade\n\n"
sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common

# download and install GPU drivers
# see https://cloud.google.com/compute/docs/gpus/add-gpus#install-gpu-driver

echo "\n\nChecking for CUDA & Installing...\n\n"
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda -y
fi

echo "\n\nVerifying GPU driver installation\n\n"
# verify that GPU driver installed
sudo modprobe nvidia
nvidia-smi

sudo apt-get install libcupti-dev

echo "\n\nDownloading Anaconda...\n\n"
# install Anaconda for current user
mkdir downloads
cd downloads
wget "https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh" -O "Anaconda3-4.3.1-Linux-x86_64.sh"

echo "\n\nInstalling Anaconda...\n\n"
bash "Anaconda3-4.3.1-Linux-x86_64.sh" -b

echo "\n\nSetting up the Anaconda path\n\n"
echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda3/bin:$PATH"
conda install -y bcolz
conda upgrade -y --all


# install and configure theano
echo "\n\nInstalling & Configuring Theano...\n\n"
conda install theano pygpu
echo "[global]
device = cuda0
floatX = float32
[cuda]
root = /usr/local/cuda" > ~/.theanorc

# install and configure keras
echo "\n\nInstalling & Configuring Keras...\n\n"
conda install keras
mkdir ~/.keras
echo '{
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano",
    "image_data_format": "channels_first"
}' > ~/.keras/keras.json

# install cudnn libraries

# echo "Downloading cuDNN libraries from Google Drive..."
# chmod +x gdown.pl
# ./gdown.pl "https://drive.google.com/uc?export=download&confirm=OlUi&id=0B1KAQ69tr5ovNi12Qk5DUEV6NDg" "cudnn.tgz"

echo "\n\nDownloading cuDNN libraries from fast.ai archive...\n\n"
wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"

echo "\n\nInstalling cuDNN...\n\n"
tar -zxf cudnn.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/

# configure jupyter and prompt for password
echo "\n\nConfiguring jupyter and will prompt for password...\n\n"
jupyter notebook --generate-config --allow-root
jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
echo "c.NotebookApp.ip = '*'
c.NotebookApp.password = u'"$jupass"'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9999" >> $HOME/.jupyter/jupyter_notebook_config.py

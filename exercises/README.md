# Exercises
The following instructions explain which tools need to be installed and in which order so that you can build and execute the exercises.

The explanations assume that you have a Windows computer with an NVIDIA graphics card. If you have a different system configuration, the individual steps must be adapted accordingly. The necessary information can be found on the respective linked websites.

## Install WSL (Windows)
Open windows command line and run the following command:
```shell
wsl --install
```
Reboot your computer.

## Install C++ Toolkits
https://developer.codeplay.com/products/oneapi/nvidia/2024.0.1/guides/get-started-guide-nvidia.html

### Install C++ Development Tools
```shell
sudo apt update
sudo apt upgrade
sudo apt -y install cmake pkg-config build-essential gdb ninja-build rsync zip

which cmake pkg-config make gcc g++
```

### Install FreeImage Libraries
https://installati.one/install-libfreeimage3-ubuntu-22-04/
https://installati.one/install-libfreeimageplus3-ubuntu-22-04/

```shell
sudo apt-get -y install libfreeimage3 libfreeimage-dev
sudo apt-get -y install libfreeimageplus3 libfreeimageplus-dev
```

### Install the Intel oneAPI Toolkits
https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager

https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager

https://www.intel.com/content/www/us/en/developer/articles/technical/intel-cpu-runtime-for-opencl-applications-with-sycl-support.html


```shell
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \ | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt upgrade

sudo apt install intel-basekit
sudo apt install intel-hpckit
```

### Install CUDA Software Stack for NVIDIA GPU
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#wsl

```shell
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cuda-toolkit
```

### Install oneAPI Extension for NVIDIA GPU
https://developer.codeplay.com/products/oneapi/nvidia/2024.0.1/guides/get-started-guide-nvidia.html#installation

Download: https://developer.codeplay.com/products/oneapi/nvidia/download/

Copy downloaded 'oneapi-for-nvidia-gpus-2024.0.2-cuda-12.0-linux.sh' to WSL home directory.

```shell
sudo sh oneapi-for-nvidia-gpus-2024.0.2-cuda-12.0-linux.sh
```

## Setup oneAPI Environment
https://developer.codeplay.com/products/oneapi/nvidia/2024.0.1/guides/get-started-guide-nvidia.html#set-up-your-environment

In order for the oneAPI compiler to find all the necessary tools and libraries, a number of environment variables must be set correctly beforehand. A shell script exists to set these environment variables correctly to the latest version of the compiler. You can execute this script with the following command;
```shell
. /opt/intel/oneapi/setvars.sh --include-intel-llvm
```

### Verify the Installation
https://developer.codeplay.com/products/oneapi/nvidia/2024.0.1/guides/get-started-guide-nvidia.html#verify-your-installation

You can use the following commands to ensure that the installation has worked:
```shell
nvidia-smi
sycl-ls
```
If nvidia-smi runs without any obvious errors in the output then your environment should be set up correctly.
If the available NVIDIA GPUs are correctly listed, then the DPC++ CUDA plugin was correctly installed and set up.


## Setup Visual Studio Code
https://code.visualstudio.com/

### Install VS Code on the Windows Side (not WSL)
https://code.visualstudio.com/docs/remote/wsl

When prompted to Select Additional Tasks during installation, be sure to check the Add to PATH option so you can easily open a folder in WSL using the code command.

### Install WSL Extension for VS Code 
https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl

### Install Necessary Extensions for VS Code in WSL Window
All additional extensions should be installed in a WSL window. Either start the Remote Explorer in VS Code or click on the green button on the bottom left corner of the VS Code window and the choose 'Connect to WSL'. The green button of a WSL window contains the text 'WSL: Ubuntu'.
Now install the following extensions in the WSL window:
- C/C++ Extension Pack
- Environment Configurator for Intel oneAPI Toolkits
- Code Sample Browser for Intel oneAPI Toolkits
- GDB with GPU Debug Support for Intel oneAPI Toolkits

### Clone a Git Repository
Create a subfolder for your Git repositories in you Linux home directory with the command 'mkdir', for example 'mkdir git'. Go to the git subfolder and create a subfolder for your ProgAlg exercises, for example 'exercises'. 

Now start Visual Studio Code and close any automatically opened folder. Then select "Clone Git Repository..." from the welcome page or use Ctrl-Shift-P to select the "Git: Clone" task. Enter the URL of the repository in the input field that opens: https://gitlab.fhnw.ch/progalg/exercises.git. Then select the previously created folder '~/git/exercises' as the destination for the repository and click OK.


### Setup IntelliSense Configurations
https://code.visualstudio.com/docs/cpp/customize-default-settings-cpp

Open the file '12_SYCL/main.cpp' in the VS Code editor. In the code editor you see the red underlined source code line '#include <sycl/sycl.hpp>'. Press on the offered code actions and choose 'Edit "includePath" settings'. In case the IntelliSense Configuration window does not open but an access error appears, then run the following command in the terminal window and try again:
```shell
sudo chown -R $USER:$USER /home/
```
VS Code opens the IntelliSense Configurations page. This page can be manually opened by pressing Ctrl+Shift+P and choosing 'C/C++: Edit Configurations (UI)'. 

In the section 'Compiler path' replace the current compiler path by the following path:
```json
/opt/intel/oneapi/compiler/2024.0/bin/icpx
```
In the section 'Include path' add the following line:
```json
/opt/intel/oneapi/compiler/2024.0/include/**
```
In the section 'C++ standard' choose c++20.
Both changes are stored in the file '.vscode/c_cpp_properties.json'.


## Build the Exercises with CMake

### Setup oneAPI Environment aforehead
Because the CMake extension of VS Code is initialized before the oneAPI environment is set by the oneAPI VS Code extension, CMake cannot find the necessary Intel executables and libraries. A good solution to solve this problem is the following approach: 
- open a command window
- start WSL
- change directory (let's assume that ~/git/exercises containes the cloned exercises)
- setup Intel oneAPI environment
- start VS Code

```shell
wsl
cd ~/git/exercises/
source /opt/intel/oneapi/setvars.sh --include-intel-llvm
code .
```

### CMake 
Choose CMake in the black side panel and
- press on 'Delete Cache and Reconfigure' and then
- click on 'Build' in the status line of VS Code.

### Run
Choose CMake in the black side panel and check the project status in the CMake window panel. 
- Under 'Configure' you can choose between different build configurations, for example 'Debug' or 'Release'.
- Under 'Launch' you can choose the current startup project. The executable of this project is started when you click on run button in status line.


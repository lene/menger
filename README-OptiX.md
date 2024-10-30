## Installing prerequisites
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2204-12-6-local_12.6.2-560.35.03-1_amd64.deb
$ sudo cp /var/cuda-repo-ubuntu2204-12-6-local/cuda-F9A63CE3-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda-toolkit-12-6
$ sudo apt-get -y install nvidia-cuda-toolkit
```

### Verifying installation
```
$ ldconfig -p | grep cuda || exit 1
$ nvcc --version || exit 1
$ cd src
$ nvcc test_cuda.cu -o test_cuda
$ ./test_cuda | grep "Hello from CUDA kernel" || exit 1
$ cd /tmp
$ git clone https://github.com/NVIDIA/cuda-samples.git
$ cd cd cuda-samples/Samples/1_Utilities/deviceQuery
$ make
$ ./deviceQuery | grep "CUDA Driver Version" || exit 1
```

### Installing NVIDIA Docker Toolkit (necessary for running in container but not for building containers)
```
$ sudo apt-get install -y nvidia-container-toolkit
$ docker run --gpus all nvidia/cuda:12.6.2-runtime-ubuntu24.04 nvidia-smi || exit 1
```

## Installing OptiX
```
$ wget https://developer.nvidia.com/downloads/designworks/optix/secure/8.0.0/nvidia-optix-sdk-8.0.0-linux64-x86_64.sh
$ chmod +x nvidia-optix-sdk-8.0.0-linux64-x86_64.sh
$ ./nvidia-optix-sdk-8.0.0-linux64-x86_64.sh
$ mv NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64 /usr/local || exit 1
$ cd /usr/local/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
$ mkdir build
$ cd build
$ ccmake ..
$ make -j$(nproc)
$ ./bin/optixConsole || exit 1
```

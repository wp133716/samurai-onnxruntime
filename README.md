# samurai-onnxruntime

This repository contains the ONNX Runtime implementation of the [SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory](https://github.com/yangchris11/samurai). The codebase includes both Python and C++ implementations for running inference using ONNX models.

https://github.com/user-attachments/assets/1ab52557-4308-47c4-96a7-a91aa3935ff9

## Getting Started

### python Installation 
The python onnxruntime code requires `python>=3.10`, as well as `torch>=2.3.1`.
```
pip install onnxruntime-gpu==1.20.0
```

#### Load ONNX

onnx模型可以在另一个项目 [samurai-onnx](https://github.com/wp133716/samurai-onnx) 中获取

#### Run Inference

```
cd python
python main.py --video_path <path_to_video> --model_path <path_to_onnx_models>
```

### C++ Installation
The C++ onnxruntime code requires `onnxruntime==1.20.0`(https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz), as well as `OpenCV>=4.5.0`.

#### Build

```
cd c++ && mkdir build
cd build
cmake ..
make -j8
```
#### Run Inference

```
./sam2_tracker_onnx <path_to_onnx_models> <path_to_video>

eg:
./sam2_tracker_onnx ../../onnx_model ../../assets/1917.mp4
```

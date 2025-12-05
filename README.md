# samurai-onnxruntime

## Getting Started

#### python Installation 
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
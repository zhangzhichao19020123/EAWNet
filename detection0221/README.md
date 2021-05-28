# RAWNet




- [x] Inference
- [x] Train
    - [x] Mocaic

```
├── README.md
├── dataset.py            dataset
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotation.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```




# 1. Train

[use yolov4 to train your own data](Use_yolov4_to_train_your_own_data.md)

1. Download weight
2. Transform data

    For coco dataset,you can use tool/coco_annotation.py.
    ```
    # train.txt
    image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
    ...
    ...
    ```
3. Train

    you can set parameters in cfg.py.
    ```
     python train.py -g [GPU_ID] -dir [Dataset direction] ...
    ```

# 2. Inference




## 2.2 Image input size for inference

Image input size is NOT restricted in `320 * 320`, `416 * 416`, `512 * 512` and `608 * 608`.
You can adjust your input sizes for a different input ratio, for example: `320 * 608`.
Larger input size could help detect smaller targets, but may be slower and GPU memory exhausting.

```py
height = 320 + 96 * n, n in {0, 1, 2, 3, ...}
width  = 320 + 96 * m, m in {0, 1, 2, 3, ...}
```

## 2.3 **Different inference options**

- Load the pretrained darknet model and darknet weights to do the inference (image size is configured in cfg file already)

    ```sh
    python demo.py -cfgfile <cfgFile> -weightfile <weightFile> -imgfile <imgFile>
    ```

- Load pytorch weights (pth file) to do the inference

    ```sh
    python models.py <num_classes> <weightfile> <imgfile> <IN_IMAGE_H> <IN_IMAGE_W> <namefile(optional)>
    ```
    
- Load converted ONNX file to do inference (See section 3 and 4)

- Load converted TensorRT engine file to do inference (See section 5)

## 2.4 Inference output

There are 2 inference outputs.
- One is locations of bounding boxes, its shape is  `[batch, num_boxes, 1, 4]` which represents x1, y1, x2, y2 of each bounding box.
- The other one is scores of bounding boxes which is of shape `[batch, num_boxes, num_classes]` indicating scores of all classes for each bounding box.

Until now, still a small piece of post-processing including NMS is required. We are trying to minimize time and complexity of post-processing.





  
  
   
Reference:
- https://github.com/eriklindernoren/PyTorch-YOLOv3
- https://github.com/marvis/pytorch-caffe-darknet-convert
- https://github.com/marvis/pytorch-yolo3

ython train.py -g 0 -dir ./data/COCO/train2017 -train_label_path ./data/COCO/train.txt

python train.py -g 0 -dir ./data/COCO/val2017 -train_label_path ./data/COCO/val.txt

python train.py -g 0 -dir ./data/COCO/val2017 -train_label_path ./data/COCO/val.txt -classes 80 -pretrained ./weights/yolov4.conv.137.pth -l 0.001

python train.py -g 0 -dir ./data/COCO/val2017 -train_label_path ./data/COCO/val.txt -classes 16 -pretrained ./weights/yolov4.conv.137.pth -l 0.001
dota
python models.py 3 weight/Yolov4_epoch166_coins.pth data/coin2.jpg data/coins.names
python models.py 16 checkpoints/Yolov4_epoch15.pth data/input/P0003__1__0___0.jpg data/coco.names data/output/P0003__1__0___0.jpg
python models.py 16 checkpoints/Yolov4_epoch15.pth data/input/P0003__1__0___0.jpg data/coco.names
python models.py 16  checkpoints/Yolov4_epoch15.pth  data/input/P0003__1__0___0.jpg   1024 1024    data/coco.names
python models.py 16  ./checkpoints/Yolov4_epoch15.pth  ./data/input/P0003__1__0___0.jpg   1024 1024    ./data/coco.names

python models.py 80  ./weights/yolov4.pth  ./data/input/P0003__1__0___0.jpg  320 320   成功了
python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320  missingkeys
(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320
Traceback (most recent call last):
  File "models.py", line 480, in <module>
    model.load_state_dict(pretrained_dict)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 830, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for Yolov4:
        Missing key(s) in state_dict: "down1.conv1.conv.0.weight", 

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./weights/yolov4.pth  ./data/input/P0003__1__0___0.jpg 256 256
Traceback (most recent call last):
  File "models.py", line 480, in <module>
    model.load_state_dict(pretrained_dict)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 830, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for Yolov4:
        size mismatch for head.conv2.conv.0.weight: copying a param with shape torch.Size([255, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([63, 256, 1, 1]).
        size mismatch for head.conv2.conv.0.bias: copying a param with shape torch.Size([255]) from checkpoint, the shape in current model is torch.Size([63]).
        size mismatch for head.conv10.conv.0.weight: copying a param with shape torch.Size([255, 512, 1, 1]) from checkpoint, the shape in current model is torch.Size([63, 512, 1, 1]).
        size mismatch for head.conv10.conv.0.bias: copying a param with shape torch.Size([255]) from checkpoint, the shape in current model is torch.Size([63]).
        size mismatch for head.conv18.conv.0.weight: copying a param with shape torch.Size([255, 1024, 1, 1]) from checkpoint, the shape in current model is torch.Size([63, 1024, 1, 1]).
        size mismatch for head.conv18.conv.0.bias: copying a param with shape torch.Size([255]) from checkpoint, the shape in current model is torch.Size([63]).

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 80  ./weights/yolov4.pth  ./data/input/P0003__1__0___0.jpg  1024 1024
Traceback (most recent call last):
  File "models.py", line 501, in <module>
    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
  File "D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4\tool\torch_utils.py", line 94, in do_detect
    output = model(img)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "models.py", line 440, in forward
    d1 = self.down1(input)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "models.py", line 122, in forward
    x2 = self.conv2(x1)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "models.py", line 65, in forward
    x = l(x)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\module.py", line 532, in __call__
    result = self.forward(*input, **kwargs)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\conv.py", line 345, in forward
    return self.conv2d_forward(input, self.weight)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\nn\modules\conv.py", line 342, in conv2d_forward
    self.padding, self.dilation, self.groups)
RuntimeError: CUDA out of memory. Tried to allocate 64.00 MiB (GPU 0; 4.00 GiB total capacity; 1.12 GiB already allocated; 62.80 MiB free; 1.13 GiB reserved in total by PyTorch)



注意demo朱能用那个固定的weights   demo.py用的是yolov4.weights，使用yolov4.pth在推理的时候会出现inf数据，导致推理结果错误
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./weights/yolov4.weights -imgfile ./data/input/P0003__1__0___0.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./checkpoints/Yolov4_epoch15.pth -imgfile ./data/input/P0003__1__0___0.jpg
P0170__1__824___0.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./checkpoints/Yolov4_epoch15.pth -imgfile ./data/input/P0170__1__824___0.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./weights/yolov4.weights -imgfile ./data/input/P0170__1__824___0.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./weights/yolov4.weights -imgfile ./data/else/dog.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./checkpoints/Yolov4_epoch17.pth -imgfile ./data/input/P0259__1__0___824.jpg


coco41
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./weights/yolov4.weights -imgfile ./data/dog.jpg   可以
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./checkpoints/Yolov4_epoch2.pth  -imgfile ./data/dog.jpg
python demo.py -cfgfile  ./cfg/yolov4.cfg -weightfile ./weights/yolov4.pth -imgfile ./data/dog.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 80  ./weights/yolov4.pth  ./data/input/P0003__1__0___0.jpg  320 320
-----------------------------------
           Preprocess : 0.000998
      Model Inference : 1.363354
-----------------------------------
-----------------------------------
       max and argmax : 0.001995
                  nms : 0.000997
Post processing total : 0.002992
-----------------------------------
-----------------------------------
           Preprocess : 0.000998
      Model Inference : 0.076795
-----------------------------------
-----------------------------------
       max and argmax : 0.001994
                  nms : 0.000997
Post processing total : 0.002991
-----------------------------------
bridge: 0.562860
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320
-----------------------------------
           Preprocess : 0.006981
      Model Inference : 1.684508
-----------------------------------
-----------------------------------
       max and argmax : 0.010963
                  nms : 0.000998
Post processing total : 0.011960
-----------------------------------
-----------------------------------
           Preprocess : 0.000998
      Model Inference : 0.103723
-----------------------------------
-----------------------------------
       max and argmax : 0.002991
                  nms : 0.000998
Post processing total : 0.003989
-----------------------------------
please give namefile
Traceback (most recent call last):
  File "models.py", line 512, in <module>
    class_names = load_class_names(namesfile)
  File "D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4\tool\utils.py", line 157, in load_class_names
    with open(namesfile, 'r') as fp:
TypeError: expected str, bytes or os.PathLike object, not NoneType


python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320 ./data/coco.names  成功了但是没有边框
(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320 ./data/coco.names
-----------------------------------
           Preprocess : 0.001995
      Model Inference : 1.695494
-----------------------------------
-----------------------------------
       max and argmax : 0.001995
                  nms : 0.000000
Post processing total : 0.001995
-----------------------------------
-----------------------------------
           Preprocess : 0.001995
      Model Inference : 0.092752
-----------------------------------
-----------------------------------
       max and argmax : 0.000997
                  nms : 0.000000
Post processing total : 0.000997
-----------------------------------
save plot results to predictions.jpg

python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/dog.jpg  320 320 ./data/coco.names
python models.py 80 ./weights/yolov4.pth  ./data/dog.jpg  320 320 ./data/coco.names 
利用别人的yolov.pth成功了
(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 80 ./weights/yolov4.pth  ./data/dog.jpg  320 320 ./data/coco.names
-----------------------------------
           Preprocess : 0.000998
      Model Inference : 1.646615
-----------------------------------
-----------------------------------
       max and argmax : 0.046877
                  nms : 0.000996
Post processing total : 0.047873
-----------------------------------
-----------------------------------
           Preprocess : 0.002992
      Model Inference : 0.197523
-----------------------------------
-----------------------------------
       max and argmax : 0.014913
                  nms : 0.000997
Post processing total : 0.015910
-----------------------------------
bicycle: 0.993608
truck: 0.880864
dog: 0.991610
tvmonitor: 0.470089
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./checkpoints/Yolov4_epoch19.pth  ./data/input/P0003__1__0___0.jpg  320 320 ./data/x.names
-----------------------------------
           Preprocess : 0.001995
      Model Inference : 1.675039
-----------------------------------
-----------------------------------
       max and argmax : 0.000997
                  nms : 0.000000
Post processing total : 0.000997
-----------------------------------
-----------------------------------
           Preprocess : 0.000998
      Model Inference : 0.116688
-----------------------------------
-----------------------------------
       max and argmax : 0.000000
                  nms : 0.000999
Post processing total : 0.000999
-----------------------------------
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv41>python models.py 80 ./weights/yolov4.pth  ./data/dog.jpg  320 320
-----------------------------------
           Preprocess : 0.001995
      Model Inference : 1.719244
-----------------------------------
-----------------------------------
       max and argmax : 0.000000
                  nms : 0.000000
Post processing total : 0.000000
-----------------------------------
-----------------------------------
           Preprocess : 0.000000
      Model Inference : 0.127398
-----------------------------------
-----------------------------------
       max and argmax : 0.001995
                  nms : 0.001995
Post processing total : 0.003990
-----------------------------------
bicycle: 0.993608
truck: 0.880864
dog: 0.991610
tvmonitor: 0.470089
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv41>python models.py 80 ./checkpoints/Yolov4_epoch2.pth  ./data/dog.jpg  320 320
-----------------------------------
           Preprocess : 0.001995
      Model Inference : 1.784540
-----------------------------------
-----------------------------------
       max and argmax : 0.002991
                  nms : 0.000000
Post processing total : 0.002991
-----------------------------------
-----------------------------------
           Preprocess : 0.000997
      Model Inference : 0.115674
-----------------------------------
-----------------------------------
       max and argmax : 0.000000
                  nms : 0.000000
Post processing total : 0.000000
-----------------------------------
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv41>python models.py 80 ./checkpoints/Yolov4_epoch2.pth  ./data/dog.jpg  320 320
-----------------------------------
           Preprocess : 0.000000
      Model Inference : 1.668415
-----------------------------------
-----------------------------------
       max and argmax : 0.000000
                  nms : 0.000000
Post processing total : 0.000000
-----------------------------------
-----------------------------------
           Preprocess : 0.000996
      Model Inference : 0.091623
-----------------------------------
-----------------------------------
       max and argmax : 0.002992
                  nms : 0.000998
Post processing total : 0.003990
-----------------------------------
save plot results to predictions.jpg

(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16  ./weights/yolo2dota.weights  ./data/input/P0003__1__0___0.jpg 320 320
Traceback (most recent call last):
  File "models.py", line 479, in <module>
    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\serialization.py", line 529, in load
    return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
  File "C:\Users\19339\AppData\Local\conda\conda\envs\pytorch\lib\site-packages\torch\serialization.py", line 692, in _legacy_load
    magic_number = pickle_module.load(f, **pickle_load_args)
_pickle.UnpicklingError: invalid load key, '\x00'.
1128
(pytorch) D:\myworklist\deeplearning\paper\objectdetection\program\pytorchYOLOv4>python models.py 16 ./checkpoints/Yolov4_epoch36.pth ./data/input/21.jpg 416 416 ./data/x.names

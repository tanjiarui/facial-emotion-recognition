# <div align='center'>Documentation</div>
<div align='center'><img src='demo.gif' alt='demo'></div>

### <div align='center'>Getting Started</div>
this is an example of how you may give instructions on setting up your project
<details open>
<summary>install</summary>
install all dependencies in Linux

```bash
sudo apt update ; sudo apt -y dist-upgrade
sudo apt install -y cmake python3-opencv
pip3 install --upgrade pip
pip3 install -r requirement
cd ./efficientdet
python3 setup.py build_ext --inplace # compile the function compute_overlap
mv efficientdet/utils/compute_overlap.cpython* utils
cd ..
```
</details>
<details open>
<summary>configuration</summary>
<ol>
<li>

download the dataset [CAER](https://caer-dataset.github.io). the static version is applied in this project
</li>
<li>

edit `config.py`</li>
</ol>
</details>

### <div align='center'>Preprocessing</div>
<details open>
<summary>annotation</summary>
due to the dataset only classifies the emotion, the script to annotate faces is necessary. only one face is classified within each image, but there might be multiple faces. thus, there must be a rule to filter out the right face that is identified as the target. in this case, the closer to the image center, the higher likelihood to be the target face

```bash
python3 bounding\ box.py
```
</details>
<details open>
<summary>json encoder</summary>
the output of annotation is csv, while the data generator loads json. that's where this script fits in

```bash
python3 encoding.py
```
</details>

### <div align='center'>Modeling</div>
<details open>
<summary>training</summary>

EfficientDet4 is applied for this application
```bash
python3 train.py
```
</details>
<details open>
<summary>evaluation</summary>

the evaluation follows on [COCO metric](https://cocodataset.org/#detection-eval)
```bash
python3 evaluation.py
```
</details>
<details open>
<summary>explainable model</summary>

explaining model features by visualizing one selected image as a heatmap
```bash
python3 grad\ cam.py
```
</details>

### <div align='center'>Deployment</div>
<details open>
<summary>convert the model</summary>
export the trained model to accelerate the efficiency

```bash
python3 model\ convert.py
```
</details>
<details open>
<summary>inference</summary>

`inference.py` applies facial emotion recognition over a video
```bash
python3 inference.py
```
</details>

### <div align='center'>Improvement</div>
According to mAP over the test set, the model fitted in a satisfactory manner, but it actually becomes weak during the inference. try to set a greater phi to lift the model capacity
```text
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.725
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.784
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.782
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.388
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.744
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.847
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.850
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.850
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.625
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.812
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.863
```
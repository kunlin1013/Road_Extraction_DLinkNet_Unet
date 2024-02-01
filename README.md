## Use deep learning to implement road extraction.

### Reference paper
[D-LinkNet](https://ieeexplore.ieee.org/document/8575492)

### Dataset 
[DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset) provide 6226 satellite images and their ground truth. 
But we found that all of them are foreign roads, lacking diversity. Therefore, we additionally labeled 3772 images from Google Map, which include roads in Taiwan and abroad.

### Data Augmentation
Because the road in the world are constantly changing, it's impossible to include all scenarios in the data. 
Therefore, we use Data Augmentation techniques such as zoom, translation, rotation, color jittering, or horizontal and vertical flip to expand the dataset, making the data more diverse.

### Reference version of packages
```
- Python                3.7.9
- scikit-learn          1.0.2
- numpy                 1.16.2
- pandas                1.3.5
- typing-extensions     4.2.0
- imgaug                0.4.0
- opencv-python         4.5.4.60
- matplotlib            3.5.3
- tensorflow-gpu        2.4.0
- segmentation-models   1.0.1
```

### Directory structure
```
|
|-- img: Images in the README.md
|
|-- Train_DLinkNet.py: Train DLinkNet
|-- Test_DLinkNet.py: Inference using DLinkNet on DeepGlobe Road Extraction 
|-- Train_Unet.py: Train Unet
|-- Test_Unet.py: Inference using Unet on DeepGlobe Road Extraction 
|-- Test_SegFormer_unseen.py: Inference using SegFormer on unseen data
|-- Find_thres_ROC.py: Use the ROC curve to analyze the threshold for binarization
```








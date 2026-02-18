## Features

This repository represents an example for deploying PyTorch models to Android with the [ExecuTorch](https://github.com/pytorch/executorch) framework and comprises:
  - conversion of a pre-trained [YOLO face detection](https://github.com/akanametov/yolo-face) model, and two ViT models for [age, gender](https://huggingface.co/abhilash88/age-gender-prediction) & [facial emotion recognition](https://huggingface.co/abhilash88/face-emotion-detection) from PyTorch to ExecuTorch
  - complete Android application utilizing the ExecuTorch models to analyze faces in pictures or from the live camera view

<img width="2398" height="948" alt="screenshot_wide" src="https://github.com/user-attachments/assets/e1bf2711-8cfb-4191-84da-baa11b875897" />


## Conversion

### Environment Setup

Python version 3.10 or newer needs to be installed on your machine.

As all employed machine learning models are based on PyTorch, you need to install torch & torchvision. Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) for your environment. The newest version 2.10 is highly recommended. For the conversion all models are mapped to the CPU for alignment, so it doesn't matter whether or not you're installing the CUDA variants.

To load the YOLO model, you need the ultralytics library, and for the ViT models the transformers library is required. 

```commandline
pip install ultralytics>=8.4.9 transformers>=4.7.0
```

For the conversion the accelerate and ExecuTorch libraries are necessary.

```commandline
pip install accelerate executorch>=1.1.0
```

ExecuTorch will also install TorchAO, but depending on your version of PyTorch, you might need to up- or downgrade. Check the [compatibility table](https://github.com/pytorch/ao/issues/2919). For ExecuTorch 1.1.0 and PyTorch 2.10, you should manually upgrade to 0.16.0.

```commandline
pip install torchao==0.16.0
```


### Lowering without quantization

Technically you can convert the models to ExecuTorch directly without quantization. For this, simply call the python routine:

```commandline
export_executorch.py
```

This will download the models if necessary and create .pte ExecuTorch packages with the default XNNPack delegates under "./android/app/src/main/assets".
The process should take no more than a minute or two after downloading.


### Lowering with quantization

The considerably reduce the file size of the ViT models, and increase the inference speed [post-training quantization](https://docs.pytorch.org/executorch/1.1/quantization-overview.html) is recommended.

For this, sample inputs for the models are required. The python routine is designed to use select parts of the datasets the models were trained on.
For YOLO face, this is [WIDER Face](https://huggingface.co/datasets/CUHK-CSE/wider_face/blob/main/data/WIDER_val.zip). You also need to get the [annotations](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip).
For the age & gender model, it is [UTKFace](https://www.kaggle.com/datasets/moritzm00/utkface-cropped).
For the emotion recognition model, download [FER2013](https://www.kaggle.com/datasets/msambare/fer2013).

Unzip all the archives into the root repository directory. Afterwards the folder structure should look like:

```folder structure
-android
-fer2013
--test
--train
-models
-utkface
--utkface_aligned_cropped
---UTKFace
-wider_face_split
-WIDER_val
--images
```

With version 1.1.0 of ExecuTorch the quantization of the YOLO model only works consistently with the Vulkan backend. On the other hand, quantization of the ViT models only with the XNNPack backend. Taking this into account, call the python routine with the following arguments:

```commandline
export_executorch.py --yolo_partitioner vulkan --yolo_quantize --vit_quantize
```


### Configuration

By default, the python routine creates two YOLO ExecuTorch models, one for portrait and one for landscape orientation - with input sizes of 768x1024 and 1024x768 respectively. ExecuTorch technically also supports dynamic input shapes, but the results with the YOLO face model were not satisfactory, as detected bounding boxes were slightly off horizontally and faces on the far right of images were not detected properly.

If you want to experiment with it, change the fixed input sizes, use a different YOLO version, or adjust other options, check out all the available arguments for the python routine:

```commandline
export_executorch.py -h
```

Be aware that when exporting with and without dynamic input shapes, you have to manually delete the unwanted models from "./android/app/src/main/assets". Otherwise, the Android application will prefer the dynamic model, if it is found.


You could also use different models than the ones from [abhilash88](https://huggingface.co/abhilash88) for the face analysis, by changing the model identifiers in the transformers ".from_pretrained" calls in the export_executorch.py file. Different facial emotion recognition models, based on the same [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) base model, have been tested successfully without any changes to the code. Other models might need more experimentation.



## Android application

To build the Android application, you need a current version of [Android Studio](https://developer.android.com/studio).

After installation, open the project from the "android" folder, and follow the prompt to "Sync" the gradle project.

When using the default export, simply press "Run 'app'" to try the app on a virtual device, or go to "Build -> Generate (Signed) App Bundle or Apk" to package the app for your Android phone.


If you have adjusted the YOLO input size during conversion, you need to set "INPUT_DIMENSION_SHORT" and "INPUT_DIMENSION_LONG" in the FaceAnalysisPipeline class accordingly. There you can also change the confidence and non-maximum suppression thresholds, in case faces are not detected as expected.

To set input sizes that do not have a 3:4 / 4:3 aspect ratio, you will have to make additional changes:
Firstly, build a different ResolutionSelector in the "bindPreviewAndAnalyzer" method of the FaceAnalysisActivity.
Secondly, adjust all the "layout_constraintDimensionRatio" values in both main_activity layouts.


If you didn't use the Vulkan backend for any of the models, you can change the dependency in the app's build.gradle.kts from
```code
implementation("org.pytorch:executorch-android-vulkan:1.1.0")
```
to
```code
implementation("org.pytorch:executorch-android:1.1.0")
```


## Disclaimer

The logo for the Android app was created with a local installation of [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image).

No other part of the source code in this repository or this documentation was created by or with the help of artificial intelligence.

import argparse
import cv2
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config as xnn_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.quantizer.vulkan_quantizer import VulkanQuantizer, get_symmetric_quantization_config as vulkan_config
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
import executorch.backends.vulkan.serialization.vulkan_graph_schema as vk_graph_schema
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.runtime import Runtime
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
import torch
import torchvision.transforms as transforms
from transformers import AutoConfig, ViTImageProcessor, ViTForImageClassification

from models.age_gender_model import AgeGenderViTModel
from models.yolo_detect_model import YOLOModel


# handle everything on CPU to keep memory aligned
DEVICE = "cpu"

def export(model, output_path, output_name, input_height, input_width, dynamic_shapes = None,
           dry_run = False, backend = "xnn", quantize = False, force_fp16 = False, transform = None, samples = []):
               
    # similar to "with torch.no_grad()", save memory
    for p in model.parameters():
        p.requires_grad = False
    
    # reduce some overhead
    for m in model.modules():
        m.export = True
        
        
    # define input dimensions
    example_inputs = (torch.randn(1, 3, input_height, input_width).to(DEVICE),)
              
    if dry_run:
        # Yolo needs a dry run before exporting
        # see https://github.com/pytorch/executorch/issues/14644
        model(*example_inputs)
        
        
    if backend == "xnn":
        partitioner = XnnpackPartitioner()
        qparams = xnn_config(is_dynamic=True, is_per_channel=True)
        quantizer = XNNPACKQuantizer()
        
    else: # vulkan
        partitioner = VulkanPartitioner(compile_options={
            "force_fp16": force_fp16,
            # if the model behaves differently on x86 and ARM, try to force a memory layout
            #"memory_layout_override": vk_graph_schema.VkMemoryLayout.TENSOR_CHANNELS_PACKED
            #"memory_layout_override": vk_graph_schema.VkMemoryLayout.TENSOR_WIDTH_PACKED
        })
        qparams = vulkan_config(is_dynamic=False, weight_bits=8)
        quantizer = VulkanQuantizer()
        
  
    if quantize:      
        print("Quantization")
        
        quantizer.set_global(qparams)

        training_ep = torch.export.export(model, example_inputs).module().to(DEVICE)
        prepared_model = prepare_pt2e(training_ep, quantizer)

        if transform is None:
            transform = transforms.ToTensor()

        for cal_sample in samples:
            # read calibration image
            image = cv2.imread(cal_sample)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (input_width,input_height), interpolation=cv2.INTER_LINEAR)

            # create tensor fitting to model input
            img_cv2 = transform(image)
            img_cv2 = img_cv2.reshape(1,3,input_height,input_width).to(DEVICE)
            
            # calibrate quantization
            prepared_model(img_cv2)

        model = convert_pt2e(prepared_model)
    

    # export
    print(f"Exporting with {backend} partitioner")
    exported_program = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes)
    
    program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[partitioner],
        compile_config=EdgeCompileConfig(_check_ir_validity=False)
    ).to_executorch(
        config=ExecutorchBackendConfig(extract_delegate_segments=False)
    )
    
    # verify XNN output
    # if you want to verify Vulkan output
    # you need to manually build the executorch runtime library with -DEXECUTORCH_BUILD_VULKAN=ON
    if backend == "xnn":
        expected_output = program.exported_program().module()(*example_inputs)
        runtime_output = Runtime.get().load_program(program.buffer).load_method("forward").execute(example_inputs)
        if isinstance(expected_output, tuple):
            print(f"expected output: {expected_output[0].shape}, {expected_output[1].shape}")
            print(f"runtime output: {runtime_output[0].shape}, {runtime_output[1].shape}")
        else:
            print(f"expected output: {expected_output[0].shape}")
            print(f"runtime output: {runtime_output[0].shape}")

    # save as .pte
    with open(Path(output_path) / f"{output_name}.pte", "wb") as f:
        f.write(program.buffer)


def divisible(value):
    ivalue = int(value)
    if ivalue % 32 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not divisible by 32")
    return ivalue



if __name__ == "__main__":

    # get arguments from command line
    parser = argparse.ArgumentParser(description="Downloads a YOLO model for face detection, as well as two vision transformer models for age, gender and emotion prediction, and exports them as .pte executorch models.")

    parser.add_argument("--input_width", type=divisible, default=768,
        help="smaller side of input for model(s) / width of sample input for dynamic model")
    parser.add_argument("--input_height", type=divisible, default=1024,
        help="larger side of input for model(s) / height of sample input for dynamic model")
    parser.add_argument("--dynamic_input", action='store_true',
        help="creates dynamic dimensions, so only one Yolo model is needed")
    parser.add_argument("--output_path", type=str, default="./android/app/src/main/assets")
        
    parser.add_argument("--yolo_model", type=str, default="yolo26n-face.pt",
        choices=["yolov8n-face.pt", "yolov8s-face.pt", "yolov11n-face.pt", "yolov11s-face.pt", "yolo26n-face.pt"])
    # yolov12_-face outputs random values after lowering
    # as of 2026-02-18: other yolo26-face models (e.g., s,m,l) are not available yet
    
    parser.add_argument("--yolo_partitioner", type=str,
        choices=["xnn", "vulkan"], default="xnn", help="xnn inference is slower")
    parser.add_argument("--yolo_forcefp16", action='store_true',
        help="force conversion from fp32 to fp16 for vulkan; can be combined with quantization; does not affect xnn")
    parser.add_argument("--yolo_quantize", action='store_true',
        help="improvements (for n model) are negligible, but dynamic inputs only work with quantization")
    parser.add_argument("--wider_face_ann", type=str,
        default="./wider_face_split/wider_face_val_bbx_gt.txt", help="path to annotations file from Wider Face dataset; needed for quantization")
    parser.add_argument("--wider_face_img", type=str,
        default="./WIDER_val/images/", help="path to folder with images from Wider Face dataset; needed for quantization")
    
    parser.add_argument("--vit_partitioner", type=str,
        choices=["xnn", "vulkan"], default="xnn", help="vulkan export works, but inference is unreliable")
    parser.add_argument("--vit_quantize", action='store_true',
        help="quantizing the ViT models is highly recommended, as it vastly reduces model size and inference times")
    parser.add_argument("--fer2013_img", type=str,
        default="./fer2013/test", help="path to folder with images from FER2013 dataset; needed for quantization")
    parser.add_argument("--utkface_img", type=str,
        default="./utkface/utkface_aligned_cropped/UTKFace", help="path to folder with images from UTKFace dataset; needed for quantization")

    args = parser.parse_args()
    
    assert not (args.yolo_partitioner == "xnn" and args.yolo_quantize), "YOLO quantization only works with Vulkan backend"
        
    assert not (args.dynamic_input and not args.yolo_quantize), "Dynamic inputs for YOLO only work with quantization"


    print("Converting face detection model")
    
    # convert YOLO face detection model
    # from https://github.com/akanametov/yolo-face
    model = YOLOModel(args.yolo_model)
    
    samples = []
    if args.yolo_quantize:
        print("Collecting samples for calibration")
        
        image = ""
        read_bbx = False
        
        samples_path = Path(args.wider_face_img)
        with open(args.wider_face_ann, 'r') as file:
            for line in file:
                a = line.strip()
                if len(line) > 1 and line[1] == '-':
                    image = a
                    read_bbx = True
                elif read_bbx:
                    if int(a) < 6: #select all images with less than 6 faces
                        samples.append(str(samples_path / image))
                    read_bbx = False
    
    if args.input_height == args.input_width:

        export(model, args.output_path, "face_detector", args.input_height, args.input_width,
            dry_run=True, backend=args.yolo_partitioner, quantize=args.yolo_quantize, samples=samples, force_fp16=args.yolo_forcefp16)

        print("Succesfully converted face detection model for square input.")
    
    elif args.dynamic_input:
        
        minDim = int(min(args.input_width, args.input_height) / 32)
        maxDim = int(max(args.input_width, args.input_height) / 32)
        dynamic_shapes = {
            "x": {
                2: 32 * torch.export.Dim("h", min=minDim, max=maxDim),
                3: 32 * torch.export.Dim("w", min=minDim, max=maxDim),
            }
        }
        
        export(model, args.output_path, "face_detector", args.input_height, args.input_width, dynamic_shapes=dynamic_shapes,
            dry_run=True, backend=args.yolo_partitioner, quantize=args.yolo_quantize, samples=samples, force_fp16=args.yolo_forcefp16)

        print("Succesfully converted face detection model for dynamic input.")
    
    else: #export separate models for portrait & landscape
        
        export(model, args.output_path, "face_detector_portrait", args.input_height, args.input_width,
            dry_run=True, backend=args.yolo_partitioner, quantize=args.yolo_quantize, samples=samples, force_fp16=args.yolo_forcefp16)
          
        print("Succesfully converted portrait face detection model.")
        
        #reload model
        model = YOLOModel(args.yolo_model)
   
        export(model, args.output_path, "face_detector_landscape", args.input_width, args.input_height,
            dry_run=True, backend=args.yolo_partitioner, quantize=args.yolo_quantize, samples=samples, force_fp16=args.yolo_forcefp16)

        print("Succesfully converted landscape face detection model.")


    print("Converting emotion recognition model")
    
    # convert facial emotion recognition model
    # from https://huggingface.co/abhilash88/face-emotion-detection
    model = ViTForImageClassification.from_pretrained(
        'abhilash88/face-emotion-detection',
        dtype="auto",
        device_map=DEVICE
    )
    
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    samples = []
    if args.vit_quantize:
        print("Collecting samples for calibration")
        
        samples_path = Path(args.fer2013_img)
        for emotion_dir in samples_path.iterdir():
            if emotion_dir.is_dir():
                img_filter = "*.jpg"
                images = emotion_dir.glob(img_filter)
                for i in range(100): #select 100 images for each emotion
                    samples.append(str(next(images)))
        
    export(model, args.output_path, "emotion", 224, 224,
        backend=args.vit_partitioner, quantize=args.vit_quantize, transform=transform, samples=samples)
    print("Succesfully converted emotion recognition model.")
    
    
    print("Converting age & gender prediction model")
    
    # convert age & gender prediction model
    # from https://huggingface.co/abhilash88/age-gender-prediction
    config = AutoConfig.from_pretrained(
        "abhilash88/age-gender-prediction",
        trust_remote_code=True
    )
        
    model = AgeGenderViTModel.from_pretrained(
        "abhilash88/age-gender-prediction",
        dtype="auto",
        device_map=DEVICE,
        config=config,
        trust_remote_code=True
    )
    
    model.eval().to(DEVICE)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    samples = []
    if args.vit_quantize:
        print("Collecting samples for calibration")
        
        samples_path = Path(args.utkface_img)
        for age in range(1,91):
            for gender in [0,1]: 
                img_filter = f"{age}_{gender}_*.jpg"
                for i in range(2): #select 2 images for each gender, from age 1 to 90
                    samples.append(str(next(samples_path.glob(img_filter))))
        
    export(model, args.output_path, "age_gender", 224, 224,
        backend=args.vit_partitioner, quantize=args.vit_quantize, transform=transform, samples=samples)
    print("Succesfully converted age & gender prediction model.")
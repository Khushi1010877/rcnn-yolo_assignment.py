#!/usr/bin/env python3
# Object Detection: R-CNN & YOLO - Theoretical & Practical Q&A
# Theoretical answers are commented (#), practical answers are active code.

# ========================== THEORETICAL QUESTIONS ==========================
# (All theoretical answers remain as comments)

#1. What is the main purpose of RCNN in object detection?
# Solution: RCNN (Regions with CNN) proposes region candidates using selective search, then classifies each region using a CNN. Its purpose is to detect multiple objects in an image by combining region proposal with deep features.

#2. What is the difference between Fast RCNN and Faster RCNN?
# Solution: Fast RCNN uses selective search for region proposals and applies a single CNN to the whole image, then RoI pooling. Faster RCNN replaces selective search with a Region Proposal Network (RPN) that shares convolutional features, making it faster and fully trainable.

#3. How does YOLO handle object detection in real-time?
# Solution: YOLO reframes detection as a single regression problem, dividing the image into a grid and predicting bounding boxes and class probabilities directly in one forward pass, enabling real-time speeds.

#4. Explain the concept of Region Proposal Networks (RPN) in Faster RCNN.
# Solution: RPN is a fully convolutional network that slides a small window over the feature map, outputs objectness scores and bounding box offsets for anchors. It generates high-quality region proposals without external algorithms.

#5. How does YOLOv9 improve upon its predecessors?
# Solution: YOLOv9 introduces Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN), improving parameter efficiency, gradient flow, and accuracy while maintaining real-time performance.

#6. What role does non-max suppression play in YOLO object detection?
# Solution: NMS removes duplicate detections by suppressing bounding boxes with lower confidence that highly overlap with the highest-confidence box, ensuring each object is detected once.

#7. Describe the data preparation process for training YOLOv9.
# Solution: Images are resized to a fixed input size (e.g., 640x640), annotations converted to YOLO format (class_id, x_center, y_center, width, height normalized). Data augmentation (Mosaic, MixUp, random flips, HSV shifts) is applied.

#8. What is the significance of anchor boxes in object detection models like YOLOv9?
# Solution: Anchor boxes are predefined bounding box shapes (width, height) that serve as references. The model predicts offsets from these anchors, making learning easier, especially for objects of varying aspect ratios.

#9. What is the key difference between YOLO and R-CNN architectures?
# Solution: R-CNN is a two-stage detector (region proposal + classification), while YOLO is a single-stage detector that directly predicts boxes and classes from the whole image in one pass. YOLO is faster, R-CNN often more accurate.

#10. Why is Faster RCNN considered faster than Fast RCNN?
# Solution: Fast RCNN uses selective search (CPU-based, slow) for proposals. Faster RCNN uses an RPN integrated into the GPU pipeline, sharing convolutional features, eliminating the separate region proposal bottleneck.

#11. What is the role of selective search in RCNN?
# Solution: Selective search generates ≈2000 region proposals by hierarchically grouping similar regions based on color, texture, size, and shape. It provides candidate object locations for classification.

#12. How does YOLOv9 handle multiple classes in object detection?
# Solution: For each grid cell, YOLOv9 predicts class probabilities (using sigmoid for multi-label) along with bounding box predictions. The final detection includes class labels with highest confidence after NMS.

#13. What are the key differences between YOLOv3 and YOLOv9?
# Solution: YOLOv9 uses advanced backbone (GELAN), better training strategies (PGI), larger input sizes, more efficient downsampling, improved loss functions, and achieves higher mAP with similar or fewer parameters.

#14. How is the loss function calculated in Faster RCNN?
# Solution: Total loss = RPN loss (binary cross-entropy for objectness + smooth L1 for bounding box regression) + Fast RCNN loss (cross-entropy for class + smooth L1 for refined boxes).

#15. Explain how YOLOv9 improves speed compared to earlier versions.
# Solution: Through architectural optimizations like reduced computation in GELAN, better gradient flow minimizing redundant calculations, and using lighter activation functions. PGI also improves convergence speed.

#16. What are some challenges faced in training YOLOv9?
# Solution: Need large annotated datasets, careful hyperparameter tuning (learning rate, anchor sizes), class imbalance, small object detection, and avoiding overfitting with strong data augmentation.

#17. How does the YOLOv9 architecture handle large and small object detection?
# Solution: YOLOv9 uses multi-scale feature fusion (FPN-like) with three detection heads (small, medium, large objects). Larger feature maps detect small objects, deeper layers detect large objects.

#18. What is the significance of fine-tuning in YOLO?
# Solution: Fine-tuning adapts a pre-trained YOLO model (e.g., on COCO) to a custom dataset with fewer images, saving training time, improving performance on domain-specific classes.

#19. What is the concept of bounding box regression in Faster RCNN?
# Solution: Bounding box regression learns to adjust the proposed anchor boxes (dx, dy, dw, dh) to better fit the ground truth. It uses smooth L1 loss to predict offsets.

#20. Describe how transfer learning is used in YOLO.
# Solution: YOLO is pre-trained on large datasets (COCO, ImageNet). The backbone weights are frozen or fine-tuned on a custom dataset, replacing the last detection layers for new classes.

#21. What is the role of the backbone network in object detection models like YOLOv9?
# Solution: The backbone extracts hierarchical visual features (edges, shapes, textures) from the input image. In YOLOv9, GELAN acts as the backbone, balancing efficiency and accuracy.

#22. How does YOLO handle overlapping objects?
# Solution: YOLO predicts multiple bounding boxes per grid cell. If two objects overlap significantly, different grid cells or anchor boxes may capture them, and NMS removes duplicates while preserving distinct objects.

#23. What is the importance of data augmentation in object detection?
# Solution: Augmentation (random flips, rotations, scaling, color jitter, mosaic) increases data diversity, improves generalization, reduces overfitting, and helps the model handle varying object sizes and lighting.

#24. How is performance evaluated in YOLO-based object detection?
# Solution: Metrics include mAP (mean Average Precision) at IoU thresholds (e.g., 0.5:0.95), precision, recall, F1-score, and inference speed (FPS) on hardware like GPU.

#25. How do the computational requirements of Faster RCNN compare to those of YOLO?
# Solution: Faster RCNN requires more computation (≈ 20-50 GFLOPS) due to two-stage pipeline, while YOLO (e.g., YOLOv9) needs much less (≈ 10-30 GFLOPS) and runs faster (30-100+ FPS vs 10-20 FPS for Faster RCNN).

#26. What role do convolutional layers play in object detection with RCNN?
# Solution: Convolutional layers extract feature maps from the whole image. In RCNN, these features are used by the RPN (Faster RCNN) and later by RoI pooling and classification heads.

#27. How does the loss function in YOLO differ from other object detection models?
# Solution: YOLO combines localization (box coordinates), objectness (confidence), and class prediction into a single loss. It weighs different components (e.g., higher weight for box coordinates) and uses sum-squared error for simplicity.

#28. What are the key advantages of using YOLO for real-time object detection?
# Solution: Single-stage unified architecture enables very high inference speed (30-100+ FPS), good accuracy for many applications, end-to-end trainability, and simpler deployment compared to two-stage detectors.

#29. How does Faster RCNN handle the trade-off between accuracy and speed?
# Solution: By using the RPN that shares features, it speeds up proposals but still retains high accuracy. Accuracy can be improved with deeper backbones (ResNet-101/152) at the cost of speed; lighter backbones (MobileNet) increase speed but reduce accuracy.

#30. What is the role of the backbone network in both YOLO and Faster RCNN, and how do they differ?
# Solution: Both use backbone (e.g., Darknet, ResNet) to extract features. YOLO uses a backbone designed for single-stage detection (Darknet‑9/59), often with cross-stage connections. Faster RCNN typically uses ResNet/ VGG, with the backbone's last features fed into RPN and later heads.

# PRACTICAL CODING QUESTIONS 
# (Using YOLOv8 as per note: "YOLOv8 model (labeled as YOLOv9)")
# Practical answers are active Python code (no leading #).

#1. How do you load and run inference on a custom image using the YOLOv8 model (labeled as YOLOv9)?

# Solution:
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg')
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy()

#2. How do you load the Faster RCNN model with a ResNet50 backbone and print its architecture?

# Solution:
import torchvision
model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
print(model_frcnn)

#3. How do you perform inference on an online image using the Faster RCNN model and print the predictions?

# Solution:
import requests
from PIL import Image
import torchvision.transforms as transforms
img = Image.open(requests.get('https://example.com/image.jpg', stream=True).raw)
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).unsqueeze(0)
model_frcnn.eval()
with torch.no_grad():
    predictions = model_frcnn(img_tensor)
print(predictions)

#4. How do you load an image and perform inference using YOLOv9, then display the detected objects with bounding boxes and class labels?

# Solution:
import cv2
from ultralytics import YOLO
yolo_model = YOLO('yolov8n.pt')  # use appropriate YOLOv9 weights
img = cv2.imread('image.jpg')
results = yolo_model(img)
annotated = results[0].plot()
cv2.imshow('Detection', annotated)
cv2.waitKey(0)

#5. How do you display bounding boxes for the detected objects in an image using Faster RCNN?

# Solution:
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig, ax = plt.subplots(1)
ax.imshow(img)
for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
    if score > 0.5:
        x, y, x2, y2 = box.detach().cpu().numpy()
        rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
plt.show()

#6. How do you perform inference on a local image using Faster RCNN?

# Solution:
img_local = Image.open('local_image.jpg').convert('RGB')
transform = transforms.Compose([transforms.ToTensor()])
img_tensor_local = transform(img_local).unsqueeze(0)
with torch.no_grad():
    pred_local = model_frcnn(img_tensor_local)

#7. How can you change the confidence threshold for YOLO object detection and filter out low-confidence predictions?

# Solution:
results_thresh = yolo_model('image.jpg', conf=0.25)
boxes = results_thresh[0].boxes
high_conf = boxes[boxes.conf > 0.5]

#8. How do you plot the training and validation loss curves for model evaluation?

# Solution:
import matplotlib.pyplot as plt
# Assuming history dict from training
history = {'train_loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4]}  # example
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#9. How do you perform inference on multiple images from a local folder using Faster RCNN and display the bounding boxes for each?

# Solution:
import glob
for img_path in glob.glob('folder/*.jpg'):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model_frcnn(img_tensor)
    # display boxes (similar to #35)
    print(f'Processed {img_path}')

#10. How do you visualize the confidence scores alongside the bounding boxes for detected objects using Faster RCNN?

# Solution:
fig, ax = plt.subplots(1)
ax.imshow(img)
for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']):
    if score > 0.5:
        x, y, x2, y2 = box.detach().cpu().numpy()
        rect = patches.Rectangle((x, y), x2-x, y2-y, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f'{score:.2f}', color='g', fontsize=10)
plt.show()

#11. How can you save the inference results (with bounding boxes) as a new image after performing detection using YOLO?

# Solution:
results_save = yolo_model('input.jpg')
results_save[0].save('output.jpg')
annotated_save = results_save[0].plot()
cv2.imwrite('output.jpg', annotated_save)
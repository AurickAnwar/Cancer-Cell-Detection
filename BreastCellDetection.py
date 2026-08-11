from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
import cv2
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

class CancerCNN(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # The 3-color input stacked on top off each other, grid of pixels is 64x64,
      #Run 16 features over the 3 sheets, could be highlight horizontal lines, dark circular shapes, etc, each detector uses a 3x3 pixel window to scan the image
      #piece by piece, add a 1-pixel fake border around the edges so the scanner doesn't shrink the 64x64 width and height while scanning
      self.relu =nn.ReLU()
      self.pool = nn.MaxPool2d(2)#after each convolution, we apply max pooling to reduce spatial dimensions, 16 x 32 x 32
      self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1)#32 output layers, 32 x 16 x 16
      self.conv3 = nn.Conv2d(32,64, kernel_size=3, padding=1)#64 output layers, 64 x 8 x 8
      self.flatten = nn.Flatten()#converts 3D tensor to 1D tensor -- > [64, 8, 8] --> [4096]
      self.fc1 = nn.Linear(4096, 256)#64 of 8*8 pixels and 256 output layers
      self.fc2 = nn.Linear(256, 2)#2 output layers, Benign and Malignant

   def forward(self, x):
      x = self.pool(self.relu(self.conv1(x))) #Takes image, runs through conv1, applies ReLU to remove negatives, shrink it in half with pool making image half the size
      x = self.pool(self.relu(self.conv2(x)))
      x = self.pool(self.relu(self.conv3(x)))
      x = self.flatten(x)#converts a 3D tensor to a 1D tensor
      x = self.relu(self.fc1(x))#applies ReLu to remove negatives. Takes 4096 numbers and compresses them down to 256 features.
      x = self.fc2(x)#2 output layers, Benign and Malignant
      return x

  
cnn_model = CancerCNN()
cnn_model.load_state_dict(torch.load("cancer_cnn.pth"))
cnn_model.eval()
cam = GradCAM(model = cnn_model, target_layers=[cnn_model.conv3]) #GradCAM object that will be used to generate heatmaps, Tells gradcam wherre to inspect the model's feature extraction
#Because conv3 is the deepest convolutional layer right before nn.Flatten(), it holds the highest-level spatial feature maps
img_path = "valid/images/116_jpg.rf.40b2319a800ddb713b9435284f9f6ffc.jpg"

yolo_model = YOLO("best.pt")
results = yolo_model(img_path)
boxes = results[0].boxes.xyxy#get the bounding box coordinates
image = cv2.imread(img_path)
classes = results[0].boxes.cls#get the class
annotated_frame = results[0].plot()


transform = transforms.Compose([
      transforms.Resize((64, 64)),#transforms the image to 64x64
      transforms.ToTensor(),#converts the image to a tensor
])

for i, (box,cls) in enumerate(zip(boxes, classes)):#iterate through the bounding boxes and classes
   x1,y1,x2,y2 = map(int, box)#convert the bounding box coordinates to integers
   crop = image[y1:y2, x1:x2]#crops image into a bounding box
   crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))#converts crop to PIL image, OpenCV reads images as BGR but PyTorch expect RGB
   crop_tensor = transform(crop_pil)#converts PIL image to tensor, uses transform to resize crop to 64 x 64 pixels and convert it to a tensor shape of [3, 64, 64]
   crop_tensor = crop_tensor.unsqueeze(0)#adds a batch dimension to the tensor, analyze 1 batch at a time [1. 3. 64. 64]

   with torch.no_grad():
      output = cnn_model(crop_tensor)#runs the crop through the CNN model, passes [1,3,64, 64] 
      predicted_class = torch.argmax(output, dim=1).item()#takes the two output scores and returns the index of the highest score
      class_names = ["Benign", "Malignant"] #1 batch 2 output logits, sample tensor([[-1.5,3.2]])
     
   grayscale_cam = cam(input_tensor=crop_tensor)#runs the crop through the GradCAM model, gradcam output --> [1, 64,64], get rid of RGB color channels
   grayscale_cam = grayscale_cam[0]#takes the first element of the grayscale_cam
   crop_resized = cv2.resize(crop, (64, 64))#resizes the crop to 64x64
   heatmap = cv2.applyColorMap((grayscale_cam*255).astype('uint8'), cv2.COLORMAP_JET)#applies the color map to the grayscale_cam
   overlay = cv2.addWeighted(crop_resized, 0.5, heatmap, 0.5, 0)#adds the crop and heatmap together
   overlay_large = cv2.resize(overlay, (image.shape[1], image.shape[0]))#resizes the overlay to the image size
   crop_resized = cv2.resize(crop, (image.shape[1], image.shape[0]))
   heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))#resizes the heatmap to the image size
   side_by_side = cv2.hconcat([crop_resized, heatmap_resized])#concatenates the crop and heatmap horizontally  
   probablities = torch.softmax(output, dim=1)#takes the two output scores and returns the probabilities 
   if predicted_class == 0:
      probability_text = f"Benign: {(probablities[0, 0].item() * 100):.2f}% "
   else:
      probability_text = f"Malignant: {(probablities[0, 1].item() * 100):.2f}%"#displays the probability of the class
   cv2.putText(side_by_side, probability_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)#displays the probability text on the side by side image
   cv2.imshow(f"Heatmap {i+1}", side_by_side)#displays the side by side image

cv2.imshow("YOLO", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

    


    

    



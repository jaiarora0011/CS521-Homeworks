import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
import matplotlib.pyplot as plt
import numpy as np

# Load the pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set model to evaluation mode

# Define the image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]   
    )
])

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

imagenet_path = './imagenet_samples'

# List of image file paths
image_paths = os.listdir(imagenet_path)

def gen_superpixels(input_image, img_name, method='slic'):
    numpy_image = np.array(input_image)
    
    def plot_superpixels(superpixels, img_name, method):
        plt.imshow(superpixels)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'superpixels/{method}_{img_name}', bbox_inches='tight', pad_inches=0)
        plt.clf()

    segments = None
    if method == 'slic':
        segments = slic(numpy_image, n_segments=100, compactness=10, start_label=1)
    elif method == 'felzenszwalb':
        segments = felzenszwalb(numpy_image, scale=100, sigma=0.5, min_size=50)
    elif method == 'quickshift':
        segments = quickshift(numpy_image, kernel_size=3, max_dist=6, ratio=0.5)
    else:
        assert False, 'Unssupported segmentation method'

    superpixels = mark_boundaries(numpy_image, segments)
    plot_superpixels(superpixels, img_name, method)

for img_path in image_paths:
    # Open and preprocess the image
    # my_img = os.path.join(img_path, os.listdir(img_path)[2])
    my_img = os.path.join(imagenet_path, img_path)
    input_image = Image.open(my_img).convert('RGB')
    gen_superpixels(input_image, img_path)
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Move the input and model to GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class index
    _, predicted_idx = torch.max(output, 1)
    predicted_idx = predicted_idx.item()
    predicted_synset = idx2synset[predicted_idx]
    predicted_label = idx2label[predicted_idx]

    print(f'Predicted label for {img_path}: {predicted_synset} ({predicted_label})')

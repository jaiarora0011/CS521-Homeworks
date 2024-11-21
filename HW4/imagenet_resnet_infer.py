import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import lasso_path

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

def plot_superpixels(superpixels, img_path):
        plt.imshow(superpixels)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.clf()

def gen_superpixels(input_image, img_name, method='slic'):
    numpy_image = np.array(input_image)

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
    plot_superpixels(superpixels, f'superpixels/{method}_{img_name}')

    return numpy_image, segments

def perturb_superpixels(segments, num_samples):
    # Perturb the superpixels uniformly at random
    # Generate a list of binary vectors with one bit for each segment
    # 1 indicates that the segment is unchanged, 0 indicates that the segment is zeroed out
    binary_perturbations = []
    num_segments = len(np.unique(segments))
    for _ in range(num_samples):
        binary_perturbation = np.random.randint(0, 2, size=num_segments)
        binary_perturbations.append(binary_perturbation)
    return binary_perturbations

def get_image_from_perturbation(input_image, segments, perturbation):
    # Create a perturbed image by zeroing out the superpixels indicated by the perturbation

    active_segments = np.where(perturbation == 1)[0] + 1 # segments are 1-indexed
    perturbed_image = input_image.copy()
    mask = np.isin(segments, active_segments)
    # Extend the mask to 3D and apply
    perturbed_image = perturbed_image * mask[..., np.newaxis]

    return perturbed_image

def get_model_prediction(model, input_image):
    # input_image is a PIL image
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

    return predicted_idx, predicted_synset, predicted_label

def get_weight(original_image, perturbed_image, width):
    distance = np.linalg.norm(original_image - perturbed_image)
    return np.exp(- distance ** 2 / width ** 2)

def train_sparse_linear_model(original_image, perturbed_dataset, width):
    # compute weights
    weights = []
    for _, perturbed_image, _ in perturbed_dataset:
        weights.append(get_weight(original_image, perturbed_image, width))

def lime(num_samples=50):
    for img_path in image_paths:
        # Open and preprocess the image
        # my_img = os.path.join(img_path, os.listdir(img_path)[2])
        my_img = os.path.join(imagenet_path, img_path)
        input_image = Image.open(my_img).convert('RGB')

        predicted_idx, predicted_synset, predicted_label = get_model_prediction(model, input_image)
        print(f'Predicted label for {img_path}: {predicted_idx} ({predicted_synset}, {predicted_label})')

        numpy_image, segments = gen_superpixels(input_image, img_path)
        # a list of binary perturbation vectors (each vector has one bit for each segment)

        binary_perturbations = perturb_superpixels(segments, num_samples)

        # List of (perturbation vector, perturbed image, predicted class index) tuples
        perturbed_dataset = []
        for perturbation in binary_perturbations:
            perturbed_image = get_image_from_perturbation(numpy_image, segments, perturbation)
            idx, _, _ = get_model_prediction(model, Image.fromarray(perturbed_image))
            perturbed_dataset.append((perturbation, perturbed_image, idx))

lime()

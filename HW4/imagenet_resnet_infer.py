import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, Ridge

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
        segments = slic(numpy_image, n_segments=100, compactness=10, start_label=0)
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
    # Generate a numpy array of shape (num_smaples, num_segments)
    # Each row is a binary vectors with one bit for each segment
    # 1 indicates that the segment is unchanged, 0 indicates that the segment is zeroed out
    num_segments = len(np.unique(segments))
    perturbations = np.random.randint(0, 2, size=(num_samples, num_segments))
    return perturbations

def get_image_from_perturbation(input_image, segments, perturbation=None, active_segments=None):
    # Create a perturbed image by zeroing out the superpixels indicated by the perturbation
    if perturbation is None and active_segments is None:
        return input_image
    if active_segments is None:
        active_segments = np.where(perturbation == 1)[0]
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
    # First normalize the images
    o = preprocess(Image.fromarray(original_image))
    p = preprocess(Image.fromarray(perturbed_image))
    distance = np.linalg.norm(o - p)
    return np.exp(- (distance ** 2) / width ** 2)

def train_sparse_linear_model(original_image, perturbed_dataset, width, K):
    # Compute weights
    weights = []
    for _, perturbed_image, _ in perturbed_dataset:
        weights.append(get_weight(original_image, perturbed_image, width))
    weights = np.array(weights)

    inputs = np.stack([x[0] for x in perturbed_dataset])
    outputs = np.array([x[2] for x in perturbed_dataset])

    ridge = Ridge(alpha=0.1, max_iter=10000)
    ridge.fit(inputs, outputs, sample_weight=weights)
    model_weights = ridge.coef_
    top_k_weights = np.argsort(model_weights)[-K:]

    return top_k_weights

# Increased the width to deal with large distances (~500)
def lime(num_samples=1000, width=600, K=10, method='slic'):
    for img_path in image_paths:
        # Open and preprocess the image
        # my_img = os.path.join(img_path, os.listdir(img_path)[2])
        my_img = os.path.join(imagenet_path, img_path)
        input_image = Image.open(my_img).convert('RGB')

        predicted_idx, predicted_synset, predicted_label = get_model_prediction(model, input_image)
        print(f'Predicted label for {img_path}: {predicted_idx} ({predicted_synset}, {predicted_label})')

        numpy_image, segments = gen_superpixels(input_image, img_path, method)
        # a list of binary perturbation vectors (each vector has one bit for each segment)

        perturbations = perturb_superpixels(segments, num_samples)

        # List of (perturbation vector, perturbed image, predicted class index) tuples
        perturbed_dataset = []
        for perturbation in perturbations:
            perturbed_image = get_image_from_perturbation(numpy_image, segments, perturbation)
            idx, _, _ = get_model_prediction(model, Image.fromarray(perturbed_image))
            perturbed_dataset.append((perturbation, perturbed_image, idx))

        top_k_segments = train_sparse_linear_model(numpy_image, perturbed_dataset, width, K)
        explanation = get_image_from_perturbation(numpy_image, segments, active_segments=top_k_segments)

        plot_superpixels(explanation, f'lime_explanations/{img_path}')

lime()

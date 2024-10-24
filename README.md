# An Implementation of Color Backdoor: A Robust Poisoning Attack in Color Space (CVPR 2023)

## Directory Structure and Files

### GTSRB_dataset
- **Images and Labels**: Contains the dataset images and their corresponding labels.

### models
- **Models**: Contains the trained models.

### accuracy_results.txt
- **ACC, ASR**: Stores the accuracy (ACC) and attack success rate (ASR) results.

### accuracy_results_WASR.txt
- **WASR under Different Filters**: Stores the weighted attack success rate (WASR) results under different filters.

### data.py
- **Data Augmentation and Parsing Functions**: Includes functions for data augmentation, image and label parsing, initialization, and generation of poisoned images. Also includes functions for selecting images based on certain ratios.

### evaluate.py
- **Model Evaluation**: Evaluates the given model's ACC and ASR, and prints the results to `accuracy_results.txt`.

### filterGenerateAndEvaluate.py
- **Filter Generation and Testing**: Generates different filters and tests them, printing the results to `accuracy_results_WASR.txt`.

### main.py
- **Training Function**: Contains the training function for the models.

### model.py
- **Custom Model**: Defines the custom neural network model.

### generateWasrImages.py
- **Generation of Adversarial Filter Images**: Generates images with adversarial filters for robustness testing.

### runAll*.py
- **Hyperparameter Setting Scripts**: Scripts for setting hyperparameters and running evaluation and training functions.

## Setup Instructions

To run the project, you need to perform the following steps:
1. Create a new `models` folder.
2. Replace the empty dataset zip file in the `GTSRB_dataset` folder with the actual dataset zip file.

## Images

### Original Image
<a href="example_images/original.ppm">
    <img src="example_images/original.png" alt="Original Image">
</a>

### Poisoned Image
<a href="example_images/poisoned.ppm">
    <img src="example_images/poisoned.png" alt="Poisoned Image">
</a>
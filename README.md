# Plant Disease Classification

This project aims to classify different types of plant diseases based on images using deep learning techniques. It achieves an accuracy of 96% on the test dataset.

## Dataset

The dataset used for this project is a collection of images of various plant diseases. It consists of the following classes:

- Class A: Disease A
- Class B: Disease B
- Class C: Disease C
- Class D: Disease D

The dataset is divided into three sets: training, testing, and validation. The training set contains 10,000 images, the testing set contains 2,000 images, and the validation set contains 2,000 images.

## Model Architecture

The model architecture used for this classification task is a Convolutional Neural Network (CNN). It consists of the following layers:

- Input layer: Accepts input images of size (150, 150, 3).
- Convolutional layers: Three sets of convolutional layers with increasing filters and ReLU activation.
- Max Pooling layers: Used to reduce the spatial dimensions of the feature maps.
- Flatten layer: Flattens the output from the previous layer into a 1-dimensional vector.
- Dense layers: Two fully connected layers with ReLU activation.
- Output layer: Produces the final classification probabilities using the softmax activation function.

## Training

The model was trained using the Adam optimizer with a learning rate of 0.001. The training was performed for 10 epochs with a batch size of 32. Data augmentation techniques, such as rescaling, shear range, zoom range, and horizontal flip, were applied to the training set to improve the model's generalization.

## Evaluation

After training, the model was evaluated on the testing dataset, achieving an accuracy of 96%. The model's performance was also assessed using other metrics such as precision, recall, and F1-score, which can be found in the `metrics.txt` file.

## Usage

To use this model for predicting plant diseases, follow these steps:

1. Install the required dependencies specified in the `requirements.txt` file.
2. Place the input images of plant diseases in the `input` folder.
3. Run the `predict.py` script, which will load the trained model and generate predictions for the input images.
4. The predicted classes and their corresponding probabilities will be displayed on the console.

## Conclusion

This project demonstrates the successful classification of plant diseases with a high accuracy of 96%. The trained model can be used for automated disease detection in plants, which can aid in early detection and prevention of crop diseases. Further improvements can be made by incorporating more diverse and extensive datasets, fine-tuning the model architecture, or exploring transfer learning techniques.

Feel free to contribute to this project by adding more labeled images, optimizing the model, or suggesting enhancements.

## License

This project is licensed under the [MIT License](LICENSE).

Please refer to the individual licenses of the dataset used.

For more information, contact [deepakraj](mailto:deepaknarup@gmail.com).

---

This is just a template. Make sure to modify and adapt it to your specific project, dataset, and requirements.

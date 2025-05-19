# 🐾 Dog Breed Classification Using InceptionResNetV2

## 📌 Project Summary

This project focuses on building an advanced deep learning model that can identify the **breed of a dog** from an input image. The model is trained to distinguish between **120 different dog breeds** using transfer learning techniques with **InceptionResNetV2**, a state-of-the-art convolutional neural network architecture.

The solution is designed to be scalable and accurate, making it suitable for practical applications in veterinary clinics, mobile apps, pet adoption platforms, and educational tools.

---

## 🧠 How It Works

The core idea is to utilize **transfer learning** by leveraging a pretrained model (InceptionResNetV2) trained on ImageNet. The model acts as a powerful feature extractor, which is then customized using a newly added classification head tailored to the dog breed dataset.

### Key Components:

- **Pretrained Base Model:** InceptionResNetV2 (excluding top layers), which provides deep visual features.
- **Custom Classification Layers:** Additional fully connected layers with dropout regularization to predict one of the 120 dog breeds.
- **Data Augmentation:** Applied to the training images to improve generalization (rotation, flipping, zooming, etc.).
- **Image Preprocessing:** All images are resized to 331x331 pixels to match the input dimensions required by the base model.
- **One-Hot Encoding:** Used for converting breed labels into a format suitable for multi-class classification.
- **Training Strategy:** The model is trained using categorical cross-entropy loss and optimized with the Adam optimizer. EarlyStopping is used to monitor the validation loss and avoid overfitting.
- **Evaluation and Visualization:** Training and validation accuracy/loss are visualized through plots. Predictions on unseen data are evaluated and compared with ground truth.

---

## 📁 Dataset Overview

The dataset consists of:

- A set of labeled training images stored in a folder.
- A set of unlabeled test images.
- A CSV file (`labels_new.csv`) that maps image filenames to their corresponding breed labels.

The dataset is divided into training and testing portions, with a subset reserved for validation during model training.

---

## 🔍 Model Training Workflow

1. **Load and preprocess images.**
2. **Perform label encoding and prepare the dataset.**
3. **Split the dataset into training and validation sets.**
4. **Construct the model using InceptionResNetV2 and custom layers.**
5. **Compile and train the model with callbacks like EarlyStopping.**
6. **Visualize the training progress.**
7. **Evaluate the model and generate predictions on the test set.**
8. **Create a submission CSV mapping image names to predicted breed labels.**
9. **Save the trained model for future inference or deployment.**

---

## ✅ Output Files

- `Model.h5`: The trained Keras model file, which can be reused without retraining.
- `submission.csv`: Contains test image IDs and predicted breed labels.
- Accuracy and loss graphs showing the model's performance over epochs.
- Optional: Visual samples comparing original dog images with their predicted breeds.

---

## 📊 Performance

The model achieves strong classification performance by combining the power of pretrained features with task-specific fine-tuning. It demonstrates:

- High training accuracy
- Stable validation accuracy due to regularization and augmentation
- Fast convergence and early stopping at optimal performance

The approach is flexible and can be retrained or fine-tuned with different datasets, breed categories, or custom dog image collections.

---

## 💡 Applications

This project has real-world use cases, such as:

- Dog breed identification apps
- Automated tagging in pet image platforms
- Animal rescue and shelter databases
- Educational tools for veterinary training
- Smart surveillance and animal monitoring

---

## 🧰 How to Use

To use this project:

1. **Clone the repository** and place the dataset (`train/`, `test/`, and `labels_new.csv`) in the working directory.
2. **Ensure the required Python libraries** (TensorFlow, Keras, Pandas, NumPy, etc.) are installed.
3. **Run the training script** to build and train the model.
4. **Use the trained model** to make predictions on new images.
5. **Submit or analyze results** using the generated `submission.csv` file.

---

## 👤 Author

This project was developed by **Susobhan Akhuli** as part of a deep learning-based image classification initiative. It showcases the use of modern AI techniques for solving multi-class visual recognition tasks.

---

## 📄 License

This project is open-source and available for educational and research use. You may adapt and extend it with appropriate attribution.

---


# Deep-Learning-for-Hyperspectral-Images

#### Notebook 1: 1D, 2D, and 3D CNN Comparison

This notebook explores the comparison of different convolutional neural network architectures (1D, 2D, and 3D) for classifying hyperspectral images.

1. **Loading Data:**
    ```python
    import scipy.io
    import numpy as np
    data = scipy.io.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = scipy.io.loadmat('Indian_pines_gt.mat')['indian_pines_gt']
    ```
    - Loads hyperspectral image data and ground truth labels using `scipy.io`.

2. **Data Preprocessing:**
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, data.shape[2])).reshape(data.shape)
    data_flattened = data_normalized.reshape(-1, data.shape[2])
    gt_flattened = gt.ravel()
    mask = gt_flattened > 0
    data_flattened = data_flattened[mask]
    gt_flattened = gt_flattened[mask]
    X_train, X_test, y_train, y_test = train_test_split(data_flattened, gt_flattened, test_size=0.3, random_state=42)
    ```
    - Normalizes the data using `StandardScaler`.
    - Flattens the data and filters out the background using a mask.

3. **Model Building and Training (1D CNN Example):**
    ```python
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    ```
    - Constructs a 1D CNN with convolutional, pooling, and dense layers.
    - Compiles the model with Adam optimizer and trains it on the training data.

4. **Evaluation and Visualization:**
    ```python
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Accuracy: {acc*100}\\nLoss: {loss}")
    ```
    - Evaluates the model on the test dataset and prints the accuracy and loss.

#### Notebook 2: Land Cover Classification of Satellite Imagery Using Convolutional Neural Networks

This notebook focuses on classifying land cover types from satellite imagery using 2D CNNs.

1. **Model Evaluation using Saved Model:**
    ```python
    from tensorflow.keras.models import load_model
    model = load_model("Salinas_Model.h5")
    pred = np.argmax(model.predict(X_test), axis=1)
    ```
    - Loads a pre-trained model and predicts classes for the test data.

2. **Classification Report and Confusion Matrix:**
    ```python
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(pred, np.argmax(y_test, 1), target_names=class_names))
    sns.heatmap(confusion_matrix(pred, np.argmax(y_test, 1)), annot=True)
    ```
    - Generates a classification report and confusion matrix to evaluate the model's performance.

# 🏛️Image Classification with Tensorflow
# 📓manual
<h2> First install </h2>

```
pip install tensorflow
```
```
pip install numpy
```
```
pip install seaborn

```
```
pip install matplotlib
```
<h2>Library</h2>

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as implt
import seaborn as sns
sns.set_style("whitegrid")
```
<h2>Path</h2>

```python
train_path = './data2/foo/main_data_test/train/'
test_path = './data2/foo/main_data_test/test/'
val_path = './data2/foo/main_data_test/valid/'
```

<h2>List Class Names in the DataSet</h2>

```python
category_names = os.listdir(train_path)
category_names
```

<h2>Image Augmentation</h2>

```python
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

<div>
  <ul>
    <li>ImageDataGenerator:</li>
      <p> Creates an instance for real-time image augmentation</p>
    <li>rotation_range:</li>
      <p>randomly rotates images by a degree range</p>
    <li>width_shift_range:</li>
      <p>Randomly shifts image Horizantally</p>
    <li>height_shift_range:</li>
      <p>Randomly shift images vertically</p>
    <li>shear_range:</li>
      <p>Shears the image</p>
    <li>zoom_range:</li>
        <p>Randomly zooms into images.</p>
    <li>horizontal_flip:</li>
       <p>Randomly flips images horizontally</p>
    <li>fill_mode:</li>
      <p>Determines how to fill in new pixels after transformations</p>
    <details>
            
        1.'constant':
            Fills the empty pixels with a constant value, specified by the cval parameter (default is 0). This is often used when you want a uniform color to fill the missing areas.
            Use Case: If you want the background to be a solid color (e.g., black or white).
        2.'nearest':
            Fills the empty pixels with the value of the nearest pixel from the original image. This is useful for preserving the appearance of the image, as it replicates nearby pixel values.
            Use Case: Good for images where you want to maintain the original colors and avoid introducing new colors into the empty areas.
        3.'reflect':
            Fills the empty pixels by reflecting the image across its edges. This means that pixels are mirrored from the edge to fill in the empty areas.
            Use Case: Often used for natural images, as it can create a more seamless transition and preserve continuity at the edges.
        4.'wrap':
              Fills the empty pixels by wrapping around the image. Essentially, it treats the image as if it's tiled infinitely, using pixels from the opposite side of the image to fill in the gaps.
              Use Case: Useful in scenarios where the image edges are intended to connect seamlessly, like textures in graphics.
                          
  </details>

  </ul>
</div>

<h2>Creating Data Generators</h2>

```python
train_generator = datagen.flow_from_directory(
    directory=train_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'  
)

validation_generator = datagen.flow_from_directory(
        val_path,
        target_size=(256, 256),
        shuffle=True,
        class_mode='categorical')

```
```
output:
Found 400 images belonging to 5 classes.
Found 75 images belonging to 5 classes.
```

<div>
  <ul>
    <li>flow_from_directory:</li>
      <p> Creates generators that load images directly from the specified directories</p>
    <li>directory:</li>
      <p>The path to the data directory</p>
    <li>target_size:</li>
      <p>Resizes images to the specified dimensions</p>
    <li>batch_size:</li>
      <p>Number of images to yield per batch</p>
    <li>class_mode:</li>
      <p>Specifies the type of label arrays (categorical for multiple classes)</p>
    <details>
      
                1. 'categorical'
                  Description: This mode is used for multi-class classification problems where each image belongs to one of multiple categories (classes).
                  Output: The labels are encoded as a one-hot vector. For example, if there are 3 classes and an image belongs to class 1, the label will be represented as [1, 0, 0].
                  Use Case: When you have more than two classes and want to classify images into one of them.
      
                2. 'binary'
                  Description: This mode is used for binary classification problems where each image can only belong to one of two classes.
                  Output: The labels are represented as a single integer (0 or 1). For example, if an image belongs to the positive class, it is labeled as 1, and if it belongs to the negative class, it is labeled as 0.
                  Use Case: When you want to classify images into one of two classes, such as distinguishing between cats and dogs.
                
                3. 'sparse'
                  Description: Similar to categorical, but the labels are not one-hot encoded; instead, they are provided as integers representing the class index.
                  Output: For example, if there are 3 classes, the label for an image belonging to class 1 will simply be 0 (not [1, 0, 0]).
                  Use Case: Useful for multi-class problems when you prefer to work with integers instead of one-hot encoding. It is less memory-intensive, especially when there are many classes.
               
                4. 'input'
                  Description: This mode is used when you want to provide both images and their corresponding labels to the model. It is mainly used for autoencoders and some other specialized tasks.
                  Output: The generator yields a tuple where both the input and output are the same (i.e., the images are used as their own labels).
                  Use Case: When you're training models that need to reconstruct the input, such as in autoencoders.
                
                5. 'None'
                  Description: If you set class_mode=None, the generator will only yield batches of images without any labels.
                  Output: This can be useful for tasks where you want to predict on images without needing the ground truth labels, such as during inference.
                  Use Case: When you only need the images for prediction or visualization and not the labels.
  </details>
  </ul>
</div>

<h2>Creating the Model</h2>

```python
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(256,256, 3))

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


```

<div>
  <ul>
    <li>VGG16:</li>
      <p>Loads a pre-trained VGG16 model (not including the top classification layers) with weights from ImageNet.</p>
      1) include_top=False: Excludes the final fully connected layer.
      2) input_shape: Sets the input shape for the model (height, width, channels).
    <li>Sequential:</li>
      <p>Creates a linear stack of layers for the model</p>
      1) Flatten: Flattens the output from the base model into a one-dimensional vector.
      2) Dense: Fully connected layers. The first layer has 128 neurons with ReLU activation, and the second has 5 neurons with softmax activation for multi-class classification.
      Relu: faster and great for decreasing a overfit
      softmax: it will use in the last part and you need the number of the class
  </ul>
</div>

<h2>Compiling the Model</h2>

```python
base_model.trainable = False 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
- trainable = False: Freezes the base model so its weights won't be updated during training.
- compile: Configures the model for training.
- optimizer: Adam optimizer, which adjusts the learning rate.
- loss: Categorical crossentropy loss function for multi-class classification.
- metrics: Tracks accuracy during training.

<details>
  
  optimizer:
    
    1) SGD (Stochastic Gradient Descent):
      Usage: Commonly used optimizer, especially in traditional machine learning and deep learning tasks.
    2) RMSprop:
      Usage: Effective for recurrent neural networks (RNNs) and other models with non-stationary objectives.
    3) Adagrad:
      Usage: Useful for sparse data and when different features need different learning rates.
    4) AdamW:
      Usage: A variation of Adam that decouples weight decay from the gradient update.
    5) Adam:
      The Adam optimizer (short for Adaptive Moment Estimation) is widely used in deep learning due to its effectiveness and efficienc
</details>
<details>

    loss Function:
      
      1) Sparse Categorical Crossentropy:
        Usage: When your target labels are integers instead of one-hot encoded vectors.
      2) Binary Crossentropy:
        Usage: When your problem involves binary classification or multi-label classification.
      3) Kullback-Leibler Divergence:
        Usage: Useful for comparing probability distributions, especially in tasks like variational autoencoders.
      4) Mean Squared Error (MSE):
        Usage: Commonly used in regression tasks.
</details>

<h2>Model Summary</h2>

```python
model.summary()

```

<h2>Training the Model</h2>

```python
history = model.fit(
      train_generator,
      steps_per_epoch=15,  
      epochs=10,
      verbose=1,
      validation_data=validation_generator)

```
- fit():
   Trains the model for a fixed number of epochs.
- train_generator:
  Data generator for training.
- steps_per_epoch:
  Number of batches to yield from the generator before declaring one epoch finished.
> [!IMPORTANT]
>   step_per_epoch = number of images /batch_size

- epochs:
  Number of epochs to train the model.
- verbose:
  Controls the display of training progress (1 shows progress bar).
- validation_data:
  Data generator for validation.

<h2>Preparing for Prediction</h2>

```python
model.save('image_classification_model.keras')
print(model.input_shape)
```
- save():
  Saves the model architecture and weights to a file.
- input_shape:
  Prints the shape of the model's input.

<h2>Making a Prediction</h2>

```python
img_path = './data2/foo/main_data_test/test/hammer/hammer (97).jpg'

img = image.load_img(img_path, target_size=(256,256))  # Ensure the size matches what the model expects
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)
print(f'Predicted class: {predicted_class}')
x = train_generator.class_indices
key = list(x.keys())
predicted_label = key[predicted_class]
print(f'Predicted label: {predicted_label}') 

```

- img_path:
  Path to the test image you want to classify.
- load_img:
  Loads and resizes the test image.
- img_to_array:
  Converts the image to a NumPy array.
- expand_dims:
  Adds a batch dimension to the image array.
- predict():
  Uses the model to make a prediction on the input image.
- argmax():
  Returns the index of the class with the highest predicted probability.
- class_indices:
  Retrieves the class indices used during training.
- keys():
  Gets the list of class labels.
- predicted_label:
  Retrieves the label corresponding to the predicted class index.

# 👨‍💻Tech
- Tensorflow
- python
- seaborn
- numpy
- os
- matplotlib

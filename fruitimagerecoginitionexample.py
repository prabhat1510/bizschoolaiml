import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Paths to training and validation directories
#URL -- https://www.kaggle.com/datasets/chrisfilo/fruit-recognition/data
#or URL-- https://data.mendeley.com/datasets/b6fftwbr2v/1
train_dir = "path_to_training_data"
val_dir = "path_to_validation_data"

#Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

#Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

#Save the model
#model.save("fruit_classifier_model.h5")
# Evaluate the model
test_loss, test_accuracy = model.evaluate(val_generator)
print(f"Validation accuracy: {test_accuracy:.4f}")
# Make predictions
sample_image = val_generator[0][0][0]  # Get a sample image from the validation set
sample_image = sample_image.reshape((1, 100, 100, 3))  # Reshape for prediction
predictions = model.predict(sample_image)
predicted_class = predictions.argmax(axis=-1)
print(f'Predicted class index: {predicted_class[0]}')
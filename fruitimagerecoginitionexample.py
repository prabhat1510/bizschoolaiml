import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Paths to training and validation directories
#D:\\dataset\\training\\apple
#D:\\dataset\\training\\banana
#D:\\dataset\\training\\orange
#D:\\dataset\\val\\apple
#D:\\dataset\\val\\banana
#D:\\dataset\\val\\orange
train_dir = "D:\\dataset\\training"
val_dir = "D:\\dataset\\val"

#Image preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
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
if train_generator.samples > 0 and val_generator.samples > 0:
    model.fit(train_generator, validation_data=val_generator, epochs=10)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(val_generator)
    print(f"Validation accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    val_generator.reset()
    sample_batch = next(val_generator)
    sample_image = sample_batch[0][0:1]  # Get first image from batch
    predictions = model.predict(sample_image)
    predicted_class = predictions.argmax(axis=-1)
    print(f'Predicted class index: {predicted_class[0]}')
else:
    print("Error: No images found in the specified directories. Please check your dataset structure.")

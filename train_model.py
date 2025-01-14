import tensorflow as tf
import matplotlib.pyplot as plt

# Configuración
DATASET_PATH = "data/vehiculos"
MODEL_SAVE_PATH = "models/vehicle_model.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Cargar dataset
print("Cargando dataset...")
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalización
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Crear modelo
print("Creando modelo...")
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation="softmax")  # 4 clases: car, truck, motorcycle, van
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
print("Entrenando modelo...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# Guardar modelo
print(f"Guardando modelo en {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Modelo guardado exitosamente.")

# Visualizar métricas
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Entrenamiento')
plt.plot(val_acc, label='Validación')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Entrenamiento')
plt.plot(val_loss, label='Validación')
plt.title('Pérdida')
plt.legend()

plt.show()

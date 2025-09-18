import tensorflow as tf 

class CommonConfig:
    SEED = 42

class MlflowConfig:
    MLFLOW_TRACKING_URI = "http://localhost:5000"
    ENABLE_SYSTEM_METRICS_LOGGING = True
    MLFLOW_EXPERIMENT_NAME = "Smart_Recycling"
    MLFLOW_RUN_NAME = "baseline"
    MLFLOW_TENSORFLOW_AUTOLOG_CONFIG = {
        "log_models": True,
        "log_datasets": True,
        "keras_model_kwargs": {"save_format": "keras"},
        "log_model_signatures": True,
        "registered_model_name": None,
        "checkpoint": True, 
        "checkpoint_monitor": "val_loss",
        "checkpoint_mode": "min",
        "checkpoint_save_best_only": True,
        "checkpoint_save_weights_only": False,
    }

class DatasetConfig:
    DATASET_FOLDER = "/home/ubuntu/dev/smart_recycling/garbage-dataset-v1"
    IMAGE_SIZE = (224, 224)
    TRAIN_BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    LABEL_MODE = "categorical"


class ModelConfig:
    ENABLE_MIXED_PRECISION = True

    BASE_MODEL = tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=None,
        include_preprocessing=True,
    )

    BASE_MODEL_TRAINABLE = False

    MODEL = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(224, 224, 3)),
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
            tf.keras.layers.RandomRotation(0.12, seed=42),
            tf.keras.layers.RandomZoom(0.12, seed=42),
            tf.keras.layers.RandomContrast(0.12, seed=42),
            tf.keras.layers.RandomBrightness(0.12, seed=42),
            BASE_MODEL,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(8, activation="softmax", dtype="float32"),
        ]
    )

    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
    LOSS = tf.keras.losses.CategoricalCrossentropy()
    METRICS = [tf.keras.metrics.F1Score(average="weighted")]


class ModelTrainingConfig:
    EPOCHS = 50

    COMPUTE_CLASS_WEIGHTS = True 
    CLASS_WEIGHT = "balanced"

    EARLY_STOPPING_CALLBACK = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    REDUCE_LR_ON_PLATEAU_CALLBACK = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=5,
        min_lr=1e-7
    )

    TRAINING_CALLBACKS = [
        EARLY_STOPPING_CALLBACK,
        REDUCE_LR_ON_PLATEAU_CALLBACK
    ]


class ModelEvaluationConfig:
    SAVE_MODEL_HISTORY = True 

    INCLUDE_EVALUATION_ON_TEST_SET = True
    SAVE_PREDICTION_TIME = True 
    SAVE_CONFUSION_MATRIX = True
    SAVE_PREDICTION_CSV = True
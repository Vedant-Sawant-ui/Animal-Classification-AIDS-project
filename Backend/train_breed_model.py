import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BACKEND_DIR / "uploads"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "Model"
IMAGE_SIZE = (224, 224)
SEED = 42


def read_split(path):
    with path.open(encoding="utf-8") as split_file:
        return [line.split()[0] for line in split_file if line.strip() and not line.startswith("#")]


def load_dataset(data_dir):
    image_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    records = {}
    breed_names = {}
    breed_animals = {}

    with (annotations_dir / "list.txt").open(encoding="utf-8") as label_file:
        for line in label_file:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            image_stem, class_id, species_id = fields[0], int(fields[1]), int(fields[2])
            breed_name = re.sub(r"_\d+$", "", image_stem)
            existing_name = breed_names.setdefault(class_id, breed_name)
            if existing_name != breed_name:
                raise ValueError(f"Class {class_id} has multiple breed names")
            animal = {1: "cat", 2: "dog"}.get(species_id)
            if animal is None:
                raise ValueError(f"Unknown species ID {species_id} for {image_stem}")
            existing_animal = breed_animals.setdefault(class_id, animal)
            if existing_animal != animal:
                raise ValueError(f"Class {class_id} has multiple species")
            records[image_stem] = (image_stem, class_id - 1)

    class_ids = sorted(breed_names)
    if class_ids != list(range(1, len(class_ids) + 1)):
        raise ValueError("Breed class IDs must be contiguous and start at 1")

    class_names = [breed_names[class_id].replace("_", " ").title() for class_id in class_ids]
    class_animals = [breed_animals[class_id] for class_id in class_ids]
    trainval_keys = read_split(annotations_dir / "trainval.txt")
    test_keys = read_split(annotations_dir / "test.txt")
    split_keys = trainval_keys + test_keys
    missing_records = [key for key in split_keys if key not in records]
    missing_images = [key for key in split_keys if not (image_dir / f"{key}.jpg").is_file()]
    if missing_records or missing_images:
        raise ValueError(
            f"Dataset split references missing records ({len(missing_records)}) "
            f"or images ({len(missing_images)})"
        )

    if len(split_keys) != len(set(split_keys)):
        raise ValueError("trainval.txt and test.txt contain duplicate or overlapping records")

    labelled_keys = set(records)
    unlisted_images = sum(1 for path in image_dir.glob("*.jpg") if path.stem not in labelled_keys)
    return image_dir, records, class_names, class_animals, trainval_keys, test_keys, unlisted_images


def split_training_data(records, trainval_keys):
    by_class = defaultdict(list)
    for key in trainval_keys:
        by_class[records[key][1]].append(key)

    rng = random.Random(SEED)
    training_keys = []
    validation_keys = []
    for keys in by_class.values():
        rng.shuffle(keys)
        validation_count = max(1, round(len(keys) * 0.15))
        validation_keys.extend(keys[:validation_count])
        training_keys.extend(keys[validation_count:])
    return training_keys, validation_keys


def make_tf_dataset(tf, image_dir, records, keys, class_count, batch_size, training):
    paths = [str(image_dir / f"{key}.jpg") for key in keys]
    labels = [records[key][1] for key in keys]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 127.5 - 1.0
        label = tf.one_hot(label, depth=class_count)
        return image, label

    if training:
        dataset = dataset.shuffle(len(keys), seed=SEED, reshuffle_each_iteration=True)
    return (
        dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def train_model(args, dataset):
    import tensorflow as tf
    from tensorflow.keras import layers

    image_dir, records, class_names, class_animals, trainval_keys, test_keys, _ = dataset
    training_keys, validation_keys = split_training_data(records, trainval_keys)
    class_count = len(class_names)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pet_breed_model.keras"
    class_metadata = {"breeds": class_names, "animals": class_animals}
    class_metadata_path = output_dir / "pet_breed_classes.json"
    test_dataset = make_tf_dataset(
        tf, image_dir, records, test_keys, class_count, args.batch_size, False
    )

    if args.evaluate_only:
        if not model_path.is_file():
            raise FileNotFoundError(f"No trained breed model found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        test_loss, test_accuracy = model.evaluate(test_dataset)
        with class_metadata_path.open("w", encoding="utf-8") as class_file:
            json.dump(class_metadata, class_file, indent=2)
        print(f"Test accuracy: {test_accuracy:.4f} (loss {test_loss:.4f})")
        print(f"Saved breed model to: {model_path}")
        return

    train_dataset = make_tf_dataset(
        tf, image_dir, records, training_keys, class_count, args.batch_size, True
    )
    validation_dataset = make_tf_dataset(
        tf, image_dir, records, validation_keys, class_count, args.batch_size, False
    )

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
    augmented = layers.RandomFlip("horizontal")(inputs)
    augmented = layers.RandomRotation(0.08)(augmented)
    features = base_model(augmented, training=False)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dropout(0.3)(features)
    features = layers.Dense(256, activation="relu")(features)
    outputs = layers.Dense(class_count, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
    )
    model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(test_dataset)
    with class_metadata_path.open("w", encoding="utf-8") as class_file:
        json.dump(class_metadata, class_file, indent=2)

    print(f"Test accuracy: {test_accuracy:.4f} (loss {test_loss:.4f})")
    print(f"Saved breed model to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train an Oxford-IIIT Pet breed classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    image_dir, records, class_names, _, trainval_keys, test_keys, unlisted_images = dataset
    training_keys, validation_keys = split_training_data(records, trainval_keys)
    print(f"Dataset: {image_dir}")
    print(f"Labeled images: {len(records)} across {len(class_names)} breeds")
    print(f"Training: {len(training_keys)}, validation: {len(validation_keys)}, test: {len(test_keys)}")
    print(f"Unlisted images ignored: {unlisted_images}")

    if not args.check_only:
        train_model(args, dataset)


if __name__ == "__main__":
    main()
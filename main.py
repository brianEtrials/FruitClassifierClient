import os, shutil
from cropper import crop_image
from classify import classify_image

TEMP_DIR = 'tmp'


def main(input_filepath):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

    # Phase 1
    paths = crop_image(input_filepath, TEMP_DIR, 'models/YOLO_weights.pt')

    results = {}

    for path in paths:
        # Phase 2
        label1, confidence1 = classify_image(path, 'models/Class1.keras', ["Apple", "Orange", "Tomato"])
        model = f"models/{label1}.keras"
        labels = []
        if label1 == "Apple":
            labels = ["Fuji", "Granny smith", "Honeycrisp", "Red delicious"]
        elif label1 == "Orange":
            labels = ["Clementine", "Mandarin", "Navel"]
        elif label1 == "Tomato":
            labels = ["Cherry", "Roma"]
        # Phase 3
        label2, confidence2 = classify_image(path, model, labels)
        if label2 in results:
            results[label2].append(confidence1 * confidence2)
        else:
            results[label2] = [confidence1 * confidence2]

    # Phase 4
    best_label = ""
    best_confidence = -1
    for l, confidences in results.items():
        c = 1
        for conf in confidences:
            c *= conf
        if conf > best_confidence:
            best_confidence = conf
            best_label = l

    # Clean up
    # shutil.rmtree(TEMP_DIR)

    return best_label, best_confidence


if __name__ == '__main__':
    a, b = classify_image("/Users/skyler/Desktop/Classifier/test_images/cherry.jpg", "/Users/skyler/Desktop/Classifier/models/Orange.keras", ["A", "B", "C", "D"])
import os
from main import main

def test(test_data_path):
    total_images = 0
    total_correct = 0
    folder_accuracies = {}

    for label in os.listdir(test_data_path):
        dir = os.path.join(test_data_path, label)
        if os.path.isdir(dir):
            correct = 0
            total = 0

            for image_name in os.listdir(dir):
                if image_name.lower().endswith(('.jpg', '.jpeg')):
                    image_path = os.path.join(dir, image_name)
                    predicted_label, confidence = main(image_path)
                    total += 1
                    total_images += 1

                    if predicted_label == label:
                        correct += 1
                        total_correct += 1

            folder_accuracies[label] = correct / total if total > 0 else 0

    # Individual  accuracies
    for label, accuracy in folder_accuracies.items():
        print(f"Accuracy for '{label}': {accuracy:.2f}")

    # Total accuracy
    total_accuracy = total_correct / total_images if total_images > 0 else 0
    print(f"Total accuracy: {total_accuracy:.2f}")


if __name__ == '__main__':
    test('../Classifier/test_data')


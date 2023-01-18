import cv2
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
ICE_SKATER = os.path.join(DATASET_DIR, "iceskater")


def main():
    images = []
    bboxes = []

    # Read the bounding box coordinates from the groundtruth.txt file
    with open(os.path.join(ICE_SKATER, "groundtruth.txt"), "r") as f:
        for line in f:
            x, y, w, h = [float(i) for i in line.strip().split(",")]
            bboxes.append((int(x), int(y), int(w), int(h)))

    for file_name in os.listdir(ICE_SKATER):
        if file_name.endswith(".jpg"):
            # Read the image
            img = cv2.imread(os.path.join(ICE_SKATER, file_name))

            # Add the image to the dataset
            images.append(img)

    # Initialize the KCF tracker
    tracker = cv2.TrackerKCF_create()

    # Train the tracker on the first image and bounding box
    tracker.init(images[0], bboxes[0])

    # Loop over the remaining images
    for img, bbox in zip(images[1:], bboxes[1:]):
        # Update the tracker with the current image and bounding box
        success, bbox = tracker.update(img)

        # Draw the bounding box of the tracked object
        if success:
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Tracking", img)
        cv2.waitKey(1)

    # Release the video capture object
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

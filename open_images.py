import os
import matplotlib.pyplot as plt

# Path to the text file containing image filenames
IMAGE_LIST_FILE = "/Volumes/My Passport/Samweg Weds Meera Amdavad/output.txt"
# Path to the directory containing the images
IMAGES_DIR = "/Volumes/My Passport/Samweg Weds Meera Amdavad"

def show_images_from_file(image_list_file, images_dir):
    """Display images listed in a text file one by one."""
    if not os.path.exists(image_list_file):
        print(f"Error: {image_list_file} does not exist.")
        return

    # Read the list of image filenames
    with open(image_list_file, 'r') as file:
        image_filenames = [line.strip() for line in file.readlines()]

    if not image_filenames:
        print("No images found in the list file.")
        return

    for image_filename in image_filenames:
        # Construct the full image path
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: {image_filename} not found in {images_dir}.")
            continue

        try:
            # Load and display the image
            image = plt.imread(image_path)
            plt.imshow(image)
            plt.axis('off')
            plt.title(image_filename)
            plt.show()
        except Exception as e:
            print(f"Error displaying {image_filename}: {e}")

if __name__ == "__main__":
    show_images_from_file(IMAGE_LIST_FILE, IMAGES_DIR)
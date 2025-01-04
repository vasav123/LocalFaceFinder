import face_recognition
import cv2
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Path to the directory containing your photos
PHOTOS_DIR = "/Volumes/My Passport/Samweg Weds Meera Amdavad/"
# Path to the directory containing your training images
TRAINING_DIR = "/Volumes/My Passport/Samweg Weds Meera Amdavad/"
# Path to save the list of filenames with matches
OUTPUT_FILE = "/Volumes/My Passport/Samweg Weds Meera Amdavad/output.txt"
# Path to save face embeddings
EMBEDDINGS_FILE = "/Volumes/My Passport/embeddings.pkl"


def load_training_faces(training_dir):
    """Load training faces and let the user select which face to use."""
    known_faces = []
    known_names = []

    for file in os.listdir(training_dir):
        filepath = os.path.join(training_dir, file)
        if file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image)
            encodings = face_recognition.face_encodings(image, face_locations)

            if encodings:
                # Display the image with labeled faces
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    plt.imshow(image)
                    plt.gca().add_patch(plt.Rectangle((left, top), right - left, bottom - top,
                                                      linewidth=2, edgecolor='green', facecolor='none'))
                    plt.text(left, top - 10, f"Face {i + 1}", color='green', fontsize=12)

                plt.title(f"Select face in {file}")
                plt.axis('off')
                plt.show()

                print(f"File: {file} contains {len(encodings)} face(s).")
                selected_face = None
                while selected_face is None:
                    try:
                        selected_face = int(input(f"Enter the number of the face to use (1-{len(encodings)}): ")) - 1
                        if selected_face == -1:
                            print("Skipping this image.")
                            selected_face = -1
                            break
                        if selected_face < 0 or selected_face >= len(encodings):
                            print("Invalid selection. Try again.")
                            selected_face = None

                    except ValueError:
                        print("Please enter a valid number.")
                if selected_face != -1:
                    known_faces.append(encodings[selected_face])
                    known_names.append(file)

    # Save embeddings to a file
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump({"faces": known_faces, "names": known_names}, f)
    print(f"Embeddings saved to {EMBEDDINGS_FILE}")

    return known_faces, known_names

def calculate_similarity(face1, face2):
    """Calculate similarity between two face encodings."""
    distance = np.linalg.norm(face1 - face2)
    similarity = max(0, 1 - distance) * 100  # Convert to percentage
    return similarity

def find_photos_of_you(photos_dir, known_faces, output_file):
    """Identify photos of you in a directory and save matches to a file."""
    matches = []

    for (root, _, files) in os.walk(photos_dir):
        if "[Candid Photo 24-11-2024]" in root:
            for file in files:
                filepath = os.path.join(root, file)

                if file.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    print(f"Processing {file}...")
                    image = face_recognition.load_image_file(filepath)
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)

                    for encoding in face_encodings:
                        similarities = [calculate_similarity(encoding, known_face) for known_face in known_faces]
                        best_match_index = np.argmax(similarities)
                        best_match_similarity = similarities[best_match_index]
                        print(best_match_similarity)
                        if best_match_similarity > 60:  # Threshold for a match
                            print(f"Match found: {file} with {best_match_similarity:.2f}% similarity")
                            matches.append(f"{file} ({best_match_similarity:.2f}% similarity)")
                            with open(output_file, 'a') as f:
                                f.write(filepath+'\n')
                            break

    # Save matches to a text file
    # with open(output_file, 'w') as f:
    #     for match in matches:
    #         f.write(match + '\n')

if __name__ == "__main__":
    # TRAIN = input("Do you want to train the model? (yes/no): ").strip().lower() == "yes"
    TRAIN = False
    if TRAIN:
        print("Loading training faces...")
        known_faces, known_names = load_training_faces(TRAINING_DIR)
    else:
        print("Loading embeddings from file...")
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                known_faces = data["faces"]
                known_names = data["names"]
            print("Embeddings loaded successfully.")
        else:
            print(f"Embeddings file not found at {EMBEDDINGS_FILE}. Please train the model first.")
            exit(1)

    print("Scanning photos for matches...")
    find_photos_of_you(PHOTOS_DIR, known_faces, OUTPUT_FILE)

    print(f"Finished! Matches are saved in {OUTPUT_FILE}.")



from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2, os, json, numpy as np

app = Flask(__name__)
CORS(app)  # Set template folder for HTML

# Path to the folder containing stored images
IMAGE_FOLDER = "images"
FEATURES_FILE = "features.json"
MATCH_RESULTS_FILE = "matches.json"

# ORB feature detector
orb = cv2.ORB_create(nfeatures=2000)

# Load stored image features or compute them if not available
def compute_features():
    features = {}
    
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.lower().endswith((".jpg", ".png")):
            image_path = os.path.join(IMAGE_FOLDER, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = orb.detectAndCompute(image, None)

            if descriptors is not None:
                features[filename] = descriptors.tolist()  # Convert to list for JSON storage

    # Save computed features
    with open(FEATURES_FILE, "w") as f:
        json.dump(features, f)

    return features

# Load or generate features
if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, "r") as f:
        stored_features = json.load(f)
else:
    stored_features = compute_features()

# Brute-force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

@app.route('/')
def index():
    return jsonify({"message": "Card Matcher API is running"})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Brute-force matcher with Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Set crossCheck to False for ratio test

@app.route("/upload", methods=["POST"])
def upload_image():
    # Read raw binary data from the request body
    binary_data = request.data
    if not binary_data:
        return jsonify({"error": "No binary data provided"}), 400

    # Convert binary data to a NumPy array
    npimg = np.frombuffer(binary_data, np.uint8)
    uploaded_image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    if uploaded_image is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Extract features from the uploaded image
    keypoints, descriptors = orb.detectAndCompute(uploaded_image, None)

    if descriptors is None:
        return jsonify({"error": "Could not extract features from image"}), 400

    matched_images = {}

    for filename, stored_descriptors in stored_features.items():
        stored_descriptors = np.array(stored_descriptors, dtype=np.uint8)
        matches = bf.knnMatch(descriptors, stored_descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_matches.append(m)

        match_count = len(good_matches)
        MIN_MATCH_COUNT = 1

        if match_count >= MIN_MATCH_COUNT:
            name, size = os.path.splitext(filename)[0].rsplit("_", 1)
            key = (name, size)

            if key not in matched_images:
                matched_images[key] = {
                    "matched_image": filename,
                    "name": name,
                    "card_size": size,
                    "count": 1
                }
            else:
                matched_images[key]["count"] += 1

    # Convert to a list and allow duplicates
    result_list = []
    for data in matched_images.values():
        for _ in range(data["count"]):
            result_list.append({
                 "name": data["name"],
                "card_size": data["card_size"]
            })

    if result_list:
        # Organize matches into top and bottom rows
        top_row = []
        bottom_row = []
        current_row_space = 10
        
        for match in result_list:
            size = 1 if match["card_size"] == "small" else 2 if match["card_size"] == "medium" else 3
            
            if current_row_space >= size:
                top_row.append(match)
                current_row_space -= size
            else:
                bottom_row.append(match)
        
        result = {
            "player_top": top_row,
            "player_bottom": bottom_row
        }

        # Save result locally
        with open(MATCH_RESULTS_FILE, "w") as f:
            json.dump(result, f, indent=4)

        return jsonify(result)
    else:
        return jsonify({"error": "No matching image found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

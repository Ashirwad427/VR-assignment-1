import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set output directory
output_dir = "stitched_images"
os.makedirs(output_dir, exist_ok=True)

def load_and_resize_images(image_paths, size=(800, 600)):
    """Loads and resizes images to a common size."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Could not load image {path}")
        img = cv2.resize(img, size)
        images.append(img)
    return images

def get_homography(kp1, kp2, matches):
    """Computes the Homography matrix using matched keypoints."""
    if len(matches) < 4:
        raise ValueError("Not enough matches found!")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def stitch_images(left, right, H):
    """Stitches two images using Homography."""
    height_l, width_l = left.shape[:2]
    height_r, width_r = right.shape[:2]

    canvas_corners = np.float32([[0, 0], [0, height_l], [width_l, height_l], [width_l, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(canvas_corners, H)
    all_corners = np.concatenate((canvas_corners, warped_corners), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]], dtype=np.float32)

    warped_left = cv2.warpPerspective(left, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    warped_right = cv2.warpPerspective(right, H_translation, (x_max - x_min, y_max - y_min))

    mask_left = np.tile(np.linspace(1, 0, width_l, dtype=np.float32), (height_l, 1))
    mask_right = np.tile(np.linspace(0, 1, width_r, dtype=np.float32), (height_r, 1))

    mask_left = cv2.warpPerspective(mask_left, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    mask_right = cv2.warpPerspective(mask_right, H_translation, (x_max - x_min, y_max - y_min))

    mask_left = np.repeat(mask_left[:, :, np.newaxis], 3, axis=2)
    mask_right = np.repeat(mask_right[:, :, np.newaxis], 3, axis=2)

    blended_image = (warped_left * mask_left + warped_right * mask_right) / (mask_left + mask_right + 1e-10)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)
    return blended_image

def remove_black_border(image):
    """Removes black areas from the stitched image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# Load images
image_paths = ["img-1.jpg", "img-2.jpg", "img-3.jpg"]
images = load_and_resize_images([os.path.join(os.getcwd(), img) for img in image_paths])

# Detect features
sift = cv2.SIFT_create()
keypoints, descriptors, keypoint_images = [], [], []

for i, img in enumerate(images):
    kp, des = sift.detectAndCompute(img, None)
    keypoints.append(kp)
    descriptors.append(des)
    
    # Draw keypoints on images
    img_with_kp = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoint_images.append(img_with_kp)
    
    # Save keypoint images
    cv2.imwrite(os.path.join(output_dir, f"keypoints_img{i+1}.jpg"), img_with_kp)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_L2)
matches12 = bf.knnMatch(descriptors[0], descriptors[1], k=2)
matches23 = bf.knnMatch(descriptors[1], descriptors[2], k=2)

good_matches12 = [m for m, n in matches12 if m.distance < 0.75 * n.distance]
good_matches23 = [m for m, n in matches23 if m.distance < 0.75 * n.distance]

# Compute Homography
H12, _ = get_homography(keypoints[0], keypoints[1], good_matches12)
stitched_img12 = stitch_images(images[0], images[1], H12)

# Match stitched image with third image
kp123, des123 = sift.detectAndCompute(stitched_img12, None)
matches123 = bf.knnMatch(des123, descriptors[2], k=2)
good_matches123 = [m for m, n in matches123 if m.distance < 0.75 * n.distance]

H123, _ = get_homography(kp123, keypoints[2], good_matches123)
stitched_img123 = stitch_images(stitched_img12, images[2], H123)
stitched_img123 = remove_black_border(stitched_img123)

# Save the final stitched output image
stitched_image_path = os.path.join(output_dir, "stitched_output.jpg")
cv2.imwrite(stitched_image_path, stitched_img123)

# Display the results
stitched_img_rgb = cv2.cvtColor(stitched_img123, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 10))
plt.imshow(stitched_img_rgb)
plt.axis("off")
plt.title("Final Stitched Panorama")
plt.show()

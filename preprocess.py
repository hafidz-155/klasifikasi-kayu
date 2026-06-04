import cv2

IMAGE_SIZE = (30, 30)


def preprocess_image(image_path):

    # Baca gambar
    image = cv2.imread(image_path)

    # Resize
    resized = cv2.resize(
        image,
        IMAGE_SIZE
    )

    # Grayscale
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # Kuantisasi 8 level
    quantized = gray // 32

    return image, resized, gray, quantized
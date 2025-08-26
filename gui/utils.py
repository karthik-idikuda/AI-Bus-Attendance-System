from PyQt5.QtGui import QImage, QPixmap
import cv2

# ----------------------------
# Convert OpenCV image (BGR) to QPixmap for PyQt display
# ----------------------------
def cv_img_to_qt_pixmap(cv_img):
    """
    Converts OpenCV BGR image to QPixmap for PyQt display.
    
    Args:
        cv_img: numpy array, BGR image from OpenCV

    Returns:
        QPixmap object for QLabel display
    """
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img)

# ----------------------------
# Resize image with aspect ratio for display
# ----------------------------
def resize_cv_img(cv_img, width=None, height=None):
    """
    Resizes OpenCV image with maintained aspect ratio.
    
    Args:
        cv_img: numpy array
        width: desired width
        height: desired height

    Returns:
        resized OpenCV image
    """
    if width is None and height is None:
        return cv_img

    h, w = cv_img.shape[:2]

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
    return resized

# ----------------------------
# Example usage (testing)
# ----------------------------
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication, QLabel

    app = QApplication(sys.argv)
    label = QLabel()

    # Load image using OpenCV
    img = cv2.imread("../data/test.jpg")
    if img is not None:
        pixmap = cv_img_to_qt_pixmap(img)
        label.setPixmap(pixmap)
        label.show()
    else:
        print("[ERROR] Image not found.")

    sys.exit(app.exec_())

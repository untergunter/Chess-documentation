import cv2
import numpy as np
import matplotlib.pyplot as plt


class FramesGetter:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        self.more_to_read = True

    def __next__(self):
        if self.more_to_read == True:
            more_to_read, frame = self.video.read()
            self.more_to_read = more_to_read
            return frame
        else:
            return None


def find_borders(gray_image)->tuple:
    """ this function returns tuple with 4 corners of the board """
    gray_image = np.copy(gray_image)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    mask = gray_image > 80
    gray_image[mask] = 255
    plt.imshow(gray_image, cmap='gray')
    plt.show()

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    plt.imshow(edges, cmap='gray')
    plt.show()

    lines = cv2.HoughLines(edges, 0.83,np.deg2rad(1),200)
    lines = lines.reshape((-1,2))
    lines_for_algebra = []
    for row_index in range(lines.shape[0]):
        rho, theta = lines[row_index]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        big_int_for_plot = 10_000
        x1 = int(x0 + big_int_for_plot * (-b))
        y1 = int(y0 + big_int_for_plot * (a))
        x2 = int(x0 - big_int_for_plot * (-b))
        y2 = int(y0 - big_int_for_plot * (a))
        single_line = ((x1, y1), (x2, y2))
        lines_for_algebra.append(single_line)
        cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255),20)

    # find 4 intersections that forms a square
    plt.imshow(rgb_image)
    plt.show()

def main(path: str) -> tuple:
    """setup file reading"""
    video_reader = FramesGetter(path)
    current_frame = video_reader.__next__()
    current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


    plt.imshow(current_gray_frame,cmap='gray')
    plt.show()

    find_borders(current_gray_frame)
    # do things to first frame

    if False:
        while video_reader.more_to_read is True:
            last_frame = current_frame
            current_frame = video_reader.__next__()

            last_rgb_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            last_gray_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)

            current_rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            plt.imshow(last_rgb_frame)
            plt.show()

            plt.imshow(current_rgb_frame)
            plt.show()



if __name__ == '__main__':
    main(r'data/chess_game_video.mp4')

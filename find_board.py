from main import FramesGetter,h_v_lines,line_intersections,cluster_points,\
    find_max_square
import cv2
import numpy as np
import matplotlib.pyplot as plt

def keep_board_from_frame_by_last_known_coords(image,last_coords,delta):
    only_board_surrounding = image.copy()
    plot_gray(only_board_surrounding)
    max_x,max_y = last_coords.max(axis=0)+delta
    min_x,min_y = last_coords.min(axis=0)-delta
    only_board_surrounding[max_y:,:] = 0
    only_board_surrounding[:min_y,:] = 0
    only_board_surrounding[:,max_x:] = 0
    only_board_surrounding[:,:min_x] = 0
    plot_gray(image-only_board_surrounding)
    return only_board_surrounding

def plot_gray(image):
    plt.imshow(image,cmap='gray')
    plt.show()
    plt.cla()

def plot_rgb(image):
    plt.imshow(image)
    plt.show()
    plt.cla()

def find_borders(gray_image,debug=True):

    if debug:plot_gray(gray_image)

    _, segmented = cv2.threshold(gray_image, 0, 255
                                , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:plot_gray(segmented)
    edges = cv2.Canny(segmented, 100, 200)
    if debug:plot_gray(edges)
    lines = cv2.HoughLines(edges, 2, np.deg2rad(1), 180)
    if lines is None:
        return None
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, 10)

    if debug:
        x = [p[0] for p in clustered_points]
        y = [p[1] for p in clustered_points]

        plt.imshow(gray_image,cmap='gray')
        plt.scatter(x, y, marker="o", color="red", s=20)
        plt.show()


    board_border_points = find_max_square(clustered_points, 100)
    if debug:
        x = [p[0] for p in board_border_points]
        y = [p[1] for p in board_border_points]

        plt.imshow(gray_image,cmap='gray')
        plt.scatter(x, y, marker="o", color="red", s=20)
        plt.show()

    return board_border_points

def main(path):
    video_reader = FramesGetter(path)
    i=0
    while video_reader.more_to_read is True:
        current_frame = video_reader.__next__()
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        board_border_points = find_borders(current_gray_frame)
        x = [p[0] for p in board_border_points]
        y = [p[1] for p in board_border_points]

        plt.imshow(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))
        plt.scatter(x, y, marker="o", color="red", s=20)
        plt.show()
        i+=1
        if i ==10:
            break

if __name__=='__main__':
    path = 'data/new_hope.mp4'
    main(path)
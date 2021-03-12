from main import FramesGetter,h_v_lines,line_intersections,cluster_points,\
    find_max_square
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def find_maximal_square(image,segment:int,debug=True):
    image = image.copy()
    image = np.uint8((image==segment)*255)
    edges = cv2.Canny(image, 1, 100)
    if debug: plot_gray(edges)
    lines = cv2.HoughLines(edges, 2, np.deg2rad(1), 180)
    if lines is None:
        return None
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    minimal_distance = int(np.min(image.shape)/10)
    clustered_points = cluster_points(intersection_points, minimal_distance)
    if len(clustered_points)>15:
        plot_frame_and_points(image,cluster_points)
        return None
    board_border_points = find_max_square(clustered_points, minimal_distance)

    return board_border_points

def find_borders(gray_image,debug=True):

    if debug:plot_gray(gray_image)

    _, segmented = cv2.threshold(gray_image, 0, 255
                                , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:plot_gray(segmented)

    _, components = cv2.connectedComponents(segmented)

    unique, counts = np.unique(components, return_counts=True)
    ascending_sizes = counts.argsort()
    descending_sizes = ascending_sizes[::-1]
    bigger_components_first = unique[descending_sizes]
    for component_number in bigger_components_first:
        square_vertices = find_maximal_square(components, component_number, debug = debug)
        if square_vertices is not None:
            return square_vertices
    return None

def plot_frame_and_points(gray_image,points):
    # x = [p[0] for p in clustered_points]
    # y = [p[1] for p in clustered_points]
    x = points[:,0]
    y = points[:,1]
    plt.imshow(gray_image,cmap='gray')
    plt.scatter(x, y, marker="o", color="red", s=30)
    plt.show()

def main(path):
    video_reader = FramesGetter(path)
    i=0
    while video_reader.more_to_read is True:
        i+=1
        current_frame = video_reader.__next__()
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        border_points = find_borders(current_gray_frame,debug=False)
        if border_points is None:
            print('no can do sir')
            continue
        plot_frame_and_points(current_gray_frame,border_points)

if __name__=='__main__':
    path = 'data/new_hope.mp4'
    path2 = 'data/chess_game_video.mp4'
    main(path)
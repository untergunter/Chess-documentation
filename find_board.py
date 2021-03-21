from main import FramesGetter,h_v_lines,line_intersections,cluster_points,\
    find_max_square
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.signal import argrelextrema

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

def plot_image_hull(image,points,hull):

    plt.imshow(image, cmap='gray')
    plt.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    plt.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')
    plt.show()

def intersection(L1, L2):
    """ from https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines """
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def get_all_intersection_points(binary_2d_image):
    lines = cv2.HoughLines(np.uint8(binary_2d_image), 2, np.deg2rad(1), 180)
    if lines is None:
        return None
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)

    for i in (horizontal, vertical):
        l = np.array(i)
        plt.scatter(l[:,0],l[:,1])
        plt.show()
        plt.cla()

    intersection_points = line_intersections(horizontal, vertical)
    return intersection_points

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
        plot_frame_and_points(image,clustered_points)
        return None
    board_border_points = find_max_square(clustered_points, minimal_distance)

    return board_border_points

def plot_with_lines(image,lines_list):
    plt.imshow(image,cmap='gray')
    for line in lines_list:
        p1,p2 = line
        plt.plot(p1,p2, color="green", linewidth=3)
    plt.show()

def lines_from_maximum(horizontal_max,vertical_max,image):
    v_max,h_max = image.shape
    horizontal_maxes = [((x,x),(0,v_max)) for x in horizontal_max]
    vertical_maxes = [((0,h_max), (y,y)) for y in vertical_max]
    all_lines = horizontal_maxes + vertical_maxes
    return all_lines

def lines_from_component(binary):
    image = np.uint8(binary*255)
    edges = cv2.Canny(image, 1, 100)
    horizontal = np.sum(edges,axis=1)
    vertical = np.sum(edges, axis=0)
    plot_gray(edges)
    n_elements = int(min(binary.shape)/20)
    y_axis = argrelextrema(horizontal, np.greater,order=n_elements)[0]
    x_axis = argrelextrema(vertical, np.greater,order=n_elements)[0]
    all_lines = lines_from_maximum(x_axis,y_axis,image)
    intersection_points = [(x,y) for x in x_axis for y in y_axis]
    return all_lines,intersection_points

def find_81_p(gray_image,debug=True):

    if debug:plot_gray(gray_image)

    _, segmented = cv2.threshold(gray_image, 0, 255
                                , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:plot_gray(segmented)

    _, components = cv2.connectedComponents(segmented)

    unique, counts = np.unique(components, return_counts=True)
    ascending_sizes = np.argsort(counts)
    descending_sizes = ascending_sizes[::-1]
    minimal_size = (gray_image.shape[0] * gray_image.shape[1]) /8
    bigger_components_first = unique[descending_sizes]
    sizes = counts[descending_sizes]
    for index,component_number in enumerate(bigger_components_first):
        if sizes[index]<minimal_size: break
        is_component = (components==component_number)*1
        all_lines,intersection_points = lines_from_component(is_component)
        if len(intersection_points)==81:
            return intersection_points

def plot_frame_and_points(gray_image,points):
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
        plot_gray(current_gray_frame)
        border_points = find_borders(current_gray_frame,debug=False)
        break
        if border_points is None:
            print('no can do sir')
            continue
        plot_frame_and_points(current_gray_frame,border_points)

def orient_first_board(first_board:np.ndarray)->np.ndarray:
    """
    rotates the board if it is on its side in a way the white are on the bottom
    :param first_board: image of the board
    :return:
    """
    best_angle = first_board.copy()
    max_diff = 0
    for _ in range(4):
        rows_sum = first_board.sum(axis=1)
        eighth = int(rows_sum.shape[0]/8)
        lower_board = np.mean(rows_sum[-eighth:])
        upper_board = np.mean(rows_sum[:eighth])
        diff = lower_board - upper_board
        if diff > max_diff:
            max_diff = diff
            best_angle = first_board.copy()
        first_board = np.rot90(first_board)
    return best_angle

if __name__=='__main__':
    path = 'data/new_hope.mp4'
    path2 = 'data/chess_game_video.mp4'
    # main(path)
    t = np.array([-1, -1, 0, 0, 0, 0, 1, 1]).repeat(8).reshape((8,8)).T
    r = orient_first_board(t)
    print(r)
    t = np.array([-1, -1, 0, 0, 0, 0, 1, 1]).repeat(8).reshape((8, 8))
    r = orient_first_board(t)
    print(r)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

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

def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines

# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)

def euclidien_distance(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def cluster_points(points_array,minimal_distance:int):
    final_points = []
    for row in range(points_array.shape[0]):
        to_keep = True
        x,y = points_array[row,:]
        for x_exist,y_exist in final_points:
            if euclidien_distance(x,y,x_exist,y_exist)<minimal_distance:
                to_keep = False
                break
        if to_keep is True:
            final_points.append((x,y))
    return final_points

def two_points_euclidien_distance(p1,p2):
    return euclidien_distance(p1[0],p1[1],p2[0],p2[1])

def is_valid_square(points_4,minimal_distance:int):
    distances = list({ two_points_euclidien_distance(p1,p2) for p1 in points_4 for p2 in points_4
                     if p1 != p2})
    distances.sort()
    small_vertices,big_vertices\
        ,small_diagonal,big_diagonal\
            =distances[0],distances[3]\
                ,distances[4],distances[5]
    if big_vertices - small_vertices <= minimal_distance:
        if abs((2**0.5)*big_vertices -small_diagonal) <= minimal_distance:
            if abs((2 ** 0.5) * small_vertices - big_diagonal) <= minimal_distance:
                return sum(distances[:4])/4
    return None

def find_max_square(points,minimal_distance:int):
    valid_squares = [(quad,is_valid_square(quad,minimal_distance))
                     for quad in combinations(points,4)
                     if is_valid_square(quad,minimal_distance) is not None]
    valid_squares.sort(key = lambda x:x[1])
    bigest_square = valid_squares[-1][0]
    return bigest_square



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
    horizontal , vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal,vertical)
    clustered_points = cluster_points(intersection_points,50)

    x = [p[0] for p in clustered_points]
    y = [p[1] for p in clustered_points]

    plt.imshow(rgb_image)
    plt.scatter(x, y, marker="o", color="red", s=5)
    plt.show()

    board_border_points = find_max_square(clustered_points, 100)

    x = [p[0] for p in board_border_points]
    y = [p[1] for p in board_border_points]

    plt.imshow(rgb_image)
    plt.scatter(x, y, marker="o", color="green", s=10)
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

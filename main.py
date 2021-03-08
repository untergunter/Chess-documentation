import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from functools import reduce
from scipy.cluster.hierarchy import fclusterdata
import itertools
import operator
import math


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


def euclidien_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def cluster_points(points_array, minimal_distance: int):
    final_points = []
    for row in range(points_array.shape[0]):
        to_keep = True
        x, y = points_array[row, :]
        for x_exist, y_exist in final_points:
            if euclidien_distance(x, y, x_exist, y_exist) < minimal_distance:
                to_keep = False
                break
        if to_keep is True:
            final_points.append((x, y))
    return final_points


def two_points_euclidien_distance(p1, p2):
    return euclidien_distance(p1[0], p1[1], p2[0], p2[1])


def is_valid_square(points_4, minimal_distance: int):
    distances = list({two_points_euclidien_distance(p1, p2) for p1 in points_4 for p2 in points_4
                      if p1 != p2})
    distances.sort()
    small_vertices, big_vertices \
        , small_diagonal, big_diagonal \
        = distances[0], distances[3] \
        , distances[4], distances[5]
    if big_vertices - small_vertices <= minimal_distance:
        if abs((2 ** 0.5) * big_vertices - small_diagonal) <= minimal_distance:
            if abs((2 ** 0.5) * small_vertices - big_diagonal) <= minimal_distance:
                return sum(distances[:4]) / 4
    return None


def find_max_square(points, minimal_distance: int):
    valid_squares = [(quad, is_valid_square(quad, minimal_distance))
                     for quad in combinations(points, 4)
                     if is_valid_square(quad, minimal_distance) is not None]
    valid_squares.sort(key=lambda x: x[1])
    bigest_square = valid_squares[-1][0]
    return bigest_square


def find_borders(gray_image) -> tuple:
    """ this function returns tuple with 4 corners of the board """
    gray_image = np.copy(gray_image)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    mask = gray_image > 80
    gray_image[mask] = 255
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    lines = cv2.HoughLines(edges, 0.83, np.deg2rad(1), 200)
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, 50)

    x = [p[0] for p in intersection_points]
    y = [p[1] for p in intersection_points]

    # plt.imshow(rgb_image)
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    x = [p[0] for p in clustered_points]
    y = [p[1] for p in clustered_points]

    # plt.imshow(rgb_image)
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    board_border_points = find_max_square(clustered_points, 100)

    x = [p[0] for p in board_border_points]
    y = [p[1] for p in board_border_points]

    # plt.imshow(rgb_image)
    # plt.scatter(x, y, marker="o", color="green", s=20)
    # plt.show()

    return board_border_points


def sort_points_clockwise(coords):
    center = tuple(
        map(operator.truediv
            , reduce(
                lambda x, y: map(operator.add, x, y), coords)
            , [4] * 2))
    points_sorted_clockwise = (sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360))

    return points_sorted_clockwise


#
# def keep_only_board(image,points):
#
#     image = np.copy(image)
#     x = [int(p[0]) for p in points]
#     y = [int(p[1]) for p in points]
#
#     max_x = max(x)
#     min_x = min(x)
#
#     max_y = max(y)
#     min_y = min(y)
#
#     xl = np.arange(0, image.shape[1])
#     yl = np.arange(0, image.shape[0])
#
#     xx, yy = np.meshgrid(xl, yl)
#     x_good = (xx >= min_x) * (xx <= max_x) * 1
#     y_good = (yy >= min_y) * (yy <= max_y) * 1
#     board = x_good*y_good*1
#     image = image * board
#     plt.imshow(image,cmap='gray')
#     plt.show()
#     return image

def board_reperspective(image, points):
    curent_corners = np.float32(sort_points_clockwise(points))
    size_of_new_image = 1800
    map_corners_to = np.float32([[0, 0]
                                    , [0, size_of_new_image]
                                    , [size_of_new_image, size_of_new_image]
                                    , [size_of_new_image, 0]])
    rotation_matrix = cv2.getPerspectiveTransform(curent_corners, map_corners_to)
    aligned = cv2.warpPerspective(image, rotation_matrix, (1800, 1800))
    # plt.imshow(aligned, cmap='gray')
    # plt.show()
    return aligned


def diff_y(p1, p2):
    diff = p1[1] - p2[1]
    return np.vdot(diff, diff) ** 0.5


def diff_x(p1, p2):
    diff = p1[0] - p2[0]
    return np.vdot(diff, diff) ** 0.5


def split_list_by_cluster_indexes(points, cluster_indexes):
    clusters = {}
    for i, val in enumerate(cluster_indexes):
        if (val not in clusters):
            clusters[val] = []
        clusters[val].append(points[i])
    for key in clusters.copy().keys():
        if (len(clusters[key]) < 3):
            del clusters[key]
    return clusters


def remove_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def cluster_avg_diff(cluster, axis):
    dif_list = []
    for points_list in cluster.values():
        points_list_axis = np.sort([pair[axis] for pair in points_list])
        dif_list.extend(np.diff(points_list_axis))
        dif_list_arr = np.array(dif_list)
    return np.average(remove_outliers(dif_list_arr))


def find_81_points(gray_image):
    """ this function returns 81 points of the board squares """
    gray_image = np.copy(gray_image)
    # plt.imshow(cropped, cmap='gray')
    # plt.show()
    v = np.median(gray_image)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray_image, lower, upper)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    lines = cv2.HoughLines(edges, 1, np.deg2rad(1), 170)
    lines = np.reshape(lines, (-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, 50)

    # x = [p[0] for p in intersection_points]
    # y = [p[1] for p in intersection_points]
    #
    # plt.imshow(cropped, cmap='gray')
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    # x = [p[0] for p in clustered_points]
    # y = [p[1] for p in clustered_points]
    #
    # plt.imshow(cropped, cmap='gray')
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    cluster_x_indexes = fclusterdata(clustered_points, 9, criterion='maxclust', metric=diff_x)
    cluster_y_indexes = fclusterdata(clustered_points, 9, criterion='maxclust', metric=diff_y)

    clusters_x = split_list_by_cluster_indexes(clustered_points, cluster_x_indexes)
    clusters_y = split_list_by_cluster_indexes(clustered_points, cluster_y_indexes)

    avg_diff_y = cluster_avg_diff(clusters_x, 1)
    avg_diff_x = cluster_avg_diff(clusters_y, 0)

    start_x = 0
    start_y = 0
    points_x = np.arange(0, 9) * avg_diff_x + start_x
    points_y = np.arange(0, 9) * avg_diff_y + start_y
    X2D, Y2D = np.meshgrid(points_y, points_x)
    final_points = np.column_stack((Y2D.ravel(), X2D.ravel()))

    x = [s[0] for s in final_points]
    y = [s[1] for s in final_points]
    plt.imshow(gray_image, cmap='gray')
    plt.scatter(x, y, marker="o", color="red", s=5)
    plt.show()

    return final_points


def crop_81_squares(gray_image, points):
    rows = []
    cluster_y_indexes = fclusterdata(points, 9, criterion='maxclust', metric=diff_y)
    clusters_y = split_list_by_cluster_indexes(points, cluster_y_indexes)
    for y_list in clusters_y.values():
        rows.append(y_list)
    rows = sorted(rows, key=lambda row: np.average([point[1] for point in row]))
    squares_list = []
    for first_row, second_row in zip(rows, rows[1:]):
        for (first_point, second_point) in zip(
                first_row, second_row[1:]):
            squares_list.append(gray_image[int(first_point[1]):int(second_point[1]), int(first_point[0]):int(second_point[0])])

    return squares_list


def main(path: str) -> tuple:
    """setup file reading"""
    video_reader = FramesGetter(path)
    for i in range(1, 350):
        current_frame = video_reader.__next__()
    current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # plt.imshow(current_gray_frame, cmap='gray')
    # plt.show()

    board_border_points = find_borders(current_gray_frame)
    # do things to first frame

    # board = keep_only_board(current_gray_frame,board_border_points)

    cropped_board = board_reperspective(current_gray_frame, board_border_points)
    # plt.imshow(croped_board, cmap='gray')
    # plt.show()

    cropped_board_no_border = cropped_board[95:-95, 95:-95]

    final_points = find_81_points(cropped_board_no_border)

    squares_list = crop_81_squares(cropped_board_no_border, final_points)

    for square in squares_list:
        plt.imshow(square, cmap='gray')
        plt.show()

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

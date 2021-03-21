import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance as dist
import chess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import imutils
from scipy.spatial.distance import cdist
import pandas as pd
import networkx as nx
from scipy.signal import argrelextrema


class FramesGetter:
    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        self.more_to_read = True

    def __next__(self):
        if self.more_to_read:
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


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


def lines_from_maximum(horizontal_max, vertical_max, image):
    v_max, h_max = image.shape
    horizontal_maxes = [((x, x), (0, v_max)) for x in horizontal_max]
    vertical_maxes = [((0, h_max), (y, y)) for y in vertical_max]
    all_lines = horizontal_maxes + vertical_maxes
    return all_lines


def lines_from_component(binary):
    image = np.uint8(binary * 255)
    edges = cv2.Canny(image, 1, 100)
    horizontal = np.sum(edges, axis=1)
    vertical = np.sum(edges, axis=0)
    n_elements = int(min(binary.shape)/12)
    y_axis = argrelextrema(horizontal, np.greater,order=n_elements)[0]
    x_axis = argrelextrema(vertical, np.greater,order=n_elements)[0]
    all_lines = lines_from_maximum(x_axis,y_axis,image)
    intersection_points = np.array([(x,y) for x in x_axis for y in y_axis])
    return all_lines,intersection_points


def find_81_p(gray_image, debug=False):
    if debug:
        plot_gray(gray_image)

    _, segmented = cv2.threshold(gray_image, 0, 255
                                 , cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if debug:
        plot_gray(segmented)

    _, components = cv2.connectedComponents(segmented)

    unique, counts = np.unique(components, return_counts=True)
    ascending_sizes = np.argsort(counts)
    descending_sizes = ascending_sizes[::-1]
    minimal_size = (gray_image.shape[0] * gray_image.shape[1]) / 8
    bigger_components_first = unique[descending_sizes]
    sizes = counts[descending_sizes]
    for index, component_number in enumerate(bigger_components_first):
        if sizes[index] < minimal_size: break
        is_component = (components == component_number) * 1
        all_lines, intersection_points = lines_from_component(is_component)
        if len(intersection_points) == 81:
            return intersection_points
    return None


def euclidien_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def cluster_points(points_array, minimal_distance: int):
    points_df = pd.DataFrame(points_array, columns=['x', 'y'])
    points_df['point_index'] = pd.Series(range(points_df.shape[0]))
    points_df['join_key'] = 0
    cross_join = pd.merge(left=points_df, right=points_df, on='join_key', suffixes=['_1', '_2'])
    cross_join['distance'] = (
                                     (cross_join['x_1'] - cross_join['x_2']) ** 2 +
                                     (cross_join['y_1'] - cross_join['y_2']) ** 2
                             ) ** 0.5
    close_points = cross_join[cross_join['distance'] <= minimal_distance]

    G = nx.Graph()
    G.add_edges_from(close_points[['point_index_1', 'point_index_2']].to_numpy().tolist())
    connected = list(nx.connected_components(G))
    points_indices = []
    for component in connected:
        element_index = next(iter(component))
        points_indices.append(element_index)
    final_points = np.take(points_array, points_indices, 0)
    return final_points


def two_points_euclidien_distance(p1, p2):
    return euclidien_distance(p1[0], p1[1], p2[0], p2[1])


def is_valid_square(points_4, minimal_distance: int):
    distances = np.unique(cdist(points_4, points_4))[1:]

    if len(distances) < 6:
        return None

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
    valid_squares = [(np.take(points, indices, 0),
                      is_valid_square(np.take(points, indices, 0), minimal_distance))
                     for indices in combinations(range(points.shape[0]), 4)
                     ]
    valid_squares = [square for square in valid_squares if not square[1] is None]
    if None in valid_squares:
        valid_squares.remove(None)
    if len(valid_squares) < 1:
        return None
    valid_squares.sort(key=lambda x: x[1])
    bigest_square = valid_squares[-1][0]
    return bigest_square


def find_borders(gray_image) -> tuple:
    """ this function returns tuple with 4 corners of the board """
    gray_image = np.copy(gray_image)

    v = np.median(gray_image)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray_image, lower, upper)

    lines = cv2.HoughLines(edges, 2, np.deg2rad(1), 180)
    if lines is None:
        return None
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, int(min(gray_image.shape) / 30))

    board_border_points = find_max_square(clustered_points, 100)

    return board_border_points

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

def rotate_board(current_board:np.ndarray,last_board:np.ndarray)->np.ndarray:
    """ this function takes a 8X8 representation of the board and the last Known one"""
    best_rotated = None
    smallest_diff = np.inf
    for _ in range(4):
        diff = current_board - last_board
        if diff<smallest_diff:
            best_rotated = current_board.copy()
        current_board = np.rot90(current_board)
    return best_rotated

def find_board_border_points(gray_image, debug=False):
    blurred = cv2.GaussianBlur(gray_image, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    if debug:
        plt.imshow(thresh, cmap='gray')
        plt.show()
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    board_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            board_cnt = approx
            break
    if board_cnt is None:
        return None
    if debug:
        output = gray_image.copy()
        cv2.drawContours(output, [board_cnt], -1, (0, 255, 0), 2)
        plot_frame_and_points(output, board_cnt.reshape(4, 2))
    return board_cnt.reshape(4, 2)


def sort_points_clockwise(coords):
    xSorted = coords[np.argsort(coords[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")


def board_reperspective(image, points, debug=False):
    curent_corners = np.float32(sort_points_clockwise(points))
    size_of_new_image = 1800
    map_corners_to = np.float32([[0, 0]
                                    , [size_of_new_image, 0]
                                    , [size_of_new_image, size_of_new_image]
                                    , [0, size_of_new_image]])
    rotation_matrix = cv2.getPerspectiveTransform(curent_corners, map_corners_to)
    aligned = cv2.warpPerspective(image, rotation_matrix, (1800, 1800))
    aligned = cv2.rotate(aligned, cv2.cv2.ROTATE_90_CLOCKWISE)
    if debug:
        plot_gray(aligned)
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
        if val not in clusters:
            clusters[val] = []
        clusters[val].append(points[i])
    for key in clusters.copy().keys():
        if len(clusters[key]) < 3:
            del clusters[key]
    return clusters


def remove_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def cluster_avg_diff(cluster, axis):
    dif_list = []
    for points_list in cluster.values():
        points_list_axis = np.sort([pair[axis] for pair in points_list])
        dif_list.extend(np.diff(points_list_axis))
    return np.average(remove_outliers(np.array(dif_list)))


def find_81_points(gray_image, debug=False):
    """ this function returns 81 points of the board squares """
    gray_image = np.copy(gray_image)
    if debug:
        plot_gray(gray_image)
    v = np.median(gray_image)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray_image, lower, upper)
    if debug:
        plot_gray(edges)

    lines = cv2.HoughLines(edges, 0.83, np.deg2rad(1), 200)

    if lines is None or len(lines) < 5:
        print("Not enough lines")
        return None

    lines = np.reshape(lines, (-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    if intersection_points is None or len(intersection_points) == 0:
        return None
    clustered_points = cluster_points(intersection_points, 80)

    if len(clustered_points) < 15:
        print("Not enough clustered points")
        return None

    if debug:
        plot_frame_and_points(gray_image, intersection_points)
        plot_frame_and_points(gray_image, clustered_points)

    cluster_x_indexes = fclusterdata(clustered_points, 9, criterion='maxclust', metric=diff_x)
    cluster_y_indexes = fclusterdata(clustered_points, 9, criterion='maxclust', metric=diff_y)

    clusters_x = split_list_by_cluster_indexes(clustered_points, cluster_x_indexes)
    clusters_y = split_list_by_cluster_indexes(clustered_points, cluster_y_indexes)

    avg_diff_y = cluster_avg_diff(clusters_x, 1)
    avg_diff_x = cluster_avg_diff(clusters_y, 0)

    if avg_diff_y > 230 or avg_diff_x > 230:
        print("distance too big")
        return None

    start_x = 0
    start_y = 0
    points_x = np.arange(0, 9) * avg_diff_x + start_x
    points_y = np.arange(0, 9) * avg_diff_y + start_y
    X2D, Y2D = np.meshgrid(points_y, points_x)
    final_points = np.column_stack((Y2D.ravel(), X2D.ravel()))

    if debug:
        plot_frame_and_points(gray_image, final_points)

    return final_points


def set_above_threshold_1_under_0(arr, percentile_threshold=75):
    threshold = np.percentile(arr, percentile_threshold)
    binary = (arr > threshold) * 1
    return binary


def naive_81_points(gray_image):
    """ if board found well and is from above - the lines are in equal spaces """
    vertical, horizontal = gray_image.shape
    vertical_guess = int(vertical / 8)
    horizontal_guess = int(horizontal / 8)
    base_intervals = np.linspace(0, 8, num=9, dtype=int)
    x_points = np.repeat(base_intervals, 9) * horizontal_guess
    y_points = np.tile(base_intervals, 9) * vertical_guess
    x_points = x_points.reshape((-1, x_points.shape[0]))
    y_points = y_points.reshape((-1, y_points.shape[0]))
    final_points = np.concatenate((x_points, y_points), axis=0).T
    return final_points


def extract_81_points(gray_image):
    """ we want to find the lines. they are almost exactly horizontal and vertical, and in semi equal spaces"""

    """ calculate delta """
    horizontal_delta = np.abs(gray_image[1:, :] - gray_image[:-1, :])
    vertical_delta = np.abs(gray_image[:, 1:] - gray_image[:, :-1])

    """ if delta is big enough keep it"""

    horizontal_delta_binary = set_above_threshold_1_under_0(horizontal_delta)
    vertical_delta_binary = set_above_threshold_1_under_0(vertical_delta)

    """ sum per axis to get the line """

    big_delta_per_row = np.sum(horizontal_delta_binary, axis=1)
    big_delta_per_column = np.sum(vertical_delta_binary, axis=0)

    """ there are 9 lines in every direction in equal spaces.
    if line is thick they are duplicated but closer """


def crop_81_squares(gray_image, points):
    rows = []
    cluster_y_indexes = fclusterdata(points, 9, criterion='maxclust', metric=diff_y)
    clusters_y = split_list_by_cluster_indexes(points, cluster_y_indexes)
    for y_list in clusters_y.values():
        rows.append(y_list)
    rows = sorted(rows, key=lambda row: np.average([point[1] for point in row]))
    squares_list = {}
    current_row = 8
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    column_index = 0
    for first_row, second_row in zip(rows, rows[1:]):
        for (first_point, second_point) in zip(
                first_row, second_row[1:]):
            if np.isnan(first_point[0]) or np.isnan(first_point[1]) or np.isnan(second_point[0]) or np.isnan(
                    second_point[1]):
                return None
            squares_list[columns[column_index] + str(current_row)] = (
            gray_image[int(first_point[1]):int(second_point[1]), int(first_point[0]):int(second_point[0])])
            column_index += 1
        current_row -= 1
        column_index = 0
    return squares_list


def plot_frame_and_points(gray_image, points):
    x = points[:, 0]
    y = points[:, 1]
    plt.imshow(gray_image, cmap='gray')
    plt.scatter(x, y, marker="o", color="red", s=30)
    plt.show()


def plot_gray(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.cla()


def handle_frame(current_frame, debug=None):
    if not debug:
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        board_border_points = find_board_border_points(current_gray_frame, debug=False)
        print("done border_points")
        if board_border_points is None:
            print("border points is None")
            return None

        cropped_board = board_reperspective(current_frame, board_border_points)

        cropped_board_no_border = cropped_board[90:-90, 90:-90]
    else:
        cropped_board_no_border = current_frame

    as_gray = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2GRAY)
    final_points = find_81_points(as_gray)

    if final_points is None:
        print("final points points is None")
        return None
    else:
        print("done final_points")

    if debug:
        plot_frame_and_points(cropped_board_no_border, final_points)

    squares_dict = crop_81_squares(adjust_gamma(cropped_board_no_border), final_points)

    if squares_dict is None:
        return None

    return squares_dict, cropped_board_no_border


def diff_squares(current_squares, model):
    square_is_contains_piece = {}
    for key in current_squares.keys():
        try:
            current_square = current_squares[key]
            current_square = cv2.resize(current_square, (150, 150))
            current_square = current_square[30:120, 30:120]
            hist_0 = cv2.calcHist([current_square], [0], None, [256], [0, 256]).flatten()
            hist_1 = cv2.calcHist([current_square], [1], None, [256], [0, 256]).flatten()
            hist_2 = cv2.calcHist([current_square], [2], None, [256], [0, 256]).flatten()
            hist = np.concatenate((hist_0, hist_1, hist_2), axis=None)
            label = model.predict(hist.reshape(1, -1))
            square_is_contains_piece[key] = label
        except Exception as e:
            print(str(e))
            return None
    return square_is_contains_piece


def add_histogram(current_piece_dict, squares_dict_current):
    histograms = []
    labels = []
    for key in current_piece_dict.keys():
        current_square = squares_dict_current[key]
        current_square = cv2.resize(current_square, (150, 150))
        current_square = current_square[30:120, 30:120]
        hist_0 = cv2.calcHist([current_square], [0], None, [256], [0, 256]).flatten()
        hist_1 = cv2.calcHist([current_square], [1], None, [256], [0, 256]).flatten()
        hist_2 = cv2.calcHist([current_square], [2], None, [256], [0, 256]).flatten()
        hist = np.concatenate((hist_0, hist_1, hist_2), axis=None)
        hist = hist.flatten()
        histograms.append(hist)
        labels.append(current_piece_dict[key][0])

    # histograms = np.array(histograms)
    # labels = np.array(labels)

    return histograms, labels


def knn_histograms(current_squares_list):
    histograms = []
    labels = []
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for current_squares in current_squares_list:
        for row in range(8, 0, -1):
            for col in columns:
                current_square = current_squares[col + str(row)]
                current_square = cv2.resize(current_square, (150, 150))
                current_square = current_square[30:120, 30:120]
                hist_0 = cv2.calcHist([current_square], [0], None, [256], [0, 256]).flatten()
                hist_1 = cv2.calcHist([current_square], [1], None, [256], [0, 256]).flatten()
                hist_2 = cv2.calcHist([current_square], [2], None, [256], [0, 256]).flatten()
                hist = np.concatenate((hist_0, hist_1, hist_2), axis=None)
                hist = hist.flatten()
                histograms.append(hist)
                labels.append("full" if row in [1, 2, 7, 8] else "empty")

    # histograms = np.array(histograms)
    # labels = np.array(labels)

    return histograms, labels


def chi_squared(p, q):
    return 0.5 * np.sum((p - q) ** 2 / (p + q + 1e-6))


def knn_model(histograms, labels):
    (trainHist, testHist, trainLabels, testLabels) = train_test_split(
        histograms, labels, test_size=0.2, random_state=99)

    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=5,
                                 n_jobs=-1, metric=chi_squared)
    model.fit(histograms, labels)
    acc = model.score(testHist, testLabels)
    print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

    return model


def update_board_state(current_piece_dict, prev_piece_dict, piece_dict_color_diff, board):
    label_change_count = 0
    to_full_change_count = 0
    to_empty_change_count = 0
    from_square = None
    to_square = None
    debug_change_dict = {}
    for key in current_piece_dict.keys():
        current_label = current_piece_dict[key]
        prev_label = prev_piece_dict[key]
        if current_label != prev_label:
            label_change_count += 1
            if current_label == 'full':
                if to_square is None:
                    to_square = []
                to_square.append(key)
                to_full_change_count += 1
            else:
                if from_square is None:
                    from_square = []
                from_square.append(key)
                to_empty_change_count += 1
            debug_change_dict[key] = current_label

    print(f"moves: {str(label_change_count)}")

    if piece_dict_color_diff and label_change_count == 1 and to_empty_change_count == 1:
        color_change_list = [k for k, v in piece_dict_color_diff.items() if v]
        if from_square[0] in color_change_list:
            color_change_list.remove(from_square[0])
        if len(color_change_list) == 1:
            move = str(from_square[0]) + str(color_change_list[0])
            if chess.Move.from_uci(move) in board.legal_moves:
                print(f"made move: {move + ' '}")
                return move, label_change_count

    elif label_change_count < 2:
        return None, label_change_count

    elif label_change_count == 2 and from_square and to_square:
        move = str(from_square[0]) + str(to_square[0])
        if chess.Move.from_uci(move) in board.legal_moves:
            print(f"made move: {move + ' '}")
            return move, label_change_count

    else:
        for key in debug_change_dict.keys():
            print(key + ": " + str(debug_change_dict[key]))
    return None, label_change_count


def start_piece_dict():
    piece_dict = {}
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    for row in range(8, 0, -1):
        for col in columns:
            piece_dict[col + str(row)] = "full" if row in [1, 2, 7, 8] else "empty"
    return piece_dict


def adjust_gamma(image, gamma=3.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def update_piece_dict(piece_dict, move):
    piece_dict[move[0:2]] = 'empty'
    piece_dict[move[2:4]] = 'full'
    return piece_dict


def color_diff(squares_dict_current, valid_squares_dict, debug = False):
    square_is_color_change = {}
    for key in squares_dict_current.keys():
        try:
            current_square = squares_dict_current[key]
            last_valid_square = valid_squares_dict[key]

            if debug:
                plot_gray(current_square)
                plot_gray(last_valid_square)

            channels = cv2.mean(cv2.resize(current_square, (150, 150))[30:120, 30:120])
            current_mean_colors = np.array([channels[0], channels[1], channels[2]])

            channels = cv2.mean(cv2.resize(last_valid_square, (150, 150))[30:120, 30:120])
            last_mean_colors = np.array([channels[0], channels[1], channels[2]])

            diff = chi_squared(current_mean_colors, last_mean_colors)

            square_is_color_change[key] = diff > 5
        except Exception as e:
            print(str(e))
            return None
    return square_is_color_change


def main(path: str) -> tuple:
    """setup file reading"""
    video_reader = FramesGetter(path)

    game_moves_file = open("game_moves.txt", "w")
    prev_frame = None
    current_frame = None
    prev_cropped_frame = None
    cropped_frame = None
    squares_dict_prev = None
    squares_dict_current = None
    current_piece_dict = None
    prev_piece_dict = None
    first_frame = True
    model = None
    frame_counter = 0
    max_frame = 10
    squares_dict_list = []
    histograms = None
    labels = None
    valid_squares_dict = None
    jump = False
    piece_dict_color_diff = None
    board = chess.Board()

    while video_reader.more_to_read is True:
        prev_frame = current_frame

        if jump:
            for i in range(0, 20):
                current_frame = video_reader.__next__()
            frame_counter += 20
            jump = False
            print("jump!")
        else:
            current_frame = video_reader.__next__()
        if current_frame is None:
            break
        print("next")
        frame_counter += 1

        if frame_counter == max_frame:
            histograms, labels = knn_histograms(squares_dict_list)
            model = knn_model(histograms, labels)

        if squares_dict_current is not None:
            prev_cropped_frame = cropped_frame
            squares_dict_prev = squares_dict_current

        res = handle_frame(current_frame)
        if res:
            squares_dict_current, cropped_frame = res
            squares_dict_list.append(squares_dict_current)
            print("done handle")
        else:
            squares_dict_current = None

        if squares_dict_current and squares_dict_prev and model:
            if first_frame:
                prev_piece_dict = start_piece_dict()
                first_frame = False
            current_piece_dict = diff_squares(squares_dict_current, model)
            print("done diff")
            if valid_squares_dict:
                piece_dict_color_diff = color_diff(squares_dict_current, valid_squares_dict)
            if prev_piece_dict and current_piece_dict:
                move, label_change_count = update_board_state(current_piece_dict, prev_piece_dict,
                                                              piece_dict_color_diff, board)
                print("done update")
                if move:
                    game_moves_file.write(move + " ")
                    board.push(chess.Move.from_uci(move))
                    temp_histograms, temp_labels = add_histogram(current_piece_dict, squares_dict_current)
                    histograms.extend(temp_histograms)
                    labels.extend(temp_labels)
                    model = knn_model(histograms, labels)
                    prev_piece_dict = update_piece_dict(prev_piece_dict, move)
                    valid_squares_dict = squares_dict_current
                    jump = True
                else:
                    if label_change_count == 0:
                        valid_squares_dict = squares_dict_current
                        jump = True
    game_moves_file.close()


if __name__ == '__main__':
    main(r'data/new_chess.mp4')

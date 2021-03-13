import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from functools import reduce
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance as dist
import operator
import math
import chess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import imutils

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
    valid_squares = [(quad, is_valid_square(quad, minimal_distance))
                     for quad in combinations(points, 4)
                     if is_valid_square(quad, minimal_distance) is not None]
    valid_squares.sort(key=lambda x: x[1])
    if len(valid_squares) < 1:
        return None
    bigest_square = valid_squares[-1][0]
    return bigest_square


def find_borders(gray_image) -> tuple:
    print("here")
    """ this function returns tuple with 4 corners of the board """
    gray_image = np.copy(gray_image)
    gray_image = cv2.blur(gray_image, (3, 3))
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    # mask = gray_image > 80
    # gray_image[mask] = 255
    # plt.imshow(gray_image, cmap='gray')
    # plt.show()

    v = np.median(gray_image)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(gray_image, lower, upper)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    lines = cv2.HoughLines(edges, 2, np.deg2rad(1), 180)
    if lines is None:
        return None
    lines = lines.reshape((-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, 60)

    x = [p[0] for p in intersection_points]
    y = [p[1] for p in intersection_points]

    plt.imshow(rgb_image)
    plt.scatter(x, y, marker="o", color="red", s=5)
    plt.show()

    x = [p[0] for p in clustered_points]
    y = [p[1] for p in clustered_points]

    plt.imshow(rgb_image)
    plt.scatter(x, y, marker="o", color="red", s=5)
    plt.show()

    board_border_points = find_max_square(clustered_points, 100)

    x = [p[0] for p in board_border_points]
    y = [p[1] for p in board_border_points]

    plt.imshow(rgb_image)
    plt.scatter(x, y, marker="o", color="green", s=20)
    plt.show()

    return board_border_points


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
        plt.imshow(output, cmap='gray')
        plt.show()
        x = [p[0] for p in board_cnt.reshape(4, 2)]
        y = [p[1] for p in board_cnt.reshape(4, 2)]

        plt.imshow(gray_image)
        plt.scatter(x, y, marker="o", color="green", s=20)
        plt.show()
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
                                    , [size_of_new_image, 0]
                                    , [size_of_new_image, size_of_new_image]
                                    , [0, size_of_new_image]])
    rotation_matrix = cv2.getPerspectiveTransform(curent_corners, map_corners_to)
    aligned = cv2.warpPerspective(image, rotation_matrix, (1800, 1800))
    aligned = cv2.rotate(aligned, cv2.cv2.ROTATE_90_CLOCKWISE)
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
    return np.average(remove_outliers(np.array(dif_list)))


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

    lines = cv2.HoughLines(edges, 2, np.deg2rad(1), 180)

    if lines is None or len(lines) < 5:
        print("Not enough lines")
        return None

    lines = np.reshape(lines, (-1, 2))
    horizontal, vertical = h_v_lines(lines)
    intersection_points = line_intersections(horizontal, vertical)
    clustered_points = cluster_points(intersection_points, 80)

    if len(clustered_points) < 15:
        print("Not enough clustered points")
        return None

    # x = [p[0] for p in intersection_points]
    # y = [p[1] for p in intersection_points]
    #
    # plt.imshow(gray_image, cmap='gray')
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    # x = [p[0] for p in clustered_points]
    # y = [p[1] for p in clustered_points]
    #
    # plt.imshow(gray_image, cmap='gray')
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

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

    # x = [s[0] for s in final_points]
    # y = [s[1] for s in final_points]
    # plt.imshow(gray_image, cmap='gray')
    # plt.scatter(x, y, marker="o", color="red", s=5)
    # plt.show()

    return final_points


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
            if  np.isnan(first_point[0]) or np.isnan(first_point[1]) or np.isnan(second_point[0]) or np.isnan(second_point[1]):
                return None
            squares_list[columns[column_index]+str(current_row)] = (gray_image[int(first_point[1]):int(second_point[1]), int(first_point[0]):int(second_point[0])])
            column_index += 1
        current_row -= 1
        column_index = 0
    return squares_list


def handle_frame(current_frame, debug=None):
    if not debug:
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        board_border_points = find_board_border_points(current_gray_frame)
        print("done border_points")
        if board_border_points is None:
            print("border points is None")
            return None

        cropped_board = board_reperspective(current_frame, board_border_points)

        cropped_board_no_border = cropped_board[90:-90, 90:-90]
    else:
        cropped_board_no_border = current_frame

    final_points = find_81_points(cv2.cvtColor(cropped_board_no_border, cv2.COLOR_BGR2GRAY))

    if final_points is None:
        print("final points points is None")
        return None
    else:
        print("done final_points")

    if debug:
        plt.imshow(cropped_board_no_border, cmap='gray')
        x = [s[0] for s in final_points]
        y = [s[1] for s in final_points]
        plt.imshow(adjust_gamma(cropped_board_no_border), cmap='gray')
        plt.scatter(x, y, marker="o", color="red", s=5)
        plt.show()

    squares_dict = crop_81_squares(adjust_gamma(cropped_board_no_border), final_points)

    if squares_dict is None:
        return None

    return squares_dict, cropped_board_no_border


def canny_edge(img, sigma=0.7):
    v = np.median(img)
    lower = 10
    upper = 25
    edges = cv2.Canny(img, lower, upper)
    return edges


def diff_squares(current_squares, model):
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    square_is_contains_piece = {}
    temp = {}
    canny = {}
    for key in current_squares.keys():
        try:
            current_square = current_squares[key]

            # plt.imshow(current_square, cmap='gray')
            # plt.show()

            # channels = cv2.mean(cv2.resize(current_square, (150, 150))[30:120, 30:120])
            # observation = np.array([(channels[2], channels[1], channels[0])])
            # temp[key] = np.mean(observation)

            # current_square_gray = cv2.cvtColor(cv2.resize(current_square, (200, 200))[70:130, 70:130], cv2.COLOR_BGR2GRAY)
            # # plt.imshow(current_square_gray, cmap='gray')
            # # plt.show()
            # res_canny = canny_edge(current_square_gray)
            # # plt.imshow(res_canny, cmap='gray')
            # # plt.show()
            # canny[key] = cv2.countNonZero(res_canny)

            current_square = cv2.resize(current_square, (150, 150))
            current_square = current_square[30:120, 30:120]
            # current_square = current_square.flatten()
            hist_0 = cv2.calcHist([current_square], [0], None, [256], [0, 256]).flatten()
            hist_1 = cv2.calcHist([current_square], [1], None, [256], [0, 256]).flatten()
            hist_2 = cv2.calcHist([current_square], [2], None, [256], [0, 256]).flatten()
            hist = np.concatenate((hist_0, hist_1, hist_2), axis=None)
            label = model.predict(hist.reshape(1, -1))
            # print(label)
            square_is_contains_piece[key] = label
        except Exception as e:
            print(str(e))
            return None
        # current_square = median = cv2.medianBlur(current_square,5)
        # prev_square = median = cv2.medianBlur(prev_square,5)
        # current_square = cv2.adaptiveThreshold(current_square, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                       cv2.THRESH_BINARY, 11, 2)
        # prev_square = cv2.adaptiveThreshold(prev_square, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                       cv2.THRESH_BINARY, 11, 2)

        # if current_square.size > prev_square.size:
        #     current_square = cv2.resize(current_square, prev_square.shape[::-1], interpolation=cv2.INTER_AREA)
        # else:
        #     prev_square = cv2.resize(prev_square, current_square.shape[::-1], interpolation=cv2.INTER_AREA)
        #
        # diff = cv2.absdiff(current_square, prev_square)
        #
        # plt.imshow(diff, cmap='gray')
        # plt.show()
        #
        # matrix, thresold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        #
        # plt.imshow(thresold, cmap='gray')
        # plt.show()

        # plt.imshow(current_square, cmap='gray')
        # plt.show()
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
        labels.append(current_piece_dict[key])

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


def chi_squared(p,q):
    return 0.5*np.sum((p-q)**2/(p+q+1e-6))

# def custom_distance(p, q):


def knn_model(histograms, labels):

    (trainHist, testHist, trainLabels, testLabels) = train_test_split(
        histograms, labels, test_size=0.2, random_state=99)

    print("[INFO] evaluating histogram accuracy...")
    model = KNeighborsClassifier(n_neighbors=5,
                                 n_jobs=-1, metric=chi_squared)
    model.fit(histograms, labels)
    # acc = model.score(testHist, testLabels)
    # print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

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
            print(f"made move: {move+' '}")
            return move, label_change_count

    # elif 2 < label_change_count < 5 and to_empty_change_count < 2 and from_square and to_square:
    #     possible_moves = []
    #     for to_empty_square in from_square:
    #         for to_full_square in to_square:
    #             move = str(to_empty_square) + str(to_full_square)
    #             if chess.Move.from_uci(move) in board.legal_moves:
    #                 possible_moves.append(move)
    #     if len(possible_moves) == 1:
    #         print(f"made move: {move+' '}")
    #         return move, label_change_count
    #     else:
    #         return None, label_change_count

    else:
        for key in debug_change_dict.keys():
            print(key + ": " + str(debug_change_dict[key]))
        # plt.imshow(current_frame, cmap='gray')
        # plt.show()
        # handle_frame(current_frame, True)
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
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def update_piece_dict(piece_dict, move):
    piece_dict[move[0:2]] = 'empty'
    piece_dict[move[2:4]] = 'full'
    return piece_dict


def color_diff(squares_dict_current, valid_squares_dict):
    square_is_color_change = {}
    for key in squares_dict_current.keys():
        try:
            current_square = squares_dict_current[key]
            last_valid_square = valid_squares_dict[key]

            # plt.imshow(current_square, cmap='gray')
            # plt.show()
            #
            # plt.imshow(last_valid_square, cmap='gray')
            # plt.show()

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
            # plt.imshow(cropped_frame, cmap='gray')
            # plt.show()

        if squares_dict_current is not None:
            prev_cropped_frame = cropped_frame
            squares_dict_prev = squares_dict_current

        res = handle_frame(current_frame)
        if res:
            squares_dict_current, cropped_frame = res
            squares_dict_list.append(squares_dict_current)
            print("done handle")
            # if first_frame:
            #     histograms, labels = knn_histograms(squares_dict_current)
            #     model = knn_model(histograms, labels)
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
                move, label_change_count = update_board_state(current_piece_dict, prev_piece_dict, piece_dict_color_diff, board)
                print("done update")
                if move:
                    game_moves_file.write(move+" ")
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
                    # if label_change_count < 4:
                    #     temp_histograms, temp_labels = add_histogram(prev_piece_dict, squares_dict_current)
                    #     # histograms = histograms[256:]
                    #     # labels = labels[256:]
                    #     histograms.extend(temp_histograms)
                    #     labels.extend(temp_labels)
                    #     model = knn_model(histograms, labels)
                    #     # model = knn_model(np.concatenate((histograms, temp_histograms), axis=0), np.concatenate((labels, temp_labels.reshape(-1)), axis=0))
                #     else:
                #         plt.imshow(cropped_frame, cmap='gray')
                #         plt.show()
    game_moves_file.close()

if __name__ == '__main__':
    main(r'data/new_chess.mp4')

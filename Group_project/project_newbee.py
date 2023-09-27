# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 16:56:56 2021
@author: COMP9517 NewBee Group
ver: 2.8
"""
import cv2
import os
import numpy as np
# from skimage.morphology import watershed
from skimage.segmentation import watershed
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import mahotas as mh
import hashlib
import math
import random

#############################################
# User Define Variables
#############################################
# the path of the images
SOURCE_PATH = "./Sequences"
# the sub folder name
SUB_DIR = "01"
# the name of the dir which to save the result images
RESULT_DIR = 'result'
# the name of the dir which to save my seq, it is no use at present
SEG_DIR = 'myseg'
# define the gaussian kernal size
Gaussian_kernal = (5, 5)
# The max distance that a cell can move in two frame
d_distance = 50
# mitosis: mark circle radius，width and color(BGR)
mitosis_radius, mitosis_width, mitosis_color = 30, 3, (0, 0, 255)
child_radius, child_width, child_color = 25, 2, (00, 160, 255)
# Fout color
font_color, font_color_mitosis, font_size, font_thickness = (255, 255, 255), (0, 255, 255), 0.8, 2
# similarity which define a new cell as an old one：0-100, the larger the value, the easier it is
similarity_reappear = 85
# rate to define mitosis both child area must be smaller than their mother cell
# child cell area < mother cell * rate, 0.0-2.0
rate_mitosis = 0.9
# threshold of similarity to define whether it is still the old cell in new frame: 0-100
similarity_threshold = 20
# permit a cell to reappear time
permit_disappear_time = 2

#############################################
# machine used variables
#############################################
# sample cell
# status: 0 disappeared  1 existing  2 in mitosis  3 a new child
sample_cell = {"id": 0, "target_index": None, "center": (200, 300), "size": (3, 2), "color": (234, 102, 65),
               "area": 6, "status": 1, "trajectory": [(180, 280), (190, 290), (200, 300)], "disappear_time": 0}
statistics = {"reappear": 0, "dividing_times": 0, "total_dividing_times": 0, "dividing_cells": 0,
              "total_dividing_cells": 0, "new_cell": 0, "disappear_cell": 0}


# create a new cell, most value are default None
def create_cell(cell_id=None, center=None, size=None, color=None, area=0, status=1):
    # status: 0 disappeared  1 existing  2 in mitosis  3 a new child
    cell = {"id": cell_id,
            "target_index": None,
            "center": center,
            "size": size,
            "color": color,
            "area": area,
            "status": status,
            "trajectory": [],
            "disappear_time": 0}

    if center:
        cell["trajectory"].append(center)
    return cell


# set default value to statistics when a new round start
def new_round_statistics():
    statistics["reappear"] = 0
    statistics["dividing_times"] = 0
    statistics["dividing_cells"] = 0
    statistics["new_cell"] = 0
    statistics["disappear_cell"] = 0


# Define the Euclidean distance between two points as the loss
def euclidean_distance(pre_center, cur_center):
    distance = ((pre_center[0] - cur_center[0]) ** 2 + (pre_center[1] - cur_center[1]) ** 2) ** 0.5
    return distance


# return the similarity between two cells
# yita: the weight of area, should between 0 to 1
# the value is between 0 to 100
def get_two_cells_similarity(cell_A, cell_B, yita=0.2):
    distance = euclidean_distance(cell_A["center"], cell_B["center"])
    distance_ratio = 1 - math.pow(distance / d_distance, 1.2)
    if distance_ratio < 0: distance_ratio = 0
    area_ratio_adjast = 25
    if cell_A["area"] > cell_B["area"]:
        area_ratio = (cell_B["area"] + area_ratio_adjast) / (cell_A["area"] + area_ratio_adjast)
    else:
        area_ratio = (cell_A["area"] + area_ratio_adjast) / (cell_B["area"] + area_ratio_adjast)
    return (distance_ratio * (1 - yita) + area_ratio * yita) * 100


# generate a random color
def get_a_random_color():
    R, G, B = 0, 0, 0
    while (R < 200 and G < 200 and B < 200) or \
            (abs(R - G) + abs(R - B) + abs(G - B) < 150):
        R, G, B = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    return B, G, R


# step 1: Get and parse a image as the target layer
def analysis_frame(img_source, frame_name=""):
    img_normalize = np.zeros_like(img_source)
    cv2.normalize(img_source, img_normalize, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    unique, counts = np.unique(img_normalize.reshape(-1, 1), axis=0, return_counts=True)
    threshold_value = unique[np.argmax(counts)][0]

    ret, img_threshold = cv2.threshold(img_normalize, threshold_value, 255, cv2.THRESH_BINARY)

    img_flood = img_threshold.copy()
    height, width = img_flood.shape

    if img_flood[0][0] != 255:
        cv2.floodFill(img_flood, None, (0, 0), 1)
    if img_flood[height - 1][0] != 255:
        cv2.floodFill(img_flood, None, (0, height - 1), 1)
    if img_flood[height - 1][width - 1] != 255:
        cv2.floodFill(img_flood, None, (width - 1, height - 1), 1)
    if img_flood[0][width - 1] != 255:
        cv2.floodFill(img_flood, None, (width - 1, 0), 1)

    cv2.floodFill(img_flood, None, (0, 0), 1)
    img_threshold[img_flood == 0] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img_eroded = cv2.erode(img_threshold, kernel, iterations=1)

    img_dilated = cv2.dilate(img_eroded, kernel)

    distance = ndi.morphology.distance_transform_edt(img_dilated)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((9, 9)), labels=img_dilated)
    markers = ndi.label(local_maxi)[0]
    img_watershed = watershed(-distance, markers, mask=img_dilated)

    img_labeled = img_watershed.astype(np.uint8)

    # Detect and drawing the outline of cells
    contours, hierarchy = cv2.findContours(img_labeled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # generate colored labeled image
    img_final = cv2.cvtColor(img_normalize, cv2.COLOR_GRAY2RGB)

    # extract cell information
    cells_target = []
    for i in range(0, len(contours)):
        pos_x, pos_y, width, height = cv2.boundingRect(contours[i])

        cell = create_cell()
        cell["center"] = (pos_x + int(width / 2), pos_y + int(height / 2))
        cell["size"] = (width, height)
        area = cv2.contourArea(contours[i])
        if area < 1: area = 1
        cell["area"] = area

        cells_target.append(cell)

    return img_final, cells_target, contours


# step 2: Compare the target layer with the cells in the cell bank
def compare_with_bank(cells_bank, cells_target):
    pair_list = []
    # 2.1
    for cell_id, cell_bank in cells_bank.items():
        if cell_bank["status"] != 1 and cell_bank["status"] != 3: continue
        cell_bank["target_index"] = None
        # 2.2
        for index in range(len(cells_target)):
            cell_target = cells_target[index]
            if euclidean_distance(cell_target["center"], cell_bank["center"]) < d_distance:
                similarity = get_two_cells_similarity(cell_target, cell_bank)
                if similarity < similarity_threshold: continue
                pair = {"id": cell_bank["id"],
                        "target_index": index,
                        "similarity": similarity}
                pair_list.append(pair)

    # 2.3
    sorted_pair_list = sorted(pair_list, key=lambda pair_list: pair_list["similarity"], reverse=True)
    for pair in sorted_pair_list:
        if cells_bank[pair["id"]]["target_index"] is None and cells_target[pair["target_index"]]["id"] is None:
            cells_bank[pair["id"]]["target_index"] = pair["target_index"]
            cells_target[pair["target_index"]]["id"] = pair["id"]

    # 2.4
    for cell_id, cell_bank in cells_bank.items():
        if cell_bank["status"] == 1 and cell_bank["target_index"] is None:
            cells_bank[cell_id]["status"] = 0
            statistics["disappear_cell"] += 1

    #
    for index in range(len(cells_target)):
        cell_target = cells_target[index]
        if cell_target["id"] is not None: continue

        # 2.5
        # find the cell which was disappeared and appear again
        reappear_cell = None
        max_similarity = similarity_reappear
        for cell_id, cell_bank in cells_bank.items():
            if cell_bank["status"] != 0: continue
            if cell_bank["disappear_time"] > permit_disappear_time: continue
            similarity = get_two_cells_similarity(cell_bank, cell_target, 0.5)
            if similarity > max_similarity:
                max_similarity = similarity
                reappear_cell = cell_bank
        if reappear_cell is not None:
            reappear_cell["status"] = 1
            reappear_cell["target_index"] = index
            cell_target["id"] = reappear_cell["id"]
            statistics["reappear"] += 1
            continue

        # 2.6
        # find mitosis cell
        max_similarity = None
        mother_cell = None
        for cell_id, cell_bank in cells_bank.items():
            if cell_bank["status"] != 1: continue
            if euclidean_distance(cell_target["center"], cell_bank["center"]) > d_distance: continue
            if cell_target["area"] > cell_bank["area"] * rate_mitosis: continue
            if cells_target[cell_bank["target_index"]]["area"] > cell_bank["area"] * rate_mitosis: continue
            similarity = get_two_cells_similarity(cell_target, cell_bank)
            if max_similarity is None or similarity > max_similarity:
                max_similarity = similarity
                mother_cell = cell_bank
        if mother_cell is not None:
            mother_cell["status"] = 2
            cell_target["status"] = 3
            statistics["dividing_times"] += 1
            statistics["total_dividing_times"] += 1
            continue

    for cell_id, cell_bank in cells_bank.items():
        if cell_bank["status"] == 2:
            cells_target[cell_bank["target_index"]]["id"] = None
            cells_target[cell_bank["target_index"]]["status"] = 3
            cell_bank["target_index"] = None
            statistics["dividing_cells"] += 2
            statistics["total_dividing_cells"] += 2
        if cell_bank["status"] == 0:
            cell_bank["disappear_time"] += 1


# step 3: label each cell state and save the result
def mark_and_save(this_statistics, cells_bank, frame_name, img, cells, contours):
    cell_count = len(cells)
    displacement_total, area_total = 0, 0
    displacement_count, area_count = 0, 0
    displacement_average, area_average = 0, 0
    img_height, img_width, _ = img.shape
    in_mitosis = 0

    for i in range(0, cell_count):
        cell = cells[i]
        cell_bank = cells_bank.get(cell["id"])

        # calculate total area
        if cell["center"][0] - cell["size"][0] > 0 and cell["center"][0] + cell["size"][0] < img_width \
                and cell["center"][1] - cell["size"][1] > 0 and cell["center"][1] + cell["size"][1] < img_height:
            area_total += cell["area"]
            area_count += 1

        # draw the cell's contour
        cv2.drawContours(img, contours[i], -1, cell_bank["color"], 2)
        # draw the cell's trajectory
        trajectory = cell_bank["trajectory"]
        if len(trajectory) > 1:
            steps = len(trajectory) - 1
            for p in range(1, len(trajectory)):
                alpha = 0.95 ** (steps - p)
                if alpha < 0.2: alpha = 0.2
                b, g, r = cell_bank["color"]
                cv2.line(img, trajectory[p - 1], trajectory[p], (b * alpha, g * alpha, r * alpha), 2)
            # calculate total displacement
            displacement_total += euclidean_distance(trajectory[-1], trajectory[-2])
            displacement_count += 1

    for i in range(0, cell_count):
        cell = cells[i]
        cell_bank = cells_bank.get(cell["id"])
        # draw the mark of cell in mitosis
        if cell_bank["status"] == 2:
            cv2.circle(img, cell_bank["center"], mitosis_radius, mitosis_color, mitosis_width)
            in_mitosis += 1

        if cell_bank["status"] == 3:
            cv2.circle(img, cell_bank["center"], child_radius, child_color, child_width)
        # draw the id of the cell
        # text_position = (
        #     int(cell_bank["center"][0] + cell_bank["size"][0] / 2 + 2),
        #     int(cell_bank["center"][1] + cell_bank["size"][1] / 2))
        # if cell_bank["status"] == 2:
        #     cv2.putText(img, str(cell_bank["id"]), text_position, cv2.FONT_HERSHEY_SIMPLEX,
        #                 font_size, font_color_mitosis, font_thickness)
        # else:
        #     cv2.putText(img, str(cell_bank["id"]), text_position, cv2.FONT_HERSHEY_SIMPLEX,
        #                 font_size, font_color, font_thickness)

    # show statistical information
    if area_count > 0:
        area_average = round(area_total / area_count, 2)
    if displacement_count > 0:
        displacement_average = round(displacement_total / displacement_count, 2)

    print()
    print(f"frame name: {frame_name}")
    print(f"cell count: {cell_count}")
    print(f"average size: {area_average} of {area_count} cells")
    print(f"average displacement: {displacement_average} of {displacement_count} cells")
    print(f'dividing times: {in_mitosis}, total {this_statistics["total_dividing_times"] + in_mitosis}')
    print(f'dividing cells count: {this_statistics["dividing_cells"]}, total {this_statistics["total_dividing_cells"]}')
    print(
        f'new/reappear/disappear cells: {this_statistics["new_cell"]}/{this_statistics["reappear"]}/{this_statistics["disappear_cell"]}')

    line_padding = 24
    line_count = 6
    pos_x = 5
    pos_y = int(img_height - line_padding * (line_count - 0.5))
    cv2.putText(img, f"frame name: {frame_name}", (pos_x, pos_y + line_padding * 0), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
    cv2.putText(img, f"cell count: {cell_count}", (pos_x, pos_y + line_padding * 1), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
    cv2.putText(img, f"average size: {area_average} of {area_count} cells", (pos_x, pos_y + line_padding * 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
    cv2.putText(img, f"average displacement: {displacement_average} of {displacement_count} cells",
                (pos_x, pos_y + line_padding * 3), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
    cv2.putText(img,
                f'dividing cells count: {this_statistics["dividing_cells"]}, total {this_statistics["total_dividing_cells"]}',
                (pos_x, pos_y + line_padding * 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)
    cv2.putText(img,
                f'new/reappear/disappear cells: {this_statistics["new_cell"]}/{this_statistics["reappear"]}/{this_statistics["disappear_cell"]}',
                (pos_x, pos_y + line_padding * 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, font_color, font_thickness)

    for cell_id, cell_bank in cells_bank.items():
        if cell_bank["status"] == 3:
            cell_bank["status"] = 1

    folder = os.path.exists(f"{RESULT_DIR}/{SUB_DIR}")
    # Check whether a folder exists. If not, create a folder
    if not folder:
        os.makedirs(f"{RESULT_DIR}/{SUB_DIR}")
    cv2.imwrite(f'{RESULT_DIR}/{SUB_DIR}/{frame_name}', img.astype(np.uint8))


# step 4: Update the new cells from the target layer to the cell bank
# step 5: Update the movement path, position, size of existing cells in the cell bank according to the target layer
def refresh_to_bank(cells_bank, cells):
    for index in range(len(cells)):
        cell = cells[index]
        if cell["id"] is None:
            id = len(cells_bank) + 1
            cell["id"] = id
            new_cell = create_cell()
            new_cell["id"] = id
            new_cell["center"] = cell["center"]
            new_cell["size"] = cell["size"]
            new_cell["area"] = cell["area"]
            new_cell["color"] = get_a_random_color()
            new_cell["status"] = cell["status"]
            new_cell["trajectory"] = [cell["center"]]

            cells_bank[id] = new_cell
            statistics["new_cell"] += 1
        else:
            id = cell["id"]
            cells_bank[id]["center"] = cell["center"]
            cells_bank[id]["size"] = cell["size"]
            cells_bank[id]["area"] = cell["area"]
            cells_bank[id]["trajectory"].append(cell["center"])

    for cell_id, cell_bank in cells_bank.items():
        if cell_bank["status"] == 2:
            cell_bank["status"] = 0


def main():
    # the follows params are used in the algorithm
    cells_bank = {}
    last_statistics = None

    cells_preface, cells_target = [], []
    img_preface, img_target = None, None
    frame_name_preface = None
    contours_preface = None

    # open all images from source path
    tiff_list = sorted(os.listdir(f'{SOURCE_PATH}/{SUB_DIR}'))

    # step 7: If the next sample graph exists, return to Step 1
    # start to traverse all tiff images
    for tiff_name in tiff_list:
        new_round_statistics()
        img = cv2.imread(f'{SOURCE_PATH}/{SUB_DIR}/{tiff_name}', 0)
        frame_name = tiff_name
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_target, cells_target, contours_target = analysis_frame(img, frame_name)
        compare_with_bank(cells_bank, cells_target)
        if img_preface is not None:
            mark_and_save(last_statistics, cells_bank, frame_name_preface, img_preface, cells_preface, contours_preface)
        refresh_to_bank(cells_bank, cells_target)
        # step 6: Use the target layer as the pre-ordered layer
        frame_name_preface, img_preface, cells_preface, contours_preface = frame_name, img_target, cells_target, contours_target
        last_statistics = statistics.copy()

    # step 8: Save the result again
    mark_and_save(last_statistics, cells_bank, frame_name_preface, img_preface, cells_preface, contours_preface)


if __name__ == "__main__":
    main()

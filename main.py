#!/usr/bin/env python3

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import os
import argparse

MAX_RATIO = 3

WORK_DIM = 1000
DEBUG_DIM = 500
PADDING = 30


def drawPoints(img, points, color=(0, 0, 255), radius=3):
    for p in points:
        cv2.circle(img, tuple(np.intp(p)), radius, color, -1)

    return img


def drawPointsWarped(img, points, color=(0, 0, 255), radius=3):
    for p in points:
        cv2.circle(img, tuple(np.intp(p * DEBUG_DIM) + PADDING), radius, color, -1)

    return img


def drawLineWarped(img, p1, p2, color=(255, 0, 0), thickness=1):
    p1 = np.array(p1) * DEBUG_DIM + PADDING
    p2 = np.array(p2) * DEBUG_DIM + PADDING

    cv2.line(img, tuple(np.intp(p1)), tuple(np.intp(p2)), color, thickness)

    return img


def findQuad(points):
    cvx = cv2.convexHull(points).reshape(-1, 2)
    quad = convertConvexHullToQuad(cvx)

    dists = [
        (i, np.min(
            [np.linalg.norm(p - p0) for p0 in cvx]
        ))
        for i, p in enumerate(quad)
    ]
    dists.sort(key=lambda x: x[1])

    v1, v2 = dists[:2]
    i1, i2 = v1[0], v2[0]

    if i1 == (i2 + 1) % 4:
        i1, i2 = i2, i1

    quad = np.float32([
        quad[(i + i1) % 4] for i in range(4)
    ])

    return quad


def findPoints(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = 255 - thresh

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    points = []

    center = np.float32([img.shape[1] // 2, img.shape[0] // 2])

    for i in range(1, num_labels):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if max(w, h) / min(w, h) > MAX_RATIO:
            continue

        if max(w, h) > max(img.shape[:2]) * 0.2:
            continue

        point = np.intp(centroids[i])

        points.append(point)

    i0, p0 = min(enumerate(points), key=lambda x: np.linalg.norm(center - x[1]))
    p1 = min(points, key=lambda x: np.linalg.norm(p0 - x) if np.linalg.norm(p0 - x) > 1 else float('inf'))
    delta = np.linalg.norm(p0 - p1)

    cluster = DBSCAN(eps=delta * 4, min_samples=1).fit(np.float32(points))
    colors = [
        (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        for _ in range(len(set(cluster.labels_)))
    ]

    debug_img = img.copy()
    for i, p in enumerate(points):
        cv2.circle(debug_img, tuple(np.intp(p)), 3, colors[cluster.labels_[i]], -1)

    points = [p for (i, p) in enumerate(points) if cluster.labels_[i] == cluster.labels_[i0]]

    return np.float32(points), debug_img, thresh


def drawGrid(img, x_cnt, y_cnt, color=(255, 0, 0)):
    dx = 1.0 / (x_cnt - 1)
    dy = 1.0 / (y_cnt - 1)

    hx = dx * 0.5
    hy = dy * 0.5

    for i in range(0, x_cnt + 1):
        x = (i - 0.5) * dx
        drawLineWarped(img, (x, -hy), (x, 1 + hy))

    for i in range(0, y_cnt + 1):
        y = (i - 0.5) * dy
        drawLineWarped(img, (-hx, y), (1 + hx, y))


def getConsecutevePoints(points, i, cnt):
    n = len(points)
    return [points[(i + j) % n] for j in range(cnt)]


def getAngleBetweenVectors(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos, -1, 1))


def getIntersection(p0, p1, p2, p3):
    A = np.array(
        [[p1[0] - p0[0], p2[0] - p3[0]],
         [p1[1] - p0[1], p2[1] - p3[1]]]
    )
    b = np.array([p2[0] - p0[0], p2[1] - p0[1]])

    x = np.linalg.solve(A, b)
    return p0 + (p1 - p0) * x[0]


def convertConvexHullToQuad(convexHull):
    convexHull = list(convexHull)

    while len(convexHull) > 4:
        print('kek')
        del_i = 0
        del_area = float('inf')

        for i in range(len(convexHull)):
            p0, p1, p2, p3 = getConsecutevePoints(convexHull, i, 4)

            try:
                intersection = getIntersection(p0, p1, p2, p3)
                area = np.linalg.norm(np.cross(p1 - intersection, p2 - intersection))

                if area < del_area:
                    del_i = i
                    del_area = area
            except:
                pass

        p0, p1, p2, p3 = getConsecutevePoints(convexHull, del_i, 4)
        mid = getIntersection(p0, p1, p2, p3)

        newConvex = []

        n = len(convexHull)
        for i, p in enumerate(convexHull):
            if i == (del_i + 1) % n:
                newConvex.append(mid)
                continue
            if i == (del_i + 2) % n:
                continue
            newConvex.append(p)

        convexHull = newConvex

    return np.float32(convexHull)


def cluster(values):
    cluster = DBSCAN(eps=0.01, min_samples=1).fit(np.float32(values).reshape(-1, 1))
    cnt = len(set(cluster.labels_))

    return cnt


def findGridCount(points):
    xs = sorted(points[:, 0])
    ys = sorted(points[:, 1])

    return cluster(xs), cluster(ys)


def getBinary(points_int, x_cnt, y_cnt):
    pts = set(map(tuple, points_int))  # Convert points to a set once
    result = []
    result_checker = []

    for y in range(y_cnt):
        row = []
        checker_row = []
        for x in range(x_cnt):
            symbol = "1" if (x, y) in pts else "0"
            row.append(symbol)
            if (x + y) % 2 == 0:
                checker_row.append(symbol)
        result.append("".join(row) + "\n")
        result_checker.append("".join(checker_row) + "\n")

    return ''.join(result), ''.join(result_checker)


def run(filename, output):
    print(f"Processing {filename}")

    base_name = os.path.basename(os.path.splitext(filename)[0])

    img = cv2.imread(filename)
    scale = WORK_DIM / max(img.shape[:2])
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    points, points_img, thresh_img = findPoints(img)
    quad = findQuad(points)

    target_points = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    transform = cv2.getPerspectiveTransform(quad, target_points)
    transform_img = cv2.getPerspectiveTransform(quad, target_points * DEBUG_DIM + np.float32([PADDING, PADDING]))

    points_nrm = cv2.perspectiveTransform(np.float32([points]), transform)[0]

    img_warped = cv2.warpPerspective(img, transform_img, (DEBUG_DIM + 2 * PADDING, DEBUG_DIM + 2 * PADDING))

    x_cnt, y_cnt = findGridCount(points_nrm)

    print(f"N: {len(points)}")
    print(f"x, y: {x_cnt}, {y_cnt}")
    drawGrid(img_warped, x_cnt, y_cnt)
    drawPointsWarped(img_warped, points_nrm)

    points_int = np.intp(np.round(points_nrm * np.float32([x_cnt - 1, y_cnt - 1])))
    binary, binary_checker = getBinary(points_int, x_cnt, y_cnt)

    print("AFTER")

    os.makedirs(f"{output}/{base_name}", exist_ok=True)

    with open(f"{output}/{base_name}/grid.txt", "w") as f:
        f.write(binary)

    with open(f"{output}/{base_name}/grid_checker.txt", "w") as f:
        f.write(binary_checker)

    print("DEBUG", output, base_name)
    cv2.imwrite(f"{output}/{base_name}/warped.png", img_warped)
    cv2.imwrite(f"{output}/{base_name}/points.png", points_img)
    cv2.imwrite(f"{output}/{base_name}/thresh.png", thresh_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="filename of the image")
    parser.add_argument("-o", "--output", help="output filename", default="output")
    args = parser.parse_args()

    targets = []

    if os.path.isfile(args.file):
        targets.append(args.file)
    elif os.path.isdir(args.file):
        targets = [
            f"{args.file}/{f}"
            for f in os.listdir(args.file)
        ]

    try:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
    except:
        print("Error creating output directory")
        return

    for target in targets:
        run(target, args.output)


if __name__ == "__main__":
    main()
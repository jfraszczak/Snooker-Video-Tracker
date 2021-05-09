import cv2
import numpy as np
from time import time

from Ball import Ball
from balls_visualization import *

# PRZETWORZENIE OBRAZU METODAMI MORFOLOGICZNYMI
# ------------------------------------------------------------------------


def apply_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask = mask > 0
    masked_frame = np.zeros_like(frame, np.uint8)
    masked_frame[mask] = frame[mask]

    return masked_frame


# Wykonanie gaussian blura, zamiana do modelu hsv aby łatwiej odfiltorwać interesujące obszary, binaryzacja obrazu
def process_frame(frame: np.ndarray, colors_range: list[tuple]) -> np.ndarray:
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 1.41)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, colors_range[0], colors_range[1])

    return mask


# Wyciągnięcie z obrazu tylko kul poprzez odfiltrowanie zielonego koloru
def process_frame_balls_extraction(frame: np.ndarray) -> np.ndarray:
    colors_range = [(36, 100, 100), (65, 255, 255)]
    img_binary = process_frame(frame, colors_range)

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(img_binary, kernel)

    return image


# Wyciągnięcie z obrazu tylko dziur poprzez odfiltorwanie obszarów, które nie są białe
def process_frame_holes_extraction(frame: np.ndarray) -> np.ndarray:
    colors_range = [(0, 0, 180), (255, 100, 255)]
    img_binary = process_frame(frame, colors_range)

    kernel = np.ones((9, 9), np.uint8)
    image = cv2.erode(img_binary, kernel)

    return image


# FUNKCJE POMOCNICZE
# ------------------------------------------------------------------------

def show_balls(balls: list):
    for ball in balls:
        ball.show()


# Wyciągnięcie koordynatów dziur
def find_holes_coordinates(frame: np.ndarray) -> list[tuple]:
    processed_frame = process_frame_holes_extraction(frame)
    cnts = cv2.findContours(processed_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    centers = []
    for cnt in cnts:
        # compute the center of the contour
        if cv2.contourArea(cnt) > 0:
            M = cv2.moments(cnt)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            centers.append((x, y))

    return centers


# Kolejne 6 funkcji służy do określenia kolorów bili i im ich przypisanie
# Podaje na wejście zbinaryzowany obraz, który zawiera tylko bile i dla kazdej bili tym algorytmem rozrostu z labków
# zbieram wszystkie piksele danej bili, zamieniam do hsv  i ibliczam ich średnie żeby potem przyporządkować konkretny
# kolor, ale strasznie to wolne i musze coś alternatywnego wymyśleć


def find_neighbours(img: np.ndarray, regions: np.ndarray, y: int, x: int) -> list[tuple]:
    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    c_neighbours = []
    for dy, dx in neighbours:
        ny, nx = y + dy, x + dx

        if ny < 0 or ny >= img.shape[0] or nx < 0 or nx >= img.shape[1]:
            continue

        if regions[ny, nx] > 0:
            continue

        if img[ny, nx] > 0:
            continue

        c_neighbours.append((ny, nx))

    return c_neighbours


def grow_mask(img: np.ndarray, regions: np.ndarray, y: int, x: int, cls: int) -> np.ndarray:
    regions[y, x] = cls

    c_neighbours = find_neighbours(img, regions, y, x)
    for ny, nx in c_neighbours:
        regions[ny, nx] = cls

    while len(c_neighbours) > 0:
        new_neighbours = []
        for ny, nx in c_neighbours:
            i_new_neighbours = find_neighbours(img, regions, ny, nx)
            for _ny, _nz in i_new_neighbours:
                regions[_ny, _nz] = cls

            new_neighbours.extend(i_new_neighbours)

        c_neighbours = new_neighbours

    return regions


def extract_ball_mask(img: np.ndarray, x: int, y: int) -> np.ndarray:
    mask = np.zeros(img.shape)
    mask = grow_mask(img, mask, y, x, 1)
    return mask


def hsv_to_color(hsv: tuple) -> str:
    h, s, v = hsv

    if v < 0.3:
        return 'black'
    if s < 0.3 and v > 0.65:
        return 'white'
    if (h < 20 or h > 351) and s > 0.7 and v > 0.3:
        return 'red'
    if 28 < h < 64 and s > 0.15 and v > 0.3:
        return 'yellow'
    if 64 < h < 80 and s > 0.15 and v > 0.3:
        return 'green'
    if 80 < h < 255 and s > 0.15 and v > 0.3:
        return 'blue'
    if 20 < h < 29 and s > 0.15 and 0.3 < v < 0.75:
        return 'brown'
    else:
        return 'pink'


def set_colors(balls: list[Ball], processed_frame: np.ndarray, frame: np.ndarray) -> list[Ball]:
    for ball in balls:
        if ball.scored:
            continue

        mask = extract_ball_mask(processed_frame, int(ball.x), int(ball.y))
        masked = apply_mask(frame, mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        h = np.asarray(hsv[:, :, 0]).reshape(-1)
        s = np.asarray(hsv[:, :, 1]).reshape(-1)
        v = np.asarray(hsv[:, :, 2]).reshape(-1)

        h_mean = np.true_divide(h.sum(), (h != 0).sum())
        s_mean = np.true_divide(s.sum(), (s != 0).sum()) / 255
        v_mean = np.true_divide(v.sum(), (v != 0).sum()) / 255

        hsv = (h_mean, s_mean, v_mean)
        color = hsv_to_color(hsv)
        ball.set_color(color)

    return balls


def colors_of_keypoints(keypoints: list[cv2.KeyPoint], frame: np.ndarray, processed_frame: np.ndarray) -> list[(tuple, str)]:
    keypoints_to_colors = []
    for keypoint in keypoints:
        x, y = keypoint.pt[0], keypoint.pt[1]

        mask = extract_ball_mask(processed_frame, int(x), int(y))

        masked = frame[mask > 0]
        masked = masked[:40]
        masked = masked.reshape(min(40,masked.shape[0]), 1, 3)

        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
        h = np.asarray(hsv[:, :, 0]).reshape(-1)
        s = np.asarray(hsv[:, :, 1]).reshape(-1)
        v = np.asarray(hsv[:, :, 2]).reshape(-1)

        h_mean = np.mean(h)
        s_mean = np.mean(s) / 255
        v_mean = np.mean(v) / 255

        hsv = (h_mean, s_mean, v_mean)
        color = hsv_to_color(hsv)

        keypoints_to_colors.append(((x, y), color))

    return keypoints_to_colors


# LOGIKA DOTYCZĄCA SAMYCH BILI
# ------------------------------------------------------------------------


def initialize_balls(keypoints: list[cv2.KeyPoint]) -> list[Ball]:
    balls = []
    for keypoint in keypoints:
        x = keypoint.pt[0]
        y = keypoint.pt[1]
        ball = Ball(x, y)
        balls.append(ball)

    return balls

# Pozycję każdej bili porównuję z znalezionymi nowymi pozycjami rozpoznanych bil, jeśli pozycja jest taka sama to uznaję
# że bila się nie poruszyła i ten rozpoznany fragment na obrazie zakłądam, że jest tą bilą, jeśli jakaś bila się poruszyła
# to dopasowuję pozycje bazując na rozpoznanym kolorze i na końcu jeśli zostaje mi tylko 1 bila bez pozycji to
# przydzielam jej ostatnia nieprzydzieloną lokalizację


def update_balls(balls: list[Ball], keypoints: list[cv2.KeyPoint], frame: np.ndarray, processed_frame: np.ndarray):
    balls_to_update = []
    unknown_keypoints = keypoints[:]

    for ball in balls:
        if ball.scored:
            ball.update()
            continue

        same_position = False
        for keypoint in keypoints:
            x, y = keypoint.pt[0], keypoint.pt[1]

            if ball.same_position(x, y):
                ball.update(x, y)
                same_position = True
                unknown_keypoints.remove(keypoint)
                break
        if not same_position:
            balls_to_update.append(ball)

    keypoints_to_colors = colors_of_keypoints(unknown_keypoints, frame, processed_frame)
    match_balls_with_remaining_keypoints(balls, balls_to_update, keypoints_to_colors)


def match_balls_with_remaining_keypoints(balls: list[Ball], balls_to_update: list[Ball], keypoints: list[cv2.KeyPoint]):
    remaining_balls = []
    for ball in balls_to_update:
        if ball.scored:
            continue

        found_corresponding_keypoint = False
        for keypoint in keypoints:
            x, y = keypoint[0]
            color = keypoint[1]

            if ball.color == color:
                ball.update(x, y)
                keypoints.remove(keypoint)
                found_corresponding_keypoint = True

        if not found_corresponding_keypoint:
            remaining_balls.append(ball)

    if len(remaining_balls) == 1 and len(keypoints) == 1:
        ball = remaining_balls[0]
        keypoint = keypoints[0]
        x, y = keypoint[0]
        ball.update(x, y)

    for ball in remaining_balls:
        ball.update()

    if len(keypoints) == 1:
        for ball in balls:
            if ball.scored and ball.color != 'red':
                ball.show()
                x, y = keypoints[0][0]
                ball.update(x, y)
                break


def detect_scores(balls: list[Ball], holes_coordinates: list[tuple], frame: np.ndarray) -> (np.ndarray, list[Ball]):
    for ball in balls:
        coord = ball.detect_score(holes_coordinates)
        if coord != ():
            frame = display_score_event(frame, coord, ball)

    return frame, balls


def balls_movement(balls: list[Ball]) -> list[bool]:
    movements = []
    for ball in balls:
        movements.append(ball.moving)
    return movements


def detect_collision(balls: list[Ball], movement_array) -> list[bool]:
    new_movement_array = []
    number_of_moving_balls = np.sum(np.array(movement_array))
    for i in range(len(balls)):
        ball = balls[i]
        new_movement_array.append(ball.moving)
        if ball.moving and not movement_array[i] and number_of_moving_balls > 1:
            print('\n\nCOLLISION')
            ball.show()
            ball.collided = True
    return new_movement_array


def process_video(video_name: str):
    cap = cv2.VideoCapture(video_name)
    first_frame = True
    while cap.isOpened():
        start = time()
        read_successful, frame = cap.read()
        if not read_successful: # end of clip
            break
        processed_frame = process_frame_balls_extraction(frame)
        img_with_keypoints, keypoints = draw_circles_around_balls(frame, processed_frame)

        if first_frame:
            balls = initialize_balls(keypoints)
            balls = set_colors(balls, processed_frame, frame)
            first_frame = False
            show_balls(balls)
            holes_coordinates = find_holes_coordinates(frame)
            movement_array = balls_movement

        img_with_keypoints = display_ball_data(img_with_keypoints, balls)
        img_with_holes = highlight_holes(img_with_keypoints, holes_coordinates)

        update_balls(balls, keypoints, frame, processed_frame)
        movement_array = detect_collision(balls, movement_array)

        img_with_collision = display_collision_event(img_with_holes, balls)
        img_with_scores, balls = detect_scores(balls, holes_coordinates, img_with_collision)

        cv2.imshow('FRAME', img_with_scores)

        if cv2.waitKey(max(1, 30 - int((time() - start) * 1000))) == ord('q'):  # Introduce 1 milisecond delay. press q to exit.
            break
        print(int((time() - start) * 1000))


process_video("video_low.mp4")

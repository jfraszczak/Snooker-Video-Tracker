import cv2
import numpy as np

from Ball import Ball


def draw_circles_around_balls(frame: np.ndarray, processed_frame: np.ndarray) -> (np.ndarray, list[cv2.KeyPoint]):
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(processed_frame)
    img_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img_with_keypoints, keypoints


def highlight_holes(frame: np.ndarray, coordinates: list[tuple]) -> np.ndarray:
    for coord in coordinates:
        cv2.circle(frame, coord, 20, (255, 0, 0), thickness=2, lineType=8, shift=0)

    return frame


def display_score_event(frame: np.ndarray, ball: Ball) -> np.ndarray:
    if ball.iterations_after_score < 50:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    'GOAL: ' + ball.color,
                    (ball.score_coordinates[0] - 50, ball.score_coordinates[1]),
                    font, 0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)

    return frame


def display_collision_event(frame: np.ndarray, balls: list[Ball]) -> np.ndarray:
    for ball in balls:
        if ball.collided:
            if ball.iterations_after_collision < 50:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,
                            'COLLISION',
                            ball.collision_coordinates,
                            font, 0.5,
                            (255, 0, 0),
                            1,
                            cv2.LINE_4)

    return frame


def quadratic_equation_roots(a: float, b: float, c: float) -> float:
    delta = b * b - 4 * a * c
    sqrt_delta = abs(delta) ** 0.5

    if delta > 0:
        x1 = (-b + sqrt_delta) / (2 * a)
        x2 = (-b - sqrt_delta) / (2 * a)

        if x1 > 0:
            return x1
        return x2

    elif delta == 0:
        x1 = -b / (2 * a)
        return x1


def display_ball_data(frame: np.ndarray, balls: list[Ball]) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX

    for ball in balls:
        if ball.known_position():
            x, y = int(ball.x), int(ball.y)
            text = ball.color + ' ' + ('moving' if ball.moving else '')

            cv2.putText(frame,
                        text,
                        (x, y),
                        font, 0.4,
                        (0, 255, 255),
                        1,
                        cv2.LINE_4)

            if ball.previous_trajectory_a is not None and ball.moving:
                x1, y1 = int(ball.previous_x), int(ball.previous_x * ball.previous_trajectory_a + ball.previous_trajectory_b)
                L = 100
                a = ball.previous_trajectory_a ** 2 + 1
                b = 0
                c = - L ** 2
                if ball.moving_forward:
                    x2 = x1 + int(quadratic_equation_roots(a, b, c))
                else:
                    x2 = x1 - int(quadratic_equation_roots(a, b, c))
                y2 = int(x2 * ball.previous_trajectory_a + ball.previous_trajectory_b)

                cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

    return frame
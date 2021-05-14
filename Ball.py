
def distance_between_point_and_line(point_coordinates: tuple, line_coefficients: tuple) -> float:
    x, y = point_coordinates
    a, b = line_coefficients

    A, B, C = a, -1, b
    distance = abs(A * x + B * y + C) / ((A ** 2 + B ** 2) ** 0.5)

    return distance


def distance(point_a: tuple, point_b: tuple) -> float:
    x1, y1 = point_a
    x2, y2 = point_b
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


class Ball:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.previous_x = x
        self.previous_y = y

        self.color = None
        self.moving = False
        self.moving_forward = None
        self.scored = False

        self.trajectory_a = None
        self.trajectory_b = None
        self.previous_trajectory_a = None
        self.previous_trajectory_b = None

        self.iterations_without_moving = 0
        self.iterations_being_invisible = 0
        self.iterations_after_score = 0
        self.iterations_back_in_game = 0

        self.collided = False
        self.iterations_after_collision = 0

        self.score_coordinates = ()
        self.collision_coordinates = ()

    def set_color(self, color: str):
        self.color = color

    def show(self):
        print('-------------------------------------------------------------------------')
        print('COLOR:', self.color)
        print('POSITION:', (self.x, self.y))
        print('PREVIOUS POSITION:', (self.previous_x, self.previous_y))
        print('TRAJECTORY: a = {}, b = {}'.format(self.trajectory_a, self.trajectory_b))
        print('PREVIOUS TRAJECTORY: a = {}, b = {}'.format(self.previous_trajectory_a, self.previous_trajectory_b))
        print('INVISIBLE:', self.iterations_being_invisible)
        print('SCORED:', self.scored)
        print('SINCE SCORING:', self.iterations_after_score)
        print('SCORE COORDINATES:', self.score_coordinates)
        print('COLLISION COORDINATES:', self.collision_coordinates)

    def same_position(self, x: float, y: float) -> bool:
        if self.x is None and self.y is None:
            return False
        return ((self.x - x) ** 2 + (self.y - y) ** 2) ** 0.5 < 2

    def moved(self) -> bool:
        if self.x is None and self.y is None:
            return True
        return not self.same_position(self.previous_x, self.previous_y)

    def update_trajectory(self):
        if self.trajectory_a is not None and self.trajectory_b is not None:
            self.previous_trajectory_a = self.trajectory_a
            self.previous_trajectory_b = self.trajectory_b

        if not (self.x is None or self.y is None) and not (self.previous_x is None or self.previous_y is None) and self.x != self.previous_x:
            self.trajectory_a = (self.y - self.previous_y) / (self.x - self.previous_x)
            self.trajectory_b = self.y - self.trajectory_a * self.x
            self.moving_forward = self.x > self.previous_x
        else:
            self.trajectory_a = None
            self.trajectory_b = None

    def update_position(self, new_x: float, new_y: float):
        if not (self.x is None and self.y is None):
            self.previous_x, self.previous_y = self.x, self.y

        if new_x is None and new_y is None:
            self.iterations_being_invisible += 1
        else:
            self.iterations_being_invisible = 0

        self.x, self.y = new_x, new_y

    def update_movement_data(self):
        if self.moved():
            self.moving = True
            self.iterations_without_moving = 0
            self.update_trajectory()
        else:
            self.iterations_without_moving += 1

    def stopped_moving(self, iterations_threshold=10):
        return self.iterations_without_moving >= iterations_threshold

    def stop_moving(self):
        self.moving = False
        self.moving_forward = None
        self.trajectory_a = None
        self.trajectory_b = None
        self.previous_trajectory_a = None
        self.previous_trajectory_b = None

    def is_invisible(self) -> bool:
        return self.x is None and self.y is None

    def back_in_game(self):
        self.scored = False
        self.previous_x, self.previous_y = self.x, self.y
        self.trajectory_a, self.trajectory_b = None, None
        self.previous_trajectory_a, self.previous_trajectory_b = None, None

        self.iterations_without_moving = 0
        self.iterations_being_invisible = 0
        self.iterations_after_score = 0
        self.iterations_back_in_game = 0

        self.moving = False
        self.moving_forward = None

        self.score_coordinates = ()

    def collide(self):
        self.collided = True
        self.collision_coordinates = (int(self.previous_x), int(self.previous_y))

    def update(self, new_x=None, new_y=None):
        self.update_position(new_x, new_y)
        self.update_movement_data()

        if self.stopped_moving():
            self.stop_moving()

        if self.scored:
            if self.iterations_after_score < 50:
                self.iterations_after_score += 1
            if not self.is_invisible():
                self.iterations_back_in_game += 1
                if self.iterations_back_in_game == 50:
                    self.back_in_game()
                    print('\n\nBACK IN GAME')
                    self.show()

        if self.collided:
            if self.iterations_after_collision < 50:
                self.iterations_after_collision += 1
            else:
                self.collided = False
                self.iterations_after_collision = 0
                self.collision_coordinates = ()

    def known_position(self) -> bool:
        return not (self.x is None and self.y is None)

    # jeśli bila nie była widoczna przez więcej niż 30 klatek to zostałą strzelona do dziury najbliższej ostatniej
    # prostej po której ta bila się poruszała
    def detect_score1(self, coordinates: list[tuple]) -> tuple:
        if self.iterations_being_invisible > 30:
            line_coefficients = None
            if self.previous_trajectory_a is not None and self.previous_trajectory_b is not None:
                line_coefficients = (self.previous_trajectory_a, self.previous_trajectory_b)
            elif self.trajectory_a is not None and self.trajectory_b is not None:
                line_coefficients = (self.trajectory_a, self.trajectory_b)

            if line_coefficients is not None:
                min_distance = distance_between_point_and_line(coordinates[0], line_coefficients)
                closest_coordinates = coordinates[0]
                for i in range(1, len(coordinates)):
                    distance = distance_between_point_and_line(coordinates[i], line_coefficients)
                    if distance < min_distance:
                        min_distance = distance
                        closest_coordinates = coordinates[i]

                    if not self.scored:
                        print('\n\nSCORE')
                        self.show()

                    self.scored = True

                return closest_coordinates
        return ()

    def detect_score(self, coordinates: list[tuple]) -> tuple:
        ball_coordinates = (self.previous_x, self.previous_y)
        if self.iterations_being_invisible > 30:
            min_distance = distance(ball_coordinates, coordinates[0])
            closest_coordinates = coordinates[0]

            for i in range(1, len(coordinates)):
                dist = distance(coordinates[i], ball_coordinates)
                if dist < min_distance:
                    min_distance = dist
                    closest_coordinates = coordinates[i]

            if not self.scored:
                print('\n\nSCORE')
                self.show()
            self.scored = True
            self.score_coordinates = closest_coordinates
            return

        self.score_coordinates = ()
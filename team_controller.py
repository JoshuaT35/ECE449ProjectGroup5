"""
ADD A PROPER DESCRIPTION LATER




-------

we need to also credit Scott Dick's work, as he stated in the email
-----



"""

from immutabledict import immutabledict
from kesslergame import KesslerController 
from typing import Dict, Tuple, Any, Type
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt


class TeamController(KesslerController):
    def __init__(self):
        super().__init__()  
        self.eval_frames = 0

        # --- variables ---
        # - max number of nearby asteroids. Theoretically no limit,
        #       this is just an arbitrarily high value to represent a high number
        #       of nearby asterouds in the membership function
        self.num_nearby_asteroids_ceiling = 40

        # - mine cooldown. When it is 0, can deploy a mine. otherwise, the value is reset
        self.mine_cooldown = 50

        # --- fuzzy systems ---
        self.ship_perimeter_fuzzy_system()
        self.ship_targeting_fuzzy_system()
        self.ship_thrust_fuzzy_system()


    def ship_targeting_fuzzy_system(self):
        # Initialize eval frames
        self.eval_frames = 0

        # Declare fuzzy variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')  # Time for bullet to reach intercept
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi / 30, math.pi / 30, 0.1), 'theta_delta')  # Angle to adjust (radians)
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Turn rate (degrees)
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')  # Fire decision
        distance_to_asteroid = ctrl.Antecedent(np.arange(0, 300, 1), "distance_to_asteroid")

        # Membership functions for distance_to_asteroid
        distance_to_asteroid["close"] = fuzz.trimf(distance_to_asteroid.universe, [0, 50, 100])
        distance_to_asteroid["medium"] = fuzz.trimf(distance_to_asteroid.universe, [50, 150, 250])
        distance_to_asteroid["far"] = fuzz.trimf(distance_to_asteroid.universe, [200, 300, 300])

        # Membership functions for bullet_time
        bullet_time['short'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['medium'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['long'] = fuzz.smf(bullet_time.universe, 0.1, 0.2)

        # Membership functions for theta_delta (turn adjustment needed)
        theta_delta['negative_large'] = fuzz.zmf(theta_delta.universe, -math.pi / 30, -math.pi / 60)
        theta_delta['negative_small'] = fuzz.trimf(theta_delta.universe, [-math.pi / 60, -math.pi / 90, 0])
        theta_delta['zero'] = fuzz.trimf(theta_delta.universe, [-math.pi / 90, 0, math.pi / 90])
        theta_delta['positive_small'] = fuzz.trimf(theta_delta.universe, [0, math.pi / 90, math.pi / 60])
        theta_delta['positive_large'] = fuzz.smf(theta_delta.universe, math.pi / 60, math.pi / 30)

        # Membership functions for ship_turn
        ship_turn['sharp_left'] = fuzz.trimf(ship_turn.universe, [-180, -180, -90])
        ship_turn['moderate_left'] = fuzz.trimf(ship_turn.universe, [-180, -90, -45])
        ship_turn['slight_left'] = fuzz.trimf(ship_turn.universe, [-90, -45, 0])
        ship_turn['slight_right'] = fuzz.trimf(ship_turn.universe, [0, 45, 90])
        ship_turn['moderate_right'] = fuzz.trimf(ship_turn.universe, [45, 90, 180])
        ship_turn['sharp_right'] = fuzz.trimf(ship_turn.universe, [90, 180, 180])

        # Membership functions for ship_fire
        ship_fire['no_fire'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0])
        ship_fire['fire'] = fuzz.trimf(ship_fire.universe, [0, 1, 1])

        # Fuzzy rules for ship_turn (independent of firing)
        rule_turn1 = ctrl.Rule(theta_delta['negative_large'], ship_turn['sharp_left'])
        rule_turn2 = ctrl.Rule(theta_delta['negative_small'], ship_turn['moderate_left'])
        rule_turn3 = ctrl.Rule(theta_delta['zero'], ship_turn['slight_left'])
        rule_turn4 = ctrl.Rule(theta_delta['positive_small'], ship_turn['slight_right'])
        rule_turn5 = ctrl.Rule(theta_delta['positive_large'], ship_turn['moderate_right'])

        # High-priority firing when alignment and bullet time are ideal
        rule_fire1 = ctrl.Rule(theta_delta['zero'] & bullet_time['short'], ship_fire['fire'])
        rule_fire2 = ctrl.Rule(theta_delta['zero'] & bullet_time['medium'], ship_fire['fire'])

        # Allow firing for minor misalignments if proximity is good
        rule_fire3 = ctrl.Rule((theta_delta['positive_small'] | theta_delta['negative_small']) & bullet_time['short'], ship_fire['fire'])
        rule_fire4 = ctrl.Rule((theta_delta['positive_small'] | theta_delta['negative_small']) & bullet_time['medium'], ship_fire['fire'])

        # Encourage firing at close targets regardless of alignment
        rule_fire5 = ctrl.Rule(distance_to_asteroid['close'], ship_fire['fire'])

        # Allow firing at medium-range targets if alignment is reasonable
        rule_fire6 = ctrl.Rule(distance_to_asteroid['medium'] & (theta_delta['zero'] | theta_delta['positive_small'] | theta_delta['negative_small']), ship_fire['fire'])

        # Fire at far targets if alignment and timing are good
        rule_fire7 = ctrl.Rule(distance_to_asteroid['far'] & theta_delta['zero'] & (bullet_time['short'] | bullet_time['medium']), ship_fire['fire'])

        # Suppress firing for poor conditions
        rule_fire8 = ctrl.Rule((bullet_time['long'] & (theta_delta['positive_large'] | theta_delta['negative_large'])) | (distance_to_asteroid['far'] & (theta_delta['positive_large'] | theta_delta['negative_large'])), ship_fire['no_fire'])

        # Allow firing for any medium or short bullet time, unless explicitly suppressed
        rule_fire9 = ctrl.Rule((bullet_time['short'] | bullet_time['medium']) & ~theta_delta['positive_large'] & ~theta_delta['negative_large'], ship_fire['fire'])

        # Create the fuzzy control system
        self.targeting_control = ctrl.ControlSystem([
            rule_turn1, rule_turn2, rule_turn3, rule_turn4, rule_turn5,  # Turning rules
            rule_fire1, rule_fire2, rule_fire3, rule_fire4, rule_fire5, rule_fire6, rule_fire7, rule_fire8, rule_fire9  # Firing rules
        ])


    def ship_thrust_fuzzy_system(self):
        # Inputs: Distance to asteroid, nearby asteroids, and theta_delta
        distance_to_asteroid = ctrl.Antecedent(np.arange(0, 1001, 1), "distance_to_asteroid")  # Expanded range
        nearby_asteroids = ctrl.Antecedent(np.arange(0, 101, 1), "nearby_asteroids")  # Expanded range
        theta_delta = ctrl.Antecedent(np.arange(-180, 181, 1), "theta_delta")  # Angle in degrees
        thrust = ctrl.Consequent(np.arange(-200, 201, 1), "thrust")  # Thrust universe [-200, 200]

        # Membership functions for distance_to_asteroid
        distance_to_asteroid["close"] = fuzz.trimf(distance_to_asteroid.universe, [0, 0, 150])
        distance_to_asteroid["medium"] = fuzz.trimf(distance_to_asteroid.universe, [50, 100, 150])
        distance_to_asteroid["far"] = fuzz.trimf(distance_to_asteroid.universe, [100, 1000, 1000])

        # Membership functions for nearby_asteroids (universe 0–100)
        nearby_asteroids["low"] = fuzz.trimf(nearby_asteroids.universe, [0, 10, 30])
        nearby_asteroids["medium"] = fuzz.trimf(nearby_asteroids.universe, [20, 50, 80])
        nearby_asteroids["high"] = fuzz.trimf(nearby_asteroids.universe, [70, 100, 100])

        # Membership functions for theta_delta (angle alignment)
        theta_delta["aligned"] = fuzz.trimf(theta_delta.universe, [-20, 0, 20])  # Head-on or slightly misaligned
        theta_delta["moderate"] = fuzz.trimf(theta_delta.universe, [-60, 0, 60])  # Moderate misalignment
        theta_delta["misaligned"] = fuzz.trimf(theta_delta.universe, [-180, -90, 90])  # Strong misalignment

        # Membership functions for thrust [-200 to 200]
        thrust["negative_high"] = fuzz.trimf(thrust.universe, [-200, -200, -100])  # Strong reverse
        thrust["negative_low"] = fuzz.trimf(thrust.universe, [-100, -50, 0])  # Slow reverse
        thrust["very_low"] = fuzz.trimf(thrust.universe, [-10, 0, 50])  # Gentle forward/reverse
        thrust["low"] = fuzz.trimf(thrust.universe, [0, 50, 100])  # Regular forward
        thrust["medium"] = fuzz.trimf(thrust.universe, [50, 100, 150])  # Moderate forward
        thrust["high"] = fuzz.trimf(thrust.universe, [100, 150, 200])  # Maximum forward

        # Updated rules for thrust

        # Negative thrust to move away from high asteroid density
        rule1 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["high"] & theta_delta["misaligned"], thrust["negative_high"])
        rule2 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["medium"] & theta_delta["misaligned"], thrust["negative_low"])

        # New rule: Slow down in close proximity even when aligned
        rule3 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["medium"] & theta_delta["aligned"], thrust["negative_low"])
        rule4 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["low"] & theta_delta["aligned"], thrust["very_low"])

        # Gradual reverse thrust to escape tight corners
        rule5 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["high"] & theta_delta["moderate"], thrust["negative_low"])
        rule6 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["medium"] & theta_delta["aligned"], thrust["low"])

        # Moderate thrust for medium proximity and alignment
        rule7 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["low"] & theta_delta["aligned"], thrust["medium"])

        # High thrust to close in on distant asteroids in low-density areas
        rule8 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["low"] & theta_delta["aligned"], thrust["medium"])
        rule9 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["medium"] & theta_delta["aligned"], thrust["medium"])

        # Suppress thrust if head-on and no escape required
        rule10 = ctrl.Rule(theta_delta["aligned"] & distance_to_asteroid["close"], thrust["very_low"])

        # Escaping large misalignment zones with controlled thrust
        rule11 = ctrl.Rule(theta_delta["misaligned"] & distance_to_asteroid["medium"], thrust["low"])

        # Control system
        self.thrust_control = ctrl.ControlSystem([
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11
        ])


    def ship_perimeter_fuzzy_system(self):
        # Antecedent: number of nearby asteroids
        nearby_asteroids = ctrl.Antecedent(np.linspace(0, 25, 25), "nearby_asteroids")

        # Antecedent: distance of closest asteroid
        closest_asteroid_distance = ctrl.Antecedent(np.linspace(0, 500, 10000), "distance_asteroid")

        # Antecedent: largest space between nearby asteroids
        largest_angle = ctrl.Antecedent(np.linspace(0, 180, 180), "largest_angle")

        # Consequent: flight (remain in place), roam, flee
        movement = ctrl.Consequent(np.linspace(0, 10, 20), "movement")

        # Membership functions for nearby_asteroids (scale of 0 - many asteroids (40 as ceiling))
        nearby_asteroids["little"] = fuzz.trimf(nearby_asteroids.universe, [0, 0, 6])
        nearby_asteroids["some"] = fuzz.trimf(nearby_asteroids.universe, [6, 11, 15])
        nearby_asteroids["many"] = fuzz.trimf(nearby_asteroids.universe, [15, self.num_nearby_asteroids_ceiling, self.num_nearby_asteroids_ceiling])

        # Membership functions for closest asteroid (scale of 0 - 500 pixels)
        closest_asteroid_distance["very_close"] = fuzz.trimf(closest_asteroid_distance.universe, [0, 0, 100])
        closest_asteroid_distance["close"] = fuzz.trimf(closest_asteroid_distance.universe, [90, 150, 210])
        closest_asteroid_distance["middle"] = fuzz.trimf(closest_asteroid_distance.universe, [200, 250, 300])
        closest_asteroid_distance["far"] = fuzz.trimf(closest_asteroid_distance.universe, [290, 350, 410])
        closest_asteroid_distance["very_far"] = fuzz.trimf(closest_asteroid_distance.universe, [400, 500, 500])

        # Membership functions for largest space between asteroids (scale of 0 - 180 degrees NOTE: math uses radians)
        largest_angle["below_60"] = fuzz.trimf(largest_angle.universe, [0, 0, 70])
        largest_angle["60_to_90"] = fuzz.trimf(largest_angle.universe, [60, 75, 90])
        largest_angle["above_90"] = fuzz.trimf(largest_angle.universe, [80, 130, 180])

        # Membership functions for running, fighting, roaming (trinary output, 0-2)
        movement["fight"] = fuzz.trimf(movement.universe, [0, 0, 1])
        movement["roam"] = fuzz.trimf(movement.universe, [1, 1.5, 2])
        movement["flee"] = fuzz.trimf(movement.universe, [2, 3, 3])

        # Define fuzzy rules

        # - little or some asteroids nearby
        # - far or very far
        # ---
        # - roam
        rule1 = ctrl.Rule(
            (nearby_asteroids["little"] | nearby_asteroids["some"])
            & (closest_asteroid_distance["far"] | closest_asteroid_distance["very_far"]),
            movement["roam"]
        )

        # - little asteroids nearby
        # - middle or close
        # ---
        # - attack
        rule2 = ctrl.Rule(
            nearby_asteroids["little"]
            & (closest_asteroid_distance["close"] | closest_asteroid_distance["middle"]),
            movement["fight"]
        )

        # - little asteroids nearby
        # - very close
        # ---
        # - flee
        rule3 = ctrl.Rule(
            nearby_asteroids["little"]
            & closest_asteroid_distance["very_close"],
            movement["flee"]
        )

        # - some asteroids nearby
        # - close/middle distance
        # - space to flee
        # ---
        # - flee
        rule4 = ctrl.Rule(
            nearby_asteroids["some"]
            & (closest_asteroid_distance["close"] | closest_asteroid_distance["middle"])
            & (largest_angle["60_to_90"] | largest_angle["above_90"]),
            movement["flee"]
        )

        # - some asteroids nearby
        # - very close distance
        # - no space to flee
        # ---
        # - fight
        rule5 = ctrl.Rule(
            nearby_asteroids["some"]
            & (closest_asteroid_distance["close"] | closest_asteroid_distance["middle"])
            & largest_angle["below_60"],
            movement["fight"]
        )

        # - many asteroids nearby
        # - space to flee
        # ---
        # - flee
        rule6 = ctrl.Rule(
            nearby_asteroids["many"]
            & (largest_angle["60_to_90"] | largest_angle["above_90"]),
            movement["flee"]
        )

        # - many asteroids nearby
        # - no space to flee
        # ---
        # - fight
        rule7 = ctrl.Rule(
            nearby_asteroids["many"]
            & (largest_angle["below_60"]),
            movement["fight"]
        )

        # create the control system
        self.ship_perimeter_situation = ctrl.ControlSystem()
        self.ship_perimeter_situation.addrule(rule1)
        self.ship_perimeter_situation.addrule(rule2)
        self.ship_perimeter_situation.addrule(rule3)
        self.ship_perimeter_situation.addrule(rule4)
        self.ship_perimeter_situation.addrule(rule5)
        self.ship_perimeter_situation.addrule(rule6)
        self.ship_perimeter_situation.addrule(rule7)
        

    def reset_mine_cooldown(self):
        self.mine_cooldown = 60

    # return list of nearby asteroids and their count
    def get_nearby_asteroids(self, ship_state: Dict, game_state: Dict, distance: int):
        # x and y coordinates of ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # default values
        list_asteroids = []
        num_asteroids = 0

        # loop through asteroids
        # find asteroid distance using a^2 + b^2 = c^2 right-triangle formula
        for asteroid in game_state["asteroids"]:
            # x and y coordinates of asteroid
            asteroid_pos_x = asteroid["position"][0]
            asteroid_pos_y = asteroid["position"][1]

            # get distance between asteroid and ship
            # a = ship_pos_x - asteroid_pos_x (doesn't matter if negative since we square)
            # b = ship_pos_y - asteroid_pos_y (doesn't matter if negative since we square)
            # c = curr_dist
            curr_dist = math.sqrt((ship_pos_x - asteroid_pos_x)**2 + (ship_pos_y - asteroid_pos_y)**2)

            # if current distance is in given distance, increase number by 1 and add asteroid to list
            if (curr_dist <= distance):
                list_asteroids.append(asteroid)
                num_asteroids += 1
        
        return list_asteroids, num_asteroids


    # return asteroid and its distance if it exists, otherwise None and -1
    def find_closest_asteroid(self, ship_state: Dict, game_state: Dict):
        # x and y coordinates of ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # default values
        asteroid_dist_list = []
        closest_asteroid = None
        closest_asteroid_dist = -1

        # loop through asteroids
        # find closest asteroid using a^2 + b^2 = c^2 right-triangle formula
        for asteroid in game_state["asteroids"]:
            # x and y coordinates of asteroid
            asteroid_pos_x = asteroid["position"][0]
            asteroid_pos_y = asteroid["position"][1]

            # get distance between asteroid and ship
            # a = ship_pos_x - asteroid_pos_x (doesn't matter if negative since we square)
            # b = ship_pos_y - asteroid_pos_y (doesn't matter if negative since we square)
            # c = curr_dist
            curr_dist = math.sqrt((ship_pos_x - asteroid_pos_x)**2 + (ship_pos_y - asteroid_pos_y)**2)

            # add to list
            asteroid_dist_list.append(curr_dist)

        # if our list is not empty (there were asteroids)
        # get the nearsest asteroid
        if (asteroid_dist_list != []):
            # get minimum distance
            closest_asteroid_dist = min(asteroid_dist_list)

            # get index of closest asteroid
            closest_asteroid_idx = asteroid_dist_list.index(closest_asteroid_dist)

            # get the asteroid at that distance
            closest_asteroid = game_state["asteroids"][closest_asteroid_idx]

        return closest_asteroid, closest_asteroid_dist


    # get bullet_t and shooting_theta for ship_targeting_fuzzy_system
    def get_bullet_t_shooting_theta(self, ship_state: Dict, game_state: Dict):
        # Get the closest asteroid
        closest_asteroid, closest_asteroid_dist = self.find_closest_asteroid(ship_state, game_state)

        # Extract x and y coordinates of the ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # Calculate the relative position between the ship and the asteroid
        asteroid_ship_x = closest_asteroid["position"][0] - ship_pos_x
        asteroid_ship_y = closest_asteroid["position"][1] - ship_pos_y

        # Angle from the ship to the asteroid
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)

        # Asteroid velocity components
        asteroid_velocity_x = closest_asteroid["velocity"][0]
        asteroid_velocity_y = closest_asteroid["velocity"][1]
        asteroid_velocity_magnitude = math.sqrt(asteroid_velocity_x**2 + asteroid_velocity_y**2)

        # Asteroid direction (angle of velocity vector)
        asteroid_direction = math.atan2(asteroid_velocity_y, asteroid_velocity_x)

        # Bullet speed (fixed for the game)
        bullet_speed = 800  # m/s

        # Solve for intercept using relative motion
        relative_angle = asteroid_ship_theta - asteroid_direction
        cos_relative_angle = math.cos(relative_angle)

        # Determinant of the quadratic equation (b^2 - 4ac)
        a = asteroid_velocity_magnitude**2 - bullet_speed**2
        b = -2 * closest_asteroid_dist * asteroid_velocity_magnitude * cos_relative_angle
        c = closest_asteroid_dist**2
        discriminant = b**2 - 4 * a * c

        # Handle edge cases where no valid solution exists
        if discriminant < 0:
            # No solution, aim directly at the asteroid's current position
            bullet_t = 0
            intercept_x = closest_asteroid["position"][0]
            intercept_y = closest_asteroid["position"][1]
        else:
            # Compute both roots of the quadratic equation
            sqrt_discriminant = math.sqrt(discriminant)
            t1 = (-b + sqrt_discriminant) / (2 * a)
            t2 = (-b - sqrt_discriminant) / (2 * a)

            # Choose the smaller positive time
            bullet_t = min(t for t in [t1, t2] if t >= 0)

            # Predict the intercept position of the asteroid
            intercept_x = closest_asteroid["position"][0] + asteroid_velocity_x * bullet_t
            intercept_y = closest_asteroid["position"][1] + asteroid_velocity_y * bullet_t

        # Calculate the firing angle needed to hit the intercept position
        firing_angle = math.atan2(intercept_y - ship_pos_y, intercept_x - ship_pos_x)

        # Calculate the angular difference (theta_delta) between the ship's heading and the firing angle
        ship_heading_rad = math.radians(ship_state["heading"])
        shooting_theta = firing_angle - ship_heading_rad

        # Wrap the angle to the range (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta


        # given 2 asteroids, find the angle created between the 2 asteroids and the ship, and the unit vector
    

    # given 2 asteroids, find the angle created between the 2 asteroids and the ship, and the unit vector inbetween
    def get_ship_2_asteroids_angle_direction(self, ship_state: Dict, asteroid1: Dict, asteroid2: Dict):
        # x and y coordinates of ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # x and y coordinates of asteroid1
        asteroid1_pos_x = asteroid1["position"][0]
        asteroid1_pos_y = asteroid1["position"][1]

        # x and y coordinates of asteroid
        asteroid2_pos_x = asteroid2["position"][0]
        asteroid2_pos_y = asteroid2["position"][1]

        # get magnitude of distance squared and magnitude of distance from ship to first asteroid
        dx_ship_to_a1 = asteroid1_pos_x - ship_pos_x
        dy_ship_to_a1 = asteroid1_pos_y - ship_pos_y
        d_ship_to_a1_square = math.pow(dx_ship_to_a1, 2) + math.pow(dy_ship_to_a1, 2)
        d_ship_to_a1 = math.sqrt(d_ship_to_a1_square)

        # get magnitude of distance squared and magnitude of distance from ship to second asteroid
        dx_ship_to_a2 = asteroid2_pos_x - ship_pos_x
        dy_ship_to_a2 = asteroid2_pos_y - ship_pos_y
        d_ship_to_a2_square = math.pow(dx_ship_to_a2, 2) + math.pow(dy_ship_to_a2, 2)
        d_ship_to_a2 = math.sqrt(d_ship_to_a2_square)

        # get magnitude of distance squared between the 2 asteroids
        dx_a1_a2 = asteroid1_pos_x - asteroid2_pos_x
        dy_a1_a2 = asteroid1_pos_y - asteroid2_pos_y
        d_a1_to_a2_square = math.pow(dx_a1_a2, 2) + math.pow(dy_a1_a2, 2)

        # law of cosines to find theta (the absolution of values means we do not have direction)
        # clamp value
        try:
            angle_theta_magnitude = math.acos(
                (d_ship_to_a1_square + d_ship_to_a2_square - d_a1_to_a2_square) / (2*d_ship_to_a1*d_ship_to_a2)
            )
        except:
            angle_theta_magnitude = 0

        # get distance unit vector v1 (ship to asteroid 1)
        # unit_v1 = magnitude(dx_ship_a1.i + dy_ship_a1.j)/magnitude
        # unit_v1_distance = tuple(dx_ship_to_a1/d_ship_to_a1, dy_ship_to_a1/d_ship_to_a1)

        # get distance unit vector v2 (ship to asteroid 2)
        # unit_v2 = magnitude(dx_ship_a2.i + dy_ship_a2.j)/magnitude
        # unit_v2_distance = tuple(dx_ship_to_a2/d_ship_to_a2, dy_ship_to_a2/d_ship_to_a2)

        # get distance unit vector v3 between v1 and v2
        unit_v3_x = (dx_ship_to_a1/d_ship_to_a1) + (dx_ship_to_a2/d_ship_to_a2)
        unit_v3_y = (dx_ship_to_a2/d_ship_to_a2) + (dx_ship_to_a2/d_ship_to_a2)
        unit_v3 = (unit_v3_x, unit_v3_y)

        return angle_theta_magnitude, unit_v3
    

    # given a list of asteroids, find the biggest theta between them
    def get_largest_ship_2_asteroids_angle_direction(self, ship_state: Dict, asteroids_list: list):
        max_theta_radians = 0
        best_unit_v3 = None

        # if at least 2 asteroids, find the best theta and unitv3 between them
        if len(asteroids_list) > 3:
            for asteroid1 in asteroids_list:
                for asteroid2 in asteroids_list:
                    theta, unit_v3 = self.get_ship_2_asteroids_angle_direction(ship_state, asteroid1, asteroid2)
                    if (theta > max_theta_radians):
                        max_theta_radians = theta
                        best_unit_v3 = unit_v3
        
        return max_theta_radians, best_unit_v3


    # what does the ship do every time
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:

        # --- take stock of perimeter ---
            # - number of asteroids nearby
        list_nearby_asteroids, num_nearby_asteroids = self.get_nearby_asteroids(ship_state, game_state, distance=500)

            # - distance of closest asteroid
        closest_asteroid, distance_to_asteroid = self.find_closest_asteroid(ship_state, game_state)

            # - if there are 2 or more asteroids, largest space is calculated, else 0
        if (num_nearby_asteroids > 2):
            # get the best theta and unitv3 pair
            theta_radians, unit_v3 = self.get_largest_ship_2_asteroids_angle_direction(ship_state, list_nearby_asteroids)
        else:
            theta_radians = 0
            unit_v3 = (0, 0)
        
        # -- from perimeter, choose what to do (fight, roam, flee) ---
        perimeter_simulation = ctrl.ControlSystemSimulation(self.ship_perimeter_situation, flush_after_run=1)

        # # pas inputs into it and receive recommendation
        perimeter_simulation.input["nearby_asteroids"] = num_nearby_asteroids
        perimeter_simulation.input["distance_asteroid"] = distance_to_asteroid
        perimeter_simulation.input["largest_angle"] = theta_radians
        perimeter_simulation.compute()

        # --- we need to fight (shoot or mine) ---
        # if perimeter_simulation.output["movement"] <= 1:
        # print(perimeter_simulation.output["movement"])

        # --- targeting fuzzy system ---
        # get bullet_t and shooting_theta inputs for ship_targeting_fuzzy_system, as well as distance to asteroid
        bullet_t, shooting_theta = self.get_bullet_t_shooting_theta(ship_state, game_state)

        # create control system simulation for ship_targeting_fuzzy_system
        # pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.input['distance_to_asteroid'] = distance_to_asteroid
        shooting.compute()

        # Get the defuzzified outputs for ship_targeting_fuzzy_system
        turn_rate = shooting.output['ship_turn']

        # --- choose to fire or place a mine ---
        # if too many asteroids nearby (adjustable), and we are not on cooldown, place a mine
        if (num_nearby_asteroids >= 7) and self.mine_cooldown == 0:
            drop_mine = True
            # reset cooldown
            self.reset_mine_cooldown()

            # do not fire
            fire = False
        else:
            # else fire
            if shooting.output['ship_fire'] >= 0:
                fire = True
            else:
                fire = False
            
            # decrement mine cooldown
            drop_mine = False
            if (self.mine_cooldown > 0):
                self.mine_cooldown -= 1


        # --- use this all the time --- 
        # Thrust System
        list_nearby_asteroids, num_nearby_asteroids = self.get_nearby_asteroids(ship_state, game_state, distance=500)

        # Debug logs for observation
        # print(f"DEBUG - Closest asteroid distance: {distance_to_asteroid}")
        # print(f"DEBUG - Nearby asteroids count: {num_nearby_asteroids}")
        # print(f"DEBUG - Theta Delta: {shooting_theta}")

        # Thrust system simulation
        thrust_sim = ctrl.ControlSystemSimulation(self.thrust_control)

        # Pass inputs to thrust system
        thrust_sim.input["distance_to_asteroid"] = distance_to_asteroid
        thrust_sim.input["nearby_asteroids"] = num_nearby_asteroids
        thrust_sim.input["theta_delta"] = shooting_theta  # Pass theta_delta for angle-aware thrust control

        # Compute thrust
        try:
            thrust_sim.compute()
            thrust = thrust_sim.output["thrust"]
            # print(f"DEBUG - Computed Thrust: {thrust}")
        except:
            # print("KeyError: 'thrust' not computed. Check inputs or rules.")
            thrust = 0.0

        # --- other stuff to do ---
        self.eval_frames +=1
        
        #DEBUG
        # print(f"thrust: {thrust}, turn_rate: {turn_rate}, fire: {fire}")
        
        return thrust, turn_rate, fire, drop_mine


    @property
    def name(self) -> str:
        return "Team 5 Controller"

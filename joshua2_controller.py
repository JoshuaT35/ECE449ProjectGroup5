from immutabledict import immutabledict
from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple, Any, Type
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib.pyplot as plt

# for now, let's create a system which detonates mines when there are asteroids nearby


class Joshua2Controller(KesslerController):
    def __init__(self):
        self.eval_frames = 30

        # create fuzzy systems
        self.ship_mine_fuzzy_system()

        # variable for mine cooldown. When it is 0, can deploy a mine
        # otherwise, the value is reset
        self.mine_cooldown = 0

    def reset_mine_cooldown(self):
        self.mine_cooldown = 80

    def ship_mine_fuzzy_system(self):
        """
        handles whether the ship should put a mine down or not
        """
        # Antecedent: number of nearby asteroids
        nearby_asteroids = ctrl.Antecedent(np.linspace(0, 25, 100), "nearby_asteroids")

        # Antecedent: largest space between nearby asteroids (TO ADD)

        # Consequent: put down a mine or not
        place_mine = ctrl.Consequent(np.linspace(-1, 1, 5), "place_mine")
        
        # Membership functions for nearby_asteroids
        nearby_asteroids["little"] = fuzz.trimf(nearby_asteroids.universe, [0, 0, 5])
        nearby_asteroids["some"] = fuzz.trimf(nearby_asteroids.universe, [4, 7, 11])
        nearby_asteroids["many"] = fuzz.trimf(nearby_asteroids.universe, [10, 25, 25])  # abitrarily high value

        # Membership functions for place_mine (binary:  < 0 no mine, > 0 place mine)
        place_mine["no_mine"] = fuzz.trimf(place_mine.universe, [-1, -1, 0])
        place_mine["mine_now"] = fuzz.trimf(place_mine.universe, [0, 1, 1])

        # Define fuzzy rules

        # - if there are little asteroids, don't place a mine
        # - should also take space into account
        rule1 = ctrl.Rule(nearby_asteroids["little"], place_mine["no_mine"])

        # - if there are some asteroids, place a mine
        # - should also take space into account
        rule2 = ctrl.Rule(nearby_asteroids["some"], place_mine["mine_now"])

        # - if there are many asteroids, place a mine
        rule3 = ctrl.Rule(nearby_asteroids["many"], place_mine["mine_now"])

        # create the control system
        self.mine_control = ctrl.ControlSystem()
        self.mine_control.addrule(rule1)
        self.mine_control.addrule(rule2)
        self.mine_control.addrule(rule3)


    # return dictionary of nearby asteroids
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
        
        return list_asteroids


    # return asteroid and its distance if it exists, otherwise None and -1
    def get_closest_asteroid(self, ship_state: Dict, game_state: Dict):
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
    

    # given 2 asteroids, find the angle created between the 2 asteroids and the ship, and the unit vector
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
        angle_theta_magnitude = math.acos(
            (d_ship_to_a1_square + d_ship_to_a2_square - d_a1_to_a2_square) / (2*d_ship_to_a1*d_ship_to_a2)
        )

        # get distance unit vector v1 (ship to asteroid 1)
        # unit_v1 = magnitude(dx_ship_a1.i + dy_ship_a1.j)/magnitude
        # unit_v1_distance = tuple(dx_ship_to_a1/d_ship_to_a1, dy_ship_to_a1/d_ship_to_a1)

        # get distance unit vector v2 (ship to asteroid 2)
        # unit_v2 = magnitude(dx_ship_a2.i + dy_ship_a2.j)/magnitude
        # unit_v2_distance = tuple(dx_ship_to_a2/d_ship_to_a2, dy_ship_to_a2/d_ship_to_a2)

        # get distance unit vector v3 between v1 and v2
        unit_v3_x = (dx_ship_to_a1/d_ship_to_a1) + (dx_ship_to_a2/d_ship_to_a2)
        unit_v3_y = (dx_ship_to_a2/d_ship_to_a2) + (dx_ship_to_a2/d_ship_to_a2)
        unit_v3 = tuple(unit_v3_x, unit_v3_y)

        return angle_theta_magnitude, unit_v3
    

    # what does the ship do every time
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # get nearby asteroids and their count within set distance
        distance = 120
        list_nearby_asteroids = self.get_nearby_asteroids(ship_state, game_state, distance)
        num_nearby_asteroids = len(list_nearby_asteroids)

        # # get nearby asteroid
        # asteroid, _ = self.get_closest_asteroid(ship_state, game_state)

        # # print velocity
        # print(asteroid["velocity"])

        # TODO: get largest space between asteroids

        # # create control system
        mine_control_sim = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)

        # # put antecedent inputs
        mine_control_sim.input["nearby_asteroids"] = num_nearby_asteroids

        # compute simulation
        mine_control_sim.compute()

        print("nearby asteroid count: {}\n".format(num_nearby_asteroids))
        print("mine CD: {}\n".format(self.mine_cooldown))
        if (mine_control_sim.output["place_mine"] > 0):
            print("wanna drop a mine\n")

        # if we need to place a mine, and we are not on cooldown, place a mine
        if mine_control_sim.output["place_mine"] > 0 and self.mine_cooldown == 0:
            drop_mine = True
            # reset cooldown
            self.reset_mine_cooldown()
        # decremenet cooldown by 1
        else:
            drop_mine = False
            if (self.mine_cooldown > 0):
                self.mine_cooldown -= 1
               
        # And return your three outputs to the game simulation. Controller algorithm complete.
        thrust = 0.0
        turn_rate = 0.0
        fire = False
        
        self.eval_frames +=1
        
        return thrust, turn_rate, fire, drop_mine
    

    @property
    def name(self) -> str:
        return "2 Controller"


if __name__ == "__main__":
    nearby_asteroids = ctrl.Antecedent(np.linspace(0, 25, 100), "nearby_asteroids")

    # Antecedent: largest space between nearby asteroids (TO ADD)

    # Consequent: put down a mine or not
    place_mine = ctrl.Consequent(np.linspace(-1, 1, 5), "place_mine")
    
    # Membership functions for nearby_asteroids
    nearby_asteroids["little"] = fuzz.trimf(nearby_asteroids.universe, [0, 0, 5])
    nearby_asteroids["some"] = fuzz.trimf(nearby_asteroids.universe, [4, 7, 11])
    nearby_asteroids["many"] = fuzz.trimf(nearby_asteroids.universe, [10, 25, 25])  # abitrarily high value

    # Membership functions for place_mine (binary: -1 = no mine, 1 = place mine)
    place_mine["no_mine"] = fuzz.trimf(place_mine.universe, [-1, -1, 0.0])
    place_mine["mine_now"] = fuzz.trimf(place_mine.universe, [0, 1, 1])

    nearby_asteroids.view()

    plt.show()

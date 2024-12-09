"""
Building off Joshua's controller, to incorporate logic for thrust, as well as genetic algorithm optimization later
"""

from immutabledict import immutabledict
from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple, Any, Type
from cmath import sqrt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np
import matplotlib as plt

# for now, let's create a controller which fires at the closest target


class AbhroController(KesslerController):
    def __init__(self):
        super().__init__()  
        self.eval_frames = 0
        self.ship_targeting_fuzzy_system()
        self.ship_mine_fuzzy_system()
        self.ship_thrust_fuzzy_system()



    def ship_targeting_fuzzy_system(self):
        # Initialize eval frames
        self.eval_frames = 0

        # Declare fuzzy variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')  # Time for bullet to reach intercept
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi / 30, math.pi / 30, 0.1), 'theta_delta')  # Angle to adjust (radians)
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Turn rate (degrees)
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')  # Fire decision

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

        # Fuzzy rules for ship_fire (firing requires alignment and proximity)
        rule_fire1 = ctrl.Rule(theta_delta['zero'] & bullet_time['short'], ship_fire['fire'])
        rule_fire2 = ctrl.Rule(theta_delta['positive_small'] & bullet_time['medium'], ship_fire['fire'])
        rule_fire3 = ctrl.Rule(theta_delta['negative_small'] & bullet_time['medium'], ship_fire['fire'])
        rule_fire4 = ctrl.Rule(bullet_time['long'] | theta_delta['positive_large'] | theta_delta['negative_large'], ship_fire['no_fire'])  # Suppress firing when conditions are poor


        # Create the fuzzy control system
        self.targeting_control = ctrl.ControlSystem([
            rule_turn1, rule_turn2, rule_turn3, rule_turn4, rule_turn5,  # Turning rules
            rule_fire1, rule_fire2, rule_fire3, rule_fire4  # Firing rules
        ])

    
    def ship_mine_fuzzy_system(self):
        """
        handles whether the ship should put a mine down or not
        """
        # if there are many asteroids nearby, put down a mine
        nearby_asteroids = ctrl.Antecedent(np.linspace(0, 8, 1), "nearby_asteroids")
        place_mine = ctrl.Consequent(np.linspace(-1, 1, 10), "place_mine")
        
        # Membership functions for nearby_asteroids
        nearby_asteroids["low"] = fuzz.trimf(nearby_asteroids.universe, [0, 0, 2])
        nearby_asteroids["medium"] = fuzz.trimf(nearby_asteroids.universe, [2, 3, 4])
        nearby_asteroids["high"] = fuzz.trimf(nearby_asteroids.universe, [4, 7, 7])

        # Membership functions for place_mine (binary: -1 = no mine, 1 = place mine)
        place_mine["no_mine"] = fuzz.trimf(place_mine.universe, [-1, -1, 0.0])
        place_mine["mine_now"] = fuzz.trimf(place_mine.universe, [0, 1, 1])

        # Define fuzzy rules
        rule1 = ctrl.Rule(nearby_asteroids["low"], place_mine["no_mine"])
        rule2 = ctrl.Rule(nearby_asteroids["medium"], place_mine["mine_now"])
        rule3 = ctrl.Rule(nearby_asteroids["high"], place_mine["mine_now"])

        self.mine_control = ctrl.ControlSystem()
        self.mine_control.addrule(rule1)
        self.mine_control.addrule(rule2)
        self.mine_control.addrule(rule3)


    def ship_thrust_fuzzy_system(self):
        # Inputs: Distance to asteroid, nearby asteroids, and theta_delta
        distance_to_asteroid = ctrl.Antecedent(np.arange(0, 300, 1), "distance_to_asteroid")
        nearby_asteroids = ctrl.Antecedent(np.arange(0, 81, 1), "nearby_asteroids")
        theta_delta = ctrl.Antecedent(np.arange(-180, 181, 1), "theta_delta")  # Angle in degrees
        thrust = ctrl.Consequent(np.arange(0, 151, 1), "thrust")  # Thrust universe 0–150

        # Membership functions for distance_to_asteroid
        distance_to_asteroid["close"] = fuzz.trimf(distance_to_asteroid.universe, [0, 50, 100])
        distance_to_asteroid["medium"] = fuzz.trimf(distance_to_asteroid.universe, [50, 150, 250])
        distance_to_asteroid["far"] = fuzz.trimf(distance_to_asteroid.universe, [200, 300, 300])

        # Membership functions for nearby_asteroids (universe 0–80)
        nearby_asteroids["low"] = fuzz.trimf(nearby_asteroids.universe, [0, 10, 20])
        nearby_asteroids["medium"] = fuzz.trimf(nearby_asteroids.universe, [15, 30, 50])
        nearby_asteroids["high"] = fuzz.trimf(nearby_asteroids.universe, [40, 60, 80])

        # Membership functions for theta_delta (angle alignment)
        theta_delta["small"] = fuzz.trimf(theta_delta.universe, [-20, 0, 20])  # Head-on or slightly misaligned
        theta_delta["medium"] = fuzz.trimf(theta_delta.universe, [-60, 0, 60])  # Moderate misalignment
        theta_delta["large"] = fuzz.trimf(theta_delta.universe, [-180, -90, 90])  # Safely aligned away

        # Membership functions for thrust (universe 0–150, with emergency thrust above 100)
        thrust["very_low"] = fuzz.trimf(thrust.universe, [0, 0, 50])
        thrust["low"] = fuzz.trimf(thrust.universe, [0, 50, 100])
        thrust["medium"] = fuzz.trimf(thrust.universe, [50, 100, 125])
        thrust["high"] = fuzz.trimf(thrust.universe, [100, 150, 150])

        # Rules for normal behavior (adjust thrust based on distance and crowding)
        rule1 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["high"] & theta_delta["large"], thrust["high"])
        rule2 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["medium"] & theta_delta["large"], thrust["medium"])
        rule3 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["low"] & theta_delta["large"], thrust["low"])

        rule4 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["high"] & theta_delta["medium"], thrust["medium"])
        rule5 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["medium"] & theta_delta["medium"], thrust["low"])
        rule6 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["low"] & theta_delta["medium"], thrust["very_low"])

        rule7 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["high"], thrust["low"])
        rule8 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["medium"], thrust["very_low"])
        rule9 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["low"], thrust["very_low"])

        # Rules for angle alignment
        rule10 = ctrl.Rule(theta_delta["small"], thrust["very_low"])  # Head-on: suppress thrust
        rule11 = ctrl.Rule(theta_delta["medium"] & nearby_asteroids["low"], thrust["low"])  # Moderate misalignment
        rule12 = ctrl.Rule(theta_delta["large"] & distance_to_asteroid["medium"], thrust["medium"])  # Safely aligned

        # Control system
        self.thrust_control = ctrl.ControlSystem(
            [
                rule1, rule2, rule3,
                rule4, rule5, rule6,
                rule7, rule8, rule9,
                rule10, rule11, rule12
            ]
        )



    # return the number of asteroids within the given distance
    def get_num_nearby_asteroids(self, ship_state: Dict, game_state: Dict, distance: int):
        # x and y coordinates of ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # default values
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

            # if current distance is in given distance, increase number by 1
            if (curr_dist <= distance):
                num_asteroids += 1
        
        return num_asteroids


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
        # get the closest asteroid
        closest_asteroid, closest_asteroid_dist = self.find_closest_asteroid(ship_state, game_state)

        # x and y coordinates of ship
        ship_pos_x = ship_state["position"][0]
        ship_pos_y = ship_state["position"][1]

        # closest asteroid distance

        asteroid_ship_x = ship_pos_x - closest_asteroid["position"][0]
        asteroid_ship_y = ship_pos_y - closest_asteroid["position"][1]
        
        asteroid_ship_theta = math.atan2(asteroid_ship_y,asteroid_ship_x)
        
        asteroid_direction = math.atan2(closest_asteroid["velocity"][1], closest_asteroid["velocity"][0]) # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(closest_asteroid["velocity"][0]**2 + closest_asteroid["velocity"][1]**2)
        bullet_speed = 800 # Hard-coded bullet speed from bullet.py
        
        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * closest_asteroid_dist * asteroid_vel * cos_my_theta2)**2 - (4*(asteroid_vel**2 - bullet_speed**2) * closest_asteroid_dist)
        
        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * closest_asteroid_dist * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel**2 -bullet_speed**2))
        intrcpt2 = ((2 * closest_asteroid_dist * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel**2-bullet_speed**2))
        
        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2
                
        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        # Velocities are in m/sec, so bullet_t is in seconds. Add one tik, hardcoded to 1/30 sec.
        intrcpt_x = closest_asteroid["position"][0] + closest_asteroid["velocity"][0] * (bullet_t+1/30)
        intrcpt_y = closest_asteroid["position"][1] + closest_asteroid["velocity"][1] * (bullet_t+1/30)

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y),(intrcpt_x - ship_pos_x))
        
        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi/180)*ship_state["heading"])
        
        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return bullet_t, shooting_theta


    # what does the ship do every time
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # get bullet_t and shooting_theta inputs for ship_targeting_fuzzy_system
        bullet_t, shooting_theta = self.get_bullet_t_shooting_theta(ship_state, game_state)
        
        # create control system simulation for ship_targeting_fuzzy_system
        # pass the inputs to the rulebase and fire it
        shooting = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        shooting.input['bullet_time'] = bullet_t
        shooting.input['theta_delta'] = shooting_theta
        shooting.compute()

        # Get the defuzzified outputs for ship_targeting_fuzzy_system
        turn_rate = shooting.output['ship_turn']
        
        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False

        # get number of nearby asteroids within set distance
        distance = 120
        num_nearby_asteroids = self.get_num_nearby_asteroids(ship_state, game_state, distance)
        mine_control_sim = ctrl.ControlSystemSimulation(self.mine_control, flush_after_run=1)
        mine_control_sim.input["nearby_asteroids"] = num_nearby_asteroids

        mine_control_sim.compute()
        print(mine_control_sim.output["place_mine"])

        # if mine_control_sim.output["place_mine"] == -1:
        #     drop_mine = False
        # else:
        #     drop_mine = True
                
 
    # Step 4: Closest Asteroid and Thrust System
        closest_asteroid, distance_to_asteroid = self.find_closest_asteroid(ship_state, game_state)
        num_nearby_asteroids = self.get_num_nearby_asteroids(ship_state, game_state, distance=500)

        # Debug logs for observation
        print(f"DEBUG - Closest asteroid distance: {distance_to_asteroid}")
        print(f"DEBUG - Nearby asteroids count: {num_nearby_asteroids}")
        print(f"DEBUG - Theta Delta: {shooting_theta}")

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
            print(f"DEBUG - Computed Thrust: {thrust}")
        except KeyError:
            print("KeyError: 'thrust' not computed. Check inputs or rules.")
            thrust = 0.0

        drop_mine = False
        
        self.eval_frames +=1
        
        #DEBUG
        print(f"thrust: {thrust}, turn_rate: {turn_rate}, fire: {fire}")
        
        return thrust, turn_rate, fire, drop_mine


    @property
    def name(self) -> str:
        return "Abhro Controller"
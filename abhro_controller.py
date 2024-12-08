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
        self.eval_frames = 0 #What is this?

        # self.targeting_control is the targeting rulebase, which is static in this controller.      
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0,1.0,0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1*math.pi/30,math.pi/30,0.1), 'theta_delta') # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180,180,1), 'ship_turn') # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1,1,0.1), 'ship_fire')
        
        #Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = fuzz.trimf(bullet_time.universe,[0,0,0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0,0.05,0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe,0.0,0.1)
        
        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1*math.pi/30,-2*math.pi/90)
        theta_delta['NM'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/30, -2*math.pi/90, -1*math.pi/90])
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-2*math.pi/90,-1*math.pi/90,math.pi/90])
        # theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [-1*math.pi/90,math.pi/90,2*math.pi/90])
        theta_delta['PM'] = fuzz.trimf(theta_delta.universe, [math.pi/90,2*math.pi/90, math.pi/30])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe,2*math.pi/90,math.pi/30)
        
        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180,-180,-120])
        ship_turn['NM'] = fuzz.trimf(ship_turn.universe, [-180,-120,-60])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-120,-60,60])
        # ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [-60,60,120])
        ship_turn['PM'] = fuzz.trimf(ship_turn.universe, [60,120,180])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [120,180,180])
        
        #Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be  thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1,-1,0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0,1,1]) 
                
        #Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        #DEBUG
        #bullet_time.view()
        #theta_delta.view()
        #ship_turn.view()
        #ship_fire.view()

        # Declare the fuzzy controller, add the rules 
        # This is an instance variable, and thus available for other methods in the same object. See notes.                         
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
        self.targeting_control = ctrl.ControlSystem()
        self.targeting_control.addrule(rule1)
        self.targeting_control.addrule(rule2)
        self.targeting_control.addrule(rule3)
        # self.targeting_control.addrule(rule4)
        self.targeting_control.addrule(rule5)
        self.targeting_control.addrule(rule6)
        self.targeting_control.addrule(rule7)
        self.targeting_control.addrule(rule8)
        self.targeting_control.addrule(rule9)
        self.targeting_control.addrule(rule10)
        # self.targeting_control.addrule(rule11)
        self.targeting_control.addrule(rule12)
        self.targeting_control.addrule(rule13)
        self.targeting_control.addrule(rule14)
        self.targeting_control.addrule(rule15)
        self.targeting_control.addrule(rule16)
        self.targeting_control.addrule(rule17)
        # self.targeting_control.addrule(rule18)
        self.targeting_control.addrule(rule19)
        self.targeting_control.addrule(rule20)
        self.targeting_control.addrule(rule21)

    
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


    """
    def ship_thrust_fuzzy_system(self):
        
        #Defines the fuzzy logic system for controlling the ship's thrust.
        
        # Input: Distance to the nearest asteroid
        distance_to_asteroid = ctrl.Antecedent(np.arange(0, 300, 1), "distance_to_asteroid")
        distance_to_asteroid["close"] = fuzz.zmf(distance_to_asteroid.universe, 0, 100)
        distance_to_asteroid["medium"] = fuzz.trimf(distance_to_asteroid.universe, [50, 150, 250])
        distance_to_asteroid["far"] = fuzz.smf(distance_to_asteroid.universe, 200, 300)

        # Input: Number of nearby asteroids
        nearby_asteroids = ctrl.Antecedent(np.arange(0, 11, 1), "nearby_asteroids")
        nearby_asteroids["low"] = fuzz.zmf(nearby_asteroids.universe, 0, 3)
        nearby_asteroids["medium"] = fuzz.trimf(nearby_asteroids.universe, [2, 5, 8])
        nearby_asteroids["high"] = fuzz.smf(nearby_asteroids.universe, 7, 10)

        # Input: Angle to the closest asteroid
        angle_to_asteroid = ctrl.Antecedent(np.arange(-math.pi, math.pi, 0.1), "angle_to_asteroid")
        angle_to_asteroid["aligned"] = fuzz.trimf(angle_to_asteroid.universe, [-math.pi / 4, 0, math.pi / 4])
        angle_to_asteroid["off_angle"] = fuzz.trimf(angle_to_asteroid.universe, [-math.pi, -math.pi / 2, -math.pi / 4])
        angle_to_asteroid["opposite"] = fuzz.trimf(angle_to_asteroid.universe, [-math.pi, 0, math.pi])

        # Output: Thrust magnitude
        thrust = ctrl.Consequent(np.arange(0, 1.1, 0.1), "thrust")
        thrust["none"] = fuzz.zmf(thrust.universe, 0, 0.3)
        thrust["low"] = fuzz.trimf(thrust.universe, [0.2, 0.4, 0.6])
        thrust["medium"] = fuzz.trimf(thrust.universe, [0.5, 0.7, 0.9])
        thrust["high"] = fuzz.smf(thrust.universe, 0.8, 1.0)

        # Define fuzzy rules
        rule1 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["high"], thrust["high"])
        rule2 = ctrl.Rule(distance_to_asteroid["close"] & angle_to_asteroid["opposite"], thrust["high"])
        rule3 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["low"], thrust["none"])
        rule4 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["medium"], thrust["medium"])
        rule5 = ctrl.Rule(distance_to_asteroid["medium"] & angle_to_asteroid["off_angle"], thrust["low"])

        # Attach rules to the control system
        self.thrust_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        #self.thrust_sim = ctrl.ControlSystemSimulation(self.thrust_control, flush_after_run=1)
    """
    def ship_thrust_fuzzy_system(self):
        # Inputs: Distance to asteroid and nearby asteroids
        distance_to_asteroid = ctrl.Antecedent(np.arange(0, 300, 1), "distance_to_asteroid")
        nearby_asteroids = ctrl.Antecedent(np.arange(0, 81, 1), "nearby_asteroids")
        thrust = ctrl.Consequent(np.arange(0, 151, 1), "thrust")  # Thrust universe 0–150

        # Membership functions for distance_to_asteroid
        distance_to_asteroid["close"] = fuzz.trimf(distance_to_asteroid.universe, [0, 50, 100])
        distance_to_asteroid["medium"] = fuzz.trimf(distance_to_asteroid.universe, [50, 150, 250])
        distance_to_asteroid["far"] = fuzz.trimf(distance_to_asteroid.universe, [200, 300, 300])

        # Membership functions for nearby_asteroids (universe 0–80)
        nearby_asteroids["low"] = fuzz.trimf(nearby_asteroids.universe, [0, 10, 20])
        nearby_asteroids["medium"] = fuzz.trimf(nearby_asteroids.universe, [15, 30, 50])
        nearby_asteroids["high"] = fuzz.trimf(nearby_asteroids.universe, [40, 60, 80])

        # Membership functions for thrust (universe 0–150, with emergency thrust above 100)
        thrust["low"] = fuzz.trimf(thrust.universe, [0, 50, 100])
        thrust["medium"] = fuzz.trimf(thrust.universe, [50, 100, 125])
        thrust["high"] = fuzz.trimf(thrust.universe, [100, 150, 150])
        thrust["emergency"] = fuzz.trimf(thrust.universe, [120, 150, 150])  # Reserved for emergencies

        # Rules for normal behavior
        rule1 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["high"], thrust["high"])
        rule2 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["medium"], thrust["medium"])
        rule3 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["low"], thrust["low"])

        rule4 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["high"], thrust["medium"])
        rule5 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["medium"], thrust["low"])
        rule6 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["low"], thrust["low"])

        rule7 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["high"], thrust["low"])
        rule8 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["medium"], thrust["low"])
        rule9 = ctrl.Rule(distance_to_asteroid["far"] & nearby_asteroids["low"], thrust["low"])

        # Rules for emergency behavior
        emergency_rule1 = ctrl.Rule(distance_to_asteroid["close"] & nearby_asteroids["high"], thrust["emergency"])
        emergency_rule2 = ctrl.Rule(distance_to_asteroid["medium"] & nearby_asteroids["high"], thrust["emergency"])

        # Control system
        self.thrust_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, emergency_rule1, emergency_rule2])


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
                

        
        # implementation for thrust
        # implementation for thrust
        closest_asteroid, distance_to_asteroid = self.find_closest_asteroid(ship_state, game_state)
        num_nearby_asteroids = self.get_num_nearby_asteroids(ship_state, game_state, distance=500)
        print(f"DEBUG - closest asteroid distance: {distance_to_asteroid}")
        print(f"DEBUG - number of nearby asteroids: {num_nearby_asteroids}")

        # New simulation instance for each step
        thrust_sim = ctrl.ControlSystemSimulation(self.thrust_control)

        # Set inputs
        thrust_sim.input["distance_to_asteroid"] = distance_to_asteroid
        thrust_sim.input["nearby_asteroids"] = num_nearby_asteroids
        # Compute thrust
        try:
            thrust_sim.compute()
            thrust = thrust_sim.output["thrust"]
            print(f"computed thrust value: {thrust}")
        except KeyError:
            print("KeyError: 'thrust' not computed. Check inputs or rules.")
            thrust = 0.0



        drop_mine = False

        if shooting.output['ship_fire'] >= 0:
            fire = True
        else:
            fire = False
        
        self.eval_frames +=1
        
        #DEBUG
        print(f"thrust: {thrust}, turn_rate: {turn_rate}, fire: {fire}")
        
        return thrust, turn_rate, fire, drop_mine


    @property
    def name(self) -> str:
        return "Abhro Controller"
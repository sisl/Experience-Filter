#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-n', '--number-of-vehicles', metavar='N', default=30, type=int, help='Number of vehicles (default: 30)')
    argparser.add_argument('-w', '--number-of-walkers', metavar='W', default=10, type=int, help='Number of walkers (default: 10)')
    argparser.add_argument('--safe', action='store_true', help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument('--filterv', metavar='PATTERN', default='vehicle.*', help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument('--generationv', metavar='G', default='All', help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument('--filterw', metavar='PATTERN', default='walker.pedestrian.*', help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument('--generationw', metavar='G', default='2', help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='Port to communicate with TM (default: 8000)')
    argparser.add_argument('--asynch', action='store_true', help='Activate asynchronous mode execution')
    argparser.add_argument('--hybrid', action='store_true', help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument('-s', '--seed', metavar='S', type=int, help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument('--car-lights-on', action='store_true', default=False, help='Enable car lights')
    argparser.add_argument('--hero', action='store_true', default=False, help='Set one of the vehicles as hero')
    argparser.add_argument('--respawn', action='store_true', default=False, help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument('--no-rendering', action='store_true', default=False, help='Activate no rendering mode')
    argparser.add_argument('--scenario', default=0, type=int,help='Traffic scenario number, 0: normal, 1: busy, 2: aggressive')
    args = argparser.parse_args()
    return args

def blueprint_setup(world, args):
    blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
    blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

    if args.safe:
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)
    return blueprints, blueprintsWalkers

def set_blueprint(blueprints, args):
    hero = args.hero
    blueprint = random.choice(blueprints)
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)
    if hero:
        blueprint.set_attribute('role_name', 'hero')
        hero = False
    else:
        blueprint.set_attribute('role_name', 'autopilot')
    
    return blueprint

def main(args):

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        #world = client.load_world('Town02')

        number_of_vehicles = args.number_of_vehicles
        number_of_walkers = args.number_of_walkers
        spawn_points = world.get_map().get_spawn_points()        
        number_of_spawn_points = len(spawn_points)
        print('spawn points number: ')
        print(number_of_spawn_points)
            

        # Traffic manager setup
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
        world.apply_settings(settings)

        blueprints, blueprintsWalkers = blueprint_setup(world, args)

        # scenario setup
        if (args.scenario == 0):
            number_of_vehicles = int(0.3 * number_of_spawn_points)
            number_of_normal_vehicles = int(0.5*number_of_vehicles)
            number_of_cauitous_vehicles = number_of_vehicles - number_of_normal_vehicles            
            number_of_aggressive_vehicles = 0
            number_of_walkers = int(0.05 * number_of_spawn_points)
        elif(args.scenario == 1):
            number_of_vehicles = int(0.8 * number_of_spawn_points)
            number_of_normal_vehicles = int(0.8*number_of_vehicles)
            number_of_cauitous_vehicles = number_of_vehicles - number_of_normal_vehicles
            number_of_aggressive_vehicles = 0
            number_of_walkers = int(0.1 * number_of_spawn_points)
        elif(args.scenario == 2):
            number_of_vehicles = int(0.5 * number_of_spawn_points)
            number_of_normal_vehicles = int(0.5*number_of_vehicles)
            number_of_cauitous_vehicles = 0
            number_of_aggressive_vehicles = number_of_vehicles - number_of_normal_vehicles         
            number_of_walkers = int(0.05 * number_of_spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        # --------------
        # Spawn vehicles
        # --------------
        hero = args.hero

        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break   

            blueprint = set_blueprint(blueprints, args)

            vehicle = world.spawn_actor(blueprint, transform)
            vehicle.set_autopilot(True,args.tm_port)
            vehicles_list.append(vehicle)

            if (args.scenario == 0 and n < number_of_cauitous_vehicles):
                # cautious cars
                speed_perc = 10*random.rand()+30 #60-70% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

            elif (args.scenario == 0):
                # normal cars
                speed_perc = 10*random.rand()+10 #80-90% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

            if (args.scenario == 1 and n < number_of_cauitous_vehicles):
                # cautious cars
                speed_perc = 10*random.rand()+30 #60-70% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)
            elif (args.scenario == 1):
                # normal cars
                speed_perc = 10*random.rand()+10 #80-90% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

            if(args.scenario == 2 and n < number_of_aggressive_vehicles):
                # aggressive cars
                speed_perc = 10*random.rand()-10 #110-120% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)
                traffic_manager.ignore_lights_percentage(vehicle,50) # ignore lights 50% of the time
                traffic_manager.ignore_vehicles_percentage(vehicle,10) # ignore vehicles 10% of the time
                traffic_manager.ignore_walkers_percentage(vehicle,5) # ignore walkers 5% of the time
            elif(args.scenario == 2):
                # normal cars
                speed_perc = 10*random.rand()+10 #80-90% of the speed limit
                traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

        if (args.scenario == 0):
             print('spawned %d cautious vehicles and %d normal vehicles.' % (number_of_cauitous_vehicles, number_of_normal_vehicles))

        if (args.scenario == 1):
             print('spawned %d cautious vehicles and %d normal vehicles.' % (number_of_cauitous_vehicles, number_of_normal_vehicles))

        if (args.scenario == 2):
             print('spawned %d aggressive vehicles and %d normal vehicles.' % (number_of_aggressive_vehicles, number_of_normal_vehicles))

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(0.2)

        while True:
            if not args.asynch and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()

    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)

if __name__ == '__main__':

    try:
        main(parse_arguments())
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

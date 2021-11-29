import carla
import sys
import numpy as np
import random
import glob
import os
import time

from numpy.lib.function_base import append
from carla import VehicleLightState as vls
from numpy import random

import argparse
import logging

class DefaultArguments:
    port = 2000
    host = '127.0.0.1'
    number_of_vehicles = 30
    number_of_walkers = 10
    safe = True
    filterv = 'vehicle.*'
    filterw = 'walker.pedestrian.*'
    generationv = 'All'
    generationw = '2' 
    tm_port = 8000
    asynch = True
    hybrid = False
    seed = None
    hero = False
    car_lights_on = False
    respawn = False
    no_rendering = False

def is_within_distance(spawn_transform, spct_transform, max_distance, min_distance):
    difference_vector = np.array([
        spawn_transform.location.x - spct_transform.location.x,
        spawn_transform.location.y - spct_transform.location.y
    ])
    distance = np.linalg.norm(difference_vector)
    if distance > max_distance or distance < min_distance:
        return False
    else:
        return True

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


def generate_traffic_func(scenario=0, spawn_radius=100, actor_id=0):
    
    args = DefaultArguments()
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    spawn_points = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    world = client.get_world()
    #world = client.load_world('Town01_Opt')

    number_of_vehicles = args.number_of_vehicles
    number_of_walkers = args.number_of_walkers
    map_spawn_points = world.get_map().get_spawn_points()        
    number_of_spawn_points = len(map_spawn_points)

    spct = world.get_spectator()
    spct_transform = spct.get_transform()
    min_dist = 2

    # only spawn at spawn positions within 100m radius of the spectator
    if (actor_id == 0):
        for n, transform in enumerate(map_spawn_points):
            if (is_within_distance(spct_transform, transform, spawn_radius, min_dist)):
                spawn_points.append(transform)       
        number_of_spawn_points = len(spawn_points)
    else:
        aoi = world.get_actor(actor_id)
        actor_transform = aoi.get_transform()
        for n, transform in enumerate(map_spawn_points):
            if (is_within_distance(actor_transform, transform, spawn_radius, min_dist)):
                spawn_points.append(transform)       
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
    if (scenario == 0):
        number_of_vehicles = int(0.3 * number_of_spawn_points)
        number_of_normal_vehicles = 0
        number_of_cauitous_vehicles = number_of_vehicles  
        number_of_aggressive_vehicles = 0
        number_of_walkers = int(0.05 * number_of_spawn_points)
    elif(scenario == 1):
        number_of_vehicles = int(0.6 * number_of_spawn_points)
        number_of_normal_vehicles = number_of_vehicles
        number_of_cauitous_vehicles = 0
        number_of_aggressive_vehicles = 0
        number_of_walkers = int(0.1 * number_of_spawn_points)
    elif(scenario == 2):
        number_of_vehicles = int(0.5 * number_of_spawn_points)
        number_of_normal_vehicles = 0
        number_of_cauitous_vehicles = 0
        number_of_aggressive_vehicles = number_of_vehicles      
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
        print(n)
        blueprint = set_blueprint(blueprints, args)

        vehicle = world.spawn_actor(blueprint, transform)
        vehicle.set_autopilot(True,args.tm_port)
        vehicles_list.append(vehicle)

        if (scenario == 0 and n < number_of_cauitous_vehicles):
            # cautious cars
            speed_perc = 10*random.rand()+30 #60-70% of the speed limit
            traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

        if (scenario == 1 and n < number_of_normal_vehicles):
            # cautious cars
            speed_perc = 10*random.rand()+10 #80-90% of the speed limit
            traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)

        if(scenario == 2 and n < number_of_aggressive_vehicles):
            # aggressive cars
            speed_perc = -10*random.rand()-10 #110-120% of the speed limit
            traffic_manager.vehicle_percentage_speed_difference(vehicle,speed_perc)
            traffic_manager.ignore_lights_percentage(vehicle,50) # ignore lights 50% of the time
            traffic_manager.ignore_vehicles_percentage(vehicle,10) # ignore vehicles 10% of the time
            traffic_manager.ignore_walkers_percentage(vehicle,5) # ignore walkers 5% of the time

    if (scenario == 0):
            print('spawned %d cautious vehicles.' % (number_of_cauitous_vehicles))

    if (scenario == 1):
            print('spawned %d normal vehicles.' % (number_of_normal_vehicles))

    if (scenario == 2):
            print('spawned %d aggressive vehicles.' % (number_of_aggressive_vehicles))

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

    '''
    while True:
        if not args.asynch and synchronous_master:
            world.tick()
        else:
            world.wait_for_tick()       
    '''
    return vehicles_list, walkers_list, all_id, all_actors

def kill_traffic(vehicles_list, walkers_list, all_id, all_actors):
    args = DefaultArguments()
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    world = client.get_world()

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
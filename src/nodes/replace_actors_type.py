#!/usr/bin/env python

"""
Replaces actors with a different type, but retains transform. E.g. replace all traffic lights with traffic cones.
"""

import carla
import glob
import os
import sys
import time
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--actor_type', type=str, default ="traffic.traffic_light", help="Type of actors to change.")
    parser.add_argument('-t', '--target_type', type=str, default="static.prop.trafficcone01", help="Target actor type.")

    parser.add_argument('-H', '--host', type=str, default='127.0.0.1' ,help="Host of running Carla server. Default: localhost.")
    parser.add_argument('-P', '--port', type=int, default=2000 ,help="Port of running Carla server.")
    
    args = parser.parse_args()
    return args


def main(args):
	client = carla.Client(args.host, args.port)
	client.set_timeout(2.0)
	world = client.get_world()

	actor_list = world.get_actors()
	aoi = actor_list.filter("*{}*".format(args.actor_type))
	tbp =  world.get_blueprint_library().find(args.target_type)

	for item in aoi:
		if "traffic_light" in args.actor_type:
			item.set_state(carla.TrafficLightState.Green)
			item.set_green_time(9999999.0)
		world.spawn_actor(tbp, item.get_transform())
		item.destroy()


if __name__ == '__main__':
	try:
		main(parse_arguments())
	except KeyboardInterrupt:
		print('\nAborted by user.')
#!/usr/bin/env python

"""
Prints the distance of nearest N traffic signs to an actor of interest. 
"""

import carla
import glob
import os
import sys
import time
import argparse
import heapq

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--actor_id', type=int, help="ID of the actor of interest.")
    parser.add_argument('-t', '--traffic_signs', type=str, default="stop", choices=["speed_limit" "stop", "traffic_light", "unknown", "yield"], help="Traffic signs to consider.")
    parser.add_argument('-n', '--number_of_signs', type=int, default=3, help="Number of traffice signs to consider.")

    parser.add_argument('-S', '--sleeptime', type=float, default=0.25, help="Sleep-time between each publication.")
    parser.add_argument('-H', '--host', type=str, default='127.0.0.1' ,help="Host of running Carla server. Default: localhost.")
    parser.add_argument('-P', '--port', type=int, default=2000 ,help="Port of running Carla server.")
    
    args = parser.parse_args()
    return args


def main(args):
	client = carla.Client(args.host, args.port)
	client.set_timeout(2.0)
	world = client.get_world()
	aoi = world.get_actor(args.actor_id)

	while True:
		actor_list = world.get_actors()
		aoi_location = aoi.get_location()
		tsoi = actor_list.filter("traffic.{}*".format(args.traffic_signs))

		distances = [(carla.Location.distance(aoi.get_location(), item.get_location()), item.get_location()) for item in tsoi]
		# import pdb; pdb.set_trace()

		# print("-----------------------------------")
		for item in heapq.nsmallest(args.number_of_signs, distances): print(f"Actor: {str(aoi_location)}   Sign: {str(item[1])}   Distance: {item[0]}")
		print("-----------------------------------")

		time.sleep(args.sleeptime)


if __name__ == '__main__':
	try:
		main(parse_arguments())
	except KeyboardInterrupt:
		print('\nAborted by user.')
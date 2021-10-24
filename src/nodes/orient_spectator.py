#!/usr/bin/env python

"""
Actively orients the spectator as a birdseye view over an actor of interest. 
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

    parser.add_argument('-a', '--actor_id', type=int, help="ID of the actor of interest.")
    parser.add_argument('-z', '--altitude', type=float, default=35.0, help="How high up in the sky is the spectator from the actor? [m]")
    parser.add_argument('-pt', '--pitch', type=float, default=-90.0, help="How much is the spectator pitched the actor? [deg]")

    parser.add_argument('-S', '--sleeptime', type=float, default=0.001, help="Sleep-time between each publication.")
    parser.add_argument('-H', '--host', type=str, default='127.0.0.1' ,help="Host of running Carla server. Default: localhost.")
    parser.add_argument('-P', '--port', type=int, default=2000 ,help="Port of running Carla server.")
    
    args = parser.parse_args()
    return args


def main(args):
	client = carla.Client(args.host, args.port)
	client.set_timeout(2.0)
	world = client.get_world()
	spct = world.get_spectator()
	spt_init_tf = spct.get_transform()

	try:
		aoi = world.get_actor(args.actor_id)
		aoi_init_yaw = aoi.get_transform().rotation.yaw
	except:
		spct.set_transform(spt_init_tf)
		sys.exit("ABORTED. Actor is not found. Spectator is reset.")

	while True:
		if not aoi.is_active:
			spct.set_transform(spt_init_tf)
			sys.exit("ABORTED. Actor is destroyed. Spectator is reset.")

		aoi_tf = aoi.get_transform()
		aoi_tf.location.z = args.altitude
		aoi_tf.rotation.roll = 0.0
		aoi_tf.rotation.pitch = args.pitch
		aoi_tf.rotation.yaw = aoi_init_yaw
		spct.set_transform(aoi_tf)
		
		time.sleep(args.sleeptime)


if __name__ == '__main__':
	try:
		print('Actively orienting spectator...')
		main(parse_arguments())
	except KeyboardInterrupt:
		print('\nAborted by user.')
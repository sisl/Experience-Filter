#!/usr/bin/env python

"""
Prints the controls of of an actor vehicle. 
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

    parser.add_argument('-S', '--sleeptime', type=float, default=0.25, help="Sleep-time between each publication.")
    parser.add_argument('-H', '--host', type=str, default='127.0.0.1' ,help="Host of running Carla server. Default: localhost.")
    parser.add_argument('-P', '--port', type=int, default=2000 ,help="Port of running Carla server.")
    
    args = parser.parse_args()
    return args

def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    actor_list = world.get_actors()
    aoi = world.get_actor(args.actor_id)

    while True:
        print(aoi.get_control())
        time.sleep(args.sleeptime)


if __name__ == '__main__':
    try:
        main(parse_arguments())
    except KeyboardInterrupt:
        print('\nAborted by user.')
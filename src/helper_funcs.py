import carla
import sys
import numpy as np
import random

def printl(L, log_name=""):
    """Print elements of an iterable object. Useful for objects like L = <carla.libcarla.ActorList>."""
    if log_name: print(log_name)
    for idx, item in enumerate(L):
        print(idx, item)
    return

def vector3D_norm(vec: carla.Vector3D) -> float:
    """Returns the norm/magnitude (a scalar) of the given carla.3D vector."""
    return np.linalg.norm(np.array([vec.x, vec.y, vec.z])) 

def move_actor(actor, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
    """Move any carla actor object by (dx, dy, dz, droll, dpitch, dyaw)."""
    tf_init = actor.get_transform()
    tf_init_loc, tf_init_rot = tf_init.location, tf_init.rotation
    tf_final = carla.Transform(carla.Location(x=tf_init_loc.x+dx,
                                              y=tf_init_loc.y+dy,
                                              z=tf_init_loc.z+dz),
                               carla.Rotation(pitch=tf_init_rot.pitch+dpitch,
                                              yaw=tf_init_rot.yaw+dyaw,
                                              roll=tf_init_rot.roll+droll))
    actor.set_transform(tf_final)
    return

def move_actor_id(actor, world, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0):
    """Same as above, but input is Int actor id, instead of the actor object itself."""
    actor = world.get_actor(actor)
    return move_actor(actor, dx=dx, dy=dy, dz=dz, droll=droll, dpitch=dpitch, dyaw=dyaw)


def distance_among_actors(actor1, actor2):
    """Get distance between two actor objects."""
    return carla.Location.distance(actor1.get_location(), actor2.get_location())

def distance_among_ids(id1, id2, world):
    """Same as above, but input is Int actor id."""
    actor1 = world.get_actor(id1)
    actor2 = world.get_actor(id2)
    return carla.Location.distance(actor1.get_location(), actor2.get_location())

def ref_angle(target_actor, reference_actor):
    """Get the reference angle between two actors. Input ordering matters!"""
    target_loc = target_actor.get_location()
    ref_tf = reference_actor.get_transform()
    ref_loc = ref_tf.location
    target_vector = np.array([
        target_loc.x - ref_loc.x,
        target_loc.y - ref_loc.y])
    norm_target = np.linalg.norm(target_vector)

    fwd = ref_tf.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    dotted = np.dot(forward_vector, target_vector)
    angle = np.rad2deg(np.arccos(np.clip(dotted / norm_target, -1., 1.)))

    return angle

def tf_distance(target_location, reference_location):
    """
    Calculate distances between two object transforms.

    :param target_location: location of the target object
    :param reference_location: location of the reference object
    """
    target_vector = np.array([
        target_location.x - reference_location.x,
        target_location.y - reference_location.y
    ])
    norm_target = np.linalg.norm(target_vector)
    return norm_target

def angle_is_approx(query_angle, target_angle, target_delta=45):
    """Retuns true if `query_angle` is approximately equal to `target_angle`, with a permitted deviation of `delta_target`."""
    return target_angle - target_delta < query_angle < target_angle + target_delta

def is_ahead_of_reference(query_actor, reference_actor):
    """
    Returns True if `query_actor` is ahead of `reference_actor`, w.r.t. to the direction `query_actor` is pointing.
    
    Map Compass:
            +Y (90)
              |
    +X (0)————o———— -X (-180)
              |
           -Y (-90)
    """
    query_yaw = query_actor.get_transform().rotation.yaw

    if angle_is_approx(query_yaw, 0):
        return query_actor.get_location().x > reference_actor.get_location().x

    elif angle_is_approx(query_yaw, 90):
        return query_actor.get_location().y > reference_actor.get_location().y

    elif angle_is_approx(query_yaw, -180):
        return query_actor.get_location().x < reference_actor.get_location().x

    elif angle_is_approx(query_yaw, -90):
        return query_actor.get_location().y < reference_actor.get_location().y


import carla
import numpy as np
import time
import pickle
import csv
import diversipy

def printl(L, log_name=""):
    """Print elements of an iterable object. Useful for objects like L = <carla.libcarla.ActorList>."""
    if log_name: print(log_name)
    for idx, item in enumerate(L):
        print(idx, item)
    return

def reverse_dict(D):
    return {v:k for (k,v) in D.items()}

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

def get_nearest_neighbor(datapoints_list, point):
    """Get the nearest element in `datapoints_list` to `point`."""
    distances = np.array(datapoints_list) - np.tile(point, (len(datapoints_list), 1))
    idx = np.argmin(np.linalg.norm(distances, axis=1))
    val = datapoints_list[idx]
    return idx, val

def is_ahead_of_reference(query_actor, reference_actor):
    """
    Returns True if `query_actor` is ahead of `reference_actor`, w.r.t. the direction `query_actor` is pointing.
    
    Yaw Compass:
            +Y (90)
              |
    +X (0)————o———— -X (+-180)
              |
           -Y (-90)
    """
    query_yaw = query_actor.get_transform().rotation.yaw

    if angle_is_approx(query_yaw, 0):
        return query_actor.get_location().x > reference_actor.get_location().x

    elif angle_is_approx(query_yaw, 90):
        return query_actor.get_location().y > reference_actor.get_location().y
        
    elif angle_is_approx(query_yaw, -90):
        return query_actor.get_location().y < reference_actor.get_location().y

    else:   # angle_is_approx(query_yaw, 180) or angle_is_approx(query_yaw, -180):
        return query_actor.get_location().x < reference_actor.get_location().x

def save_with_pkl(data, savename, is_stamped):
    """
    Save data with pkl.
    Setting `is_stamaped` to True will append a timestamp to `savename`.
    """
    if is_stamped:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{savename}_{timestamp}.pkl"
    else:
        filename = f"{savename}.pkl"

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    return

def load_with_pkl(loadname):
    """Load a single pkl."""
    with open(loadname, 'rb') as f:
        L = pickle.load(f)
    return L

def load_many_with_pkl(list_of_loadnames, len_of_each_loadname=2):
    """Load many files with pkl."""
    result = [[] for _ in range(len_of_each_loadname)]   # pre-allocation

    for loadname in list_of_loadnames:
        L = load_with_pkl(loadname)
        for idx in range(len_of_each_loadname):
            result[idx].extend(L[idx])
    print(f"## INFO: Loaded {len(list_of_loadnames)} pkl files")
    return result

def log_to_file(filename, row_data_list):
    """Dump data into a readable file."""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data_list)

def read_log_from_file(filename):
    """Load data from a readable file."""
    rows = []
    with open(filename) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            rows.append(row)
    return np.array(rows)

def get_scenario_score(agent, score_args=None):
    """Get the score of a completed scenario."""
    vel_hist, time_hist = agent.velocity_history, agent.time_history
    safety_val = agent.min_distance_to_any_rival

    def get_acceleration(vel_hist, time_hist):
        v, t = np.array(vel_hist), np.array(time_hist)
        dv = v[:-1] - v[1:]
        dt = t[:-1] - t[1:]
        return dv / dt

    def get_time_taken(time_hist):
        return time_hist[-1] - time_hist[0]

    discomfort_val = np.sum(np.abs(get_acceleration(vel_hist, time_hist)))
    time_val = get_time_taken(time_hist)

    contributors = {'safety':safety_val, 'discomfort':discomfort_val, 'time':time_val}
    if not score_args:
        return contributors
    else:
        score = score_args.safety * safety_val + score_args.discomfort * discomfort_val + score_args.time * time_val
        return score, contributors

def get_coverage_points(datapoints, tef):
    """Select best `tef` number of elements in `datapoints` that provide best coverage throughout the hypercube."""
    return diversipy.psa_select(datapoints, tef)

def get_furthest_points(datapoints, datapoint_to_benchmark, tef):
    """Select the furthest `tef` points in `datapoints` w.r.t. to `datapoint_to_benchmark`."""
    distances = np.array(datapoints) - np.tile(datapoint_to_benchmark, (len(datapoints), 1))
    idxs = np.argsort(np.linalg.norm(distances, axis=1))[-tef:]
    return [datapoints[i] for i in idxs]

def can_be_float(element):
    """
    Check if `element` can be a float or not.
    Useful during broadcasting to check whether strings can be parsed as floats.
    """
    try:
        float(element)
        return True
    except ValueError:
        return False

def normalize_from_0_to_1(x, min_val=None, max_val=None):
    """Normalize a np.array between 0 and 1."""
    if not min_val: min_val = np.min(x)
    if not max_val: max_val = np.max(x)
    return (x - min_val) / (max_val - min_val)

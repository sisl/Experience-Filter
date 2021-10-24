import carla

client = carla.Client('localhost', 2000)
world = client.get_world()    # world = client.load_world('Town01')

# Apply batch commands for speed. E.g.:
# client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

world.set_weather(carla.WeatherParameters.WetCloudySunset)

# Snapshots of the world

world_snapshot = world.get_snapshot()
timestamp = world_snapshot.timestamp # Get the time reference 

for actor_snapshot in world_snapshot: # Get the actor and the snapshot information
    actual_actor = world.get_actor(actor_snapshot.id)
    actor_snapshot.get_transform()
    actor_snapshot.get_velocity()
    actor_snapshot.get_angular_velocity()
    actor_snapshot.get_acceleration()  

actor_snapshot = world_snapshot.find(actual_actor.id) # Get an actor's snapshot

# Find a convenient location to spawn a walker
spawn_point = carla.Transform()
spawn_point.location = world.get_random_location_from_navigation()

# Get blueprint library
blueprint_library = world.get_blueprint_library()

# Attach camera to a vehicle, and save RGB output to disk
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera = world.spawn_actor(camera_bp, relative_transform, attach_to=my_vehicle)
camera.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

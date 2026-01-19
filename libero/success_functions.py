import numpy as np

_max_mug_height_reached = 0.0

def lift_white_yellow_mug(env):
    # lifted mug
    return env.env.object_states_dict["white_yellow_mug_1"].get_geom_state()["pos"][2] > 0.50

def lift_white_yellow_mug_and_place_on_plate(env):
    global _max_mug_height_reached
    
    # Reset tracker at start of episode
    if env.t < 10:
        _max_mug_height_reached = 0.0
    
    # Track the maximum height the mug has reached
    current_height = env.env.object_states_dict["white_yellow_mug_1"].get_geom_state()["pos"][2]
    _max_mug_height_reached = max(_max_mug_height_reached, current_height)
    
    # Require that the mug reached at least 0.5 height at some point
    mug_was_lifted = _max_mug_height_reached > 0.50

    ontop = check_ontop(env, "plate_2", "white_yellow_mug_1")
    robot_in_air = env.unwrapped.robots[0].recent_ee_pose.current[2] > 0.65

    return mug_was_lifted and ontop and robot_in_air

def check_ontop(env, this_object_name, other_object_name):
    """Check if this_object is on top of other_object."""
    this_object_position = env.env.sim.data.body_xpos[
        env.env.obj_body_id[this_object_name]
    ]
    other_object_position = env.env.sim.data.body_xpos[
        env.env.obj_body_id[other_object_name]
    ]
    return (
        (this_object_position[2] <= other_object_position[2])
        and check_contact(env, this_object_name, other_object_name)
        and (
            np.linalg.norm(this_object_position[:2] - other_object_position[:2])
            < 0.03
        )
    )

def check_contact(env, object_name_1, object_name_2):
    """Check if two objects are in contact."""
    object_1 = env.env.get_object(object_name_1)
    object_2 = env.env.get_object(object_name_2)
    return env.env.check_contact(object_1, object_2)


def false(env):
    return False
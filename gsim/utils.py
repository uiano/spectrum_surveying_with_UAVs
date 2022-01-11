def time_to_str(time_delta):

    hours = time_delta.seconds // 3600
    reminder = time_delta.seconds % 3600
    minutes = reminder // 60
    seconds = (time_delta.seconds - hours * 3600 -
               minutes * 60) + time_delta.microseconds / 1e6
    time_str = ""
    if time_delta.days:
        time_str = "%d days, " % time_delta.days
    if hours:
        time_str = time_str + "%d hours, " % hours
    if minutes:
        time_str = time_str + "%d minutes, " % minutes
    if time_str:
        time_str = time_str + "and "

    return time_str + "%.3f seconds" % seconds
    #set_trace()


def print_obj_attributes(obj):

    for name in dir(obj):
        print("==============================================")
        print(f"Attr. name: \"{name}\"")

        try:
            attr = getattr(obj, name)
            print(f"Attr. type: {type(attr)}")
            print(attr)
        except Exception:
            pass


def instr(
    obj,
    expand_lists=False,
    level=0,
    prefix="",
):
    """Inspect the structure of an object.

    Args:
        `obj`: object to inspect

        `expand_lists`: if False, only the first item of a list is
        expanded. This is helpful when all items in a list have the
        same structure. 

    """
    def nprint(text):
        sp = "    "
        print(sp * level + "- ", prefix, text)

    if hasattr(obj, 'shape'):
        # Typically tf.Tensor or np.ndarray
        nprint(f"{obj.__class__} with shape {obj.shape}")
    elif isinstance(obj, list):
        nprint(f"list of length {len(obj)}")
        for ind, item in enumerate(obj):
            instr(item,
                  level=level + 1,
                  prefix=f"{ind}:",
                  expand_lists=expand_lists)
            if not expand_lists:
                break

    elif isinstance(obj, tuple):
        nprint(f"tuple of length {len(obj)}")
        for ind, item in enumerate(obj):
            instr(item,
                  level=level + 1,
                  prefix=f"{ind}: ",
                  expand_lists=expand_lists)
    elif isinstance(obj, dict):
        nprint(f"dict with keys")
        for key, val in obj.items():
            instr(val,
                  level=level + 1,
                  prefix=f"\"{key}\":",
                  expand_lists=expand_lists)
    else:
        nprint(f"{obj.__class__}")

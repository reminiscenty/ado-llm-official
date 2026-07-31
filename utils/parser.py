import numpy as np
import re


PARAM_UNITS = ["f", "m", "n", "p", "u", "k", "G", "M"]


def extract_param_sets_from_response(
    response,
    params_list,
    params_units=None,
    max_sets=None,
):
    """Extract complete parameter sets from an LLM response, bottom-first."""
    if params_units is None:
        params_units = PARAM_UNITS

    units_pattern = f"[{''.join(re.escape(unit) for unit in params_units)}]?"
    value_pattern = rf"\d+(?:\.\d+)?{units_pattern}"
    param_patterns = {
        param: re.compile(rf"\b{re.escape(param)}\s*=\s*({value_pattern})\b")
        for param in params_list
    }

    def add_set_from_chunk(chunk, sets, seen):
        found = {}
        for param, pattern in param_patterns.items():
            matches = pattern.findall(chunk)
            if matches:
                found[param] = matches[-1]
        if len(found) == len(params_list):
            key = tuple(found[param] for param in params_list)
            if key not in seen:
                seen.add(key)
                sets.append({param: found[param] for param in params_list})

    def extract_from_text(text):
        sets = []
        seen = set()
        text = re.sub(r"```\w*\n?", "", text)

        for line in text.splitlines():
            add_set_from_chunk(line, sets, seen)

        block = []
        for line in text.splitlines():
            if any(pattern.search(line) for pattern in param_patterns.values()):
                block.append(line)
            elif block:
                add_set_from_chunk(" ".join(block), sets, seen)
                block = []
        if block:
            add_set_from_chunk(" ".join(block), sets, seen)

        return sets

    text_lower = response.lower()
    section_text = None
    for marker in ("genetic adjustment:", "genetic adjustment"):
        idx = text_lower.rfind(marker)
        if idx != -1:
            section_text = response[idx:]
            break

    sets = extract_from_text(section_text) if section_text else []
    if not sets:
        sets = extract_from_text(response)

    sets = sets[::-1]
    if max_sets is not None:
        sets = sets[:max_sets]
    return sets


def get_param_bounds(param_name, ranges):
    if "w" in param_name[0]:
        return ranges["w"]
    if "l" in param_name[0]:
        return ranges["l"]
    if "r" in param_name[0]:
        return ranges["r"]
    if "c" in param_name[0]:
        return ranges["c"]
    raise ValueError(f"Unknown parameter type for '{param_name}'")


def clip_params_dict(params_dict, params_list, ranges):
    """Clip parameter values to the allowed design-space bounds."""
    clipped = {}
    clipped_names = []
    for param in params_list:
        value = convert_value_to_float(params_dict[param])
        lo, hi = get_param_bounds(param, ranges)
        clipped_value = min(max(value, lo), hi)
        if clipped_value != value:
            clipped_names.append(param)
        clipped[param] = format_float(clipped_value)
    return clipped, clipped_names


def nparray_to_params_dict(params_numpy, params_list):
    params_dict = {}
    for i, param in enumerate(params_list):
        params_dict[param] = format_float(params_numpy[i])

    return params_dict

def params_dict_to_nparray(params_dict):
    values = []
    for k, v in params_dict.items():
        v_fp = convert_value_to_float(v)
        values.append(v_fp)

    params_numpy = np.array(values)

    return params_numpy




def numpy2params(numpy_array, params_list) -> str:
    """convert numpy array to params string

    Args:
        numpy_array (np.array): numpy array
        params_list (list): list of params

    Returns:
        str: params string
    """
    # params_query = ".param "
    params_query = ""
    for j, param in enumerate(params_list):
        params_query += f"{param}={format_float(numpy_array[j])} "
    return params_query


def params2numpy(params_query, params_list):
    """convert params string to numpy array

    Args:
        params_query (str): params string
        params_list (list): list of params

    Returns:
        np.array: numpy array
    """
    params = params_query.split(" ")
    numpy_array = np.zeros(len(params_list))
    for param in params:
        if "=" in param:
            key, value = param.split("=")
            numpy_array[params_list.index(key)] = convert_value_to_float(value)
    return numpy_array


def format_float(value:float, precision=1) -> str:
    """format float value, convert to string with {precision} decimal points, and add unit if necessary

    Args:
        value (float): a float value
        precision (int, optional): precision of the float value. Defaults to 1.

    Returns:
        str: formatted string
    """
    value_abs = abs(value)
    if value_abs == 0:
        return "0"
    if value_abs < 1e-12:
        return f"{value*1e15:.{precision}f}f"
    if value_abs < 1e-9:
        return f"{value*1e12:.{precision}f}p"
    if value_abs < 1e-6:
        return f"{value*1e9:.{precision}f}n"
    if value_abs < 1e-3:
        return f"{value*1e6:.{precision}f}u"
    if value_abs < 1e3:
        return f"{value:.{precision}f}"
    if value_abs < 1e6:
        return f"{value*1e-3:.{precision}f}k"
    if value_abs < 1e9:
        return f"{value*1e-6:.{precision}f}M"
    if value_abs < 1e12:
        return f"{value*1e-9:.{precision}f}G"
    return f"{value:.{precision}f}"


def convert_value_to_float(value_with_unit:str) -> float:
    """ convert the string value with unit to float

    Args:
        value_with_unit (str): a string value with unit

    Returns:
        float: a float value
    """
    # convert the values to float; eg. 1.0u to 1e-6, 1.0k to 1e3, etc.
    # Find the last character of the value;
    unit = value_with_unit[-1]
    # If the last character is a unit, convert the value to a float;
    if unit in PARAM_UNITS:
        # Convert the value to a float;
        value = float(value_with_unit[:-1])
        # Convert the value to the correct unit;
        if unit == "f":
            value *= 1e-15
        elif unit == "p":
            value *= 1e-12
        elif unit == "n":
            value *= 1e-9
        elif unit == "u":
            value *= 1e-6
        elif unit == "k":
            value *= 1e3
        elif unit == "M":
            value *= 1e6
        elif unit == "G":
            value *= 1e9
        # Update the value in the dictionary;
        return value
    else:
        # If the last character is not a unit, convert the value to a float;
        return float(value_with_unit)

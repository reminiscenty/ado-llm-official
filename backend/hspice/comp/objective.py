def read_results(f_val, val_name, args):
    '''
    read f_val as a float nubmer: return the float number if it is a float number, otherwise return the failed value in args
    Args: f_val (str), string containing the float number
          val_name (str), name of the value
          args (dict), dictionary containing circuit information
    '''
    try:
        f_val = float(f_val)
        return float(f_val)
    except ValueError:
        print(f'Failed to read {val_name} from the file')
        return args[f'{val_name}_failed']

def normalize_fvals(f_vals: dict, ckt: dict, check_specs=False):
    '''
    Normalize the objective function values
    Args: f_vals (dict), dictionary containing objective function values
          ckt (dict), dictionary containing circuit information
    Returns: f_vals (dict), dictionary containing normalized objective function values
    '''
    def get_fval(val_name, check_specs=False):
        '''
        Check if the value meets the specs
        Args:
              val_name (str), name of the value
        Returns: f_val (float), value to be checked
        '''
        f_val = f_vals[val_name]
        if check_specs:
            if val_name in ['pow', 'vhy', 'v_offset']:
                f_val = abs(f_val)
                if f_val > ckt[f'{val_name}_spec']:
                    return ckt[f'{val_name}_failed']
            else:
                if f_val < ckt[f'{val_name}_spec']:
                    return ckt[f'{val_name}_failed']
                else:
                    return min(f_val, 2*ckt[f'{val_name}_norm'][1])
        return f_val
    # get the objective function values
    ugain_freq = get_fval('ugain_freq', check_specs)
    gain = get_fval('gain', check_specs)
    hy_error = get_fval('vhy', check_specs)
    # offset is the absolute value
    offset = get_fval('v_offset', check_specs)
    pow = get_fval('pow', check_specs)

    # get the normalization values
    gain_range = ckt['gain_norm']
    ugain_freq_range = ckt['ugain_freq_norm']
    hy_error_range = ckt['vhy_norm']
    offset_range = ckt['v_offset_norm']
    pow_range = ckt['pow_norm']

    # normalize the objective function values
    f_vals['gain_normed'] = (gain - gain_range[0]) / (gain_range[1] - gain_range[0])
    f_vals['pow_normed'] = (pow - pow_range[0]) / (pow_range[1] - pow_range[0])
    f_vals['ugain_freq_normed'] = (ugain_freq - ugain_freq_range[0]) / (ugain_freq_range[1] - ugain_freq_range[0])
    f_vals['vhy_normed'] = (hy_error - hy_error_range[0]) / (hy_error_range[1] - hy_error_range[0])
    f_vals['v_offset_normed'] = (offset - offset_range[0]) / (offset_range[1] - offset_range[0])




    # leave the power, vhy, v_offset as it is

    return f_vals

def objective(f_vals: dict, ckt: dict, check_specs=True):
    '''
    Calculate the objective function value
    Args: f_vals (dict), dictionary containing normalized objective function values
          ckt (dict), dictionary containing circuit information
    Returns: objective (float), objective function value
    '''
    # normalize the objective function values
    f_vals = normalize_fvals(f_vals, ckt, check_specs)
    # merits
    merits = ['gain', 'ugain_freq', 'vhy', 'v_offset', 'pow']
    fom = 0
    for merit in merits:
        fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed'] 
        # fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}'] 
    f_vals['fom'] = fom
    return f_vals
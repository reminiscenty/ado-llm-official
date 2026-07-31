
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
        print(f'Failed to read {val_name} from the file, returning failed value: {args[f"{val_name}_failed"]}')
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
            if val_name == 'pow':
                if f_val > ckt[f'pow_spec']:
                    return ckt[f'pow_failed']
            elif val_name == 'pm':
                # 45 < pm < 60
                if f_val < 45:
                    return ckt[f'pm_failed']
                
            else:
                if f_val < ckt[f'{val_name}_spec']:
                    return ckt[f'{val_name}_failed']
        return f_val
    # get the objective function values
    gbw = get_fval('gbw', check_specs)
    gain = get_fval('gain', check_specs)
    cmrr = get_fval('cmrr', check_specs)
    pm = get_fval('pm', check_specs)
    pow = get_fval('pow', check_specs)

    # get the normalization values
    gbw_range = ckt['gbw_norm']
    gain_range = ckt['gain_norm']
    cmrr_range = ckt['cmrr_norm']
    pm_range = ckt['pm_norm']
    pow_range = ckt['pow_norm']

    # normalize the objective function values
    f_vals['gbw_normed'] = (gbw - gbw_range[0]) / (gbw_range[1] - gbw_range[0])
    f_vals['gain_normed'] = (gain - gain_range[0]) / (gain_range[1] - gain_range[0])
    f_vals['cmrr_normed'] = (cmrr - cmrr_range[0]) / (cmrr_range[1] - cmrr_range[0])
    f_vals['pm_normed'] = (pm - pm_range[0]) / (pm_range[1] - pm_range[0])
    f_vals['pow_normed'] = (pow - pow_range[0]) / (pow_range[1] - pow_range[0])

    # clip the normalized values to 2
    f_vals['gbw_normed'] = min(1.25, f_vals['gbw_normed'])
    f_vals['gain_normed'] =  min(1.25, f_vals['gain_normed'])
    f_vals['cmrr_normed'] = min(1.25, f_vals['cmrr_normed'])
    f_vals['pm_normed'] = min(1.25, f_vals['pm_normed'])

    # leave the power as it is

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
    merits = ['gbw', 'gain', 'cmrr', 'pm', 'pow']
    # merits = ['gbw', 'gain', 'cmrr',]
    fom = 0
    for merit in merits:
        fom += ckt[f'{merit}_weight'] * f_vals[f'{merit}_normed']
    f_vals['fom'] = fom
    return f_vals
import os
import subprocess
from typing import Any

from backend.hspice.comp.aux_info import read_work_region
from backend.hspice.comp.objective import objective, read_results


def hspice_eval_f_comp(
    point_to_evaluate: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Evaluate a hysteresis-comparator design with HSPICE."""
    fvals = {
        "fom": 0.0,
        "ugain_freq": 0.0,
        "gain": 0.0,
        "vhy": 0.0,
        "v_offset": 0.0,
        "aux_info": "",
    }

    ckt_dir = args["ckt_dir"]
    param_file = os.path.join(ckt_dir, "param.inc")

    with open(param_file, "w", encoding="utf-8") as f:
        f.write(point_to_evaluate)

    ckt_name = args["ckt_name"]

    subprocess.run(
        [
            "hspice",
            f"{ckt_dir}/{ckt_name}.sp",
            "-o",
            f"{ckt_dir}/{ckt_name}.lis",
        ],
        check=True,
        text=True,
    )

    # read results from {ckt_name}.ma1, gain, gbw, pm is the 1st, 2nd, 3rd values in the fifth line
    with open(f"{ckt_dir}/{ckt_name}.ma0", encoding="utf-8") as f:
        lines = f.readlines()
        gain = read_results(lines[4].split()[0], 'gain', args)
        ugain_freq = read_results(lines[4].split()[2], 'ugain_freq', args)
    # read results from {ckt_name}.mt0, 
    with open(f"{ckt_dir}/{ckt_name}.mt0", encoding="utf-8") as f:
        lines = f.readlines()
        hy_error = read_results(lines[4].split()[2], 'vhy', args)
        offset = read_results(lines[4].split()[3], "v_offset", args)
    # read results from {ckt_name}.dp0
    with open(f"{ckt_dir}/{ckt_name}.dp0", encoding="utf-8") as f:
        lines = f.readlines()
        pow_vol = read_results(lines[25].split()[1], 'pow_vol', args)
        pow_cur = read_results(lines[49].split()[1], 'pow_cur', args)
        work_regions = read_work_region(lines)
    # update fvals
    fvals['gain'] = gain                        # dB
    fvals['ugain_freq'] = ugain_freq / 1e6      # MHz
    fvals['vhy'] = abs(hy_error) * 1e3               # mV
    fvals['v_offset'] = abs(offset) *1e3        # mV

    fvals['pow_vol'] = pow_vol
    fvals['pow_cur'] = pow_cur
    fvals['pow'] = pow_vol * 1e6                # uW
    fvals['work_regions'] = work_regions
    fvals['aux_info'] += f'The following mosfets are not in saturation region:\n '
    # fvals['metrics'] = {'gbw': gbw, 'gain': gain, 'cmrr': cmrr, 'pm': pm, 'pow': pow}
    # append (mosfet, work_region) to aux_info
    for mosfet, work_region in work_regions.items():
        fvals['aux_info'] += f'{mosfet}: {work_region}\n '
    # calculate fom
    fvals = objective(fvals, args)
    return point_to_evaluate, fvals

# main function
if __name__ == '__main__':
    # a two-stage opamp example, 8 transistors, 1 resistor, 1 capacitor

    point_to_evaluate = ".param w1=25u l1=2.5u w2=20u l2=2.5u w3=25u l3=2u w4=20u l4=2u w5=10u l5=2u w6=25u l6=2u w7=30u l7=2.5u w8=2u l8=2u r1=20k c1=0.2p"
    point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=200u l1=1u w2=200u l2=1u w3=20u l3=1u w4=20u l4=1u w5=15u l5=1u w6=15u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=250u l1=1u w2=250u l2=1u w3=25u l3=1u w4=25u l4=1u w5=20u l5=1u w6=20u l6=1u w7=250u l7=1u w8=25u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=300u l1=1u w2=300u l2=1u w3=30u l3=1u w4=30u l4=1u w5=20u l5=1u w6=20u l6=1u w7=200u l7=1u w8=20u l8=1u c1=2p r1=1k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=1.5k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=2k'
    point_to_evaluate = '.param w1=120u l1=1u w2=120u l2=1u w3=12u l3=1u w4=12u l4=1u w5=8u l5=1u w6=8u l6=1u w7=120u l7=1u w8=12u l8=1u c1=1.5p r1=800'
    point_to_evaluate = '.param w1=150u l1=1u w2=150u l2=1u w3=15u l3=1u w4=15u l4=1u w5=10u l5=1u w6=10u l6=1u w7=150u l7=1u w8=15u l8=1u c1=4p r1=800'

    args = {
        'ckt_dir': '/home/yuwang/Coding/LLMBO/amp2',
        'ckt_name': 'amp2',
        'gbw_norm': [1, 1e6],
        'gain_norm': [-20, 60],
        'cmrr_norm': [0, 80],
        'pm_norm': [0, 180],
        "pow_norm": [0, 1.5e-5],
        'gbw_weight': 0.25,
        'gain_weight': 0.25,
        'cmrr_weight': 0.25,
        'pm_weight': 0.25,
        "pow_weight": -0.25,
        'gbw_failed': 0.0,
        'gain_failed': 0.0,
        'pm_failed': 0.0,
        'acm_failed': 0.0,
    }
    point_to_evaluate, fvals = hspice_eval_f_comp(point_to_evaluate, args)
    print(f'point_to_evaluate: {point_to_evaluate}')
    print(f'fvals: {fvals}')

    print(f"gain={fvals['gain']:.3f} cmrr={fvals['cmrr']:.3f} gbw={fvals['gbw']:.3f} pm={fvals['pm']:.3f}")
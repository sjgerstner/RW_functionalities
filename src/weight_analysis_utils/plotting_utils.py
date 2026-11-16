import numpy as np

from .utils import (
    COMBO_TO_NAME, VANILLA_CATEGORIES,
    make_combo_name_dict, floats_to_strings,
)

# CATEGORY_COLORS = {
#     (1,1,1):(1,1,0,1),
#     (1,1,0):(1,1,0,.5),
#     (1,0,1):(.5,.5,0,1),
#     (1,0,0):(.5,.5,0,0.5),
#     (0,1,1):(0,0,0,1),
#     (0,1,0):(0.25,.25,0.25,0.5),#new: (0,0,0,.5) grey
#     (0,0,1):(0.9,0.9,0.9,1),#new: (0,.5,0,1) greenish
#     (-1,1,1):(0,0,1,1),
#     (-1,1,0):(0,0,1,.5),
#     (-1,0,1):(0,.5,.5,1),
#     (-1,0,0):(0,.5,.5,0.5),
# }

CATEGORY_COLORS = {
    key: (
        max(0,key[0]) * (key[1]+1) *.5,
        .5 if key[1]==0 else max(0,key[0]),
        max(0,-key[0]) * (key[1]+1)/2,
        (key[2]+1)/2,
    )
    for key in COMBO_TO_NAME
}
CATEGORY_COLORS[(0,0,1)]=(0.9,0.9,0.9,1)

VANILLA_COLORS = {
    1: CATEGORY_COLORS[(1,1,1)],
    0: CATEGORY_COLORS[(0,0,1)],
    -1: CATEGORY_COLORS[(-1,1,1)],
}


def make_full_key_list(key_list:list[float])->list[str]:
    old_len = len(key_list)
    step_size = min(abs(key_list[i+1]-key_list[i]) for i in range(old_len-1))
    float_list = np.linspace(-1,1, num = round(2/step_size), endpoint=False).tolist()
    return floats_to_strings(float_list)

def make_color_dict(category_keys:list[str])->dict[str, tuple[float,float,float,float]]:
    value_list = np.linspace(-1,1, num=len(category_keys))
    return {
        key: (
            (1-max(0,-value_list[i])),
            (1-max(0,-value_list[i])),
            (1-max(0,value_list[i])),
            abs(value_list[i])**0.5 if value_list[i]<0 else 1,
        )
        for i,key in enumerate(category_keys)
    }

def get_names(results:dict):
    if isinstance(next(iter(results)), tuple):
        names_and_colors, combo_to_name = CATEGORY_COLORS, COMBO_TO_NAME
        new_results={COMBO_TO_NAME[key]:value for key,value in results.items()}
    elif isinstance(next(iter(results)), int):
        names_and_colors, combo_to_name = VANILLA_COLORS, VANILLA_CATEGORIES
        new_results={VANILLA_CATEGORIES[key]:value for key,value in results.items()}
    else:
        new_results = {}
        #print(results.keys())
        key_list = list(results.keys())
        for key in key_list:
            new_results[f"{key:.2f}"] = results[key]#add string-formatted keys
        key_list = make_full_key_list(key_list)
        combo_to_name = make_combo_name_dict(key_list, make_string_keys=False)
        names_and_colors = make_color_dict(key_list)
    return names_and_colors, combo_to_name, new_results

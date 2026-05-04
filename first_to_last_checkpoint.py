from torch import save

from weight_analysis_utils import utils, plotting
from weight_analysis_utils.loading import load_model_data, MODEL_TO_CHECKPOINTS, load_data_if_exists

DEVICE = 'cuda:0'
MODEL_NAME = "allenai/OLMo-7B-0424-hf"
PATH = f"../RW_functionalities_results/results/{MODEL_NAME}"

print("loading other cosine data...")
full_data = load_data_if_exists(f'{PATH}/refactored')

if "W_in_start_to_end" not in full_data:
    print("loading weights of initial model...")
    initial_data = load_model_data(
        MODEL_NAME,
        checkpoint_value=MODEL_TO_CHECKPOINTS[MODEL_NAME][0],
        device='cpu',
        processing=False,
    )

    print("loading weights of final model...")
    final_data = load_model_data(
        MODEL_NAME,
        device='cpu',
        processing=False,
    )

    print("computing similarities between checkpoints...")
    start_to_end_sims = {}
    for key in initial_data.keys():
        if key=='d_model':
            continue
        print(key)
        start_to_end_sims[key] = utils.cos(
            initial_data[key].to(DEVICE),
            final_data[key].to(DEVICE)
        ).cpu()

    print("add to full data and save...")
    for key,value in start_to_end_sims.items():
        full_data[key+'_start_to_end'] = value
    save(full_data, f'{PATH}/refactored/data.pt')

#print(full_data.keys())

print("plotting...")
for start_to_end_key in ["W_gate", "W_in", "W_out"]:
    print(start_to_end_key)
    for wcos_key in ["gatelin", "gateout", "linout"]:
        print(wcos_key)
        plotting.plot_any_vs_any(
            data=full_data,
            keys=(start_to_end_key+'_start_to_end', wcos_key),#TODO switch keys
            arrangement=(8,4),
            savefile=f'{PATH}/refactored/{start_to_end_key}_change_vs_{wcos_key}.pdf'
        )

EXPERIMENT_LIST = [
    "beta",
    "randomness", #compute 95 percent randomness regions (95 percent of 'mismatched' weight cosines are in this region)
    "norms", #norms of weight vectors
    "categories", #categorize the neurons
    "category_stats",#compute statistics of RW classes by layer
    #"quartiles",#compute quartiles of cosine similarities (by layer)
    "plot_fine",#create fine-grained plot
    "plot_selected",
    "plot_coarse",#create coarse-grained plot (categories by layer)
    "make_table",
    "plot_boxplots",#make boxplots of cosine similarities by layer
    "plot_all_medians",#make one plot with the median cos(w_in,w_out) similarities (y) across layers (x) of all models (one line per model)
    "plot_selected_medians",
    "plot_norms",
    "plots_cosines_vs_norms",
    "plot_norm_in_norm_out",
    "plot_half_coarse",
    "half_coarse_table",
]
MODEL_LIST = [
    "allenai/OLMo-7B-0424-hf",
    "allenai/OLMo-1B-hf",
    "google/gemma-2-2b",
    "google/gemma-2-9b",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.1",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-7B",
    "01-ai/Yi-6B",
]
VANILLA_MODELS = [
    "gpt2",#gpt2-small
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "distilgpt2",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-7b1",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-cased",
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "Baidicoot/Othello-GPT-Transformer-Lens",
]

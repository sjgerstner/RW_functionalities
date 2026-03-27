from argparse import ArgumentParser
import nltk
from nltk.corpus import stopwords
import pandas as pd
from transformer_lens import HookedTransformer #TODO we just need the tokeniser!
from transformers import PreTrainedTokenizerFast

def clean_df(df_wiki:pd.DataFrame, model_or_tokenizer:HookedTransformer|PreTrainedTokenizerFast, stopwords0_:dict[str,str]):
    """
    From a dataframe of subjects and paragraphs, do tokenisation and deduplication.
    
    This function was largely extracted from
    https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions
    Copyright 2023 Google LLC.
    Licensed under the Apache License, Version 2.0 (the "License").

    Args:
        df_wiki (pd.DataFrame): The original dataframe. 2 columns and a header of column names "subject" and "paragraphs".
            Each entry should have (a) a subject (string) from the "knowns" data (knowns_df)
            and (b) paragraphs concatenated with space about the subject (a single string).
        model (HookedTransformer): this is relevant for tokenisation.
        stopwords0_ (dict[str,str]): dict of stopwords, usually the English stopwords from nltk.
        The keys are the stopwords, the values are irrelevant (legacy).

    Returns:
        pd.DataFrame: The cleaned dataframe (with tokenized and deduped attributes)
    """
    if isinstance(model_or_tokenizer, HookedTransformer):
        def tokenize(x:str)->list[str]:
            return list(set(model_or_tokenizer.to_str_tokens(x)))
    elif isinstance(model_or_tokenizer, PreTrainedTokenizerFast):
        def tokenize(x:str)->list[str]:
            return list(set(model_or_tokenizer.batch_decode(model_or_tokenizer.encode(x))))
    else:
        raise TypeError(
            f"model_or_tokenizer should be HookedTransformer or PreTrainedTokenizerFast, but is {type(model_or_tokenizer)}"
        )
    df_wiki = df_wiki.fillna('')
    # Tokenize, remove duplicate tokens, stopwords, and subwords.
    df_wiki["context_tokenized_dedup"] = df_wiki["paragraphs"].progress_apply(
        tokenize
    )
    df_wiki["context_tokenized_dedup_len"] = df_wiki.context_tokenized_dedup.apply(
        len#lambda x: len(x)
    )
    df_wiki["context_tokenized_dedup_no-stopwords"] = df_wiki.context_tokenized_dedup.apply(
        lambda x: [
            y for y in x
            if y.strip() not in stopwords0_ and len(y.strip())>2
        ]
    )
    df_wiki["context_tokenized_dedup_no-stopwords_len"] = df_wiki["context_tokenized_dedup_no-stopwords"].apply(
        len#lambda x: len(x)
    )
    return df_wiki

def clean_and_save_df(path:str, model_or_tokenizer:HookedTransformer|PreTrainedTokenizerFast, stopwords0_:dict[str,str], model_name="cleaned"):
    # This should be a path to a csv file
    # with 2 columns and a header of column names "subject" and "paragraphs".
    # Each entry should have (a) a subject (string) from the "knowns" data (knowns_df)
    # and (b) paragraphs concatenated with space about the subject (a single string).
    df_wiki = pd.read_csv(f'{path}/wiki.csv')
    df_wiki = clean_df(df_wiki, model_or_tokenizer, stopwords0_)
    df_wiki.to_pickle(f'{path}/wiki_{model_name}.pickle')
    return df_wiki

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--model_name', default='allenai/OLMo-1B-hf')
    args = parser.parse_args()
    # List of stopwords from NLTK, needed only for the attributes rate evaluation.
    nltk.download('stopwords')
    stopwords0_ = stopwords.words('english')
    stopwords0_ = {word: "" for word in stopwords0_}#TODO this was copied from Geva et al., but is it necessary?
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name)
    df_wiki = clean_and_save_df(path=args.data_dir, model_name=args.model_name, stopwords0_=stopwords0_, model_or_tokenizer=tokenizer)

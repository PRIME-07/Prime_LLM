import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input='data/lyrics_corpus.txt',
    model_prefix='data/lyrics_bpe',
    vocab_size=8000,
    model_type='bpe',
    character_coverage=0.9995,
    user_defined_symbols=[
        # control tokens
        "<prompt>",
        "<song>",
        "</song>",
        
        # emotion tokens
        "<happy>",
        "<sad>",
        "<angry>",
        "<romantic>",
        
        # genre tokens
        "<genre_pop>",
        "<genre_rap>",
        "<genre_rock>",
        "<genre_rb>",
    
        # structure tokens
        "[intro]",
        "[verse]",
        "[chorus]",
        "[bridge]",
        "[outro]",
    ],

)

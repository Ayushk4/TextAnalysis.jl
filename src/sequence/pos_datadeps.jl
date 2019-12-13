function pos_datadep_register()
    register(DataDep("POS Model Weights",
        """
        The weights for POS Sequence Labelling Model.
        """,
        "https://github.com/Ayushk4/POS.jl/releases/download/v0.0.1/pos_weights.tar.xz",
        "8c15163fdcd9f6f1882fbb20b2502a7cc729eab4152332b7959a3918452d3a03";
        post_fetch_method = function(fn)
            unpack(fn)
            dir = "pos_weights"
            innerfiles = readdir(dir)
            mv.(joinpath.(dir, innerfiles), innerfiles)
            rm(dir)
        end
    ))

    register(DataDep("POS Model Dicts",
        """
        The character and words dict for POS Sequence Labelling Model.
        """,
        "https://github.com/JuliaText/TextAnalysis.jl/releases/download/v0.6.0/pos_model_dicts.tar.xz",
        "4d7fe8238ff0cfb92d195dfa745b4ed08f916d4707e3dbe27a1b3144c9282f41";
        post_fetch_method = function(fn)
            unpack(fn)
            dir = "model_dicts"
            innerfiles = readdir(dir)
            mv.(joinpath.(dir, innerfiles), innerfiles)
            rm(dir)
        end
    ))
end

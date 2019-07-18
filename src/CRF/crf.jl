using Flux
using Flux: param, identity, onehot, onecold, @treelike

"""
Linear Chain - CRF Layer.

For input sequence `x`,
predicts the most probable tag sequence `y`,
over the set of all possible tagging sequences `Y`.

In this CRF, two kinds of potentials are defined,
emission and Transition.
"""
mutable struct CRF{S, F} # Calculates Argmax( log ∑ )
    W::S    # Transition Scores
    n::Int  # Num Labels
    f::F    # Feature function
end

"""
Second last index for start tag,
last one for stop tag .
"""
CRF(num_labels::Integer) = CRF(num_labels::Integer, identity)

function CRF(n::Integer, f::Function; initW = rand)
    W = initW(n + 2, n + 2)
    W[:, n + 1] .= -10000
    W[n + 2, :] .= -10000
    # W[n + 1, n + 1] = 0
    # W[n + 2, n + 2] = 0

    return CRF(param(W), n, f)
end

function Base.show(io::IO, c::CRF)
    print(io, "CRF with `", c.n, "` distinct tags and feature function `", c.f,"`")
end

function (a::CRF)(x_seq)
    viterbi_decode(a, x_seq)
end

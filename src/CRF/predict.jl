# Decoding is done by using Viterbi Algorithm
# Computes in polynomial time

"""
Scores for the first tag in the tagging sequence.
"""
function preds_first(c::CRF, y)
    c.W[c.n + 1, onecold(y, 1:length(y))]
end

function preds_last(c::CRF, y)
    c.W[onecold(y, 1:length(y)), c.n + 2]
end

"""
Scores for the tags other than the starting one.
"""
function preds_single(c::CRF, y, y_prev)
    c.W[onecold(y, 1:length(y)), onecold(y_prev, 1:length(y_prev))]
end

# Helper for forward pass, returns max_probs and corresponding arg_max for all the labels
function forward_unit_max(a::CRF, x, prev)
    p = preds_single(a, x) .+ an
    m = maximum(p, dims=1)
    return m, [i[1] for i in indexin(m,p)]
end

"""
Computes the forward pass for viterbi algorithm.
"""
function forward_pass(c::CRF, x)
    n = length(x)
    α = preds_first(c, x[1])
    α_idx = [zeros(Int, size(c.s, 2)) for i in 1:length(x)]

    for i in 2:length(x)
        α, α_idx[i] = forward_unit_max(c, x[i], α)
    end

    return findmax(α)[2], α_idx
end

"""
Computes the backward pass for viterbi algorithm.
"""
function backward_pass(α_idx_last, α_idx)
    labels = Array{Flux.OneHotVector, 1}(undef, size(α_idx,1))
    labels[end] = α_idx_last

    for i in reverse(2:size(α_idx,1))
        labels[i-1] =  α_idx[i, labels[i]]
    end

    return reverse(labels)
end

"""
    viterbi_decode(::CRF, input_sequence)

Predicts the most probable label sequence of `input_sequence`.
"""
function viterbi_decode(c::CRF, x_seq)
    size(x_seq,1) == 0 && throw("Input sequence is empty")
    α_star, α_max = backward_pass(forward_pass(c, x_seq)...)
end

function predict(c::CRF, x_seq)
    viterbi_decode(c, x_seq)
end

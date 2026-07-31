module MLUtils

using Statistics

export standardise,
       destandardise,
       standardise_matrix,
       predict_linear,
       mse

"""
    standardise(x)

Z-score standardise a vector.
Returns (x_std, μ, σ).
"""
function standardise(x)
    μ = mean(x)
    σ = std(x)
    x_std = (x .- μ) ./ σ
    return x_std, μ, σ
end

"""
    destandardise(x_std, μ, σ)

Undo z-score standardisation.
"""
destandardise(x_std, μ, σ) = μ .+ σ .* x_std

"""
    standardise_matrix(X_raw)

Standardise every column of a DataFrame and return
a Float64 matrix.
"""
function standardise_matrix(X_raw)
    X = Matrix{Float64}(undef, size(X_raw))

    for (i, column) in enumerate(eachcol(X_raw))
        X[:, i], _, _ = standardise(column)
    end

    return X
end

"""
    predict_linear(X, w, b)

Linear prediction: Xw + b
"""
predict_linear(X, w, b) = X * w .+ b

"""
    mse(ŷ, y)

Mean squared error.
"""
mse(ŷ, y) = mean((ŷ .- y).^2)
"""
    binary_cross_entropy(ŷ, y)

Compute the mean binary cross-entropy (BCE) loss between predicted
probabilities `ŷ` and binary labels `y`.

The predictions are clipped to the interval `[ϵ, 1-ϵ]` to avoid
numerical issues caused by `log(0)`.

Returns a single scalar representing the average loss over all samples.
"""
function binary_cross_entropy(ŷ, y)
    ϵ = 1e-15
    ŷ = clamp.(ŷ, ϵ, 1 - ϵ)

    return -mean(
        y .* log.(ŷ) .+
        (1 .- y) .* log.(1 .- ŷ)
    )
end
"""
    sigmoid(Z::Matrix)

Apply the sigmoid activation function elementwise.
"""
function sigmoid(Z::Matrix)
    return 1.0 ./ (1.0 .+ exp.(-Z))
end


"""
    softmax(Z::Matrix)

Apply the softmax activation function to each row of `Z`.

Each row is treated as a separate sample. The row maximum is
subtracted before exponentiation to improve numerical stability.
"""
function softmax(Z::Matrix)
    Z = Z .- maximum(Z, dims=2)
    expZ = exp.(Z)

    return expZ ./ sum(expZ, dims=2)
end


"""
    categorical_cross_entropy(Ŷ::Matrix, Y::Matrix)

Compute the mean categorical cross-entropy loss between the
predicted probabilities `Ŷ` and the one-hot encoded labels `Y`.
"""
function categorical_cross_entropy(Ŷ::Matrix, Y::Matrix)
    ϵ = 1e-15
    Ŷ = clamp.(Ŷ, ϵ, 1 - ϵ)

    return -mean(sum(Y .* log.(Ŷ), dims=2))
end
end

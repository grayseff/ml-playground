include(joinpath(@__DIR__, "src","utils.jl"))
using .MLUtils

using CSV, DataFrames,Statistics, RDatasets
using Random,LinearAlgebra 

Random.seed!(42)

iris = dataset("datasets","iris")


#This is the iris dataset, with columns SepalLength, Width, PetalLength Width Species
#
#We will train a model to classify the following by species.
#Start with setosa/not setosa
#
y = iris.Species .== "setosa"
X_raw = select(iris,Not(:Species))
X = standardise_matrix(X_raw)
y = Float64.(y)


# classification into binary means sigmoid function 
#
sigmoid(z) = 1 ./(1 .+ exp.(-z))
# Now we define the predict function for this case:
# X is a 150x4 matrix, 150 flowers, 4 datapoints, 
#Features -> linearise -> sigmoid -> probability 

#
function predict(X,w,b)
	z = X*w .+ b 
	return sigmoid(z)
end

# Note, loss is not related to mse now, it's how close is our sigmoid probability to the truth.
function binary_cross_entropy(ŷ,y)
	#to prevent log(0) problem 
	ϵ = 1e-15
	ŷ=clamp.(ŷ,ϵ,1-ϵ)
	return -mean( y .* log.(ŷ) .+ (1 .-y).* log.(1 .-ŷ))
end
# the gradient works out to 1/nX'(ŷ-y) and mean( ŷ - y )
#
function gradient(difference,X)
	dw = (1/length(difference)) * X' * difference 
	db = mean(difference)
	return dw,db 
end
w_i = randn(4)
b_i = rand()

ŷ_i = predict(X,w_i,b_i)
l_i = binary_cross_entropy(ŷ_i,y)
η = 0.001
w,b = w_i,b_i 
losses = [] 

for epoch in 1:5000
	ŷ = predict(X,w,b)
	loss = binary_cross_entropy(ŷ,y)
	push!(losses,loss)

	pred = ŷ.>=0.5
	accuracy = mean(pred .== y)
	diff = ŷ.-y
	dw,db = gradient(diff,X)
	w .-= η.*dw 
	b -= η*db 
	if epoch %100 == 0 
		println("Epoch $epoch: loss is $loss, Accuracy = $(accuracy*100)%")
	end
end
using Plots 
plot(losses)
savefig("plots/loss.png")

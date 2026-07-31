include(joinpath(@__DIR__,"utils.jl"))
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
# using Plot    s
using GLMakie
GLMakie.activate!()
fig = Figure()
ax = Axis(fig[1,1])
lines!(ax, losses)
# savefig(fig, "plots/loss.png")
display(fig)

#--------------------------------------------------#
#------------For Ternary classification------------#
#--------------------------------------------------#

Y = Matrix{Float64}(undef, nrow(iris),3)
for (i,species) in enumerate(iris.Species)
	if species == "setosa"
		Y[i,:] = [1. , 0. , 0. ]
	elseif species == "versicolor"
		Y[i,:] = [0. , 1. , 0. ]
	elseif species == "virginica"
		Y[i,:] = [0. , 0. , 1. ]
	else
		println("Could not classify $i")
	end
end
# We need the softmax as the replacement for the sigmoid for three-classification
function softmax(Z)
	Z = Z .- maximum(Z,dims=2)
	expZ = exp.(Z)
	return expZ ./ sum(expZ,dims=2)
end
#predict now depends on the softmax function


function predict(X::Matrix,W::Matrix,b)
	Z = X*W .+b
	return softmax(Z)
end
#categorical_cross_entropy replaces binary_cross_entropy where one-hot encoding filters, the clamp prevents inf, sum across rows to return loss
function categorical_cross_entropy(Ȳ , Y)
	ϵ = 1e-15
	Ȳ = clamp.(Ȳ,ϵ,1-ϵ)

	return -mean(sum( Y .* log.(Ȳ),dims=2 ))
end
function gradient(difference::Matrix,X::Matrix )
	n = size(X,1)

	dW = X' * difference/n
	dB = mean(difference,dims=1)
	return dW,dB
end
W_i = randn(4,3)
B_i = randn(1,3)
Ȳ_i = predict(X,W_i,B_i )
L_i = categorical_cross_entropy(Ȳ_i , Y )
W,B = W_i,B_i
losses = []
η = 0.01




for epoch in 1:5000
	Ŷ = predict(X,W,B )
	loss = categorical_cross_entropy(Ŷ,Y)
	push!(losses,loss)
	pred = argmax.(eachrow(Ŷ))
	truth = argmax.(eachrow(Y))
	accuracy = mean(pred .== truth)
	diff = Ŷ .- Y
	dW,dB = gradient(diff,X)
	W .-= (η.*dW)
	B .-= (η.*dB)
	if epoch % 100 == 0
		println("Epoch $epoch: loss is $loss accuracy is $accuracy")
	end
end

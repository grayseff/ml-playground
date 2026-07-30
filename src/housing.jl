using CSV, DataFrames,Statistics, RDatasets
using Random,LinearAlgebra 
Random.seed!(42)

housing = dataset("MASS","Boston")
first(housing,5)


y_raw = housing.MedV #housing values
X_raw = housing[:,Not(:MedV)] #data regarding housing


any(ismissing,eachcol(housing))

function standardise(xin)
	μ = mean(xin)
	σ = std(xin)

	x_std = (xin .- μ)./σ 

	return x_std,μ,σ 
end
destandardise(x_std,μ,σ) = μ + σ*x_std 

X = Matrix{Float64}(undef,size(X_raw))
for (i,column) in enumerate(eachcol(X_raw))
	X[:,i],μ,σ = standardise(column)
end
# We now have a 506x13 matrix and a 506element vector to perform regression on
#
function mse(ŷ,y)
	return mean((ŷ .- y).^2)
end

# basic premise is predict -> loss -> grad -> update prediction 
# This remains y = Xw + b as linear regression 

function predict(X,w,b) 
	return X* w .+ b 
end

w_i = randn(13)
b_i = 0. 

# now we train: 
function gradients(X,ŷ,y)
	residual = ŷ .- y 
	dw= (2/length(y)) * X'  * residual
	db = 2*mean(residual) 
	return dw,db
end

w,b = w_i,b_i 
l_i = mse(predict(X,w,b),y_raw )
# println("Initial values are $w and $b with loss of $l_i")
η = 0.01
losses = Float64[]
for epoch in 1:5000
	ŷ = predict(X,w,b)
	loss = mse(ŷ,y_raw)
	dw,db = gradients(X,ŷ,y_raw)
	w = w .- (dw.*η)
	b -= db*η
	push!(losses,loss)
	if epoch % 100 == 0
		println("Epoch $epoch: loss = $loss")
	end

end
# println("final values are $w, $b with a loss of $(mse(predict(X,w,b),y_raw))")	
#--------------------------------------#
# Or direct linear regression: #
 

X_aug = hcat(ones(size(X,1)),X)

θ = X_aug\y_raw 
println("b = $(θ[1])")
println("w = $(θ[2:end])")
#with comparison of the methods:
#
wdiff = norm(w .- θ[2:end])
println(norm)
bdiff = abs(b .- θ[1])
println(wdiff," ",bdiff)

#-----------------------------------#
#------Generate synthetic data------#
#-----------------------------------#

using Random,Plots,Statistics
Random.seed!(42)
# We will fake data where 2br apartments are 3*sqm+min 
# sqm only range from 60-200m^2 in this case 
# base land value is 150,000 building value is 4000/sqm 
# working in thousands 
fakeprice(ix) = 4 * ix + 150 

sqm = rand( 60:200,1000 ) #sqm of house 

# functions to massage data into nice numbers
normalise(x) = x./maximum(x)
minmax(x) = (x .- minimum(x)) ./ (maximum(x) - minimum(x))
standardise(x) = ( x .- mean(x)) ./ std(x) 
destandardsie(x_std,μ,σ) = μ + σ*x_std


price = fakeprice.(sqm) #prices of house 

price = price .+ 20 .*randn(length(price) ) #add noise 

scatter(sqm, price;
    title="Synthetic Housing Data",
    xlabel="Feature",
    ylabel="Target")


savefig("plots/synthetic_data.png")

#create the normalised target x 
x = standardise(sqm)
μ,σ = mean(sqm) , std(sqm)

#-------------------------------------#
#-------------------------------------#

# knowing the data is linear, we make a bad initial guess for ŷ = wx + b 

wi = rand()
bi = rand() 

function predict(x, w , b ) 
	w *x + b
end

ŷ = predict.(x,wi,bi)


function mse(y_true, y_pred)
	mean((y_true .- y_pred).^2 )
end
loss = mse(price,ŷ)
println(loss)


 
#-------------------------#
# this is the basic premise, now we need to add the predictive step
# calc mse, improve w,b 

#first calc gradient, we do this numerically for practice
# This won't work numerically unless x is in -1:1 so we scale by max(x) 
#


function gradients(x,y,w,b)
	ϵ = 1e-6 
	loss_1 = mse(y,predict.(x,w,b))
	loss_2w = mse(y,predict.(x,w+ϵ,b))
	loss_2b = mse(y,predict.(x,w,b+ϵ))

	return (loss_2w -loss_1)/ϵ , (loss_2b-loss_1)/ϵ
end
w,b = wi,bi	
println( wi,bi )
η = 0.001
initial_loss = mse(price,predict.(x,w,b))
for epoch in 1:1000 
	dw,db = gradients(x,price,w,b)
	# println(dw,db)

	w -= η * dw 
	b -= η * db 
	# println(mse(price,predict.(x,w,b)))
end
final_loss = mse(price,predict.(x,w,b))
println(initial_loss," ",final_loss," ",w," ",b)
#-------------------------------------------#
#Using actual gradients
#for linear regression this means dw = 2*mean((yhat .- y).*x) and db = 2*mean(yhat .- y)
#
function gradients_analytical(x,y,w,b)
	ŷ = predict.(x,w,b)

	dw = 2*mean((ŷ .- y).* x)
	db = 2*mean(ŷ .- y)
	return dw,db
end
w,b = wi,bi	
println( wi,bi )
η = 0.001
initial_loss = mse(price,predict.(x,w,b))
for epoch in 1:1000 
	dw,db = gradients_analytical(x,price,w,b)
	# println(dw,db)

	w -= η * dw 
	b -= η * db 
	# println(mse(price,predict.(x,w,b)))
end
final_loss = mse(price,predict.(x,w,b))
println(initial_loss," ",final_loss," ",w," ",b)


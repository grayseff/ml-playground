using MLDatasets, Statistics,  Random, LinearAlgebra, Flux
# include("utils.jl")
# using .MLUtils
#instantiate the dataset
mnist_train = MNIST(split=:train)
mnist_test = MNIST(split=:test)
#Split into features and targets
X_train = mnist_train.features
y_train = mnist_train.targets
X_test = mnist_test.features
y_test = mnist_test.targets
#Reshape features to be 2D
X_train = reshape(X_train, :, size(X_train, 3))
X_test = reshape(X_test, :, size(X_test, 3))
#----------------------------------#
#----- Developing the model -------#
# ---------------------------------#

#here we determine the architecture of the model:
# We choose ReLU and 724->128->10 outputs
model = Chain(
    Dense(784 => 128, relu),
    Dense(128 => 10)
)
# and the categorical cross entropy to turn 0->9 digits into probabilities
loss(model,x, y) = Flux.logitcrossentropy(model(x), y)
#but logit cross entropy requires logits, not integers as y_train is composed on
y_train = Flux.onehotbatch(y_train, 0:9)
y_test = Flux.onehotbatch(y_test, 0:9)




#----------------------------------#
#----- Training the model ---------#
#----------------------------------#
opt = Flux.setup(Flux.Adam(), model)

train_loader = Flux.DataLoader((X_train, y_train), batchsize=68, shuffle=true)
test_loader = Flux.DataLoader((X_test, y_test), batchsize=68, shuffle=true)
for epoch in 1:5
    for (x, y) in train_loader
        Flux.train!(loss, model, train_loader, opt)
    end
    println("Epoch $epoch, loss: $(round(loss(model,X_test, y_test), digits=4))")
    ȳ = model(X_test)
    predictions = Flux.onecold(ȳ, 0:9)
    truth = Flux.onecold(y_test, 0:9)
    accuracy = mean(predictions .== truth)
    println("Accuracy: $accuracy")
end
#----- Evaluating the model -------#


using Images
using Flux: onehotbatch, crossentropy, softmax
using Flux.Data: DataLoader
using Flux
using Base.Iterators: repeated, partition

function count_images(folder_path)
    return length(readdir(folder_path))
end

function load_images(path)
    @time begin
        num_files = sum(count_images("$path/$label") for label in readdir(path))
        X = zeros(Float32, 28, 28, 1, num_files)
        y = Int64[]
        
        index = 1
        for label in readdir(path)
            for file in readdir("$path/$label")
                img = load("$path/$label/$file")
                img_resized = imresize(img, (28, 28))
                img_gray = Gray.(img_resized)
                data = reshape(Float32.(channelview(img_gray)), 28, 28, 1)
                X[:, :, :, index] .= data
                push!(y, parse(Float32, label) == 1.0 ? 1 : 2)
                index += 1
            end
        end
    end
    
    return X, y
end

@time begin
    path = "trening"
    x_train, y_train = load_images(path)
    path2 = "testing"
    x_test, y_test = load_images(path2)
end

train_cats_dir = "trening/2"
train_dogs_dir = "trening/1"
validation_cats_dir = "testing/2"
validation_dogs_dir = "testing/1"

# Print the number of all images
println("Total training cat images: ", count_images(train_cats_dir))
println("Total training dog images: ", count_images(train_dogs_dir))
println("Total validation cat images: ", count_images(validation_cats_dir))
println("Total validation dog images: ", count_images(validation_dogs_dir))

# Network model 
model = Chain(
    Conv((5, 5), 1 => 6, leakyrelu),
    MaxPool((2, 2)),
    Conv((5, 5), 6 => 16, leakyrelu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, leakyrelu),
    Dropout(0.5),
    Dense(120 => 84, leakyrelu),
    Dropout(0.5),
    Dense(84 => 2),
    softmax
)

# Function to measure the model accuracy
function accuracy()
    correct = 0
    for index in 1:length(y_test)
        probs = model(Flux.unsqueeze(x_test[:, :, :, index], dims=4))
        predicted_digit = argmax(probs)[1] == 1 ? 1 : 2
        if predicted_digit == y_test[index]
            correct += 1
        end
    end
    return correct / length(y_test)
end

# Reshape the data
x_train = reshape(x_train, 28, 28, 1, :)
x_test = reshape(x_test, 28, 28, 1, :)

# Assemble the training data
train_data = DataLoader((x_train, onehotbatch(y_train, 1:2)), shuffle=true)

# Initialize the ADAM optimizer with default settings
optimizer = Flux.setup(ADAM(0.000094), model)

# Define the loss function
function loss(model, x, y)
    loss_value = Flux.crossentropy(model(x), y)
    return loss_value
end

# Train model 20 times in a loop
for epoch in 1:20
    @time begin
        Flux.train!(loss, model, train_data, optimizer)
        println("Epoch: $epoch, Loss: $(loss(model, x_train, onehotbatch(y_train, 1:2)))")
        println("Accuracy: $(accuracy())")
    end
end

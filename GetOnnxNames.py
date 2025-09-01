import onnx

model = onnx.load("pokemon_prediction_model/1/pokemon_model.onnx")
output = [node.name for node in model.graph.output]
input_all = [node.name for node in model.graph.input]

input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)

print('Input all: ', input_all)
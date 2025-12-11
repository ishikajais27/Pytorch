import numpy as np


class Neuron:
    """A single neuron - the smallest thinking unit"""
    
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs) * 0.1   # initial wight 
        self.bias = 0.0

    def think(self, inputs):    #predict the output using weight,bias and activation function
        inputs = np.array(inputs)       
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        return self._sigmoid(weighted_sum)
    
    def _sigmoid(self, x):  #calculate sigmoid_ (activation function) ,loss function
        return 1 / (1 + np.exp(-x))
    
    def train(self, inputs, correct_answer, learning_rate=0.1):
        inputs = np.array(inputs)      
        
        prediction = self.think(inputs)   #get the predicted output from above function
        error = prediction - correct_answer  
        
        sigmoid_derivative = prediction * (1 - prediction)  
        gradient = error * sigmoid_derivative  #calculation of gradient derivative of loss function wrt parameters
        
        self.weights -= learning_rate * gradient * inputs  #optimization 
        self.bias -= learning_rate * gradient  #optimization
        
        return error ** 2

def teach_neuron_spam():
    print("\n" + "="*60)
    print("TEACHING NEURON TO RECOGNIZE SPAM")
    print("="*60)
    
    neuron = Neuron(num_inputs=3)
    
    training_data = [
        ([1, 1, 1], 1),
        ([1, 0, 0], 1),
        ([0, 1, 0], 0),
        ([0, 0, 0], 0),
        ([1, 1, 0], 1),
    ]
    
    for round in range(1000):
        total_error = 0
        for inputs, is_spam in training_data:
            total_error += neuron.train(inputs, is_spam)
        
        if round % 200 == 0:
            print(f"Round {round}: Avg error = {total_error/len(training_data):.4f}")
    
    print("\nTEST RESULTS")
    print("-" * 60)
    
    tests = [
        ([1, 1, 1], "SPAM - free money from unknown"),
        ([1, 0, 1], "SPAM - free from unknown"),
        ([0, 0, 1], "UNKNOWN SENDER"),
        ([0, 0, 0], "NOT SPAM - clean email")
    ]
    
    for inp, desc in tests:
        conf = neuron.think(inp)
        print(f"{desc}: {conf:.2f}")

teach_neuron_spam()

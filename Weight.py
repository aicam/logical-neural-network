import numpy as np
import random as rnd
import gc
import time

class Layer:
    weights = []
    training_weights = []
    changing_layers = []
    change_rate = 0

    def change_training_weight(self,training_weights,changing_layers):
        self.training_weights = training_weights
        self.changing_layers = changing_layers

    def addWeight(self, i):
        return self.weights.append(i)

    def layers_len(self):
        return len(self.weights)

    def generate_random_weights(self, length, layer_number, weight_number):
        if (layer_number in [item[0] for item in self.changing_layers]):
            changing_weight_numbers = 0
            previous_weight = []
            changing_indexes = []
            weight_index = 0
            for item in self.changing_layers:
                if item[0] == layer_number:
                    changing_weight_numbers = item[1]
            for weight in self.training_weights:
                if layer_number == weight[0] and weight_number == weight[1]:
                    previous_weight = weight[2]
                    continue
                weight_index += 1
            while len(changing_indexes) < changing_weight_numbers:
                index = rnd.randint(0, length - 1)
                if (index not in changing_indexes):
                    changing_indexes.append(index)
                    previous_weight[index] = (rnd.randint(0, 1)) if index == 0 else rnd.randint(2, 6)
            self.training_weights[weight_index][2] = previous_weight
            return previous_weight

        else:
            random_weight = []
            for i in range(length - 1):
                random_weight.append(rnd.randint(2, 6))
            random_weight = np.array(random_weight)
            np.random.shuffle(random_weight)
            random_weight = np.concatenate([np.array([rnd.randint(0, 1)]), random_weight])
            if layer_number not in [item[0] for item in self.training_weights] and weight_number not in [item[1] for item in self.training_weights]:
                self.training_weights.append([layer_number, weight_number, random_weight])
            return random_weight


class AliTrainer():
    layer = None
    accuracy = 80
    batch_size = 128
    data = []
    data_size = 0
    answers = []
    batch_loss = 1
    batch_counter = 0

    def __init__(self, layer, accuracy, batch_size, datasize):
        self.layer = layer
        self.accuracy = accuracy
        self.batch_size = batch_size    
        self.data_size = datasize
        self.batch_counter = self.batch_size

    @staticmethod
    def logical_operation(operand, x, y):
        return {
            2: np.bitwise_and(x,y),
            3: np.bitwise_or(x,y),
            4: np.bitwise_and(x,np.bitwise_not(y)),
            5: np.bitwise_or(x,np.bitwise_not(y)),
            6: x
        }[operand]
    def feed_data(self, data, ans):
        if (str(type(ans[0])) != "<class 'list'>"):
            exit("answers must be at list shape [None,1]")
        if (len(data) != len(ans)):
            exit("answers and datas should be same size")
        if (len(data) % self.batch_size != 0):
            exit("group data length should be a multiplied of batch size")
        self.data.append(data)
        self.answers.append(ans)

    def calculate_layer(self, i, data):
        new_data = []
        weights = []
        #this condition happens because weights shouldn't change in a batch of data
        if(self.batch_counter == self.batch_size):
            weights = [self.layer.generate_random_weights(self.layer, length=len(data), layer_number=i, weight_number=k) for k
                    in range(self.layer.weights[i])]
        elif(self.batch_counter != self.batch_size):
            for k in range(self.layer.weights[i]):
                for item in self.layer.training_weights :
                    if item[0] == i and item[1] == k :
                        weights.append(item[2])
        for weight in weights:
            new_data.append(AliTrainer.calculate_weights_data(data, weight))
        return new_data, weights

    @staticmethod
    def calculate_weights_data(data, weights):
        last_digit = 1
        # initialize first operation out of loop
        last_digit = data[0] if (weights[0] == 0) else not data[0]
        for i in range(1, len(data)):
            last_digit = AliTrainer.logical_operation(weights[i], last_digit, data[i])
            
        return last_digit

    # this method is defualt loss calculation , any other methods can be implemented
    # the result of method should be loss of a single batch size data and answers
    @staticmethod
    def calculate_loss_with_percent(size, predicted, answers):
        loss_simple_array = []
        for i in range(size):
            loss_simple_array.append(np.abs(np.bitwise_xor(np.array(predicted[i]), np.array(answers[i]))))
        loss_ndarray = np.array(loss_simple_array)
        return np.sum(loss_ndarray) / (size * len(predicted[0]))



    def train(self, loss):
        layers_count =  self.layer.layers_len(self.layer)
        changeable_layers = int(np.floor(loss * layers_count)) if layers_count > 1 else 1
        self.layer.changing_layers = []
        while len(self.layer.changing_layers) < changeable_layers:
            layer_number = rnd.randint(0, layers_count - 1)
            if layer_number not in self.layer.changing_layers:
                self.layer.changing_layers.append([layer_number, (int(np.floor(self.data_size * loss)))
                if layer_number == 0 else (int(np.floor(loss * self.layer.weights[layer_number])))])


    def loss(self):
        now = time.time()
        logical_calculation = 0
        while len(self.data) > 0:
            gc.collect()
            result = []
            batch_data = self.data.pop()
            batch_answer = self.answers.pop()
            batch_weights = []
            for cn in range(self.batch_size):
                raw_data = batch_data[cn]
                logical_nn_weights_by_layer = []
                # get first layer result for repeat
                new_data, weights = self.calculate_layer(0, raw_data)
                logical_nn_weights_by_layer.append(weights)
                for k in range(1, Layer.layers_len(self.layer)):
                    new_data, new_weight = self.calculate_layer(k, new_data)
                    logical_nn_weights_by_layer.append(new_weight)
                result.append(new_data)
                self.batch_counter = self.batch_size if len(result) < self.batch_size else 1 if self.batch_counter == self.batch_size else  self.batch_counter + 1
            self.batch_loss = self.calculate_loss_with_percent(self.batch_size, result, batch_answer)
            self.train(loss=self.batch_loss)
            
            if self.batch_loss <= 1 - self.accuracy:
                diff = time.time() - now
                minutes, seconds = diff // 60, diff % 60
                print('time left : ' + str(minutes) + ':' + str(seconds))
                print(self.batch_loss," weight:",logical_nn_weights_by_layer," data :",batch_data," answer:",result," correct answer: ",batch_answer)
                exit("training finished :)")

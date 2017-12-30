function [z] = sigmoid(z)
%The sigmoid function
z = exp(-z);
z = 1+z;
z = 1./z;
end



function [z] = sigmoid_prime(z)
%Derivative of the sigmoid function.
z = 1-sigmoid(z);
z = z.*sigmoid(z);
end


function [ret] = cost_derivative(output_activations, y)

%"""Return the vector of partial derivatives \partial C_x /
%        \partial a for the output activations."""
       ret = (output_activations.-y);
end


function [a] = feedforward(a)
%   If a is the input to the network, what would be the output ?
%         global psizes;
%         global biases;
%         global weights;
        y=psizes(2:end);
        for i =1:numel(y)
        z = weights(i).weight*a + biases(i).bias;
        a = sigmoid(z);
        end

end

function [n_test_score] = evaluate(test_data)

%        """Return the number of test inputs for which the neural
%        network outputs the correct result. Note that the neural
%        network's output is assumed to be the index of whichever
%        neuron in the final layer has the highest activation."""
        n_test_score=0;
         
        for i=1:numel(test_data)
            output_x= feedforward(test_data(i).x);
            predicted_y = max_value(output_x);
                        
            if (predicted_y == test_data(i).y)
            n_test_score +=1;
            end
        endfor

end

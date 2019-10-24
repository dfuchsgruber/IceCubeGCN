import torch
import torch.nn as nn

class RecurrentNeuralNetwork(nn.Module):
    """ Standard Recurrent Neural Network using LSTMs and self-attention. """

    def __init__(self, num_input_features, num_classes, hidden_size, num_layers, dropout_rate=0.5):
        """ Creates a RNN model to parse event hits. 
        
        Parameters:
        -----------
        num_input_features : int
            Number of features for the input.
        num_classes : int
            The number of different classes.
        hidden_size : int
            Dimensionality of the hidden state of each LSTM.
        num_layers : int
            Number of layers to the stacked LSTM.
        droput_rate : float
            Dropout rate.
        """
        super().__init__(**kwargs)
        self.number_classes = num_classes
        if dropout_rate is None:
            dropout_rate = 0
        self.number_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(num_input_features, hidden_size, num_layers=num_layers, dropout=dropout_rate, bidirectional=True)
        self.logistic_regression = nn.Linear(num_input_features, num_classes)
        

    def forward(self, X):
        """ Forward pass.
        
        Parameters:
        -----------
        X : torch.FloatTensor, shape [sequence_length, batch_size, D]
            The DOM hits.

        Returns:
        --------
        y : torch.FloatTensor, shape [batch_size, 1]
            Soft class labels.
        """
        # Initialize LSTM states
        batch_size = X.size()[-2]
        h_0 = Variable(torch.zeros(2 * self.number_layers, batch_size, self.hidden_size).cuda())
		c_0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).cuda())

        # Stacked LSTM
        output, (h_n, c_n) = self.lstm(X, (h_0, c_0))
        

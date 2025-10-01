import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC



class Bert_SVM_Model(object):
    def __init__(self, base_model, svm_shared_layers, device):
        self.base_model = base_model
        self.svm_shared_layers = svm_shared_layers
        self.device = device
    
    # Function to extract features using TinyBert Model
    # Extract representations/features from the base model (without the final classification layers)
    def extract_features(self, input_ids_batch, attention_mask_batch):
        output_batch = self.base_model(input_ids_batch, attention_mask_batch)
        CLS_token_batch = output_batch.last_hidden_state[:,0,:]  # Shape = [batch_size, hidden_dim]

        return CLS_token_batch
    
    # Functions to train and test SVM model

    def train_svm(self, X_train, y_train):
        self.svm_shared_layers.fit(X_train, y_train)

        # Predictions (raw decision function, not classes)
        scores = self.svm_shared_layers.decision_function(X_train)

        # Hinge loss
        hinge_losses = np.maximum(0, 1 - y_train * scores)
        total_loss = 0.5 * np.sum(self.svm_shared_layers.coef_ ** 2) + self.svm_shared_layers.C * np.sum(hinge_losses)

        #print("Hinge losses:", hinge_losses)
        #print("Total SVM loss:", total_loss)

        avg_loss = total_loss / len(y_train)
        print("Average loss:", avg_loss)

        return avg_loss

    def evaluate_svm(self, test_features):
        predictions = self.svm_shared_layers.predict(test_features)
        return predictions
    
    
    def get_model_parameters(self):
        """Returns the parameters of a sklearn model."""

        model = self.svm_shared_layers
        
        if model.fit_intercept:
            params = {
                'coef': model.coef_,
                'intercept': model.intercept_, # bias
            }
        else:
            params = {
                'coef': model.coef_,
            }
        return params

    def set_model_params(self, params):  
        """Sets the parameters of a sklean model."""
        self.svm_shared_layers.coef_ = params['coef'] # since it only consists of one layer
        if self.svm_shared_layers.fit_intercept:
            self.svm_shared_layers.intercept_ = params['intercept']



def initialize_model(base_model, svm_shared_layers, device):
    model = Bert_SVM_Model(base_model, svm_shared_layers, device)
    return model
# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
import torch
import copy
from tqdm import tqdm
from utils.initilaize_model import initialize_model
from sklearn.metrics import classification_report, accuracy_score


class Client():

    def __init__(self, id, train_loader, test_loader, base_model, svm_shared_layers, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(base_model, svm_shared_layers, device)
        self.receiver_buffer = {}
        self.epoch = 0   # record local update epoch
        self.clock = []  # record the time

    def local_update(self, num_iter, device):
        all_features = []
        all_labels = []

        itered_num = 0
        loss = 0.0
        end = False
        
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            with torch.no_grad():
                loop = tqdm(self.train_loader, leave=True)

                for batch in loop:
                    # Step 1: Move inputs to the same device as the base_model (i.e., CUDA device)
                    input_ids_batch, attention_mask_batch, labels_batch = batch['input_ids'].to(device, non_blocking=True), batch['attention_mask'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True) 

                    # Step 2: Forward pass (no gradient needed)
                    CLS_token_batch = self.model.extract_features(input_ids_batch, attention_mask_batch)
                    
                    all_features.append(CLS_token_batch.cpu())
                    all_labels.append(labels_batch.cpu())

                    itered_num += 1
                    if itered_num >= num_iter: # end of local updates 
                        end = True
                        self.epoch += 1

                        # Step 3: Stack all the mini-batches into one tensor of shape 
                        # torch.Size([number of samples = num of batches * BATCH_SIZE, 768 = hidden size of tokenized input])
                        all_features = torch.cat(all_features, dim=0)
                        all_labels = torch.cat(all_labels, dim=0)

                        # Step 4: Convert to NumPy
                        train_features = all_features.numpy()
                        train_labels = all_labels.numpy()
                        
                        # pass extracted features as inputs to SVM model
                        print(f"Training SVM on Client {self.id}...")
                        loss += self.model.train_svm(train_features, train_labels)
                        
                        break
                if end: break
                self.epoch += 1

        loss /= num_iter
        return loss
        
    def test_model(self, device):
        all_features = []
        all_labels = []

        correct = 0.0
        total = 0.0
        
        with torch.no_grad():
            loop = tqdm(self.test_loader, leave=True)

            for batch in loop:
                # Step 1: Move inputs to the same device as the base_model (i.e., CUDA device)
                input_ids_batch, attention_mask_batch, labels_batch = batch['input_ids'].to(device, non_blocking=True), batch['attention_mask'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True) 

                # Step 2: Forward pass (no gradient needed)
                CLS_token_batch = self.model.extract_features(input_ids_batch, attention_mask_batch)
                
                all_features.append(CLS_token_batch.cpu())
                all_labels.append(labels_batch.cpu())

                total += labels_batch.size(0)

            # Step 3: Stack all the mini-batches into one tensor of shape 
            # torch.Size([number of samples = num of batches * BATCH_SIZE, 768 = hidden size of tokenized input])
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Step 4: Convert to NumPy
            test_features = all_features.numpy()
            test_labels = all_labels.numpy()

            print(f"Testing SVM on Client {self.id}...")
            predictions = self.model.evaluate_svm(test_features)

            #print("Classification Report:")
            #print(classification_report(test_labels, predictions))
            #acc = accuracy_score(test_labels, predictions)
            #print(f"Accuracy: {acc * 100:.2f}%")

            correct += (predictions == test_labels).sum().item()

        return total, correct
    

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.get_model_parameters())
                                        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict   # get the intermediate model parameters from the respective edge server 
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        self.model.set_model_params(self.receiver_buffer)
        return None


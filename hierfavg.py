from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from client import Client
from cloud import Cloud
from edge import Edge
from options import args_parser
from utils.get_dataloaders import get_dataloaders, show_distribution
from utils.initialize_bert_model import create_bert_model
import copy
from sklearn.metrics import classification_report, accuracy_score



# This information is needed to create a correct scikit-learn model
NUM_UNIQUE_LABELS = 2  # 5G-NIDD has 2 classes according to binary classification
NUM_FEATURES = 2  # Number of features extracted from the bert model, i.e., input_ids and attention_mask


# Metrics for Evaluation
COMMUNICATION_OVERHEAD = 0
COMMUNICATION_ROUND_TO_CONVERGENCE = 0
TRAINING_TIME = 0
CPU/GPU_UTILIZATION = 0



def set_initial_params(model: SGDClassifier) -> None:
    """Sets initial parameters as zeros Required since model params are not initialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.arange(NUM_UNIQUE_LABELS)

    model.coef_ = np.zeros((NUM_UNIQUE_LABELS, NUM_FEATURES))
    if model.fit_intercept:
        model.intercept_ = np.zeros((NUM_UNIQUE_LABELS,))


def create_svm_and_instantiate_parameters():
    model = SGDClassifier(loss="hinge", alpha=0.01, max_iter=1000, tol=1e-3)
    # Setting initial parameters, akin to model.compile for keras models
    set_initial_params(model)
    return model

def set_model_params(svm_shared_layers, params):  # svm_shared_layers is a model not just parameters
    """Sets the parameters of a sklean model."""
    svm_shared_layers.coef_ = params['coef'] # since it only consists of one layer
    if svm_shared_layers.fit_intercept:
        svm_shared_layers.intercept_ = params['intercept']

# Testing at the edge level
def all_clients_test(server, clients, cids, device):  
    [server.send_to_client(clients[cid]) for cid in cids]
    for cid in cids:
        server.send_to_client(clients[cid])
        # The following sentence!
        clients[cid].sync_with_edgeserver()
    correct_edge = 0.0
    total_edge = 0.0
    for cid in cids:
        correct, total = clients[cid].test_model(device)
        correct_edge += correct
        total_edge += total
    return correct_edge, total_edge


# Function to extract features using TinyBert Model
# Extract representations/features from the base model (without the final classification layers)
def extract_features(dataloader, base_model, device):
    all_features = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(dataloader, leave=True)

        for batch in loop:
            # Step 1: Move inputs to the same device as the base_model (i.e., CUDA device)
            input_ids, attention_mask, labels = batch['input_ids'].to(device, non_blocking=True), batch['attention_mask'].to(device, non_blocking=True), batch['label'].to(device, non_blocking=True) 

            # Step 2: Forward pass (no gradient needed)
            outputs = base_model(input_ids, attention_mask=attention_mask)
            CLS_token = outputs.last_hidden_state[:,0,:]  # Shape = [batch_size, hidden_dim]

            all_features.append(CLS_token.cpu())
            all_labels.append(labels.cpu())

    # Step 3: Stack all the mini-batches into one tensor of shape 
    # torch.Size([number of samples = num of batches * BATCH_SIZE, 768 = hidden size of tokenized input])
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Step 4: Convert to NumPy
    data_features = all_features.numpy()
    data_labels = all_labels.numpy()

    return data_features, data_labels

# Testing at the cloud level
def evaluate_global_model(v_test_loader, base_model, model, device):
    correct = 0.0
    total = 0.0
    
    # Extract features for training and testing data
    print("Extracting features from testing data...")
    test_features, test_labels = extract_features(v_test_loader, base_model, device) # logits

    print("Predicting...")
    predictions = model.predict(test_features)

    print("Classification Report:")
    print(classification_report(test_labels, predictions))
    acc = accuracy_score(test_labels, predictions)
    print(f"Accuracy: {acc * 100:.2f}%")

    total = len(test_labels)
    correct += (predictions == test_labels).sum().item()

    return correct, total




def HierFAVG(args):
    #make experiments repeatable + set cuda device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        #cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Set TensorBoard Summary
    FILEOUT = f"clients{args.num_clients}_edges{args.num_edges}_" \
              f"t1-{args.num_local_update}_t2-{args.num_edge_aggregation}" \
              f"epoch{args.num_communication}" \
              f"bs{args.batch_size}"
    writer = SummaryWriter(comment=FILEOUT)

    # Build Dataloaders
    train_loaders, test_loaders, v_test_loader = get_dataloaders(args)

    if args.show_dis:
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(f"Dataset Length: ", len(train_loader.dataset))
            distribution = show_distribution(train_loader)
            print("train dataloader {} distribution".format(i))
            print(distribution)

        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            distribution = show_distribution(test_loader)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")
            print(distribution)

        v_size = len(v_test_loader.dataset)
        print(v_size)
        distribution = show_distribution(v_test_loader)
        print("validation dataloader distribution")
        print(f"validation dataloader size {v_size}")
        print(distribution)


    # Step 1: Create, optimize and quantize TinyBERT model using ONNX Runtime
    
    print("Create Base Model...")
    base_model = create_bert_model()
    print(base_model)

    # Step 2: Initialize the global model (SVM model) for the shared layers
    print("Create and Tune hyperparameters of SVM model...")
    svm_shared_layers = create_svm_and_instantiate_parameters()
    #svm_shared_layers = SVC(C= 1.0, kernel='linear', gamma = 'scale')
    #svm_shared_layers = SGDClassifier(loss="hinge", alpha=0.01, max_iter=1000, tol=1e-3)
    #svm_shared_layers = svm_shared_layers.to(device)

    # Step 3: Initialize clients and server
    print("Initialize Clients...")
    clients = []
    for i in range(args.num_clients):
        clients.append(Client(id=i,
                            train_loader=train_loaders[i],
                            test_loader=test_loaders[i],
                            base_model=base_model,
                            svm_shared_layers = copy.deepcopy(svm_shared_layers),
                            args=args,
                            device=device)
                    )
        
    # Step 4: Initialize edge server and assign clients to the edge server
    edges = []
    cids = np.arange(args.num_clients)
    clients_per_edge = int(args.num_clients / args.num_edges)
    p_clients = [0.0] * args.num_edges
    
    # This is randomly assign the clients to edges
    for i in range(args.num_edges):
        #Randomly select clients and assign them
        selected_cids = np.random.choice(cids, clients_per_edge, replace=False)
        cids = list (set(cids) - set(selected_cids))
        edges.append(Edge(id = i,
                            cids = selected_cids,
                            shared_layers = copy.deepcopy(clients[0].model.svm_shared_layers)))
        [edges[i].client_register(clients[cid]) for cid in selected_cids]
        edges[i].all_trainsample_num = sum(edges[i].sample_registration.values())
        p_clients[i] = [sample / float(edges[i].all_trainsample_num) for sample in
                list(edges[i].sample_registration.values())]
        edges[i].refresh_edgeserver()

    # Step 5: Initialize cloud server
    cloud = Cloud(shared_layers=copy.deepcopy(clients[0].model.svm_shared_layers))
    # Register edge servers to the cloud server
    [cloud.edge_register(edge=edge) for edge in edges]
    p_edge = [sample / sum(cloud.sample_registration.values()) for sample in
                list(cloud.sample_registration.values())]
    cloud.refresh_cloudserver()


    # Step 6: Begin training
    for num_comm in tqdm(range(args.num_communication)):
        cloud.refresh_cloudserver()
        [cloud.edge_register(edge=edge) for edge in edges]
        for num_edgeagg in range(args.num_edge_aggregation):
            edge_loss = [0.0]* args.num_edges
            edge_sample = [0]* args.num_edges
            correct_all = 0.0
            total_all = 0.0
            # no edge selection included here
            # for each edge, iterate
            for i,edge in enumerate(edges):
                edge.refresh_edgeserver()
                client_loss = 0.0
                selected_cnum = max(int(clients_per_edge * args.frac),1)
                selected_cids = np.random.choice(edge.cids,
                                                 selected_cnum,
                                                 replace = False,
                                                 p = p_clients[i])
                for selected_cid in selected_cids:
                    edge.client_register(clients[selected_cid])
                for selected_cid in selected_cids:
                    edge.send_to_client(clients[selected_cid])
                    clients[selected_cid].sync_with_edgeserver()
                    client_loss += clients[selected_cid].local_update(num_iter=args.num_local_update,
                                                                      device = device)
                    clients[selected_cid].send_to_edgeserver(edge)
                edge_loss[i] = client_loss
                edge_sample[i] = sum(edge.sample_registration.values())

                print("Intermediate Aggregation at the Edge Level...")
                edge.aggregate(args)
                correct, total = all_clients_test(edge, clients, edge.cids, device)
                correct_all += correct
                total_all += total
            # end interation in edges
            all_loss = sum([e_loss * e_sample for e_loss, e_sample in zip(edge_loss, edge_sample)]) / sum(edge_sample)
            avg_acc = correct_all / total_all

            print(f"All Loss at Edge Layer: {all_loss}")
            print(f"Average Accuracy at Edge Layer: {avg_acc}")
            writer.add_scalar(f'Partial_Avg_Train_loss',
                          all_loss,
                          num_comm* args.num_edge_aggregation + num_edgeagg +1)
            writer.add_scalar(f'All_Avg_Test_Acc_edgeagg',
                          avg_acc,
                          num_comm * args.num_edge_aggregation + num_edgeagg + 1)

        # Now begin the cloud aggregation
        for edge in edges:
            edge.send_to_cloudserver(cloud)
        cloud.aggregate(args)
        for edge in edges:
            cloud.send_to_edge(edge)

        # Testing Global Model
        set_model_params(svm_shared_layers, cloud.shared_state_dict)
        correct_all_v, total_all_v = evaluate_global_model(v_test_loader, base_model, svm_shared_layers, device)
        avg_acc_v = correct_all_v / total_all_v

        print(f"Average Accuracy at Cloud Layer: {avg_acc_v}")
        
        writer.add_scalar(f'All_Avg_Test_Acc_cloudagg_Vtest',
                            avg_acc_v,
                            num_comm + 1)
        
    writer.close()
    print(f"The final virtual acc is {avg_acc_v}")


    '''#Begin training
    for i in range(args.num_clients):
        client_loss = 0.0
        client_loss += clients[i].local_update(num_iter=args.num_local_update,
                                                            device = device)
        print(f"Client num {i}, loss: {client_loss}")

    print(f"Final Clients Loss per 1 communication round: {client_loss}")

    total = 0.0
    correct = 0.0

    print('Testing Models at Clients side...')
    for i in range(args.num_clients):
        client_total = 0.0
        client_correct = 0.0

        client_total, client_correct = clients[i].test_model(device)
        print(f"Client total {client_total}, Client correct {client_correct}")
        total += client_total
        correct += client_correct

    avg_acc = correct / total
    print(f"Average Accuracy: {avg_acc}")'''

    
        
    
    


def main():
    args = args_parser()
    HierFAVG(args)

if __name__ == '__main__':
    main()
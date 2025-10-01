import os
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


# Args to be passed to make the model flexible: num_clients, batch_size, cuda


class LoaderDataset(Dataset): # Divide data into batches
    def __init__(self, encodings, labels):
        self.encodings = encodings        # X
        self.labels = labels              # y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert each row to tensor
        input_ids = torch.tensor(self.encodings[idx][0], dtype=torch.long)
        attention_mask = torch.tensor(self.encodings[idx][1], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}


def row_to_text(row):
    return (
        f"This flow lasted for {row['Dur']} units of duration with a runtime of {row['RunTime']}. "
        f"The statistical features include a mean of {row['Mean']}, a sum of {row['Sum']}, "
        f"a minimum of {row['Min']}, and a maximum of {row['Max']}. "
        f"It used the {row['Proto']} protocol with source type of service {row['sTos']} "
        f"and destination type of service {row['dTos']}. "
        f"The differentiated service bits were {row['sDSb']} at the source and {row['dDSb']} at the destination. "
        f"The source time-to-live was {row['sTtl']} with {row['sHops']} hops, while the destination TTL was {row['dTtl']} with {row['dHops']} hops. "
        f"The termination cause was {row['Cause']}. "
        f"The flow contained {row['TotPkts']} total packets ({row['SrcPkts']} from the source and {row['DstPkts']} from the destination) "
        f"and {row['TotBytes']} total bytes ({row['SrcBytes']} from the source and {row['DstBytes']} from the destination). "
        f"The offset was {row['Offset']}, the mean packet size at the source was {row['sMeanPktSz']} and at the destination {row['dMeanPktSz']}. "
        f"The overall load was {row['Load']} with {row['SrcLoad']} from the source and {row['DstLoad']} from the destination. "
        f"Packet loss included {row['Loss']} total, with {row['SrcLoss']} from the source and {row['DstLoss']} from the destination, "
        f"which corresponds to a percentage loss of {row['pLoss']}. "
        f"The average gap between packets was {row['SrcGap']} at the source and {row['DstGap']} at the destination. "
        f"The overall rate was {row['Rate']} with a source rate of {row['SrcRate']} and a destination rate of {row['DstRate']}. "
        f"The connection state was {row['State']}, with a source window size of {row['SrcWin']} and a destination window size of {row['DstWin']}. "
        f"The source VLAN ID was {row['sVid']} and the destination VLAN ID was {row['dVid']}. "
        f"The source TCP base sequence number was {row['SrcTCPBase']} and the destination TCP base {row['DstTCPBase']}. "
        f"The TCP round trip time was {row['TcpRtt']}, the SYN-ACK delay was {row['SynAck']}, and the ACK-data delay was {row['AckDat']}. "
        f"Finally, this flow is considered as {row['Label']}."
    )


from datasets import Dataset

def convert_to_text(dataset):
    texts = dataset.apply(row_to_text, axis=1)

    # Combine texts and label attributes and convert into a huggingface dataset
    hf_dataset = Dataset.from_dict({
        "text": texts.tolist(),
        "label": dataset["Label"].tolist()
    })

    return hf_dataset


def tokenize(hf_dataset):
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def convert_to_dataloader(tokenized_dataset, args, is_shuffle, kwargs):
    tokenized_dataset = tokenized_dataset.to_pandas()

    # Split data into features (X) and labels (y)
    X = tokenized_dataset[['input_ids', 'attention_mask']]  # tokenized result
    y = tokenized_dataset['label']  # Target labels
    y = y.replace({'Benign': 0, 'Malicious': 1}).astype(int)

    # Convert data to lists for compatibility
    X = X.values.tolist()
    y = y.values.tolist()

    # Create PyTorch datasets
    data = LoaderDataset(X, y)
    
    # Create DataLoader instances for batching
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=is_shuffle, **kwargs) 

    return dataloader


def pre_processing(data, args, is_shuffle, kwargs):
    # Step 1: Preprocess the dataset (Convert Network Flows into Texts)
    hf_dataset = convert_to_text(data)

    # Step 2: Tokenization
    tokenized_dataset = tokenize(hf_dataset)

    # Step 3: Convert to DataLoader
    data_loader = convert_to_dataloader(tokenized_dataset, args, is_shuffle, kwargs)

    return data_loader


# Split the dataset according to the number of clients, instead of fixed partitioning 
# dataset = resampled_binary_training_data.csv

def split_data(dataset, args, kwargs, is_shuffle = True):  
    data_loaders = [0] * args.num_clients

    # each client has one shard
    num_shards = args.num_clients

    # the number of inputs in one shard
    num_inputs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_inputs)

    #divide and assign
    for i in range(args.num_clients):
        # Select one shard per client randomly
        rand_set = set(np.random.choice(idx_shard, 1, replace= False))

        # Remove the selected shard from the list
        idx_shard = list(set(idx_shard) - rand_set)

        # Get the inputs within the selected shard
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_inputs: (rand + 1) * num_inputs]), axis=0) # indexes of rows at the original dataset
            dict_users[i] = dict_users[i].astype(int)

        # Get the dataset for each client
        user_data = dataset.iloc[dict_users[i]]

        # Convert to DataLoader
        data_loaders[i] = pre_processing(user_data, args, is_shuffle, kwargs)
    return data_loaders


def concatenate_test_loaders(test_loaders, args, kwargs, is_shuffle = False):
    # Collect datasets from the loaders
    datasets = [dataloader.dataset for dataloader in test_loaders]

    # Extract encodings and labels from all datasets
    all_encodings = []
    all_labels = []

    for ds in datasets:
        all_encodings.extend(ds.encodings)
        all_labels.extend(ds.labels)

    # Rebuild dataset
    combined_data = LoaderDataset(all_encodings, all_labels)


    # Wrap into one DataLoader
    combined_loader = DataLoader(combined_data, args.batch_size, shuffle=is_shuffle, **kwargs)

    return combined_loader



def get_dataloaders(args):
    train_loaders, test_loaders, v_test_loader = {}, {}, {}

    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    # get the trainloaders and testloaders for all the clients
    script_dir = os.path.dirname(os.path.abspath(__file__))  # folder of current script
    train_path = os.path.join(script_dir, "..", "Unified_Dataset", "resampled_binary_training_data.csv")
    test_path = os.path.join(script_dir, "..", "Unified_Dataset", "binary_testing_data.csv")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    #note: is_shuffle here also is a flag for differentiating train and test
    train_loaders = split_data(train, args, kwargs, is_shuffle = True)
    test_loaders = split_data(test, args, kwargs, is_shuffle = False)

    #get the validation dataloader for all the clients
    v_test_loader = concatenate_test_loaders(test_loaders, args, kwargs, is_shuffle = False)
    

    return train_loaders, test_loaders, v_test_loader




def show_distribution(dataloader):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    labels = np.array(dataloader.dataset.labels) 

    num_samples = len(labels)
    idxs = [i for i in range(num_samples)]
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        label = labels[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution
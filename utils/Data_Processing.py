from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


#Global Variables
NUM_CLIENTS = 10
BATCH_SIZE = 8
FEATURES_NUM = 48



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
    

def load_dataset(partition_id: int):
    partition_filename = f'Partitions/client_{partition_id+1}.csv'
    dataset = pd.read_csv(partition_filename)
    return dataset


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

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    return tokenized_datasets


def split_dataset(tokenized_datasets):
    tokenized_datasets = tokenized_datasets.to_pandas()

    # Split data into features (X) and labels (y)
    X = tokenized_datasets[['input_ids', 'attention_mask']]  # tokenized result
    y = tokenized_datasets['label']  # Target labels
    y = y.replace({'Benign': 0, 'Malicious': 1})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert data to numpy arrays for compatibility
    X_train = X_train.values.tolist()
    y_train = y_train.values.tolist()
    X_test = X_test.values.tolist()
    y_test = y_test.values.tolist()

    return X_train, y_train, X_test, y_test


def convert_to_dataloaders(X_train, y_train, X_test, y_test):
    # Create PyTorch datasets
    train_dataset = LoaderDataset(X_train, y_train)
    val_dataset = LoaderDataset(X_test, y_test)
    
    # Create DataLoader instances for batching
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) #num_workers=4, pin_memory=True)
    testloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True) #num_workers=4, pin_memory=True)

    return trainloader, testloader



# First Step: Load the dataset partition and pre-process it for training and validation

def pre_processing(partition_id: int):

    # Step 1: Load partition data from CSV file
    dataset = load_dataset(partition_id)

    # Step 2: Preprocess the dataset (Convert Network Flows into Texts)
    hf_dataset = convert_to_text(dataset)

    # Step 3: Tokenization
    tokenized_datasets = tokenize(hf_dataset)

    # Step 4: Split Data into training and testing sets
    X_train, y_train, X_test, y_test = split_dataset(tokenized_datasets)

    # Step 5: Convert to DataLoaders
    trainloader, testloader = convert_to_dataloaders(X_train, y_train, X_test, y_test)

    return trainloader, testloader
    
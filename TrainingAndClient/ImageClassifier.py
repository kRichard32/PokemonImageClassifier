import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import json

global_label_set = []
class PokemonDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, labels = self.data.iloc[idx]
        if self.transform:
            img = self.transform(img)
            return img, labels
        return img, labels
def create_dataset(path):
    images = []
    labels = []
    pathlist = Path(path).glob('*.png')
    for path in pathlist:
        temp_image = Image.open(str(path)).convert('RGB')
        images.append(temp_image)
        filename = path.stem
        if "-" in filename:
            filename = filename.split("-")[0]
        if filename not in global_label_set:
            global_label_set.append(filename)
        temp_num = global_label_set.index(filename)
        labels.append(temp_num)
    dictionary = {'images': images, 'labels': labels}
    df = pd.DataFrame(dictionary)
    return df
def debug_setup():
    random_seed = 45
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

def get_triple_datasets(dataset):
    test_pct = 0.3
    test_size = int(len(dataset) * test_pct)
    dataset_size = len(dataset) - test_size

    val_pct = 0.1
    val_size = int(dataset_size * val_pct)
    train_size = dataset_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    #return train_ds, val_ds, test_ds
    return dataset, dataset, dataset
def run_models():
    debug_setup()

    train_ds, val_ds, test_ds = get_triple_datasets(create_dataset("./resources/images"))

    train_transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(220, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.ToTensor(),
    ])

    train_dataset = PokemonDataset(train_ds, train_transform)
    val_dataset = PokemonDataset(val_ds, test_transform)
    test_dataset = PokemonDataset(test_ds, test_transform)
    device = get_default_device()

    pokemon_classifier = PokemonClassifierCNN(len(global_label_set))

    batch_size = 64

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    test_dl = DataLoader(test_dataset, batch_size * 2, num_workers=0, pin_memory=True)

    train_dl = DeviceDataLoader(train_dl, device=device)
    val_dl = DeviceDataLoader(val_dl, device=device)
    test_dl = DeviceDataLoader(test_dl, device=device)
    to_device(pokemon_classifier, device)

    num_epochs = 100
    opt_func = torch.optim.Adam
    max_lr = 0.004
    grad_clip = 0.1
    weight_decay = 1e-4
    history = fit_one_cycle(num_epochs, max_lr, pokemon_classifier, train_dl, val_dl, weight_decay, grad_clip, opt_func)
    torch.save(pokemon_classifier.state_dict(), "./pokemon_model.pth")
    with open("./model_metadata.txt", "w+") as f:
        f.write(str(len(global_label_set)))
    with open("./global_list.json", "w") as f:
        json.dump(global_label_set, f)
def draw_history(history):
    val_loss = []
    train_loss = []
    train_acc = []
    val_acc = []

    time = list(range(len(history)))
    for h in history:
        val_loss.append(h['loss'])
        train_loss.append(h['train_loss'])
        train_acc.append(h['train_acc'])
        val_acc.append(h['acc'])

    plt.plot(time, train_acc, c='red', label='train_acc', marker='x')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0, grad_clip=None, opt_func = torch.optim.Adam):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_accs = []
        lrs = []
        for batch in tqdm.tqdm(train_loader):
            acc,loss = model.training_step(batch)
            train_losses.append(loss)
            train_accs.append(acc)

            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))

            sched.step()
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = sum(train_accs)/len(train_accs)
        result['lrs'] = lrs
        model.epoch_end(epoch,result)
        history.append(result)
    return history
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_default_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d, device) for d in data]
    else:
        return data.to(device, non_blocking=True)
class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __len__(self):
        return len(self.dl)
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        img, targets = batch
        out = self(img)
        loss = F.nll_loss(out, targets)
        acc = accuracy(out, targets)
        return acc, loss
    def validation_step(self, batch):
        acc, loss = self.training_step(batch)
        return {'acc':acc.detach(), 'loss':loss.detach()}
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    def epoch_end(self, epoch, result):
        print("Epoch [{}] : train_loss: {:.4f}, "
              "train_acc: {:.4f}, val_loss: {:.4f}, "
              "val_acc: {:.4f}".format(epoch,result["train_loss"],
              result["train_acc"], result["loss"], result["acc"]))
class PokemonClassifierCNN(ImageClassificationBase):
    def __init__(self, out):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3,16,3,1,1), #220 * 220 * 16
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16,16,3,1,1), #220 * 220 * 16
            nn.MaxPool2d(2,2), #110 * 110 * 16
            nn.ReLU(),

            nn.Conv2d(16,8,3,1,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,8,3,1,1),
            nn.MaxPool2d(2,2), #55 * 55 * 8
            nn.ReLU(),

            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(55 * 55 * 8, out),
            nn.Dropout(0.5),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        return self.network(x)
def load_model(model_path, model_metadata):
    with open(model_metadata, "r") as f:
        model_predict_size = int(f.readline().strip())
    pokemon_classifier = PokemonClassifierCNN(model_predict_size)
    pokemon_classifier.load_state_dict(torch.load(model_path, weights_only=True))
    pokemon_classifier.eval()
    print("Model loaded")
    return pokemon_classifier

def run_inference_on_image(image):
    script_path = os.path.dirname(os.path.abspath(__file__))
    abs_model = os.path.join(script_path, "pokemon_model.pth")

    model = load_model(abs_model, os.path.join(script_path, "model_metadata.txt"))


    img_tensor =  transforms.Resize((220, 220))(image)
    img_tensor = img_tensor.unsqueeze(0)

    tensor = model(img_tensor)
    max_element = tensor.argmax().tolist()
    return max_element

def get_pokemon_label(index):
    script_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_path, "global_list.json"), "r") as f:
        global_label_set = json.load(f)
    return global_label_set[index]

if __name__ == "__main__":
    print("start")
    script_path = os.path.dirname(os.path.abspath(__file__))
    rel_path = "pokemon_model.pth"
    abs_file_path_model = os.path.join(script_path, rel_path)
    file = Path(abs_file_path_model)
    if file.is_file():
        img = Image.open(os.path.join(script_path, "ab2.png")).convert('RGB')
        index = run_inference_on_image(transforms.ToTensor()(img))
        print(get_pokemon_label(index))
    else:
        run_models()



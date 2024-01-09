from torch.utils.data import DataLoader
class DataloaderWrapper:
    def __init__(self, dataset, batch_size, num_workers, shuffle, pin_memory=False, drop_last=False):
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     pin_memory=pin_memory, drop_last=drop_last)
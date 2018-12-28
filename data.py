class SubSample:

    def __init__(self, dataset, nb):
        nb = min(len(dataset), nb)
        self.dataset = dataset
        self.nb = nb
        self.transform = dataset.transform
        self.classes = dataset.classes

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return self.nb

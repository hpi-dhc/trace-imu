import numpy as np
import cv2
from sklearn.decomposition import PCA
from keras.preprocessing import sequence


class ImageDataset:
    def __init__(self, path, source, params, modalities=None, modifier=''):
        self.params = params
        self.labels = self.params[source]['labels']
        self.exclude = self.params[source]['exclude']
        self.subjects = self.params[source]['subjects']
        self.window_length = self.params[source]['window_length']
        self.data = dict(np.load(path + source + modifier + '.npz', allow_pickle=True))
        self.modalities = modalities

    def window_split(self):
        if self.modalities is None:
            self.modalities = np.arange(6)
        windowed_set = {}
        for key, value in self.data.items():
            windowed_set[key] = np.vsplit(value[:, self.modalities],
                                          [i for i in range(self.window_length, len(value), self.window_length)])
        self.data = windowed_set

    def flatten_dict(self):
        kv = [[], []]
        for key, value in self.data.items():
            kv[0].extend([key] * len(value))
            kv[1].extend(value)
        return kv

    def to_image(self, ss, size=64):
        images = []
        for s in range(ss.shape[1]):
            if len(ss[:, s, :]) < 2:
                images.append(np.zeros((size, size)))
                continue
            pca = PCA(n_components=2)
            s2d = pca.fit_transform(ss[:, s, :])
            #s2d = ss[:, s, 1:]
            image = np.zeros((size, size))
            for point in s2d:
                coords = np.floor((point + 1) * (size // 2)).astype(int)
                image[coords[0], coords[1]] += 1.0
            images.append(image / np.max(image))
        return np.array(images)

    def prepare_labels(self):
        labels = []
        file_ids = []
        for label, seqs in self.data.items():
            current_label = (np.inf, '')
            lab = label.lower()
            for l in self.labels:
                try:
                    if lab.index(l) < current_label[0]:
                        current_label = (lab.index(l), l)
                except ValueError:
                    pass # substring not found -> label is ''

            for _ in range(len(seqs)):
                labels.append(current_label[1])
                file_ids.append(label)

        x = np.array(self.flatten_dict()[1])
        selector = np.invert(np.isin(labels, list(self.exclude)))
        x = np.array([self.to_image(i) for i in np.array(x)[selector]])
        y = np.array(labels)[selector]
        file_ids = np.array(file_ids)[selector]

        return x, y, file_ids

    def k_fold(self, file_ids, include=None):
        subjects = sorted(list(self.subjects))
        if include is not None:
            subjects = [subjects[i] for i in include]
        folds = []
        indices = [[n for n, l in enumerate(file_ids) if l.startswith(subjects[i])] for i in range(len(subjects))]
        for i in range(len(subjects)):
            folds.append((np.random.permutation(np.concatenate(indices[:i] + indices[i + 1:])).astype(int),
                          np.random.permutation(indices[i]).astype(int)))
        return folds

    def encode(self, y):
        targets = np.zeros((len(y), len(self.enc)))
        for i in range(len(y)):
            targets[i, self.enc.index(y[i])] = 1
        return targets

    def get_dataset(self, k_fold=True, include=None):
        self.window_split()
        x, y, file_ids = self.prepare_labels()
        self.enc = sorted([l for l in self.labels if l not in self.exclude])
        if k_fold:
            folds = self.k_fold(file_ids, include)
            return folds, x, self.encode(y)
        return x, self.encode(y)


class TimeSeriesDataset(ImageDataset):
    def __init__(self, path, source, params, modalities=None, kind='trajectory'):
        self.kind = kind
        modifier = ''
        if self.kind == 'raw':
            modifier = '_acc_ori'
        super().__init__(source, path, params, modalities, modifier)

    def window_split(self):
        if self.kind == 'trajectory':
            if self.modalities is None:
                self.modalities = np.arange(6)
        elif self.kind == 'raw':
            if self.modalities is None:
                self.modalities = np.arange(60)
        else:
            raise ValueError("No such kind: " + self.kind)
        super().window_split()

    def prepare_labels(self):
        labels = []
        file_ids = []
        for label, seqs in self.data.items():
            current_label = (np.inf, '')
            lab = label.lower()
            for l in self.labels:
                try:
                    if lab.index(l) < current_label[0]:
                        current_label = (lab.index(l), l)
                except ValueError:
                    pass # substring not found -> label is ''
            if current_label[1] == '':
                print(label, lab, self.labels)
                exit(1)
            for _ in range(len(seqs)):
                labels.append(current_label[1])
                file_ids.append(label)

        x = sequence.pad_sequences((self.flatten_dict()[1]), padding='post', dtype='float')
        selector = np.invert(np.isin(labels, list(self.exclude)))
        x = x[selector]
        y = np.array(labels)[selector]
        file_ids = np.array(file_ids)[selector]

        if len(x.shape) > 3:
            s = x.shape
            x = x.reshape(s[0], s[1], s[3])

        return x, y, file_ids

class ImageTimeSeriesDataset(ImageDataset):
    def window_split(self):
        if self.modalities is None:
            self.modalities = np.arange(6)
        windowed_set = {}
        for key, value in self.data.items():
            windowed_set[key] = np.vsplit(value[:, self.modalities],
                                          [i for i in range(self.window_length // 2, len(value), self.window_length // 2)])
        self.data = windowed_set

    def prepare_labels(self):
        labels = []
        file_ids = []
        for label, seqs in self.data.items():
            current_label = (np.inf, '')
            lab = label.lower()
            for l in self.labels:
                try:
                    if lab.index(l) < current_label[0]:
                        current_label = (lab.index(l), l)
                except ValueError:
                    pass  # substring not found -> label is ''

            for _ in range(len(seqs)):
                labels.append(current_label[1])
                file_ids.append(label)



        x = np.array(self.flatten_dict()[1])
        selector = np.invert(np.isin(labels, list(self.exclude)))
        #from keras.applications.xception import preprocess_input
        x = np.array([self.to_image(i) for i in x[selector]])
        
        s_length = 4
        current_label = ''
        output = ([], [], [])
        for img, label, file_id in zip(x, np.array(labels)[selector], np.array(file_ids)[selector]):
            if label != current_label:
                buf = np.zeros((s_length,) + img.shape)
                current_label = label
            buf[:s_length-1] = buf[1:]
            buf[s_length-1] = img
            output[0].append(buf)
            output[1].append(current_label)
            output[2].append(file_id)
        '''
        from scipy import ndimage
        # Data Augmentation
        for i in range(len(output[0])):
            flip = np.random.randint(0, 2) * 2 - 1
            rot = np.random.randint(5, 90)
            output[0].append([[cv2.flip(img, flip) for img in imgs] for imgs in output[0][i]])
            output[1].append(output[1][i])
            output[2].append(output[2][i])

            output[0].append([[ndimage.rotate(img, rot, reshape=False) for img in imgs] for imgs in output[0][i]])
            output[1].append(output[1][i])
            output[2].append(output[2][i])
        print(np.array(output[0]).shape)
        '''
        #y = np.array(labels)[selector][3:]
        #file_ids = np.array(file_ids)[selector][3:]
        #x = np.array([[x[i+j] for j in range(4)] for i in range(len(x)-3)])
        x, y, f = output
        return np.array(x), y, f

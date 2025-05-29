import numpy as np

# kelas LabelEncoder yang dibuat tanpa menggunakan library LabelEncoder
class LabelEncoder:

    # fungsi untuk inisialisasi atribut kelas, mapping label ke index, dan mapping index ke label
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}
        self.index_to_class = {}

    # fungsi untuk menyimpan label unik yang telah diurutkan, serta membuat mapping bagi label dan index
    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.classes_ = np.array(unique_labels)
        self.class_to_index = {label: idx for idx, label in enumerate(self.classes_)}
        self.index_to_class = {idx: label for idx, label in enumerate(self.classes_)}

    # fungsi untuk mengubah daftar label menjadi daftar indeks numerik
    def transform(self, labels):
        if self.classes_ is None:
            raise ValueError("Encoder belum di-fit. Panggil 'fit()' atau 'fit_transform()' terlebih dahulu.")
        
        try:
            return np.array([self.class_to_index[label] for label in labels])
        except KeyError as e:
            raise ValueError(f"Label tidak dikenali: {e}")

    # fungsi kombinasi dari fit() dan transform()
    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    # fungsi untuk mengubah kembali daftar indeks numerik menjadi daftar label asli
    def inverse_transform(self, indices):
        if self.classes_ is None:
            raise ValueError("Encoder belum di-fit.")
        
        try:
            return [self.index_to_class[int(idx)] for idx in indices]
        except KeyError as e:
            raise ValueError(f"Indeks tidak dikenali: {e}")

# fungsi untuk membagi dataset menjadi data train dan data test
def train_test_split(X, y, images=None, test_size=0.2, random_state=None):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError("X dan y harus berupa numpy array.")
    
    if images is not None and not isinstance(images, np.ndarray):
        raise TypeError("images harus berupa numpy array atau None.")
    
    if len(X) != len(y) or (images is not None and len(X) != len(images)):
        raise ValueError("Panjang X, y, dan images harus sama.")
    
    if not (0 < test_size < 1):
        raise ValueError("test_size harus berupa nilai antara 0 dan 1.")

    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if images is not None:
        img_train, img_test = images[train_indices], images[test_indices]
    else:
        img_train, img_test = None, None

    return X_train, X_test, y_train, y_test, img_train, img_test
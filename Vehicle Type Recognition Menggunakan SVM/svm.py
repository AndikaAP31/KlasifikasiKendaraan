import numpy as np
from itertools import product

# kelas Support Vector Machine (SVM) yang dibuat tanpa menggunakan library
class SVM:

    # fungsi untuk inisialisasi parameter-parameter yang ada di dalam SVM
    def __init__(self, kernel='linear', C=1.0, tol=1e-3, max_passes=5, max_iter=1000, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.gamma = gamma
        self._fitted = False

    # fungsi untuk menghitung nilai gamma berdasarkan metode 'scale' dan 'auto' ketika diperlukan
    def _compute_gamma(self, X):
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * np.var(X, axis=0).mean() + 1e-8)
        elif self.gamma == 'auto':
            return 1.0 / (X.shape[1] + 1e-8)
        else:
            return self.gamma

    # fungsi untuk menghitung kernel matrix antara X1 dan X2 yang mendukung kernel RBF dan Linear
    def _kernel_function(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            gamma_val = self._compute_gamma(X1)
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            K = np.exp(-gamma_val * (X1_sq + X2_sq - 2 * np.dot(X1, X2.T)))
            return K
        else:
            raise ValueError(f"Kernel '{self.kernel}' tidak didukung")

    # fungsi untuk melatih model SVM dengan strategi One vs One (OVO) untuk kasus multiclass
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models = []

        if len(self.classes_) == 2:
            y_bin = np.where(y == self.classes_[0], -1, 1)
            model = self._train_binary(X, y_bin)
            self.models.append((model, self.classes_[0], self.classes_[1]))
        else:
            for i in range(len(self.classes_)):
                for j in range(i + 1, len(self.classes_)):
                    c1 = self.classes_[i]
                    c2 = self.classes_[j]
                    idx = np.where((y == c1) | (y == c2))[0]
                    X_pair = X[idx]
                    y_pair = y[idx]
                    y_bin = np.where(y_pair == c1, -1, 1)
                    model = self._train_binary(X_pair, y_bin)
                    self.models.append((model, c1, c2))
        self._fitted = True

    # fungsi untuk melatih model SVM biner menggunakan algoritma Sequential Minimal Optimization (SMO)
    def _train_binary(self, X, y):
        n = X.shape[0]
        alpha = np.zeros(n)
        b = 0
        passes = 0

        K = self._kernel_function(X)

        for it in range(self.max_iter):
            num_changed = 0
            for i in range(n):
                E_i = np.dot((alpha * y), K[:, i]) + b - y[i]

                cond1 = (y[i] * E_i < -self.tol and alpha[i] < self.C)
                cond2 = (y[i] * E_i > self.tol and alpha[i] > 0)
                if not (cond1 or cond2):
                    continue

                j = i
                while j == i:
                    j = np.random.randint(0, n)

                E_j = np.dot((alpha * y), K[:, j]) + b - y[j]

                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]

                if y[i] != y[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - self.C)
                    H = min(self.C, alpha[i] + alpha[j])

                if L == H:
                    continue

                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alpha[j] -= y[j] * (E_i - E_j) / eta
                alpha[j] = np.clip(alpha[j], L, H)

                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    continue

                alpha[i] += y[i] * y[j] * (alpha_j_old - alpha[j])

                b1 = b - E_i - y[i] * (alpha[i] - alpha_i_old) * K[i, i] - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                b2 = b - E_j - y[i] * (alpha[i] - alpha_i_old) * K[i, j] - y[j] * (alpha[j] - alpha_j_old) * K[j, j]

                if 0 < alpha[i] < self.C:
                    b = b1
                elif 0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

            if passes >= self.max_passes:
                break

        sv = alpha > 1e-5
        model = {
            'support_vectors': X[sv],
            'alpha': alpha[sv],
            'y_sv': y[sv],
            'b': b
        }
        return model

    # fungsi untuk menghitung decision function (jarak ke hyperlane) untuk input X
    def _decision_function(self, model, X):
        K = self._kernel_function(X, model['support_vectors'])
        return np.dot(K, model['alpha'] * model['y_sv']) + model['b']

    # fungsi untuk menghasilkan prediksi kelas untuk input X
    def predict(self, X):
        if not self._fitted:
            raise Exception("Model belum dilatih. Panggil fit() terlebih dahulu.")

        if len(self.classes_) == 2:
            decision = self._decision_function(self.models[0][0], X)
            return np.where(decision >= 0, self.models[0][2], self.models[0][1])
        else:
            votes = np.zeros((X.shape[0], len(self.classes_)))
            for model, c1, c2 in self.models:
                decision = self._decision_function(model, X)
                idx1 = np.where(self.classes_ == c1)[0][0]
                idx2 = np.where(self.classes_ == c2)[0][0]
                votes[:, idx1] += (decision < 0).astype(int)
                votes[:, idx2] += (decision >= 0).astype(int)
            return self.classes_[np.argmax(votes, axis=1)]
        
# kelas Grid Search Cross-Validation (GridSearchCV) yang dibuat tanpa menggunakan library
class GridSearchCV:

    # fungsi untuk inisialisasi variabel-variabel yang akan dibutuhkan 
    def __init__(self, estimator, param_grid, cv=3):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_score_ = -float('inf')

    # fungsi untuk membentuk kombinasi parameter dari param_grid
    def _generate_param_combinations(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        all_combinations = list(product(*values))
        return [dict(zip(keys, combo)) for combo in all_combinations]

    # fungsi untuk melatih model berdasarkan seluruh kombinasi parameter menggunakan cross-validation
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples = len(X)
        fold_size = n_samples // self.cv
        param_combinations = self._generate_param_combinations()

        for params in param_combinations:
            total_score = 0

            rng = np.random.default_rng(seed=42)
            indices = rng.permutation(n_samples)
            folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(self.cv)]

            for fold in range(self.cv):
                val_indices = folds[fold]
                train_indices = np.hstack([folds[i] for i in range(self.cv) if i != fold])

                X_train, X_val = X[train_indices], X[val_indices]
                y_train, y_val = y[train_indices], y[val_indices]

                model = self.estimator.__class__(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                score = np.mean(y_pred == y_val)
                total_score += score

            avg_score = total_score / self.cv

            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

        self.best_estimator_ = self.estimator.__class__(**self.best_params_)
        self.best_estimator_.fit(X, y)

        return self

    # fungsi yang menggunakan estimator terbaik untuk melakukan prediksi
    def predict(self, X):
        if self.best_estimator_ is None:
            raise Exception("Model belum dilatih. Panggil fit() terlebih dahulu.")
        return self.best_estimator_.predict(X)
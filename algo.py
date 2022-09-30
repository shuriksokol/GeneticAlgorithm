import numpy as np
from sklearn.model_selection import train_test_split

"""
X_train, X_val, y_train, y_val = load_data()
genalgo = GeneticAlgorithm(LogisticRegression, accuracy_score)
genalgo.fit(X_train, y_train, X_val, y_val, select = 10)
important = genalgo.return_features()
"""

class GeneticAlgorithm:
    """
    Genetic algorithm for features selection
    """
    def __init__(self, model_, score_, asc_ = True):
        self.model = model_
        self.score = score_
        self.asc = 1 if asc_ else -1


    def fit(self, Xt_, yt_, Xv_ = None, yv_ = None, split = None, **kwargs):
        if Xv_ is None or yv_ is None:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(Xt_, yt_, train_size = split)
        else:
            self.X_train = Xt_
            self.X_val = Xv_
            self.y_train = yt_
            self.y_val = yv_
        self.n_features = self.X_train.shape[1]
        self.params = kwargs
        self.generate()
        self.accs = self.calc_accs(self.species)
        order = np.argsort(self.accs)
        self.accs = self.accs[order]
        self.species = self.species[order]
        self.max_accs = [self.accs[-1]]
        self.min_accs = [self.accs[0]]
        self.avg_accs = [np.average(self.accs)]
        if 'stop_abs' in self.params:
            stop_abs = self.params['stop_abs']
            while np.amax(self.accs) - np.amin(self.accs) > stop_abs:
                self.breed()
                self.selection()
                self.max_accs.append(self.accs[-1])
                self.min_accs.append(self.accs[0])
                self.avg_accs.append(np.average(self.accs))
        else:
            if 'stop_ratio' in self.params:
                stop_ratio = self.params['stop_ratio']
            else:
                stop_ratio = 0.01
            while (np.amax(self.accs) - np.amin(self.accs)) / np.amax(self.accs) > stop_ratio:
                self.breed()
                self.selection()
                self.max_accs.append(np.amax(self.accs))
                self.min_accs.append(np.amin(self.accs))
                self.avg_accs.append(np.average(self.accs))


    def generate(self):
        if 'select' in self.params:
            select = self.params['select']
        else:
            select = self.n_features // 2
        if 'overlap' in self.params:
            overlap = self.params['overlap']
        else:
            overlap = 4
        population = self.n_features * overlap // select
        species = []
        for _ in range(population):
            species.append(np.random.choice(self.n_features, size = select, replace = False))
        self.species = np.array(species)


    def calc_accs(self, species):
        accs = []
        for specy in species:
            model = self.model()
            X_part_train = self.X_train[:, specy]
            X_part_val = self.X_val[:, specy]
            model.fit(X_part_train, self.y_train)
            accs.append(self.score(self.y_val, model.predict(X_part_val)) * self.asc)
        return np.array(accs)


    def breed(self):
        if 'child_rate' in self.params:
            child_rate = self.params['child_rate']
        else:
            child_rate = 0.2
        children = int(child_rate * self.species.shape[0])
        parents = np.random.choice(self.species.shape[0], size = 2 * children, replace = False, p = self.accs / np.sum(self.accs))
        dads = self.species[parents[0::2]]
        moms = self.species[parents[1::2]]
        children = np.empty(shape = dads.shape, dtype = int)
        for i in range(dads.shape[0]):
            same = np.intersect1d(dads[i], moms[i], assume_unique = True)
            diff = np.concatenate((np.setdiff1d(dads[i], same, assume_unique = True), np.setdiff1d(moms[i], same, assume_unique = True)))
            children[i] = np.concatenate((same, np.random.choice(diff, size = diff.shape[0] / 2, replace = False)))
        self.species = np.concatenate((self.species, children))
        self.accs = np.concatenate((self.accs, self.calc_accs(children)))


    def selection(self):
        if 'child_rate' in self.params:
            child_rate = self.params['child_rate']
        else:
            child_rate = 0.2
        delete = int(child_rate * self.species.shape[0])
        if not np.all(self.accs[:-1] <= self.accs[1:]):
            order = np.argsort(self.accs)
            self.accs = self.accs[order]
            self.species = self.species[order]
        self.species = self.species[delete:]
        self.accs = self.accs[delete:]

    def get_best_features(self):
        return self.species[-1]


    def get_history(self):
        return self.max_accs, self.min_accs, self.avg_accs


    def get_max_accs(self):
        return self.max_accs


    def get_min_accs(self):
        return self.min_accs


    def get_avg_accs(self):
        return self.avg_accs

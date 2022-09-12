import numpy as np
from sklearn.model_selection import train_test_split

'''
X_train, X_val, y_train, y_val = load_data()
genalgo = GeneticAlgorithm(LogisticRegression, accuracy_score)
genalgo.fit(X_train, y_train, X_val, y_val, select = 10)
important = genalgo.return_features()
'''

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
        # Ещё параметры?
        self.generate()
        self.accs = self.calc_accs(self.species)
        


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

        parents = np.random.choice(self.species.shape[0], size = self.species.shape[0] // self.alpha * 2, replace = False, p = accs / np.sum(accs))
        dads = species[parents[0::2]]
        moms = species[parents[1::2]]
        bind = np.full(shape = (dads.shape[0], x_train.shape[1]), fill_value = False)
        bind[np.arange(bind.shape[0])[:, None], dads] = True
        binm = np.full(shape = (moms.shape[0], x_train.shape[1]), fill_value = False)
        binm[np.arange(binm.shape[0])[:, None], moms] = True
        same = bind * binm
        diff = bind ^ binm
        children = np.empty(shape = dads.shape, dtype = int)
        for i in range(dads.shape[0]):
            det = np.nonzero(same[i])[0]
            if diff[i].any():
                und = np.random.choice(np.nonzero(diff[i])[0], size = dads.shape[1] - det.shape[0], replace = False)
            else:
                und = np.array([], dtype = int)
            child = np.concatenate((det, und))
            if child.shape[0] > children.shape[1]:
                child = child[:children.shape[1]]
            elif child.shape[0] < children.shape[1]:
                temp = np.full(shape = x_train.shape[1], fill_value = 1.0)
                temp[child] = 0.0
                temp /= np.sum(temp)
                add = np.random.choice(temp.shape[0], size = children.shape[1] - child.shape[0], replace = False, p = temp)
                child = np.concatenate((child, add))
            children[i] = child
        new_species = np.concatenate((species, children))
        new_accs = np.concatenate((accs, calc_accs(children)))
        return new_species, new_accs'''
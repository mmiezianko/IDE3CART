
import collections
from decision_tree.algorithm import *
import itertools
class DecisionTree:
    id_iter = itertools.count()
    """Binary tree implementation with true and false branch. """
    def __init__(self, col=-1, value=None, branch_with_value=None, branch_with_others=None, outputs=None):
        self.id = next(DecisionTree.id_iter)
        self.branch_with_value = branch_with_value
        self.branch_with_others = branch_with_others
        self.value = value
        self.col = col # kolumny
        self.outputs = outputs # None for nodes, not None for leaves


def grow_tree(data, algorithm_fun = entropy):
    """Grows and then returns a binary decision tree.
    algorithm_fun: entropy or gini"""

    # data -> rekordy tabeli; jeżeli są = 0 to zwracamy puste drzewo
    if len(data) == 0: return DecisionTree()
    # algorithm_fun dla zbioru danych w pierwszej interacji bedzie sie odnosił do entropii całego zbioru danych.
    # w kolenych iteracjach będzie to entropia poprzedniej decyzji
    current_result = algorithm_fun(data) # obliczanie entropii ukadu

    # best gain to najwyzszy wskaznik jakosci
    best_info_gain = 0.0
    best_value_labelled = None
    best_subsets = None

    col_num = len(data[0]) - 1  # zliczanie ilości zmiennych opisujących. ostatnia kolumna to target dlatego - 1
    for col in range(col_num):
        # iterowanie po zmiennych opisujących. zmienna values_of_column będzie zawierać listę wszystkich zmiennych dla danej kolumny
        # czyli dla ostatniej kolumny [zmeczony, srednie, wypoczety, srednie, srednie ... itd ]
        values_of_column = [row[col] for row in data]

        for value in values_of_column:
            # wyciągamy unikalne wartości (np. kobieta/mężczyzna, przedziały wiekowe itd.)
            # subset1, subset2 to są podzbiory: w jednym jest dana unikalna wartość,
            # w drugim znajduje się pozostała reszta unikalnych wartości
            (subset1, subset2) = set_splitter(data, col, value)

            # p to prawdopodobieństwo wystąpienia danej wartości
            p = float(len(subset1)) / len(data)
            #wyliczenie zysku  -> szukamy max
            info_gain = current_result - p*algorithm_fun(subset1) - (1-p)*algorithm_fun(subset2)
            #jeśli podzbiory nie są puste (czyli mamy z czego podział) i info_gain jest większy od best_info_gain to:
            if info_gain > best_info_gain and len(subset1)>0 and len(subset2) > 0:
                best_info_gain = info_gain
                best_value_labelled = (col, value) #col to nazwa zmiennej decyzyjnej value to jej wartość np. płeć, kobieta
                best_subsets = (subset1, subset2)

    # jeżeli wystąpił zysk to dzielimy dalej -> powtarzamy proces
    if best_info_gain > 0: # gdyby dać tu 0.5 to byłby prepruning
        #rekurencja
        branch_with_value = grow_tree(best_subsets[0]) # gałąź ktora zawiera daną cechę (bardziej informatywną)
        branch_with_others =grow_tree(best_subsets[1]) # gałąź z resztą cech
        # branch_with_value i branch_with_others zapewniają rekurencję, czyli "zadajemy" kolejne pytania, tak długo az zysk informacyjny = 0
        # depth drzewa rozwija się właśnie na tym etapie. W sytuacji gdy zysk informacyjny = 0, uruchamiany jest else, w którym obliczane są
        # wystąpienia danej etykiety w liściu
        # depth -> czyli głębokość drzewa to kolejne instancje klasy Decision Tree - węzły
        return DecisionTree(col=best_value_labelled[0], value=best_value_labelled[1], branch_with_value=branch_with_value, branch_with_others=branch_with_others)
    else:
        # zwraca liczebnośći labeli w formie słownika np. skały: 10, dom: 9, ścianka: 5
        return DecisionTree(outputs=unique_labels_counter(data))
    # DecisionTree zawiera wskaźniki na instancje klasy DecisionTree ( te wskaźniki to gałęxie - "małe drzewa")

# jest to tzw. post pruning. Robimy to aby zrozumieć lepiej działanie drzewa, ale równie dobrze można by zrobić
# pre pruning czyli obcinanie już na poziomie tworzenia drzewa ustwiając if best_info_gain > 0.5:
def prune(tree, minGain, algorithm_fun=entropy, noticication=False): # tree to jest węzeł i dwa podproblemy. Drzewo zawiera w
    """Prunes the obtained tree according to the minimal gain (entropy or Gini). """
    
    # wywołujemy funkcje prune rekurencyjnie dla każdego brancha - dziecka (jest nim drzewo, argument tree)
    
    # Jeśli output gałązi z wartością jest pusty (czyli jeśli nie jest to liść)  
    if tree.branch_with_value.outputs == None: 
        prune(tree.branch_with_value, minGain, algorithm_fun, noticication) # przechodzimy do koljenego poziomu

    # Jeśli output gałązi z resztą wartości jest pusty (czyli jeśli nie jest to liść)  
    if tree.branch_with_others.outputs == None: 
        prune(tree.branch_with_others, minGain, algorithm_fun, noticication) # przechodzimy do koljenego poziomu

    # merge leaves (potentionally)
    # jeśli jesteśmy już w liściu:
    
    if tree.branch_with_value.outputs != None and tree.branch_with_others.outputs != None:
        output_of_branch_with_value, output_of_branch_with_others = [], []  #

        # tutaj iterujemy po key oraz po value outputs (czyli np. skały:2, dom:3, ściana:4) i otrzymujemy listę
        # gdzie jest tyle wsytąpień labela, co wartość w słowniku (czyli np. skały, skały, dom, dom, dom)
        for v, c in tree.branch_with_value.outputs.items(): # items odnosi się do key i value w słowniku, czyli label wraz z liczebnością
            for i in range(c):
                output_of_branch_with_value.append(v)
        for v, c in tree.branch_with_others.outputs.items():
            for i in range(c):
                output_of_branch_with_others.append(v)

        # powtarzamy operacje z funkcji grow_tree

        # p to prawdopodobieństwo wystąpienia danej wartości
        p = float(len(output_of_branch_with_value)) / len(output_of_branch_with_value + output_of_branch_with_others)
        # wyliczenie zysku  -> szukamy max. feature_importance to inaczej zysk informacyjny, który wskazuje na to czy liście powinny zostać obcięte (czy podział na gałęzie niesie zysk informacyjny)
        feature_importance = algorithm_fun(output_of_branch_with_value + output_of_branch_with_others) - p*algorithm_fun(output_of_branch_with_value) - (1-p)*algorithm_fun( output_of_branch_with_others)
        if feature_importance < minGain: # minGain to nasz treshold
            if noticication: print('Nastąpił pruning: zysk informacyjny = %f' % feature_importance)
            tree.branch_with_value, tree.branch_with_others = None, None # ucięcie liści
            tree.outputs = unique_labels_counter(output_of_branch_with_value + output_of_branch_with_others) # zawiązanie liścia w miejsce następnika


def predict(samples, tree_model, dataMissing=False):
    """Classifies the sampless according to the tree.
    dataMissing: true or false if data are missing or not. """

    # w tej funkcji sprawdzam które prawdopodobieństwo z outputs było największe
    # czyli np. jak mamy w outputs 2/10 kobiet w wieku > 30 lat mieszkających  na wsi = internet, 6/10 ... = prasa, 2/10 telewizja
    # to zaklasyfikuje nam, że  kobiety w wieku > 30 lat mieszkające  na wsi czytają prasę
    def classifyWithoutMissingData(samples, tree_model):
        if tree_model.outputs != None:  # liść
            return tree_model.outputs
        else:
            v = samples[tree_model.col] # col=best_value_labelled[0] czyli label
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree_model.value:
                    branch = tree_model.branch_with_value
                else:
                    branch = tree_model.branch_with_others
            else:
                if v == tree_model.value:
                    branch = tree_model.branch_with_value
                else:
                    branch = tree_model.branch_with_others
        return classifyWithoutMissingData(samples, branch)


    def classifyWithMissingData(samples, tree_model):
        if tree_model.outputs != None:  # leaf
            return tree_model.outputs
        else:
            v = samples[tree_model.col]
            if v == None:
                tr = classifyWithMissingData(samples, tree_model.branch_with_value)
                fr = classifyWithMissingData(samples, tree_model.branch_with_others)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount)/(tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                result = collections.defaultdict(int) # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
                for k, v in tr.items(): result[k] += v*tw
                for k, v in fr.items(): result[k] += v*fw
                return dict(result)
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree_model.value: branch = tree_model.branch_with_value
                    else: branch = tree_model.branch_with_others
                else:
                    if v == tree_model.value: branch = tree_model.branch_with_value
                    else: branch = tree_model.branch_with_others
            return classifyWithMissingData(samples, branch)

    # function body
    if dataMissing:
        return classifyWithMissingData(samples, tree_model)
    else:
        return classifyWithoutMissingData(samples, tree_model)


def plot(decisionTree):

    def toString(decisionTree, indent=''):
        if decisionTree.outputs != None:  # leaf node
            return str(decisionTree.outputs)
        else:
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = f'id={decisionTree.__hash__()} Column {decisionTree.col}: x >= {decisionTree.value}?'
            else:
                decision = f'id={decisionTree.__hash__()},Column {decisionTree.col}: x == {decisionTree.value}?'
            branch_with_value = indent + 'yes -> ' + toString(decisionTree.branch_with_value, indent + '\t\t')
            branch_with_others = indent + 'no  -> ' + toString(decisionTree.branch_with_others, indent + '\t\t')
            return (decision + '\n' + branch_with_value + '\n' + branch_with_others)

    print(toString(decisionTree))



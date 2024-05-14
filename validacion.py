import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold

# Función para verificar si dos conjuntos de datos son disjuntos
def are_disjoint(set1, set2):
    intersection = set(set1).intersection(set(set2))
    return len(intersection) == 0

# Función para comprobar la proporción de clases en un conjunto de datos
def check_class_proportion(data):
    _, counts = np.unique(data, return_counts=True)
    return counts

# Método de Hold Out 70/30 estratificado
def hold_out_stratified(data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, stratify=target)
    return X_train, X_test, y_train, y_test

# Método de 10-Fold Cross-Validation estratificado
def cross_validation_stratified(data, target):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    splits = list(skf.split(data, target))
    return splits

# Cargar los conjuntos de datos
iris_data = load_iris()
wine_data = load_wine()
cancer_data = load_breast_cancer()

# Conjuntos de datos
iris_X = iris_data.data
iris_y = iris_data.target

wine_X = wine_data.data
wine_y = wine_data.target

cancer_X = cancer_data.data
cancer_y = cancer_data.target

# Aplicar métodos de validación
print("Validación con Iris Dataset:")
print("Hold Out 70/30:")
iris_X_train, iris_X_test, iris_y_train, iris_y_test = hold_out_stratified(iris_X, iris_y)
print("Conjunto de entrenamiento:", check_class_proportion(iris_y_train))
print("Conjunto de prueba:", check_class_proportion(iris_y_test))
print("Son disjuntos:", are_disjoint(range(len(iris_X_train)), range(len(iris_X_test))))

print("\n10-Fold Cross-Validation:")
iris_splits = cross_validation_stratified(iris_X, iris_y)
for i, (train_index, test_index) in enumerate(iris_splits):
    print("Fold", i+1, ":")
    print("Conjunto de entrenamiento:", check_class_proportion(iris_y[train_index]))
    print("Conjunto de prueba:", check_class_proportion(iris_y[test_index]))
    print("Son disjuntos:", are_disjoint(train_index, test_index))

print("\nValidación con Wine Dataset:")
print("Hold Out 70/30:")
wine_X_train, wine_X_test, wine_y_train, wine_y_test = hold_out_stratified(wine_X, wine_y)
print("Conjunto de entrenamiento:", check_class_proportion(wine_y_train))
print("Conjunto de prueba:", check_class_proportion(wine_y_test))
print("Son disjuntos:", are_disjoint(range(len(wine_X_train)), range(len(wine_X_test))))

print("\n10-Fold Cross-Validation:")
wine_splits = cross_validation_stratified(wine_X, wine_y)
for i, (train_index, test_index) in enumerate(wine_splits):
    print("Fold", i+1, ":")
    print("Conjunto de entrenamiento:", check_class_proportion(wine_y[train_index]))
    print("Conjunto de prueba:", check_class_proportion(wine_y[test_index]))
    print("Son disjuntos:", are_disjoint(train_index, test_index))

print("\nValidación con Breast Cancer Dataset:")
print("Hold Out 70/30:")
cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = hold_out_stratified(cancer_X, cancer_y)
print("Conjunto de entrenamiento:", check_class_proportion(cancer_y_train))
print("Conjunto de prueba:", check_class_proportion(cancer_y_test))
print("Son disjuntos:", are_disjoint(range(len(cancer_X_train)), range(len(cancer_X_test))))

print("\n10-Fold Cross-Validation:")
cancer_splits = cross_validation_stratified(cancer_X, cancer_y)
for i, (train_index, test_index) in enumerate(cancer_splits):
    print("Fold", i+1, ":")
    print("Conjunto de entrenamiento:", check_class_proportion(cancer_y[train_index]))
    print("Conjunto de prueba:", check_class_proportion(cancer_y[test_index]))
    print("Son disjuntos:", are_disjoint(train_index, test_index))
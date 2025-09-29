# model.py - Enhanced Heart Disease Prediction Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import random


# ===== Load and Prepare the Data =====
def load_data():
    """
    Reads the heart disease dataset and splits it into:
    - df: complete dataframe
    - X: the features (input variables)
    - y: the target (output we want to predict)
    """
    df = pd.read_csv("dataset.csv")  # Load the dataset from a CSV file
    X = df.drop("target", axis=1)  # Remove the target column to get features
    y = df["target"]  # Target column shows presence of heart disease
    return df, X, y


# ===== Feature Selection using Statistical Method =====
def select_best_features(X, y, k=10):
    """
    Selects the k best features using statistical test (f_classif).
    This helps remove less important features and improve model performance.
    """
    # Use SelectKBest to find the most important features
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    # Get the names of selected features
    selected_features = X.columns[selector.get_support()].tolist()

    return X_selected, selected_features, selector


# ===== Feature Extraction - Create New Features =====
def extract_features(df):
    """
    Creates new meaningful features from existing ones.
    This can help the model find better patterns in the data.
    """
    df_enhanced = df.copy()

    # Risk score based on multiple factors
    df_enhanced['risk_score'] = (
            df_enhanced['age'] * 0.1 +  # Age factor
            df_enhanced['chol'] * 0.01 +  # Cholesterol factor
            df_enhanced['trestbps'] * 0.01 +  # Blood pressure factor
            df_enhanced['cp'] * 2 +  # Chest pain importance
            df_enhanced['exang'] * 3  # Exercise angina importance
    )

    # Age groups (young, middle-aged, elderly)
    df_enhanced['age_group'] = pd.cut(df_enhanced['age'],
                                      bins=[0, 40, 60, 100],
                                      labels=[0, 1, 2])

    # Cholesterol risk level
    df_enhanced['chol_risk'] = pd.cut(df_enhanced['chol'],
                                      bins=[0, 200, 240, 1000],
                                      labels=[0, 1, 2])

    # Blood pressure categories
    df_enhanced['bp_category'] = pd.cut(df_enhanced['trestbps'],
                                        bins=[0, 120, 140, 300],
                                        labels=[0, 1, 2])

    return df_enhanced


# ===== Simple Genetic Algorithm for Feature Selection =====
class SimpleGeneticAlgorithm:
    """
    A simple genetic algorithm to find the best combination of features.
    This is a metaheuristic optimization technique.
    """

    def __init__(self, X, y, population_size=20, generations=10):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.feature_count = X.shape[1]

    def create_individual(self):
        """Create a random individual (feature combination)"""
        # Each individual is a binary list: 1=use feature, 0=don't use
        return [random.randint(0, 1) for _ in range(self.feature_count)]

    def fitness(self, individual):
        """Calculate fitness (accuracy) of an individual"""
        # Select features based on individual's genes
        selected_features = [i for i, gene in enumerate(individual) if gene == 1]

        # Need at least 2 features to train a model
        if len(selected_features) < 2:
            return 0

        X_selected = self.X.iloc[:, selected_features]

        # Quick train-test split for fitness evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.y, test_size=0.3, random_state=42
        )

        # Train a simple model and get accuracy
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)

    def crossover(self, parent1, parent2):
        """Create offspring by combining two parents"""
        crossover_point = random.randint(1, self.feature_count - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, individual, mutation_rate=0.1):
        """Randomly change some genes in an individual"""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # Flip the bit
        return individual

    def run(self):
        """Run the genetic algorithm"""
        # Create initial population
        population = [self.create_individual() for _ in range(self.population_size)]

        best_fitness = 0
        best_individual = None

        for generation in range(self.generations):
            # Calculate fitness for each individual
            fitness_scores = [self.fitness(ind) for ind in population]

            # Find the best individual in this generation
            max_fitness = max(fitness_scores)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = population[fitness_scores.index(max_fitness)]

            # Create new population
            new_population = []

            # Keep the best individuals (elitism)
            sorted_pop = sorted(zip(population, fitness_scores),
                                key=lambda x: x[1], reverse=True)
            new_population.extend([ind for ind, _ in sorted_pop[:2]])

            # Generate new individuals through crossover and mutation
            while len(new_population) < self.population_size:
                # Select parents (tournament selection)
                parent1 = random.choice([ind for ind, _ in sorted_pop[:10]])
                parent2 = random.choice([ind for ind, _ in sorted_pop[:10]])

                # Create child and mutate
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        return best_individual, best_fitness


# ===== Apply Genetic Algorithm for Feature Selection =====
def genetic_feature_selection(X, y):
    """
    Use genetic algorithm to find the best features.
    Returns selected features and their names.
    """
    ga = SimpleGeneticAlgorithm(X, y, population_size=15, generations=8)
    best_features, best_score = ga.run()

    # Get selected feature indices and names
    selected_indices = [i for i, gene in enumerate(best_features) if gene == 1]
    selected_feature_names = X.columns[selected_indices].tolist()

    # Apply feature selection
    X_selected = X.iloc[:, selected_indices]

    return X_selected, selected_feature_names, best_score


# ===== Split Data into Train/Test Sets =====
def split_data(X, y, test_size=0.2):
    """
    Divides the dataset into training and testing parts.
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


# ===== Train the Selected Machine Learning Model =====
def train_model(model_name, X_train, y_train):
    """
    Trains either a Random Forest or Logistic Regression model,
    depending on what the user selected.
    """
    if model_name == "Random Forest":
        # Random Forest: combines many decision trees
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Logistic Regression":
        # Logistic Regression: good for binary classification
        model = LogisticRegression(max_iter=1000, random_state=42)

    # Train the model with the training data
    model.fit(X_train, y_train)
    return model


# ===== Model Evaluation =====
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and returns accuracy and predictions.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, predictions
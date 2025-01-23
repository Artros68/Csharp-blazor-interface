#!/usr/bin/env python
# coding: utf-8

# # Alogrithme d'optimisation énergétique

# ## Traitement des données

# Le prétraitement des données, également connu sous le nom de traitement des données, est une phase essentielle de l'apprentissage automatique. Son objectif principal est de garantir la qualité, la cohérence et la pertinence des données exploitées par les algorithmes. Cette étape constitue le point de départ de notre projet.

# ### Import et chargement des données

# Nous allons à présent commencer avec l'import des bibliothèques nécessaire à notre algorithme

# In[29]:


import numpy as np #pour la manipulation des données sous forme de DataFrame
import matplotlib.pyplot as plt #pour les calculs numériques
import pandas as pd #pour la visualisation des données
import seaborn as sns #pour la visualisation des données
import zipfile #pour extraire les fichiers
import random
import json
import os

from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score,precision_score, recall_score, make_scorer, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
from datetime import timedelta


# Puis nous allons importer et charge notre dataset

# In[49]:


# Vérification que le fichier existe
if os.path.exists(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path, sep=';')  # Lecture avec le bon séparateur
        df = df.copy()  # Création d'une copie après la lecture
        print(df.head(10))  # Affichage des 10 premières lignes sous forme de tableau
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
else:
    print(f"Le fichier {csv_file_path} n'a pas été trouvé.")


# ### Lecture et visualisation des données

# Jetons un premier coup d'oeil aux données et voyons à quoi elles ressemblent.

# In[31]:


# Vérification des types de données
df.info()


# In[32]:


# Affichage des statistiques descriptives
df.describe().T.style.background_gradient(cmap="Dark2")


# In[33]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# In[34]:


for col in df.select_dtypes(include=['number']).columns:
# Visualisation (boîte à moustaches)
        plt.figure()
        plt.boxplot(df[col])
        plt.title(f"Boxplot for {col}")
        plt.show()


# In[35]:


# Correlation heatmap
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# ### Nettoyage des données

# Commmençons le nettoyage, afin d'avoir un jeu de données propre.

# In[36]:


# Vérification de la présence de valeurs manquantes
df.isnull().sum()


# In[37]:


# Vérification de la présence de valeurs en double dans les données et suppressure de celles-ci :
duplicated_features=df.duplicated().sum()
print("Number of duplicates ----->>> ",duplicated_features)
df = df.drop_duplicates()
duplicated_features=df.duplicated().sum()
print("Number of duplicates of cleaning it ----->>> ",duplicated_features)


# In[38]:


# Création d'une nouvelle colonne avec les résultats de la division
df['Rendement'] = df['Puissance utile (kW)'] / df['Puissance absorbée (kW)']

# Affichage du DataFrame avec la nouvelle colonne
print(df)


# ### Détection d'anomalies

# In[39]:


# Fonction pour créer dynamiquement le dictionnaire des consommations par équipement, en excluant celles à 0
def create_equipment_consumptions_dict_by_row(df, period='1Y'):
    equipment_consumptions = {}
    # Combiner la colonne 'Date' avec la colonne 'Horodatage' pour créer une colonne complète de datetime
    df['Horodatage_complet'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Horodatage'].astype(str), format='%d/%m/%Y %H:%M')
    
    # Calculer la date limite pour la période spécifiée (par exemple: 1 an)
    period_date = df['Horodatage_complet'].max() - timedelta(days=365)  # Période d'un an
    df_recent = df[df['Horodatage_complet'] >= period_date]  # Filtrer les données pour la période récente

    # Identifier les équipements uniques
    unique_equipment = df['Nom de l\'équipement'].unique()

    # Boucle pour attribuer les consommations pour chaque équipement
    for equipment in unique_equipment:
        equipment_mask = df["Nom de l'équipement"] == equipment
        df_equipment = df.loc[equipment_mask]

        # Liste pour stocker les consommations valides pour cet équipement
        equipment_consumptions[equipment] = []

        # Pour chaque ligne de l'équipement, vérifier les consommations non nulles
        for _, row in df_equipment.iterrows():
            # Chercher dynamiquement les colonnes contenant des consommations pour cet équipement
            for col in df_equipment.columns:
                if "Consommation" in col and row[col] > 0:
                    if col not in equipment_consumptions[equipment]:
                        equipment_consumptions[equipment].append(col)

    return equipment_consumptions

# Exemple d'utilisation
equipment_consumptions = create_equipment_consumptions_dict_by_row(df, period='1Y')

# Afficher le dictionnaire créé
print(equipment_consumptions)


# In[40]:


# Extraire l'heure et définir jour/nuit
df['hour'] = df['Horodatage_complet'].dt.hour
df['minute'] = df['Horodatage_complet'].dt.minute
df['day_night'] = df['hour'].apply(lambda x: 'day' if 6 <= x < 22 else 'night')  # Modifiée pour exclure 22h00

# Fonction pour détecter les anomalies et afficher un message spécifique pour chaque anomalie, jour/nuit pris en compte
def detect_anomalies_and_message_with_night_adjustment(df, equipment_consumptions):
    anomalies = pd.Series([0] * len(df), index=df.index)  # Initialiser la série des anomalies
    anomaly_messages = []  # Liste pour stocker les messages des anomalies
    
    # Boucle sur chaque équipement et ses consommations associées
    for equipment, consumptions in equipment_consumptions.items():
        equipment_mask = df["Nom de l'équipement"] == equipment
        df_equipment = df.loc[equipment_mask]
        
        # Appliquer la réduction de consommation de nuit (seulement sur les périodes de nuit)
        for col in consumptions:
            if col not in df.columns:
                continue  # Si la colonne n'existe pas, on la saute

            # Distinguer les périodes de jour et de nuit
            for period in ['day', 'night']:
                df_period = df_equipment[df_equipment['day_night'] == period].copy()  # Créer une copie

                # Appliquer un facteur de réduction pour la nuit (par exemple 0.5) sur les consommations de nuit
                if period == 'night':
                    df_period[col] = df_period[col] * 0.5  # Réduire la consommation de nuit

                # Appliquer Isolation Forest pour détecter les anomalies
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)  # 10% d'anomalies attendues
                scores = isolation_forest.fit_predict(df_period[[col]])  # Appliquer IsolationForest sur les données
                anomaly_indices = df_period.index[scores == -1]  # Anomalies détectées par IsolationForest
                anomalies.loc[anomaly_indices] = 1  # Marquer les anomalies comme 1

                # Ajouter les messages spécifiques pour les anomalies
                for index in anomaly_indices:
                    timestamp = df_period.loc[index, 'Horodatage_complet']
                    value = df_period.loc[index, col]
                    reference = df_period.loc[index, 'Référence de l\'équipement']
                    
                    # Construire le type de consommation
                    if 'électrique' in col.lower():
                        consumption_type = 'consommation électrique'
                    elif 'gaz' in col.lower():
                        consumption_type = 'consommation de gaz'
                    elif 'fioul' in col.lower():
                        consumption_type = 'consommation de fioul'
                    elif 'eau' in col.lower():
                        consumption_type = 'consommation d\'eau'
                    else:
                        consumption_type = 'autre consommation'
                    
                    # Ajouter l'indication jour/nuit dans le message
                    anomaly_messages.append(f"La {consumption_type} de la {equipment} (Référence: {reference}) le {timestamp} ({value}) en période {period} constitue une anomalie.")

    return anomalies, anomaly_messages

# Appliquer la détection des anomalies et obtenir les messages
df['Anomalies'], anomaly_messages = detect_anomalies_and_message_with_night_adjustment(df, equipment_consumptions)

# Afficher les messages des anomalies
# A l'exportation de l'algorithme, il faudra enlever le size
sample_size = 5
for message in anomaly_messages[:sample_size]:
    print(message)

# Calculer le pourcentage d'anomalies détectées
anomaly_percentage = df['Anomalies'].mean() * 100

# Afficher le résultat
print(f"Pourcentage d'anomalies détectées : {anomaly_percentage:.2f}%")

# Afficher un échantillon des résultats
print(df[['Horodatage_complet', 'Nom de l\'équipement', 'Référence de l\'équipement', 'Anomalies']])


# ### Exploitation des résultats de la détection

# In[41]:


# Filtrer les anomalies
df_anomalies = df[df['Anomalies'] == 1].copy()  # Utiliser copy() pour éviter l'avertissement

# Convertir les timestamps en chaîne de caractères dans un format lisible
df_anomalies['Horodatage_complet'] = df_anomalies['Horodatage_complet'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Exporter les anomalies triées par date
df_anomalies = df_anomalies.sort_values(by=['Nom de l\'équipement','Date','Horodatage'])

# Convertir le DataFrame en dictionnaire, puis en JSON avec une mise en forme lisible
json_data = df_anomalies.to_dict(orient='records')
with open('anomalies.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)  # Mise en forme lisible


# ### Recommandation personalisée 

# In[42]:


def generate_recommendations(anomaly_messages):
    recommendations = []  # Liste pour stocker les recommandations
    for message in anomaly_messages:
        if "consommation électrique" in message:
            if "en période night" in message:
                recommendations.append("Vérifiez si l'équipement est programmé pour fonctionner la nuit ou s'il y a un problème de fuite d'énergie.")
            else:
                recommendations.append("Considérez des optimisations d'énergie ou vérifiez la charge de travail de l'équipement.")
        
        elif "consommation de gaz" in message or "consommation de fioul" in message:
            recommendations.append("Inspectez les brûleurs ou les conduites pour des fuites possibles ou des dysfonctionnements.")
        
        elif "consommation d'eau" in message:
            recommendations.append("Recherchez des fuites ou des vannes laissées ouvertes par inadvertance.")
        
        else:
            recommendations.append("Consultez un technicien pour une inspection approfondie.")

    # Associer les recommandations aux messages
    return [{"anomaly": msg, "recommendation": rec} for msg, rec in zip(anomaly_messages, recommendations)]

# Utiliser les recommandations sur les anomalies détectées
anomaly_recommendations = generate_recommendations(anomaly_messages)

# Afficher quelques recommandations pour vérification
for item in anomaly_recommendations[:10]:
    print(f"Anomalie: {item['anomaly']}\nRecommandation: {item['recommendation']}\n")


# ### Prédiction de la consommation

# Pour cette partie, nous allons utilisé l'agorithme XGboost, et des métriques pour savoir si notre algorithme est fiable.
# Un MAE faible (proche de zéro) signifie que le modèle fait des prédictions proches des valeurs réelles.
# Le MSE est un indicateur complémentaire qui nous permets de mieux cerner l’efficacité de votre modèle. Plus il est faible, mieux c'est.
# Le code est divisé en 3 parties :
# - Boucle principale : Pour chaque équipement et chaque type de consommation (électricité, gaz, fioul, etc.), le modèle XGBoost prédit la consommation sur des données historiques.
# - Prévision : Une fois le modèle entraîné, il effectue une prédiction pour les 720 heures suivantes.
# - Enregistrement des résultats : Les résultats sont enregistrés dans des fichiers CSV pour chaque équipement et type de consommation.

# In[43]:


# Générer le dictionnaire des consommations par équipement
equipment_consumptions = create_equipment_consumptions_dict_by_row(df, period='1Y')

# Prévision pour chaque équipement et consommation
for equipment, consumptions in equipment_consumptions.items():
    print(f"\nPrédictions pour l'équipement : {equipment}")
    
    for consumption in consumptions:
        print(f"  Prévisions pour : {consumption}")
        
        # Préparation des données
        df_equipment = df[df["Nom de l'équipement"] == equipment]
        df_consumption = df_equipment[['Horodatage_complet', consumption]].dropna()
        df_consumption['hour'] = df_consumption['Horodatage_complet'].dt.hour
        df_consumption['day'] = df_consumption['Horodatage_complet'].dt.day
        df_consumption['month'] = df_consumption['Horodatage_complet'].dt.month
        df_consumption['year'] = df_consumption['Horodatage_complet'].dt.year

        X = df_consumption[['hour', 'day', 'month', 'year']]
        y = df_consumption[consumption]

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modèle XGBoost
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"    MAE pour {consumption}: {mae}")
        print(f"    MSE pour {consumption}: {mse}")

        # Prévision future
        future_dates = pd.date_range(start=df_consumption['Horodatage_complet'].max(), periods=720, freq='h')
        future_data = pd.DataFrame({
            'hour': future_dates.hour,
            'day': future_dates.day,
            'month': future_dates.month,
            'year': future_dates.year
        })
        future_pred = model.predict(future_data)

        # Stocker les résultats
        future_forecast = pd.DataFrame({
            'Horodatage_complet': future_dates,
            f'{consumption} prévue': future_pred
        })
        future_forecast.to_csv(f'Previsions_{equipment}_{consumption.replace(" ", "_")}.csv', index=False)

        print(f"    Prévisions enregistrées dans Previsions_{equipment}_{consumption.replace(' ', '_')}.csv")

print("Prédictions terminées pour tous les équipements et consommations.")


# ### Optimisation énergétique

# In[44]:


# Exemple de prix énergétiques (en euros)
prix_electricite = 0.2516  # Prix par kWh
prix_gaz = 0.1184  # Prix par kWh ou m³
prix_fioul = 0.1257  # Prix par L ou kWh

# Générer le dictionnaire des consommations par équipement
equipment_consumptions = create_equipment_consumptions_dict_by_row(df, period='1Y')

# Optimisation pour chaque équipement et chaque consommation
for equipment, consumptions in equipment_consumptions.items():
    print(f"\nOptimisation pour l'équipement : {equipment}")
    
    for consumption in consumptions:
        print(f"  Traitement pour : {consumption}")
        
        # Chargement des prévisions de consommation (doit être produit par le code de prédiction)
        forecast_file = f'Previsions_{equipment}_{consumption.replace(" ", "_")}.csv'
        forecast = pd.read_csv(forecast_file)

        # Vérifier les noms des colonnes disponibles
        print(f"Colonnes disponibles dans le fichier de prévision : {forecast.columns.tolist()}")

        # Assurer que 'Horodatage_complet' est en datetime
        forecast['Horodatage_complet'] = pd.to_datetime(forecast['Horodatage_complet'])

        # Calcul des coûts énergétiques avant optimisation
        forecast['Coût électricité'] = forecast[f'{consumption} prévue'] * prix_electricite
        forecast['Coût gaz'] = forecast[f'{consumption} prévue'] * prix_gaz
        forecast['Coût fioul'] = forecast[f'{consumption} prévue'] * prix_fioul
        forecast['Coût total'] = forecast['Coût électricité'] + forecast['Coût gaz'] + forecast['Coût fioul']

        # Optimisation : réduction de la consommation d'électricité pendant certaines heures
        forecast['Optimisation électricité'] = forecast.apply(
            lambda row: row[f'{consumption} prévue'] * 0.8 if 22 <= row['Horodatage_complet'].hour <= 6 else row[f'{consumption} prévue'],
            axis=1
        )
        
        # Recalcule du coût après optimisation
        forecast['Coût électricité optimisé'] = forecast['Optimisation électricité'] * prix_electricite
        forecast['Coût total optimisé'] = forecast['Coût électricité optimisé'] + forecast['Coût gaz'] + forecast['Coût fioul']

        # Calcul des économies
        forecast['Économies totales'] = forecast['Coût total'] - forecast['Coût total optimisé']

        # Résultats
        print(f"    Économies totales sur la période d'optimisation pour {consumption} : {forecast['Économies totales'].sum()} euros")

        # Sauvegarder les résultats dans un fichier CSV
        forecast.to_csv(f'Previsions_Optimisation_{equipment}_{consumption.replace(" ", "_")}.csv', index=False)

        print(f"    Résultats d'optimisation enregistrés dans Previsions_Optimisation_{equipment}_{consumption.replace(' ', '_')}.csv")

print("Optimisation terminée pour tous les équipements et consommations.")


# # Bonus 

# In[50]:


"""import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Simuler un monitoring énergétique en temps réel

def monitor_energy_consumption(df, equipment_col='Nom de l\'équipement', consumption_cols=['Consommation'], interval=5):
    
    Simule un monitoring énergétique en lisant les données de consommation à des intervalles définis.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données historiques.
    equipment_col (str): Le nom de la colonne contenant les équipements.
    consumption_cols (list): Liste des colonnes de consommation à surveiller.
    interval (int): Temps en secondes entre les mises à jour simulées.


    df_monitor = df.copy()
    df_monitor['Horodatage_complet'] = pd.to_datetime(df_monitor['Horodatage_complet'])
    df_monitor.set_index('Horodatage_complet', inplace=True)

    for i, row in df_monitor.iterrows():
        # Affichage des informations de consommation actuelles
        print(f"\nÉquipement: {row[equipment_col]} - Horodatage: {i}")
        for consumption_col in consumption_cols:
            consommation = row[consumption_col]
            print(f"{consumption_col}: {consommation:.2f} kWh")

        # Détection des pics en fonction d'un seuil dynamique basé sur les 24 dernières heures
        last_24h = df_monitor.loc[i - pd.Timedelta(hours=24):i]
        if not last_24h.empty:
            threshold = last_24h[consumption_cols].mean() + 2 * last_24h[consumption_cols].std()
            for consumption_col in consumption_cols:
                if consommation > threshold[consumption_col]:
                    print(f"ALERTE: Pic de consommation détecté pour {consumption_col}! (Valeur: {consommation:.2f}, Seuil: {threshold[consumption_col]:.2f})")

        # Pause pour simuler le temps réel
        time.sleep(interval)

# Chargement des données
csv_file_path = './equipment_dataset3.csv'
df = pd.read_csv(csv_file_path, sep=';')

# Simulation du monitoring
monitor_energy_consumption(
    df,
    equipment_col="Nom de l'équipement",
    consumption_cols=["Consommation électrique (kWh)", "Consommation de gaz (kWh ou m3)", "Consommation de fioul (L ou kWh)"],
    interval=2  # Changer cette valeur pour ajuster la vitesse de la simulation
)


# In[ ]:





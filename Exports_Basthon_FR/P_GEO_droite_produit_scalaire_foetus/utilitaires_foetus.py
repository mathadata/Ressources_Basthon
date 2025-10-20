#pip install hyperopt

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display
import sys
import json
import pickle
import importlib.util
from zipfile import ZipFile
from io import BytesIO
from math import sqrt
import requests

# permet de charger les fichiers matlab (*.mat)
from typing import Tuple
import builtins # to use builtins.RAW, just for dev test

import utilitaires_common as common
from utilitaires_common import *

DATASET = getattr(builtins, 'DATASET', 'artificiel_rebiased_2_cheated') # 'original', 'relabelled_std', 'relabelled_std_10s'

LOCAL = getattr(builtins, 'LOCAL', False)  # Pour utiliser les données locales (travail hors ligne)

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# function to interpolate NaN values in a numpy array
def interpolate_nans(data, max_nan_span=10):
    nans = np.isnan(data)
    if not np.any(nans):
        return data
    
# le répertoire de travail
directory = os.path.dirname(__file__)

def load_data(source=DATASET, local=LOCAL):
    if local:
        root = parent_dir+"/challenges/foetus/Data/"
        d_train_path = f'{root}d_train_{source}.pickle'
        r_train_path = f'{root}r_train_{source}.pickle'
        with open(d_train_path, 'rb') as f:
            d_train = pickle.load(f)

        with open(r_train_path, 'rb') as f:
            r_train = pickle.load(f)
        print(f"LOCAL données {source} chargées")
    else:
        inputs_zip_urls = [f"{files_url}/foetus.zip", "https://raw.githubusercontent.com/mathadata/data_challenges_lycee/main/foetus.zip"]
        loading_errors = []

        for url in inputs_zip_urls:
            try:
                inputs_zip = requests.get(url)
                break
            except Exception as e:
                loading_errors.append(f"erreur de chargement {url} : {e}")
                if debug:
                    print_error(f"failed to load from {url} : {e}")
        else:
            error = '\n'.join(loading_errors)
            pretty_print_error(f"Erreur du chargement des données. Essayez de recharger la page, relancer le notebook en suivant les instructions SOS ou sur un autre navigateur internet.")
            pretty_print_error(f'Si le problème persiste, transmettez cette erreur à votre professeur pour qu\'il nous l\'envoie :\n\n{error}')
            exit(1)

        zf = ZipFile(BytesIO(inputs_zip.content))
        zf.extractall()
        zf.close()
        d_train_path = f'd_train.pickle'
        r_train_path = f'r_train.pickle'
        with open(d_train_path, 'rb') as f:
            d_train = pickle.load(f)

        with open(r_train_path, 'rb') as f:
            r_train = pickle.load(f)
        # Suppresion du mot surce pour ne pas afficher le nom du dataset dans le notebook    
        print(f"données chargées")

    d_animation = d_train[10].copy()
    
    if source == 'relabelled_std':
        # reordonner les données : on veut que les 10 premières données soient [130 202 271 206 57 21 91 58 67 84]
        indices_debut = [130, 202, 271, 206, 57, 21, 91, 58, 67, 84]
        indexes = indices_debut + [i for i in range(0, len(d_train)) if i not in indices_debut]
        d_train_new = d_train[indexes]
        r_train_new = r_train[indexes]

        d_train = d_train_new.copy()
        r_train = r_train_new.copy()

    if source == 'relabelled_balanced':
        # reordonner les données : on veut que les 10 premières données soient [11, 0, 1, 6, 9, 19, 20, 22, 17, 12]
        indices_debut = [11, 0, 1, 6, 46, 19, 20, 22, 17, 12]
        indexes = indices_debut + [i for i in range(0, len(d_train)) if i not in indices_debut]
        d_train_new = d_train[indexes]
        r_train_new = r_train[indexes]
        d_train = d_train_new.copy()
        r_train = r_train_new.copy()
        
    return d_train, r_train, d_animation

# chargement des données
d_train, r_train, d_animation = load_data(source=DATASET, local=LOCAL)


# Import des strings pour lire l'export dans jupyter - A FAIRE AVANT IMPORT utilitaires_common

strings = {
    "dataname": "cas",
    "dataname_plural": "cas",
    "feminin": False,
    "contraction": False,
    "classes": ["Normal", "ALARMANT"],
    "r_petite_caracteristique": "Normal",
    "r_grande_caracteristique": "ALARMANT",
    "train_size": int(round(len(d_train), -2)),  # arrondi à la dizaine
    "objectif_score_droite": 21,
    "objectif_score_droite_custom": 10,
    "pt_retourprobleme": "(40; 20)"
}

classes = strings['classes']

#Nouvelle variable pour checker la validation de l'exercice de classification
exercice_classification_ok = False

# Fonction de mise à jour
def set_exercice_classification_ok():
    global exercice_classification_ok
    exercice_classification_ok = True
    #print(f"DEBUG: exercice_classification_ok mis à jour -> {exercice_classification_ok}")

d = d_train[0]
# d2 = d_train[167]
d2 = d_train[10]

v = 100

class Foetus(common.Challenge):
    def __init__(self):
        super().__init__()
        self.strings = strings
        self.d_train = d_train
        self.r_train = r_train
        self.d = d
        self.d2 = d2
        self.classes = strings['classes']
        self.r_petite_caracteristique = strings['r_petite_caracteristique']
        self.r_grande_caracteristique = strings['r_grande_caracteristique']
        self.vmin = 100
        self.vmax = 150

        # STATS
        # Moyenne histogramme
        self.carac_explanation = f"C'est la bonne réponse ! Les foetus {classes[1]} ont souvent une fréquence cardiaque avec plus de baisses que les foetus {classes[0]}. C'est pourquoi leur caractéristique est souvent plus grande."

        # GEO
        # Tracer 10 points
        self.dataset_10_points = d_train[0:10]
        self.labels_10_points = r_train[0:10]
        self.droite_10_points = {
            'm': 1,
            'p': 20,
        }

        # Droite produit scalaire
        # Pour les versions question_normal
        self.M_retourprobleme=(40,20)

        # Objectif de score avec les 2 caracs de référence pour passer à la suite
        self.objectif_score_droite = strings['objectif_score_droite']
        self.objectif_score_droite_custom = strings['objectif_score_droite_custom']

    def remplacer_d_train_sans_nan(self):
        global d_train, d
        d_train = d_train_no_nan.copy()
        d = d_train_no_nan[0].copy()
        self.d_train = d_train_no_nan.copy()
        self.d = d
        print()
        print("--- Nettoyage des données ---")

    def affichage_dix(self):
        for d in self.d_train[:10]:
            affichage(d)


    def display_custom_selection(self, id):
        d_sain = self.d_train[np.where(self.r_train == self.classes[0])][:2]
        d_malade = self.d_train[np.where(self.r_train == self.classes[1])][:2]

        data = np.concatenate([d_sain, d_malade])
        labels = [self.classes[0]] * len(d_sain) + [self.classes[1]] * len(d_malade)

        run_js(f"setTimeout(() => window.mathadata.setup_zone_selection('{id}', '{json.dumps(data, cls=NpEncoder)}', '{json.dumps(labels, cls=NpEncoder)}'), 500)")

    def display_custom_selection_2d(self, id):
        self.display_custom_selection(id)
    
    def caracteristique(self, d):
        # Calculate the percentage of values less than v (v = 100)
        # Filter out NaN values
        valid_values = d[~np.isnan(d)]
        # Count values less than v
        count_less_than_v = np.sum(valid_values < v)
        # Calculate percentage
        percentage = (count_less_than_v / len(valid_values)) * 100
        return percentage

    def caracteristique_ecart_type(self, d):
        return d[~np.isnan(d)].std()

    def caracteristique_vmin(self, d, vmin):
        """% de valeurs < vmin"""
        valid_values = d[~np.isnan(d)]
        count = np.sum(valid_values < vmin)
        return (count / len(valid_values)) * 100
        
    def caracteristique_vmax(self, d, vmax):
        """% de valeurs > vmax"""
        valid_values = d[~np.isnan(d)]
        count = np.sum(valid_values > vmax)
        return (count / len(valid_values)) * 100
        
    def deux_caracteristiques_custom(self, d):
        """Retourne un tuple (x,y) où:
           x = % valeurs < vmin
           y = % valeurs > vmax"""
        valid_values = d[~np.isnan(d)]
        
        if len(valid_values) == 0:
            return (0, 0)
            
        x = (np.sum(valid_values < self.vmin) / len(valid_values)) * 100
        y = (np.sum(valid_values > self.vmax) / len(valid_values)) * 100
        
        return (x, y)


    def caracteristique2(self, d):
        return np.nanmax(d) - np.nanmin(d)

    def deux_caracteristiques(self, d):
        if (self.vmin > self.vmax):
            return 100

        valid_values = d[~np.isnan(d)]
        
        y, x = np.sum(valid_values < self.vmin)/len(valid_values) * 100, np.sum(valid_values > self.vmax)/len(valid_values) * 100
        return x, y
    
    def caracteristique_custom(self, d):
        # Compute percentage of values in the range [vmin, vmax]
        if (self.vmin > self.vmax):
            return 100

        valid_values = d[~np.isnan(d)]
        
        count_out_of_range = np.sum(valid_values < self.vmin) + np.sum(valid_values > self.vmax)
        return count_out_of_range / len(valid_values) * 100

    # def deux_caracteristiques_custom(self, d):
    #     return (0,0)

    def cb_custom_score(self, score):
        if score < 0.2:
            # Avant de passer au bac à sable, on retire les NaN de d_train
            self.remplacer_d_train_sans_nan()
            return True
        else:
            print_error("Modifie vmin et vmax pour faire moins de 20% d'erreur et passer à la suite. Cherche les valeurs qui sont plus fréquentes pour les foetus malades.")
        
        return False

    def import_js_scripts(self):
        run_js("""
            define('chartjs/helpers', ['chartjs'], function(Chart) {
                return Chart.helpers;
            });

            require.config({
                paths: {
                    'chartjs-zoom-plugin': 'https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min',
                    'hammerjs': 'https://cdn.jsdelivr.net/npm/hammerjs@2.0.8/hammer.min' // Added Hammer.js path
                },
                map: {
                    'chartjs-zoom-plugin': {
                        'chart.js': 'chartjs',
                        'chartjs/helpers': 'chartjs' // Map 'chartjs/helpers' to 'chartjs'
                    }
                },
                shim: {
                    'chartjs-zoom-plugin': {
                        deps: ['chartjs', 'hammerjs'] // Specify dependencies
                    }
                }
            })

            require(['chartjs-zoom-plugin'])
        """)    

init_challenge(Foetus())

# Créer une version de la liste d_train sans les NaN
d_train_no_nan = [d[~np.isnan(d)] for d in d_train]

def affichage(d, start_minut:float = -60, duration_in_minuts: float = 30, display_mean: bool = True, interpolate_missing_values:bool = True, min_y_value: int = None, max_y_value: int = None):
    # fhr_full = loadmat(id_to_path[id])['fhr'].ravel()
    # start_idx = int(start_minut*4*60)
    # if start_idx<0: 
    #     start_idx+=len(fhr_full)
    # start_idx = max(0, start_idx-1)
    # end_idx = min( start_idx+1+int(duration_in_minuts*4*60) , len(fhr_full) )
    
    # fhr = fhr_full[start_idx:end_idx]
    fhr = d
    if interpolate_missing_values:
        fhr = interpolate_nans(fhr)
    
    # nombre d'éléments dans l'électrocardiogramme
    n = len(fhr)
    
    # Create an array of time points (assuming each heart rate measurement is taken at regular intervals)
    time_in_minuts = np.arange(n)/(60*4)
    # Plot the heart rate data
    plt.figure(figsize=(20, 10))
    plt.plot(time_in_minuts, fhr, linestyle='-', color='black', linewidth=1)
    # Formater les ticks pour afficher le temps au format hh:mm
    def format_func(time_in_minuts, tick_number):
        #time_in_minuts = int(time_in_minuts/(60*4))
        hours = int(time_in_minuts) // 60
        minutes = int(time_in_minuts) % 60
        return f'{hours:02d}:{minutes:02d}'
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    #comment = 'sujet sain' if id_to_target[id] == 0 else 'sujet malade'
    
    if display_mean:
        # observed_mean = int(moyenne(fhr_full))
        observed_mean = int(moyenne(fhr))
        plt.axhline(y=observed_mean, color='red', linewidth=1, linestyle='-', label='Moyenne: '+str(observed_mean))
        # Annotate the y-value on the vertical axis
        plt.text(x=0, y=observed_mean, s=f'{observed_mean:.0f}', color='red', va='center', ha='right')

    # Add labels and title
    plt.xlabel('Temps (minutes)', fontsize=15)
    plt.ylabel('Rythme cardiaques', fontsize=15)
    #plt.title(f"Electrocardiogramme pour un {comment} ({id})", fontsize=15)
    plt.xlim(0, max(time_in_minuts))
    plt.ylim(min_y_value or 60, max_y_value or len(d))
    # Display grid
    plt.grid(True)

    if display_mean:
        plt.legend()
    # Show the plot
    plt.show()

def affichage_tableau(d):
  df = pd.DataFrame(d,columns=['fréquence']).round(1)
  dligne =  df.transpose()
  display(dligne)

def affichage_donnee_courbe(d):
        df = pd.DataFrame(d).round(1)
        dligne =  df.transpose()
        display(dligne)
        # Utile ici la suite ? 
        affichage_html(d) 


def get_histogramme_data(d=None, id=None):
    # Define custom bin edges
    bin_edges = np.arange(np.nanfloor(d), np.nanmax(d) + 1, 1)  # Bins from 0 to max value in d with width 1
    data = d[~np.isnan(d)]
    counts, bins = np.histogram(data, bins=bin_edges)
    return counts, bins

# calcule la moyenne de la séquence ''fhr' en ignorant les NaN
def moyenne(fhr):
    return fhr[~np.isnan(fhr)].mean()

# calcul des 3 quartiles Q1 , Q2 (=mediane), Q3 (en ignorant les NaN)
def compute_quartiles_Q1_Q2_Q3(fhr) -> Tuple[float,float,float]:
    # Remove NaN values
    cleaned_data = fhr[~np.isnan(fhr)]
    Q1 = np.percentile(cleaned_data, 25)
    Q2 = np.percentile(cleaned_data, 50)  # This is the median
    Q3 = np.percentile(cleaned_data, 75)
    return Q1,Q2,Q3

# nombre d elements NaN dans la séquence 'fhr'
def nan_count(fhr):
    return np.count_nonzero(np.isnan(fhr))

random_frequencies = []
def affichage_10_frequences():
    random_frequencies.clear()
    while np.sum(np.array(random_frequencies) < v) == 0:
        d = d_train[np.random.randint(0, len(d_train))]
        random_frequencies.clear()
        for i in range(0, 10):
            fr = np.NAN
            while np.isnan(fr):
                fr = d[np.random.randint(0, len(d))].round(1)
            random_frequencies.append(fr)
        
    df = pd.DataFrame(random_frequencies, columns=['fréquence'])
    #dligne = pd.DataFrame([random_frequencies], columns=[f'fréquence_{i+1}' for i in range(10)])
    dtranspose= df.transpose()
    #display(df)
    #display(dligne)
    display(dtranspose)

def update_custom_epsilon(epsilon):
    common.challenge.custom_epsilon = epsilon

def exercice_classification():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}" style="display: flex; gap: 2rem;">
            <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem; align-items: center;">
                <canvas id="{id}-1-chart"></canvas>
                <!--<canvas id="{id}-1-histo"></canvas>-->
                <div>
                    <label><input type="radio" name="{id}-1-input" value="0" checked> {classes[0]}</label>
                    <label><input type="radio" name="{id}-1-input" value="1"> {classes[1]}</label>
                </div>
            </div>
            <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem; align-items: center;">
                <canvas id="{id}-2-chart"></canvas>
                <!--<canvas id="{id}-2-histo"></canvas>-->
                <div>
                    <label><input type="radio" name="{id}-2-input" value="0" > {classes[0]}</label>
                    <label><input type="radio" name="{id}-2-input" value="1" checked> {classes[1]}</label>
                </div>
            </div>
        </div>
        <div style="flex: 1; display: flex; flex-direction: column; gap: 1rem; align-items: center; margin-top: 3rem; margin-bottom: 3rem;">
            <label id="{id}-count"> ==> Commencez l'exercice en prédisant quel foetus est malade <== </label>
        </div>
        <button id="{id}-submit" type="button">Valider</button>
    '''))

    run_js(f"""
    setTimeout(() => {{
        const labelCount = document.getElementById('{id}-count')
        const radios_1 = document.getElementsByName('{id}-1-input')
        const radios_2 = document.getElementsByName('{id}-2-input')
        const submit = document.getElementById('{id}-submit')

        // Vérifions si l'exercice est déjà terminé

        if(localStorage.getItem('exercice_classification_ok') === 'true') {{
            labelCount.style.color = 'green';
            labelCount.innerHTML = "Bravo ! L'exercice est terminé, vous pouvez passer à la suite"
            submit.style.display = 'none'   
        }}


        function syncRadios(selectedRadios, otherRadios) {{
            selectedRadios.forEach(radio => {{
                radio.addEventListener('change', () => {{
                    const selectedValue = parseInt(radio.value)
                    otherRadios.forEach(r => {{
                        r.checked = (parseInt(r.value) !== selectedValue)
                    }})
                }})
            }})
        }}

        syncRadios(radios_1, radios_2)
        syncRadios(radios_2, radios_1)

        
        let countBonEchantillon = 0
        submit.addEventListener('click', () => {{
            const selected_1 = parseInt(Array.from(radios_1).find(r => r.checked).value)
            const selected_2 = parseInt(Array.from(radios_2).find(r => r.checked).value)

            if (selected_1 === selected_2) {{
                alert('Les deux échantillons doivent être de classes différentes')
                return
            }}
            
            if (selected_1 === window.mathadata.exercice_classification_order[0]) {{
                if (countBonEchantillon < 4) {{
                    countBonEchantillon++
                    labelCount.style.color = 'green'; 
                    labelCount.innerHTML = 'Bravo ! Vous avez trouvé le foetus malade, il vous reste <span style="font-size: 2em;">' + (5 - countBonEchantillon) + '</span> paires à classifier !';
                }} else {{
                    labelCount.style.color = 'green';
                    labelCount.innerHTML = "Bravo ! L'exercice est terminé, vous pouvez passer à la suite"
                    localStorage.setItem('exercice_classification_ok', 'true')
                    window.mathadata.run_python('set_exercice_classification_ok()');
                    submit.style.display = 'none'
                    return
                }}
                window.mathadata.run_python('get_exercice_data_json()', (res) => {{
                    window.mathadata.exercice_classification('{id}', res[0], res[1], res[2])
                }})
            }} else {{
                labelCount.style.color = 'red';
                labelCount.innerHTML=('Mauvaise réponse, réessayez !')
            }}
        }})
    }}, 500)
    """)

    
    [random_order, d1, d2] = get_exercice_data()
    run_js(f"setTimeout(() => window.mathadata.exercice_classification('{id}', {random_order.tolist()}, '{json.dumps(d1, cls=NpEncoder)}', '{json.dumps(d2, cls=NpEncoder)}'), 500)")

def get_exercice_data():
    random_order = np.random.permutation(2)

    index_1 = np.random.choice(np.where(np.array(r_train) == classes[random_order[0]])[0])
    index_2 = np.random.choice(np.where(np.array(r_train) == classes[random_order[1]])[0])

    return [random_order, d_train[index_1], d_train[index_2]]

def get_exercice_data_json():
    return json.dumps(get_exercice_data(), cls=NpEncoder)

def animation_battement():
    id = uuid.uuid4().hex
    display(HTML(f'''
    <div style="display: flex; flex-direction: column; gap: 1rem; align-items: center;">
        <canvas id="{id}-chart"></canvas>
        <div>Temps : <span id="{id}-time">0</span>  -  Fréquence : <span id="{id}-freq">120</span></div>
        <div style="display: flex; gap: 1rem;">
            <button id="{id}-play" style="display: none; background: none; border: none; padding: 0; cursor: pointer; width: 32px; height: 32px;" onclick="togglePlayPause('{id}')">
                ▶️
            </button>
            <button id="{id}-pause" style="background: none; border: none; padding: 0; cursor: pointer; width: 32px; height: 32px;" onclick="togglePlayPause('{id}')">
                ⏸️
            </button>
        </div>
        <div class="heart" id="{id}-heart" style="display: flex; justify-content: center; align-items: center;">
            <img id="{id}-heart-img" src="{files_url}/heart-illustration.png" />
        </div>
    <script>
        window.animationState = window.animationState || {{}};
        window.animationState['{id}'] = {{ isPlaying: true }};

        function togglePlayPause(id) {{
            window.animationState[id].isPlaying = !window.animationState[id].isPlaying;
            const playBtn = document.getElementById(id + '-play');
            const pauseBtn = document.getElementById(id + '-pause');

            if (window.animationState[id].isPlaying) {{
                playBtn.style.display = 'none';
                pauseBtn.style.display = 'block';
            }} else {{
                playBtn.style.display = 'block';
                pauseBtn.style.display = 'none';
            }}

            const event = new CustomEvent('animation-state-change', {{ 
                detail: {{ 
                    id: id, 
                    isPlaying: window.animationState[id].isPlaying 
                }} 
            }});
            window.dispatchEvent(event);
        }}

        (function() {{
            const testImg = document.getElementById('{id}-heart-img');
            testImg.onload = function() {{
                // Image chargée avec succès
            }};
            testImg.onerror = function() {{
                const heartDiv = document.getElementById('{id}-heart');
                if (heartDiv) {{
                    heartDiv.innerHTML = '<div style="font-size: 4rem; color: red;"></div>';
                }}
            }};
        }})();
    </script>
    '''))
    run_js(
        f"setTimeout(() => window.mathadata.animation_battement('{id}', '{json.dumps(d_animation, cls=NpEncoder)}'), 500)")

def set_v_limits(vmin, vmax):
    common.challenge.vmin = vmin
    common.challenge.vmax = vmax
    # print(f"Seuils mis à jour : vmin={vmin}, vmax={vmax}")

# JS

styles = '''
.data-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 1rem;
    min-height: 360px;
    max-height: 500px;
    height: 100%;
    width: 100%;
    min-width: 600px;
}

.heart {
    position: relative;
    width: 100px;
    height: 100px;
    margin: 1rem;
    display: inline-block;
}

.heart-animate {
    animation: heartbeat 200ms;
}

@keyframes heartbeat {
    0%, 20%, 40%, 60%, 80%, 100% {
        transform: scale(1);
    }
    10% {
        transform: scale(1.1);
    }
    30% {
        transform: scale(1.15);
    }
    50% {
        transform: scale(1.1);
    }
    70% {
        transform: scale(1);
    }
}

'''

run_js(f"""
    let style = document.getElementById('mathadata-style-fhr');
    if (style !== null) {{
        style.remove();
    }}

    style = document.createElement('style');
    style.id = 'mathadata-style-fhr';
    style.innerHTML = `{styles}`;
    document.head.appendChild(style);
""")

run_js("""   
    if( localStorage.getItem('exercice_classification_ok') === 'true') {{
       window.mathadata.run_python('set_exercice_classification_ok()');
    }} 
    const min = 20
    const max = 240

    let vmin = 100
    let vmax = 150
       
    //window.mathadata.preset_zoom_enabled = true;  // Valeur par défaut (le faire passer à false si on ne veut pas le zoom)

    
    window.mathadata.affichage_chart = (id, data, with_selection, preset_zoom, isContainerSmall = false) => {
        if (typeof data === 'string') {
            data = JSON.parse(data)
        }
        const small = !!isContainerSmall
        const config = {
            type: 'scatter',
            data: {
                labels: data.map((_, i) => i),
                datasets: [{
                    label: 'Rythme cardiaque',
                    data: data,
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    pointStyle: 'cross',
                    order: 2,
                }]
            },
            options: {
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Temps (secondes)'
                        },
                        min: 0,
                        max: data.length,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Rythme cardiaque'
                        },
                        min,
                        max,
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    zoom: {
                        zoom: {
                            wheel: {
                                enabled: preset_zoom, //window.mathadata.preset_zoom_enabled pour la globale
                            },
                            pinch: {
                                enabled: false,
                            },
                            mode: 'x',
                        },
                        pan: {
                            enabled: preset_zoom, //window.mathadata.preset_zoom_enabled pour la globale
                            mode: 'x',
                        },
                        limits: {
                            x: { min: 0, max: data.length },
                        }
                    }
                },
            },
            plugins: [{}],
        }

        // Si le container est petit, exécuter ceci , hehehee et bien cela nous aidera bien pour l'animation tapis algo.
        if (small) {
            // Cacher les titres d'axes
            if (config.options.scales && config.options.scales.x && config.options.scales.x.title) {
                config.options.scales.x.title.display = false
            }
            if (config.options.scales && config.options.scales.y && config.options.scales.y.title) {
                config.options.scales.y.title.display = false
            }

            // Cacher la line de la grid et les ticks
            config.options.scales.x.grid = { display: false }
            config.options.scales.y.grid = { display: false }
            config.options.scales.x.ticks = { display: false, maxTicksLimit: 2 }
            config.options.scales.y.ticks = { display: false, maxTicksLimit: 2 }

            // Désactiver le zoom et le pan
            if (config.options.plugins && config.options.plugins.zoom) {
                config.options.plugins.zoom.zoom.wheel.enabled = false
                config.options.plugins.zoom.pan.enabled = false
            }
        }
        
        if (with_selection) {
            config.data.datasets = config.data.datasets.concat([
                {
                    type: 'line',
                    label: 'vmin',
                    data: [{x: 0, y: vmin}, {x: data.length, y: vmin}],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 0,
                    pointHoverRadius: 0,
                    order: 1,
                },
                {
                    type: 'line',
                    label: 'vmax',
                    data: [{x: 0, y: vmax}, {x: data.length, y: vmax}],
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0,
                    pointHitRadius: 0,
                    pointHoverRadius: 0,
                    order: 1,
                },
            ])
        }

        window.mathadata.create_chart(id, config)
    }

    window.mathadata.affichage_histogramme = (id, data) => {
        if (typeof data === 'string') {
            data = JSON.parse(data)
        }
        const roundedData = data.filter(v => v !== null).map(d => Math.floor(d))
        const counts = {}
        for (const val of roundedData) {
            counts[val] = (counts[val] || 0) + 1
        }
        const labels = new Array(max - min + 1).fill().map((_, i) => max - i) // Max to min in reverse order
        window.mathadata.create_chart(id, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'Nombre de battements de coeur par minute',
                    data: labels.map(l => counts[l] ?? 0),
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Nombre de secondes avec cette fréquence'
                        },
                        min: 0,
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Rythme cardiaque'
                        },
                    }
                }
            },
        })
    }
        
    window.mathadata.affichage = (id, data, params) => {
        if (typeof data === 'string') {
            data = JSON.parse(data)
        }

        if (params) {
            if (typeof params === 'string') {
                params = JSON.parse(params)
            }
        } else {
            params = {}
        }

        const {with_selection, mode} = params

        const container = document.getElementById(id)

        if (!container.innerHTML) {
            container.innerHTML = `
                <div style="height: 100%; width: 100%; display: flex; flex-direction: column; justify-content: center; gap: 1rem">
                    <canvas id="${id}-chart"></canvas>
                    ${mode === 2 ? `<canvas id="${id}-histo"></canvas>` : ''}
                </div>
            `
        }

        //container.classList.add('data-container')
        
        /*
                    <div style="height: 100%; overflow-y: scroll; min-width: min-content; flex: 1;">
                        <table id="${id}-table">
                        </table>
                    </div>
        const table = document.getElementById(`${id}-table`)
        const content = data.map((d, i) => `<tr><td>${i + 1}</td><td>${d && d.toFixed(2)}</td></tr>`).join('')
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Seconde</th>
                    <th>Fréquence cardiaque</th>
                </tr>
                ${content}
            </thead>
        `
        */

        // Detect small container (under 100px on either dimension) to adapt chart rendering
        // Use offsetWidth/offsetHeight (not getBoundingClientRect) for notebook environment compatibility
        const width = container.offsetWidth || container.clientWidth || 0
        const height = container.offsetHeight || container.clientHeight || 0
        const isContainerSmall = Math.min(width, height) < 100

        window.mathadata.affichage_chart(`${id}-chart`, data, with_selection, true, isContainerSmall)
        if (with_selection) {
        /*
            const input = document.getElementById(`${id}-input`)
            input.addEventListener("change", (e) => {
                let val = parseFloat(e.target.value)
                if (val < 0) {
                    val = -val
                }
                
                const chart = window.mathadata.charts[`${id}-chart`]
                const length = chart.data.datasets[0].data.length
                const mean = chart.data.datasets[1].data[0]
                chart.data.datasets[2].data = Array(length).fill(mean - val)
                chart.data.datasets[3].data = Array(length).fill(mean + val)
                chart.update()
                window.mathadata.run_python(`update_custom_epsilon(${val})`)
            })
        */
        }
        
        if (mode === 2) {
            window.mathadata.affichage_histogramme(`${id}-histo`, data)
        }

        return window.mathadata.charts[`${id}-chart`]
    }

    window.mathadata.exercice_classification = (id, random_order, d1, d2) => {
        if (typeof d1 === 'string') {
            d1 = JSON.parse(d1)
        }
        if (typeof d2 === 'string') {
            d2 = JSON.parse(d2)
        }
        
        window.mathadata.affichage_chart(`${id}-1-chart`, d1,false, true)
        //window.mathadata.affichage_histogramme(`${id}-1-histo`, d1)
        window.mathadata.affichage_chart(`${id}-2-chart`, d2,false, true)
        //window.mathadata.affichage_histogramme(`${id}-2-histo`, d2)
        window.mathadata.exercice_classification_order = random_order
    }

    window.mathadata.animation_battement = (id, values) => {
        values = JSON.parse(values);
        window.mathadata.affichage_chart(`${id}-chart`, values, false, true); // 1er false pour ne pas afficher la sélection, 2ème true pour zoomer par défaut
        const chart = window.mathadata.charts[`${id}-chart`];   
        
        const intervals = [];
        let remaining_frac = 1;
        let waited_time = 0;
        for (const heart_rate of values) {
            if (heart_rate === null) {
                remaining_frac = 0;
                waited_time += 1000;
                continue;
            }
            
            const time_between_beat_ms = (60 / heart_rate) * 1000;
            const time_to_next = remaining_frac * time_between_beat_ms;
            if (time_to_next <= 1000) {
                intervals.push(waited_time + time_to_next);
                let time_in_sec = time_to_next;
                while (time_in_sec + time_between_beat_ms <= 1000) {
                    intervals.push(time_between_beat_ms);
                    time_in_sec += time_between_beat_ms;
                }
                
                waited_time = 1000 - time_in_sec;
                remaining_frac = 1 - (waited_time / time_between_beat_ms);
            }
        }
        
        let play = true;
        let index = 0;
        let time_ms = 0;
        let next_beat_to = null;
        
        const time = document.getElementById(`${id}-time`);
        const freq = document.getElementById(`${id}-freq`);

        chart.options.onClick = (e) => {
            const canvasPosition = Chart.helpers.getRelativePosition(e, chart);

            // Substitute the appropriate scale IDs
            const dataX = chart.scales.x.getValueForPixel(canvasPosition.x);
            time_ms = dataX * 1000;
            chart.update()

            const second = Math.floor(time_ms / 1000)
            time.innerHTML = `${second}`
            freq.innerHTML = `${values[second]}`

            clearTimeout(next_beat_to)
            let time_sum = 0
            index = 0
            for (const to of intervals) {
                if (time_sum + to >= time_ms) {
                    next_beat_to = setTimeout(beat, time_sum + to - time_ms)
                    break
                }
                index++    
                time_sum += to
            }
        }
        
        let current_max = 100000;
        chart.options.scales.x.min = 0;
        chart.options.scales.x.max = current_max / 1000;
        
        const updateChartScale = () => {
            if (time_ms >= current_max) {
                const next_max = current_max + 30000;
                const max_limit = values.length * 1000;
                chart.options.scales.x.min = current_max / 1000;
                chart.options.scales.x.max = Math.min(next_max, max_limit) / 1000;
                current_max = next_max;
                chart.update();
            }
        };

        
        chart.config.plugins[0].beforeDraw = function(_chart, args, options) {
            const ctx = _chart.ctx;
            const chartArea = _chart.chartArea;
            const x = _chart.scales.x.getPixelForValue(time_ms / 1000);
            ctx.save();
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x, chartArea.top);
            ctx.lineTo(x, chartArea.bottom);
            ctx.stroke();
            ctx.restore();
        };
        chart.update();
        
        const time_up_interval = setInterval(() => {
            if (play) {
                time_ms = Math.min(time_ms + 100, values.length * 1000);
                chart.update('none');
                
                const second = Math.floor(time_ms / 1000);
                time.innerHTML = `${second}`;
                freq.innerHTML = `${values[second]}`;
                
                //updateChartScale();
            }
        }, 100);
        
        const heart = document.getElementById(`${id}-heart`);
        const animationTime = 200;
        const audio = new Audio(`${mathadata.files_url}/heartbeat-loop-96879.mp3`);
        audio.addEventListener('error', () => {
            console.error('[audio] ERREUR : Impossible de charger l’audio.');
            alert("⚠️ ATTENTION: Le fichier audio du battement de coeur n’a pas pu être chargé. Continuez l'exercice sans audio ou réexécutez la cellule pour réessayer.");
            return;
        });
        const beat = () => {
            if (!play) return;
            if (index < intervals.length) {
                next_beat_to = setTimeout(beat, intervals[index++]);
            } else {
                next_beat_to = null;
            }
            audio.pause();
            audio.currentTime = 0;
            audio.play();
            heart.classList.add('heart-animate');
            setTimeout(() => heart.classList.remove('heart-animate'), animationTime);
        };
        
        document.getElementById(`${id}-pause`).addEventListener('click', () => {
            play = false;
        });
        
        document.getElementById(`${id}-play`).addEventListener('click', () => {
            if (!play) {
                play = true;
                beat();
            }
        });

        beat();
    };

    window.mathadata.setup_zone_selection = (id, data, labels) => {
        data = JSON.parse(data)
        labels = JSON.parse(labels)
    
        const container = document.getElementById(id)
        container.innerHTML = `
            <div>
                <div id="${id}-data" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; justify-content: center;"></div>
                <div style="display: flex; flex-direction: row; gap: 3rem; margin-top: 2rem; justify-content: center;">
                    <div>
                        <label for="${id}-input-vmin">v<sub>min</sub></label>
                        <input id="${id}-input-vmin" type="number" min="0"></input>
                    </div>
                    <div>
                        <label for="${id}-input-vmax">v<sub>max</sub></label>
                        <input id="${id}-input-vmax" type="number" min="0"></input>
                    </div>
                </div>
            <div>
        `;

        mathadata.foetus_tables = []
        const dataContainer = document.getElementById(`${id}-data`)
        for (let i = 0; i < data.length; i++) {
            const div = document.createElement('div')
            div.style.display = 'flex'
            div.style.flexDirection = 'column'
            div.style.gap = '1rem'
            div.style.alignItems = 'center'
            div.style.width = '100%'
            div.innerHTML = `
                <div id=${id}-data-${i} style="width: 100%"></div>
                <span style="text-align: center;">${labels[i]}</span>
            `

            dataContainer.appendChild(div)
            mathadata.foetus_tables.push(mathadata.affichage(`${id}-data-${i}`, data[i], {with_selection: true}))
        }

        const vminInput = document.getElementById(`${id}-input-vmin`)
        const vmaxInput = document.getElementById(`${id}-input-vmax`)

        // init inputs
        vminInput.value = vmin
        vmaxInput.value = vmax
        
        const displayVRange = () => {
            const min = parseFloat(vminInput.value)
            const max = parseFloat(vmaxInput.value)
            for (let i = 0; i < mathadata.foetus_tables.length; i++) {
                const chart = mathadata.foetus_tables[i]
                chart.config.data.datasets[1].data = [{x: 0, y: min}, {x: data[i].length, y: min}]
                chart.config.data.datasets[2].data = [{x: 0, y: max}, {x: data[i].length, y: max}]
                chart.update()
            }

            window.mathadata.run_python(`set_v_limits(${min}, ${max})`, ()=>{
                window.mathadata.on_custom_update();
            })
        }

        vminInput.addEventListener('change', () => {
            displayVRange();
        })

        vmaxInput.addEventListener('change', () => {
            displayVRange();
        })

        displayVRange();
    }

    window.mathadata.interface_data_gen = (id) => {
        const container = document.getElementById(id);
        container.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 1rem; align-items: center; margin: 20px 0;">
                <div style="text-align: center;">
                    <h3>Générateur de Rythme Cardiaque Foetal</h3>
                    <p style="font-size: 14px; max-width: 600px; margin: 10px auto;">
                        Dessinez une courbe de rythme cardiaque foetal en cliquant et glissant sur le graphique. 
                        L'axe horizontal représente le temps (0-7.5 minutes), l'axe vertical la fréquence cardiaque (60-200 bpm).
                    </p>
                </div>
                
                <div style="display: flex; gap: 2rem; align-items: center;">
                    <div id="${id}-chart-container" style="position: relative; width: 600px; height: 300px; border: 1px solid #ccc;">
                        <canvas id="${id}-chart" width="600" height="300" style="cursor: crosshair;"></canvas>
                    </div>
                    
                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <!--<div>
                            <label for="${id}-baseline">Ligne de base (bpm):</label>
                            <input type="range" id="${id}-baseline" min="60" max="160" value="100" style="width: 150px;">
                            <span id="${id}-baseline-value">100</span>
                        </div>
                        -->

                        <div>
                            <label for="${id}-variability">Variabilité:</label>
                            <input type="range" id="${id}-variability" min="0" max="50" value="10" style="width: 150px;">
                            <span id="${id}-variability-value">10</span>
                        </div>
                        
                        <div>
                            <label for="${id}-anomalies">Anomalies:</label>
                            <input type="checkbox" id="${id}-anomalies">
                        </div>
                        
                        <button id="${id}-clear" style="padding: 8px 16px; background-color: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer;">
                            Effacer
                        </button>
                    </div>
                </div>
            </div>
        `;

        const canvas = document.getElementById(`${id}-chart`);
        const ctx = canvas.getContext('2d');
        const variabilitySlider = document.getElementById(`${id}-variability`);
        const anomaliesCheckbox = document.getElementById(`${id}-anomalies`);
        const clearButton = document.getElementById(`${id}-clear`);
        
        // Configuration du graphique
        const width = 600;
        const height = 300;
        const padding = 40;
        const chartWidth = width - 2 * padding;
        const chartHeight = height - 2 * padding;
        
        // Plages de valeurs
        const timeRange = 7.5; // minutes
        const hrMin = 60;
        const hrMax = 200;
        
        // Données du rythme cardiaque
        let heartRateData = [];
        let isDrawing = false;
        let lastIndex = null;
        let lastHeartRate = null;
        
        function initializeData() {
            heartRateData = new Array(1800).fill(null); //Pour ne plus avoir les longues lignes qui depassent on pase à null aqu lieu de 0 en fill
            drawChart();
        }
        
        // Dessiner le graphique
        function drawChart() {
            ctx.clearRect(0, 0, width, height);
            
            // Dessiner les axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding, padding);
            ctx.lineTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding);
            ctx.stroke();
            
            // Dessiner la grille
            ctx.strokeStyle = '#eee';
            ctx.lineWidth = 0.5;
            
            // Lignes horizontales (fréquence cardiaque)
            for (let i = 0; i <= 5; i++) {
                const y = padding + (chartHeight / 5) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }
            
            // Lignes verticales (temps)
            for (let i = 0; i <= 6; i++) {
                const x = padding + (chartWidth / 6) * i;
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, height - padding);
                ctx.stroke();
            }
            
            // Dessiner les labels
            ctx.fillStyle = '#333';
            ctx.font = '12px Arial';
            ctx.textAlign = 'center';
            
            // Labels Y (fréquence cardiaque)
            for (let i = 0; i <= 5; i++) {
                const value = hrMax - (hrMax - hrMin) * i / 5;
                const y = padding + (chartHeight / 5) * i;
                ctx.fillText(value.toString(), padding - 10, y + 4);
            }
            
            // Labels X (temps)
            for (let i = 0; i <= 6; i++) {
                const value = (1800 / 6) * i;  // échelle en secondes
                const x = padding + (chartWidth / 6) * i;
                ctx.fillText(value.toFixed(0) + 's', x, height - padding + 20);
            }
            
            // Dessiner la courbe de rythme cardiaque
            if (heartRateData.length > 0) {
                    ctx.strokeStyle = '#68BBBA';
                    ctx.lineWidth = 2;
                    ctx.beginPath();

                    let firstPoint = true;
                    for (let i = 0; i < heartRateData.length; i++) {
                        if (heartRateData[i] == null) continue;

                        const x = padding + (chartWidth * i) / (heartRateData.length - 1);
                        const y = padding + chartHeight - (chartHeight * (heartRateData[i] - hrMin)) / (hrMax - hrMin);

                        // Dessiner une croix pour le point
                        const size = 4; // taille de la croix
                        ctx.strokeStyle = '#68BBBA';
                        ctx.lineWidth = 1;

                        ctx.beginPath();
                        ctx.moveTo(x - size, y - size);
                        ctx.lineTo(x + size, y + size);
                        ctx.moveTo(x - size, y + size);
                        ctx.lineTo(x + size, y - size);
                        ctx.stroke();
                    }
                    ctx.stroke();
            }
        }
        
        // Convertir les coordonnées de la souris en données
        function mouseToData(mouseX, mouseY) {
            const x = Math.max(0, Math.min(chartWidth, mouseX - padding));
            const y = Math.max(0, Math.min(chartHeight, mouseY - padding));
            
            const timeIndex = Math.round((x / chartWidth) * (heartRateData.length - 1));
            const heartRate = hrMax - (y / chartHeight) * (hrMax - hrMin);
            
            return { timeIndex, heartRate };
        }
        
        // Appliquer la variabilité et les anomalies
        function applyVariability() {
            const variability = parseInt(variabilitySlider.value);
            const hasAnomalies = anomaliesCheckbox.checked;
            
            for (let i = 0; i < heartRateData.length; i++) {
                // Variabilité normale
                const randomVariation = (Math.random() - 0.5) * variability;
                heartRateData[i] = Math.max(hrMin, Math.min(hrMax, heartRateData[i] + randomVariation));
                
                // Anomalies occasionnelles
                if (hasAnomalies && Math.random() < 0.02) {
                    if (Math.random() < 0.5) {
                        // Décélération
                        for (let j = Math.max(0, i - 20); j < Math.min(heartRateData.length, i + 20); j++) {
                            const distance = Math.abs(j - i);
                            const intensity = Math.max(0, 1 - distance / 20);
                            heartRateData[j] = Math.max(hrMin, heartRateData[j] - intensity * 30);
                        }
                    } else {
                        // Accélération
                        for (let j = Math.max(0, i - 15); j < Math.min(heartRateData.length, i + 15); j++) {
                            const distance = Math.abs(j - i);
                            const intensity = Math.max(0, 1 - distance / 15);
                            heartRateData[j] = Math.min(hrMax, heartRateData[j] + intensity * 20);
                        }
                    }
                }
            }
        }
        
        // Événements de la souris
        canvas.addEventListener('mousedown', (e) => {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            const { timeIndex, heartRate } = mouseToData(mouseX, mouseY);

            lastIndex = timeIndex;
            lastHeartRate = heartRate;
            heartRateData[timeIndex] = heartRate;
            drawChart();
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (isDrawing) {
                const rect = canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                const { timeIndex, heartRate } = mouseToData(mouseX, mouseY);

                if (lastIndex !== null) {
                    const start = Math.max(0, Math.min(lastIndex, timeIndex));
                    const end = Math.min(heartRateData.length - 1, Math.max(lastIndex, timeIndex));
                    const startVal = lastHeartRate;
                    const endVal = heartRate;
                    const variability = parseInt(variabilitySlider.value);

                    for (let i = start; i <= end; i++) {
                        const t = (end === start) ? 1 : (i - start) / (end - start);
                        let v = startVal + t * (endVal - startVal);
                        if (variability > 0) {
                            const randomVariation = (Math.random() - 0.5) * variability;
                            v = (v + randomVariation);
                        }
                        heartRateData[i] = Math.max(hrMin, Math.min(hrMax, v));
                    }
                } else {
                    heartRateData[timeIndex] = heartRate;
                }

                lastIndex = timeIndex;
                lastHeartRate = heartRate;
                drawChart();
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            isDrawing = false;
            lastIndex = null;
            lastHeartRate = null;
        });
        
        variabilitySlider.addEventListener('input', (e) => {
            document.getElementById(`${id}-variability-value`).textContent = e.target.value;
        });
        
        clearButton.addEventListener('click', () => {
            initializeData();
        });
        
        // Initialiser
        initializeData();

        return () => heartRateData;
    }
""")

# Validation

# Pour valider differente orthographes sans les guillemets
Normal = "Normal"
normal = "Normal"
NORMAL = "Normal"
ALARMANT = "ALARMANT"
alarmant = "ALARMANT"
Alarmant = "ALARMANT"
chat = "0"
CHAT = "0"
Chat = "0"
chaton = "0"
CHATON = "0"



def validate_etendue(errors, answers):
    return answers['etendue'] - (max(random_frequencies) - min(random_frequencies)) < 1e-3

def validate_caracteristique(errors, answers):
    x = answers['x']
    count = np.sum(np.array(random_frequencies) < v)
    
    if x == count:
        errors.append(f"Attention, la caractéristique est le pourcentage de valeurs avec une fréquence cardiaque inférieure à {v} et non le nombre.")
        return False
    
    if x == count * 10:
        pretty_print_success(f"Bravo ! Il y a {count} valeurs avec une fréquence cardiaque inférieure à {v} soit {x}% du temps total.")
        return True
    
    errors.append(f"Ce n'est pas la bonne réponse. La caractéristique est le pourcentage de valeurs avec une fréquence cardiaque inférieure à {v}.")
    return False

def validate_exercice(errors, answers):
    global exercice_classification_ok
    if not exercice_classification_ok :
        errors.append("Terminez l'exercice ci-dessus avant de continuer.")
        return False
    return True
validation_animation_battement = common.MathadataValidate(success="")
validation_execution_affichage = common.MathadataValidate(success="")
validation_execution_affichage_tableau = common.MathadataValidate(success="")
def validate_question_frequence(errors, answers):
    f = answers['frequence']
    if not isinstance(f, (int, float)):
        errors.append("La fréquence doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if f==Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if f!= d[3].round(1):
        errors.append("Ce n'est pas la bonne valeur. Reprends le tableau et vérifie la valeur de la fréquence cardiaque après 3 secondes d'enregistrement.")
        return False 
    return True

validation_question_frequence = common.MathadataValidateVariables({
    'frequence':None
}, function_validation=validate_question_frequence, success="Bravo ! La fréquence cardiaque est "+ str(d[3].round(1)))

validation_exercice_classification = common.MathadataValidate(
    success="Bravo ! Tu commences à savoir distinguer les différentes classes.",
    function_validation= validate_exercice
)
validation_execution_affichage_etendue = common.MathadataValidate(success="")
validation_execution_affichage_10_frequences = common.MathadataValidate(success="")
validation_question_etendue = common.MathadataValidateVariables({
    'etendue': None
}, function_validation=validate_etendue)

validation_execution_exercice_classification = common.MathadataValidate(success="")

validation_question_caracteristique = common.MathadataValidateVariables({
    'x': None
}, function_validation=validate_caracteristique, success="")

validation_affichage_banque_ecart_type = common.MathadataValidate(success="")

# Notebook histogramme

def get_names_and_values_hist_2():
    c_train_by_class = compute_c_train_by_class()
    
    # c_train_by_class[0] and c_train_by_class[1] are already numpy arrays
    c_train_class_0 = c_train_by_class[0]
    c_train_class_1 = c_train_by_class[1]

    # Condition: characteristic is between 8 (inclusive) and 10 (exclusive)
    # For class 0 (e.g., "Normal")
    # Use element-wise '&' for combining numpy boolean arrays
    condition_class_0 = (c_train_class_0 >= 8) & (c_train_class_0 < 10)
    count_class_0 = np.sum(condition_class_0)
    
    # For class 1 (e.g., "ALARMANT")
    condition_class_1 = (c_train_class_1 >= 8) & (c_train_class_1 < 10)
    count_class_1 = np.sum(condition_class_1)

    return {
        f'nombre_{classes[0]}': count_class_0,
        f'nombre_{classes[1]}': count_class_1,
    }

def get_names_and_values_hist_3():
    c_train_by_class = compute_c_train_by_class()

    c_train_class_0 = c_train_by_class[0]
    c_train_class_1 = c_train_by_class[1]

    condition_class_0 = c_train_class_0 < 6
    count_class_0 = np.sum(condition_class_0)

    condition_class_1 = c_train_class_1 < 6
    count_class_1 = np.sum(condition_class_1)

    return {
        f'nombre_{classes[0]}_inf_6': count_class_0,
        f'nombre_{classes[1]}_inf_6': count_class_1,
    }

validation_question_hist_2 = MathadataValidateVariables(get_names_and_values=get_names_and_values_hist_2, tips=[
    {
      'seconds': 30,
      'tip': 'En passant la souris sur les barres de l\'histogramme, tu peux voir le nombre d\'enregistrements qui ont une caractéristique dans l\'intervalle correspondant.'
    }
  ])

validation_question_hist_3 = MathadataValidateVariables(get_names_and_values=get_names_and_values_hist_3,tips=[
    {
      'seconds': 30,
      'tip': 'As-tu bien tenu compte du mot inférieur ?'
    }
  ])

def caracteristique_etendue_correction(d):
    """
    Calcule la caractéristique étendue pour une séquence de battements de coeur.

    """
    minimum = min(d)
    maximum = max(d)
    etendue = maximum-minimum
    return etendue

def on_success_etendue(answers):
    if has_variable('afficher_histogramme'):
        print("Voici l'histogramme des fréquences cardiaques pour l'ensemble des données d'entraînement avec cette caractéristique. Comme tu vas le voir elle n'est pas très discriminante et peu utile pour classer les enregistrements.")
        get_variable('afficher_histogramme')(legend=True,caracteristique=get_variable('caracteristique'))


validation_caracteristique_etendue_et_affichage=MathadataValidateFunction(
    'caracteristique',
    test_set=lambda: common.challenge.d_train[0:100],
    expected=lambda: [caracteristique_etendue_correction(d) for d in common.challenge.d_train[0:100]],
    on_success=on_success_etendue
)


def validate_caracteristique_libre(errors, answers):
    """
    Validation de la caractéristique libre.
    La caractéristique doit être un nombre.
    """
    caracteristique = answers['caracteristique'] 
    for d in common.challenge.d_train[0:5]:
            if not isinstance(caracteristique(d), float) and not isinstance(caracteristique(d), int):
                errors.append("La caractéristique doit être un nombre. Ta fonction ne semble pas renvoyer un nombre.")
                return False
    return True

def on_success_histogramme(answers):
    if has_variable('afficher_histogramme'):

        get_variable('afficher_histogramme')(legend=True,caracteristique=get_variable('caracteristique'))

validation_caracteristique_libre_et_affichage=MathadataValidateVariables(name_and_values={'caracteristique': None}, function_validation=validate_caracteristique_libre,success="Ta fonction renvoie bien un nombre. Testons ta proposition",on_success=on_success_histogramme)



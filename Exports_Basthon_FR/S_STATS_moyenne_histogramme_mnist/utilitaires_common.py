import os
import sys
import matplotlib.pyplot as plt
import requests
import secrets
import __main__
import json
import time
import pandas as pd
import uuid
from IPython.display import display, HTML
import importlib.util
import math
import numpy as np

### --- AJOUT DE TOUS LES SUBDIRECTIRIES AU PATH ---
base_directory = os.path.abspath('.')

# Using os.listdir() to get a list of all subdirectories
subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Adding all subdirectories to the Python path
sys.path.extend(subdirectories)

### --- Import du validation_kernel ---
# Ne marche que si fourni et si va avec le notebook en version séquencé. Sinon, ignorer :
sequence = False

try:
    from capytale.autoeval import Validate, validationclass
    from capytale.random import user_seed

    sequence = True
except ModuleNotFoundError:
    try:
        from basthon.autoeval import Validate, validationclass
        sequence = True
    except ModuleNotFoundError:
        pass

# Import des strings pour lire l'export dans jupyter
try:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
except Exception as e: 
    if not sequence:
        print(f"Error loading strings: {e}")
    pass

## Pour valider l'exécution d'une cellule de code, dans le cas du notebook sequencé :
if sequence:
    validation_execution = Validate()
else:
    def validation_execution():
        return True

        
# Données spécifiques au challenge
challenge = None
class Challenge:
    def __init__(self):
        self.d_train = []
        self.d_test = []
        self.r_train = []
        self.d = None
        self.classes = [0,1]
        self.r_petite_caracteristique = 0
        self.r_grande_caracteristique = 1
        self.strings = {
            'dataname': "donnée",
            'dataname_plural': "données",
            'feminin': True,
            'contraction': False,
            'classes': ["classe1", "classe2"],
            'train_size': "1000"
        }

    def affichage_banque(self, carac=None, mode=1, showPredictions=False, estimations=None):
        id = uuid.uuid4().hex
        display(HTML(f'''
            <div style="display: flex; height: 500px; gap: 2rem;">
                <div id="{id}-bank" class="ag-theme-quartz" style="flex: 1; min-width: 300px;"></div>
                <div style="flex: 1;" id="{id}-selected"></div>
            </div>
        '''))

        if carac is not None:
            if carac == 1:
                carac = self.caracteristique
            elif carac == 2:
                carac = self.caracteristique2
            c_train = compute_c_train(carac, self.d_train)
            
        if showPredictions and estimations is None:
            estimations = get_estimations(self.d_train)

        params = {
            'labels': self.r_train,
            'c_train': c_train if carac else None,
            'estimations': estimations if showPredictions else None,
            'mode': mode
        }

        run_js(f"setTimeout(() => window.mathadata.setup_test_bank('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")
    
    def get_data(self, index=None, dataClass=None):
        d = None
        if index is not None:
            if dataClass == None:
                d = self.d_train[index]
            else:
                class_index = self.classes.index(dataClass)
                d = self.d_train_by_class[class_index][index]
        if d is None:
            d = self.d

        return json.dumps(d, cls=NpEncoder)

    def import_js_scripts(self):
        pass

    def affichage_html(self, d=None):
        id = uuid.uuid4().hex
        display(HTML(f'<div id="{id}" style="min-height: 300px; min-width: 300px;"></div>'))
        
        if d is None:
            d = self.d
            
        run_js(f"setTimeout(() => window.mathadata.affichage('{id}', {json.dumps(d, cls=NpEncoder)}), 500)")

def init_challenge(challenge_instance):
    global challenge
    challenge = challenge_instance

    challenge.d_train_by_class = [challenge.d_train[challenge.r_train==k] for k in challenge.classes]

    run_js(f"""
        window.mathadata.classes = {challenge.classes};
        window.mathadata.challenge = window.mathadata.challenge || {{}};
        window.mathadata.challenge.classes = {challenge.classes};
        window.mathadata.challenge.strings = JSON.parse('{json.dumps(challenge.strings, cls=NpEncoder)}');
    """)

    if sequence:
        run_js("""
            var script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/requirejs@2.3.7/require.min.js';
            script.onload = () => window.mathadata.run_python('import_js_scripts()');
            document.head.appendChild(script);
        """)
    else:
        import_js_scripts()

## Alias pour simplifier la syntaxe

def affichage_html(*args, **kwargs):
    challenge.affichage_html(*args, **kwargs)

def affichage_dix(*args, **kwargs):
    challenge.affichage_dix(*args, **kwargs)

def affichage_banque(*args, **kwargs):
    challenge.affichage_banque(*args, **kwargs)

def caracteristique(d):
    return challenge.caracteristique(d)

def get_data(*args, **kwargs):
    return challenge.get_data(*args, **kwargs)

def display_custom_selection(*args, **kwargs):
    challenge.display_custom_selection(*args, **kwargs)

def import_js_scripts():
    run_js("""
            define('chartjs/helpers', ['chartjs'], function(Chart) {
            return Chart.helpers;
        });
        require.config({
            paths: {
                'ag-grid-community': 'https://cdn.jsdelivr.net/npm/ag-grid-community/dist/ag-grid-community.min',
                'chartjs': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min',
                'drag-data-plugin': 'https://cdn.jsdelivr.net/npm/chartjs-plugin-dragdata@latest/dist/chartjs-plugin-dragdata.min'
            },
           map: {
                    'drag-data-plugin': {
                        'chart.js': 'chartjs',
                        'chartjs/helpers': 'chartjs' // Map 'chartjs/helpers' to 'chartjs'
                    }
                },
                shim: {
                    'drag-data-plugin': {
                        deps: ['chartjs'] // Specify dependencies
                    }
                }
        });

        require(['ag-grid-community', 'chartjs','drag-data-plugin'], function(agGrid) {
            window.agGrid = agGrid;
        });
    """)
    challenge.import_js_scripts()

# Fonctions avec comportement different sur Capytale ou en local

if sequence:
    from js import fetch, FormData, Blob, eval, Headers
    import asyncio
    
    debug = False
    mathadata_endpoint = "https://mathadata.fr/api"
    challengedata_endpoint = "https://challengedata.ens.fr/api"
    
    Validate()() # Validate import cell

    def call_async(func, cb, *args):
        try:
            loop = asyncio.get_event_loop()
            task = loop.create_task(func(*args))

            def internal_cb(future):
                try:
                    data = future.result()
                    if cb is not None:
                        cb(data)
                except Exception as e:
                    if debug:
                        print_error("Error during post request")
                        print_error(e)
                        
            task.add_done_callback(internal_cb)
        except Exception as e:
            if debug:
                print_error("Error during post request")
                print_error(e)

    async def fetch_async(url, method='GET', body=None, files=None, fields=None, headers=None):
        if body:
            body = json.dumps(body)
        elif files or fields:
            body = FormData.new()
            if files:
                for key in files:
                    body.append(key, Blob.new([files[key].getvalue()], {'type' : 'text/csv'}))
            if fields:
                for key in fields:
                    body.append(key, str(fields[key]))
                    
        if headers:
            js_headers = Headers.new()
            for key in headers:
                js_headers.append(key, headers[key])
        else:
            js_headers = None

        response = await fetch(url, method=method, body=body, headers=js_headers)
        if response.status >= 200 and response.status < 300:
            data = await response.text()
            data_json = json.loads(data)
            return data_json
        else:
            raise Exception(f"Fetch failed with status: {response.status}")

    # Récupère l'id capytale, le statut prof/élève et le nom de classe depuis l'API capytale
    async def get_profile():
        return await fetch_async('/web/c-auth/api/me?_format=json')
    
else:
    from IPython.display import display, Javascript, HTML

    debug = True
    mathadata_endpoint = "http://localhost:3000/api"
    challengedata_endpoint = "http://localhost:8000/api"
    #analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"

analytics_endpoint = mathadata_endpoint + "/notebooks"

### Utilitaires requêtes HTTP ###

if sequence:
    mathadata_url = "https://mathadata.fr"
else:
    mathadata_url = "https://dev.mathadata.fr"

files_url = mathadata_url + "/assets/fichiers_notebooks/"

# Send HTTP request. To send body as form data use parameter files (dict of key, StringIO value) and fields (dict of key, value)
def http_request(url, method='GET', body=None, files=None, fields=None, headers=None, cb=None):
    if body is not None and (files is not None or fields is not None):
        raise ValueError("Cannot have both body and files in the same request")
    try:
        if sequence:
            call_async(fetch_async, cb, url, method, body, files, fields, headers)
        else:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=body, files=files, data=fields, headers=headers)
            elif debug:
                raise ValueError(f"Invalid method: {method}")
            else:
                return None
        
            if cb is not None:
                cb(response.json())
                
    except Exception as e:
        if debug:
            print_error("Error during http request")
            print_error(e)

challengedata_token = 'yjQTYDk8d51Uq8WcDCPUBK1GPEuEDi6W/3e736TV7qGAmmqn7CCyefkdL+vvjOFY'

def http_request_cd(endpoint, method='GET', body=None, files=None, fields=None, cb=None):
    headers = {
        'Authorization': f'Bearer {challengedata_token}'
    }
    http_request(challengedata_endpoint + endpoint, method, body, files, fields, headers, cb)


### Fonctions utilitaires génériques ###

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            if np.isnan(obj):
                return None
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.integer) or np.issubdtype(obj.dtype, np.floating):
                return np.where(np.isnan(obj), None, obj).tolist()
            else:
                return obj.tolist()
        return super(NpEncoder, self).default(obj)

def compute_c_train(fonction_caracteristique, d_train):
    c_train = np.array([fonction_caracteristique(d) for d in d_train]) 
    return c_train

def compute_c_train_by_class(fonction_caracteristique=None, d_train=None, r_train=None, classes=None, c_train=None):
    if fonction_caracteristique is None:
        fonction_caracteristique = challenge.caracteristique
    if d_train is None:
        d_train = challenge.d_train
    if r_train is None:
        r_train = challenge.r_train
    if classes is None:
        classes = challenge.classes
    if c_train is None:
        c_train = compute_c_train(fonction_caracteristique, d_train)
    c_train_par_population = [c_train[r_train==k] for k in classes]
    return c_train_par_population

def affichage_dix_caracteristique(predictions=False):
    challenge.affichage_dix()
    df = pd.DataFrame()
    df['$r$ (classe)'] = challenge.r_train[0:10]   
    caracteristique = get_variable('caracteristique')
    df['$k$ (caracteristique)'] = [caracteristique(d) for d in challenge.d_train[0:10]]
    if predictions:
        df['$\hat{r}$ (prediction)'] = '?'
    df.index+=1

    display(df)
    return

def get_algorithme_func(error=None):
    if has_variable('algorithme'):
        algorithme = get_variable('algorithme')
    else:
        algorithme = None
        
    if algorithme is None or not callable(algorithme):
        if error:
            print_error(error)
        else:
            print_error("Vous avez remplacé autre chose que les ... . Revenez en arrière avec le raccourci clavier Ctrl+Z pour annuler vos modifications.")
        return None

    return algorithme

def get_estimations(data, algorithme=None):
    if algorithme is None:
        algorithme = get_algorithme_func()
    if not algorithme:
        return None

    r_prediction = np.array([r for r in map(algorithme, data)])
    return r_prediction

def calcul_caracteristiques(d_train, caracteristique):
    """Fonction qui calcule les caractéristiques de toutes les images de d_train"""
    # vec_caracteristique = np.vectorize(caracteristique, signature="(m,n)->()")
    return np.array([caracteristique(d) for d in d_train])

# Calculer l'estimation et l'erreur : 
def erreur_train(d_train, r_train, x, classification, caracteristique):
    """Fonction qui calcule l'erreur d'entraînement pour un seuil t donné"""
    return erreur_train_optim(calcul_caracteristiques(d_train, caracteristique),r_train,x,classification)

# Calculer l'estimation et l'erreur a partir du tableau de caractéristique des images : 
def erreur_train_optim(k_d_train, r_train, x, classification):
    # Vectorize the classification function if it's not already vectorized
    r_train_est = np.vectorize(classification)(k_d_train,x)
    
    # Calculate the mean error by comparing the estimated y values with the actual r_train values
    return np.mean(r_train_est != r_train)

def compute_erreur(func_carac=None):
    if func_carac is None:
        func_carac = challenge.caracteristique
    
    func_classif = get_variable('classification')

    # pas_x = 4
    pas_x = 1
    
    # Slice d_train and r_train using numpy's advanced slicing
    d_train_sliced = challenge.d_train[::pas_x]
    r_train_sliced = challenge.r_train[::pas_x]

    k_d_train_sliced = calcul_caracteristiques(d_train_sliced, func_carac)
    t_min = int(k_d_train_sliced.min())-1
    t_max = int(k_d_train_sliced.max())+1
    
    # Vectorize the erreur_train function to apply it over an array of t_values
    vec_erreur_train = np.vectorize(lambda t: 100 * erreur_train_optim(k_d_train_sliced, r_train_sliced, t, func_classif))
    
    # Calculer pas_t pour avoir environ 50 points:
    pas_t = int((t_max - t_min) / 50)+1
    # Create a range of t values using numpy's arange function
    t_values = np.arange(t_min, t_max, pas_t)

    # Apply the vectorized function to all t_values
    scores_array = vec_erreur_train(t_values)

    return (t_values, scores_array)
    
def calculer_score(algorithme, method=None, parameters=None, cb=None, a=None, b=None, caracteristique=None):
    try:
        print("Calcul du pourcentage d'erreur en cours...")

        r_prediction_train = get_estimations(challenge.d_train, algorithme=algorithme)
        score = np.mean(r_prediction_train != challenge.r_train)
        set_score(score)

        print(f"Voici les prédictions r^ de ton algorithme pour chaque {challenge.strings['dataname']}")
        affichage_banque(carac=caracteristique, showPredictions=True, estimations=r_prediction_train)
        
        if cb is not None:
            cb(score)

    except Exception as e:
        print_error("Il y a eu un problème lors du calcul de l'erreur. Vérifie ta réponse.")
        if debug:
            raise(e)
        
def calculer_score_2(algorithme, method=None, parameters=None, cb=None, a=None, b=None, caracteristique=None):
    try:
        print("Calcul du pourcentage d'erreur en cours...")

        r_prediction_train = get_estimations(challenge.d_train, algorithme=algorithme)
        score = np.mean(r_prediction_train != challenge.r_train)
        set_score(score)

        
        if cb is not None:
            cb(score)

    except Exception as e:
        print_error("Il y a eu un problème lors du calcul de l'erreur. Vérifie ta réponse.")
        if debug:
            raise(e)        

# Fonctions trame notebook générique

def calculer_score_etape_1():
    algorithme = get_algorithme_func()
    if algorithme is None:
        return
    
    def cb(score):
        validation_score_fixe()
        set_step(1)

    calculer_score(algorithme, method="fixed", cb=cb)  

def calculer_score_carac():
    t = get_variable('t')
    caracteristique = get_variable('caracteristique')
    classification = get_variable('classification')

    def algorithme(d):
        x = caracteristique(d)
        return classification(x, t)
    
    def cb(score):
        validation_score_carac()
        set_step(2)

    calculer_score(algorithme, method="carac ref", parameters=f"t={t}", cb=cb, caracteristique=caracteristique) 

def calculer_score_custom():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable('r_grande_caracteristique'):
        print_error('Remplacez tous les ... par vos paramètres.')
        return
    
    t = get_variable('t')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    
    def algorithme(d):
        x = challenge.caracteristique_custom(d)
        if x <= t:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
        
    def cb(score):
        res = challenge.cb_custom_score(score)
        if res:
            validation_custom()
    
    calculer_score(algorithme, method="carac custom", parameters=f"t={t}", caracteristique=challenge.caracteristique_custom, cb=cb)

def calculer_score_code_eleve():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable('r_grande_caracteristique'):
        print_error('Remplacez tous les ... par vos paramètres.')
        return
    
    t = get_variable('t')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    
    def algorithme(d):
        x = get_variable('caracteristique')(d)

        if x <= t:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
        
    def cb(score):
        if score < 0.04:
            validation_code_eleve()
        else:
            print_error("Essaie de trouver une zone qui fait moins de 5% d'erreur.")
    
    calculer_score(algorithme, method="code eleve", caracteristique=get_variable('caracteristique'))

def get_erreur_plot(func_carac=None):
    (t_values, scores_array) = compute_erreur(func_carac)

    best_score_index = np.argmin(scores_array)
    global best_t
    best_t = t_values[best_score_index]

    return [t_values, scores_array]
    

def tracer_erreur(id=None, func_carac=None):

    [t_values, scores_array] = get_erreur_plot(func_carac)

    if id is None:
        id = uuid.uuid4().hex + '-graph'
        display(HTML(f'''
            <canvas id="{id}"/>
        '''))

    run_js(f'setTimeout(() => window.mathadata.tracer_erreur("{id}", {t_values.tolist()}, {scores_array.tolist()}), 500)')

def update_graph_erreur(id="graph_custom", func_carac=None):
    if func_carac is None:
        func_carac = challenge.caracteristique_custom
        
    [t_values, scores_array] = get_erreur_plot(func_carac)
    run_js(f'window.mathadata.tracer_erreur("{id}", {t_values.tolist()}, {scores_array.tolist()})')

def exercice_droite_carac():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div>
            <canvas id="{id}-chart"></canvas>
            <p id="{id}-status"></p>
        </div>
    '''))

    size = 10
    set = challenge.d_train[0:size]
    c_train = compute_c_train(challenge.caracteristique, set)
    params = {
        'c_train': c_train,
        'labels': [0 if r == challenge.classes[0] else 1 for r in challenge.r_train[0:size]],
    }
    
    run_js(f"setTimeout(() => window.mathadata.exercice_droite_carac('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

def calculer_score_seuil_optimise():
    if not validation_question_seuil_optimise():
        return

    t = get_variable('t')
    caracteristique = get_variable('caracteristique')
    classification = get_variable('classification')

    def algorithme(d):
        x = caracteristique(d)
        return classification(x, t)

    def cb(score):
        set_step(3)
    
    calculer_score(algorithme, method="carac ref seuil optim", parameters=f"t={t}", cb=cb) 

### Analytics ###

session_id = None
capytale_id = None
capytale_classroom = None

def start_analytics_session(notebook_id):
    return
    global capytale_id, capytale_classroom
    if sequence:
        seed = user_seed()
        capytale_id = seed
        if seed == 609507 or seed == 609510:
            capytale_classroom = "dev_capytale"
            debug = True
        
        def profile_callback(profile):
            global capytale_id, capytale_classroom
            try:
                capytale_id = profile['uid']
                if capytale_classroom is None:
                    capytale_classroom = profile['classe']
                get_highscore()
                if profile['profil'] == 'teacher':
                    return

                # create analytics session for students except mathadata accounts
                create_session(notebook_id)
            except Exception as e:
                if debug:
                    print_error("Error during post request")
                    print_error(e)
        
        call_async(get_profile, profile_callback)
        
    else:
        capytale_id = -1
        capytale_classroom = "dev"
        create_session(notebook_id)
        get_highscore()


def create_session(notebook_id):
    def cb(data):
        global session_id
        session_id = data['id']

    http_request(analytics_endpoint + '/session', 'POST', {
        'notebook_id': notebook_id,
        'user_id': capytale_id,
        'classname': capytale_classroom
    }, cb=cb)


### Gestion du score ###

highscore = None
session_score = None

# Display score as percentage with one decimal
def score_str(score):
    percent = score * 100
    return f"{percent:.1f}%"

def get_highscore(challenge_id=116):
    return
    def cb(data):
        if data is not None and isinstance(data, dict) and 'highscore' in data:
            global highscore
            highscore = data['highscore']
            update_score()
        elif debug:
            print_error("Failed to get highscore. Received data" + str(data))
        
    http_request_cd(f'/participants/challenges/{challenge_id}/highscore?capytale_id={capytale_id}', 'GET', cb=cb)

def set_score(score):
    print('Nouveau pourcentage d\'erreur : ' + score_str(score))
    global session_score, highscore
    if session_score is None or score < session_score:
        session_score = score
        if highscore is None or session_score < highscore:
            highscore = session_score
        update_score()
    
    
def submit(csv_content, challenge_id=116, method=None, parameters=None, cb=None):
    if (capytale_id is None):
        return
    
    def internal_cb(data):
        if data is not None and isinstance(data, dict) and 'score' in data:
            set_score(data['score'])
            
            if cb is not None:
                cb(data['score'])
        else:
            print('Il y a eu un problème lors du calcul de l\'erreur. Ce n\'est probablement pas de ta faute, réessaye dans quelques instants.')
            if debug:
                print_error("Received data" + str(data))

    http_request_cd(f'/participants/challenges/{challenge_id}/submit', 'POST', files={
        'file': csv_content,
    }, fields={
        'method': method,
        'parameters': parameters,
        'capytale_id': capytale_id,
        'capytale_classroom': capytale_classroom,
    }, cb=internal_cb)
    
    
### Customisation page web ###

def run_js(js_code):
    if sequence:
        eval(js_code)
    else:
        display(Javascript(js_code))

styles = """
#sidebox {
    position: fixed;
    top: 20vh;
    left: 0;
    max-height: 60vh;
    width: 20vw;
    background-color: white;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    z-index: 1000;
    transition: left 0.5s;
}

.sidebox-main {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
    flex: 1;
}

.sidebox-collapse-button {
    position: absolute;
    left: 100%;
    top: 30%;
    bottom: 30%;
    margin: auto;
    padding: 0.5rem;
    display: flex;
    justify-content: center;
    align-items: center;
    cursor: pointer;
    background-color: black;
    border-left: none;
    border-radius: 0 5px 5px 0;
}

.sidebox-button-icon {
    transition: transform 0.2s;
}

.sidebox-header {
    display: flex;
    gap: 3px;
    align-items: center;
}

.sidebox-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;

}

.score {
    font-size: 2rem;
    font-weight: bold;
}

.exercise {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin: 1rem 0;
}

.exercise .question {
    font-size: 1.2rem;
}

.exercise .chart {
    display: flex;
    justify-content: center;
}

.exercise .answers {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.exercise .answer {
    display: flex;
    gap: 1rem;    
}

.exercise .error {
    color: red;
}

.exercise .success {
    color: green;
}

.exercise .info {
    color: black;    
}

.bank-row-misclassed {
    background-color: #ffcccc !important;
}

#step-card {
    position: fixed;
    z-index: 1000;
    top: 120px;
    right: 25px;
    max-width: 200px;
    padding: 1rem 2rem;
    background-color: white;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    text-align: center;
}

#sos-box {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
}

.sos-button {
    background-color: #ff4444;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px;
    font-size: 24px;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s, background-color 0.2s;
    display: flex;
    justify-content: center;
    align-items: center;
}

.sos-button:hover {
    transform: scale(1.1);
    background-color: #ff6666;
}

.sos-details {
    background-color: white;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    max-width: 300px;
    display: none;
}

.sos-details.visible {
    display: block;
}

.sos-details h3 {
    margin-top: 0;
    color: #ff4444;
}

.sos-details p {
    margin: 10px 0;
    line-height: 1.4;
}

.sos-details code {
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: monospace;
}

.sos-details .shortcut {
    background-color: #e0e0e0;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-weight: bold;
}

.tip {
    background-color: lightyellow; /* Light yellow background */
    padding: 15px; /* Increased padding */
    border-radius: 5px;
    border-left: 5px solid #ffcc00; /* Yellow-orange left border */
    margin-bottom: 10px; /* Margin at the bottom */
}

.mcq-question {
    font-weight: bold;
    margin-bottom: 10px;
}
.mcq-answers {
    margin-bottom: 15px;
}
.mcq-choice {
    margin-bottom: 5px;
}
.mcq-choice label {
    margin-left: 5px;
}
.mcq-validate-button {
    padding: 8px 15px;
    background-color: #4CAF50; /* Green */
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    margin-bottom: 10px;
}
.mcq-validate-button:hover {
    background-color: #45a049;
}
.mcq-message {
    font-weight: bold;
    padding: 5px;
    border-radius: 3px;
}
"""

run_js(f"""
    let style = document.getElementById('mathadata-style');
    if (style === null) {{
        const style = document.createElement('style');
        style.id = 'mathadata-style';
        style.innerHTML = `{styles}`;
        document.head.appendChild(style);
    }}
""")

# Fonctions utilitaires JS
run_js("""

    function chartjs_title(context) {
        return context[0]?.dataset?.label
    }
    
    function chartjs_label(context) {
        return `abscisse: ${context.label || context.parsed?.x.toFixed(2)} , ordonnée: ${context.parsed?.y.toFixed(2)}`;
    }

    let i_exercice_droite_carac = 0
    const mathadata = {

        data(article, params) {
            const { plural, uppercase } = params || {};
            let res = "";
            if (article === "de") {
                if (window.mathadata.challenge.strings.contraction) {
                    res += "d'";
                } else {
                    res += "de ";
                }
            } else if (article === "un") {
                if (window.mathadata.challenge.strings.feminin) {
                    res += "une ";
                } else {
                    res += "un ";
                }
            } else if (article === "du") {
                if (window.mathadata.challenge.strings.contraction) {
                    res += "de l'";
                } else if (window.mathadata.challenge.strings.feminin) {
                    res += "de la ";
                } else {
                    res += "du ";
                }
            } else if (article === "le") {
                if (window.mathadata.challenge.strings.contraction) {
                    res += "l'";
                } else if (window.mathadata.challenge.strings.feminin) {
                    res += "la ";
                } else {
                    res += "le ";
                }
            }
            
            if (plural) {
                res += window.mathadata.challenge.strings.dataname_plural;
            } else {
                res += window.mathadata.challenge.strings.dataname;
            }

            if (uppercase) {
                res = res.charAt(0).toUpperCase() + res.slice(1);
            }
            
            return res;
        },

        ac_fem(fem, mas) {
            if (window.mathadata.challenge.strings.feminin) {
                return fem;
            } else {
                return mas;
            }
        },

        e_fem() {
            return window.mathadata.ac_fem("e", "");
        },

        run_python(python, onResult, onStream) {
            console.log('Execute python code', python)
            const onPythonResult = (data) => {
                try {
                    if (data.msg_type === "stream") {
                        if (onStream !== undefined) {
                            onStream(data.content.text)
                        } else {
                            window.alert('Python output : ' + data.content.text)
                        }
                    }
                    else if (data.msg_type === "error") {
                        console.log(data)
                        window.alert('Python error : ' + data.content.evalue)
                    }
                    else if (data.msg_type === "execute_result") {
                        if (onResult !== undefined && onResult.length > 0) { // check if callback provided expects parameters
                            const res_py = data.content.data['text/plain']
                            // remove surrounding quotes and escape added by Python to js (regex is to replace \' by ' with escaping for python and js)
                            const res_json = res_py.substring(1, res_py.length - 1).replace(/\\\\'/g, "'")
                            const res = JSON.parse(res_json)
                            onResult(res)
                        }
                    }
                } catch(e) {
                    console.error('Error processing python result', data)
                    console.error(e)
                }
            }
            
            const onShellMessage = (data) => {
                if (data.msg_type === "execute_reply") {
                    if (onResult !== undefined && onResult.length === 0) {
                        // use this websocket message to trigger callback for functions that don't return anything
                        onResult()
                    }
                }
            }

            Jupyter.notebook.kernel.execute(python, {
                iopub: {
                    output: onPythonResult
                },
                shell: {
                    reply: onShellMessage
                }
            }, {silent:false})
        },

        pass_breakpoint() {
            if (typeof basthon !== 'undefined') {
                basthon.breakpointMoveOn()
            }
        },

        create_chart(div_id, config) {

            for (const dataset of config.data.datasets) {
                if (dataset.pointHoverRadius === undefined && typeof dataset.pointRadius === 'number') {
                    dataset.pointHoverRadius = dataset.pointRadius + 2
                }
            }   
        
            // Ensure nested objects exist
            config.options = config.options || {};
            config.options.plugins = config.options.plugins || {};
            config.options.plugins.tooltip = config.options.plugins.tooltip || {};

            if (config.options.plugins.tooltip.callbacks === undefined) {
                config.options.plugins.tooltip.callbacks = {
                    title: chartjs_title,
                    label: chartjs_label
                }
            }

            if (config.options.plugins.tooltip.filter === undefined && (config.options.plugins.tooltip.mode || config.options.interaction?.mode || 'nearest') === 'nearest') {
                config.options.plugins.tooltip.filter = function (item, index) {
                    return index === 0;
                }
            }

            if (config.options.plugins.dragData === undefined) {
                config.options.plugins.dragData = {
                   dragY: false, 
                }
            }

            if (window.mathadata.charts[div_id] !== undefined) {
                console.log('update chart')
                Object.assign(window.mathadata.charts[div_id], config)
                window.mathadata.charts[div_id].update()
            } else {
                console.log('create chart')
                const wrapper = document.createElement('div');
                wrapper.style.width = '100%';
                wrapper.style.maxHeight = '60vh';
                wrapper.style.aspectRatio = config.options.aspectRatio;
                wrapper.style.display = 'flex';
                wrapper.style.justifyContent = 'center';
                wrapper.style.alignItems = 'center';
                wrapper.style.backgroundColor = 'white';

                // move inside wrapper
                const canvas = document.getElementById(div_id)
                canvas.style.width = '100%'
                canvas.parentNode.insertBefore(wrapper, canvas)
                wrapper.appendChild(canvas)
                
                const ctx = canvas.getContext('2d');
                window.mathadata.charts[div_id] = new Chart(ctx, config);
            }

            console.log(Object.keys(window.mathadata.charts))
        },

        create_exercise(div_id, config) {
            window.mathadata.exercises[div_id] = {}
            window.mathadata.setup_exercise_template(div_id, config)
            window.mathadata.setup_exercise_validation(div_id, config.questions) 
            window.mathadata.display_exercise_step(div_id, config.questions[0])
        },

        setup_exercise_template(div_id, config) {
            const template = `
                <div class="exercise">
                    <div id="${div_id}-question" class="question"></div>
                    <canvas id="${div_id}-canvas"></canvas>
                    <div id="${div_id}-answers" class="answers"></div>
                    <button id="${div_id}-submit">Valider</button>
                    <p id="${div_id}-status" class="status"></p>
                </div>
            `

            const wrapper = document.getElementById(`${div_id}`);
            wrapper.innerHTML = template;

            window.mathadata.create_chart(`${div_id}-canvas`, config.chart)
        },

        setup_exercise_validation(div_id, config) {
            const nb_steps = config.length
            let current_step = 0

            const setStatus = (msg, className) => {
                const status = document.getElementById(`${div_id}-status`);
                status.innerHTML = msg;
                status.className = className;
            }

            const onPythonResult = (data) => {
                if (data.msg_type === "stream") {
                    setStatus(data.content.text, "info")
                }
                else if (data.msg_type === "error") {
                    setStatus(data.content.evalue, "error")
                }
                else if (data.msg_type === "execute_result") {
                    const res_py = data.content.data['text/plain']
                    // remove surrounding quotes and escape added by Python to js (regex is to replace \' by ' with escaping for python and js)
                    const res_json = res_py.substring(1, res_py.length - 1).replace(/\\\\'/g, "'")
                    const res = JSON.parse(res_json)
                    
                    if (res.is_correct) {
                        if (current_step < nb_steps - 1) {
                            setStatus("Bravo, c'est la bonne réponse !", "success")
                            current_step++
                            window.mathadata.display_exercise_step(div_id, config[current_step])
                        } else {
                            setStatus("Bravo, c'est la bonne réponse ! Vous pouvez passer à la suite du notebook", "success")
                        }
                    }
                    else {
                        setStatus(res.errors.join('<br>'), "error")
                    }
                }
            }

            const submitButton = document.getElementById(`${div_id}-submit`);
            submitButton.addEventListener('click', () => {
                const answers = config[current_step].answers
                const userAnswers = Object.keys(answers).map(name => {
                    let value = null
                    switch (answers[name].type) {
                        case 'number':
                            value = parseFloat(document.getElementById(`${div_id}-answer-${name}`).value)
                            break;
                        case 'radio':
                            value = document.querySelector(`input[name=${div_id}-answer-${name}]:checked`).value
                            break;
                    }
                    return {
                        name,
                        value,
                    }
                });

                const params = {
                    step: current_step,
                    answers: userAnswers,
                }
                const python = `submit_exercise('${div_id}', '${JSON.stringify(params)}')`
                Jupyter.notebook.kernel.execute(python, {
                    iopub: {
                        output: onPythonResult
                    }
                }, {silent:false})
            });
        },
            
        display_exercise_step(div_id, step) {
            const question = document.getElementById(`${div_id}-question`)
            question.innerHTML = step.question
            
            window.mathadata.create_chart(`${div_id}-canvas`, step.chart)
            
            const answers = document.getElementById(`${div_id}-answers`)
            let answers_html = ''
            for (const [name, answer] of Object.entries(step.answers)) {
                if (answer.type === 'number') {
                    answers_html += `
                        <div class="answer">
                            <label for="${div_id}-answer-${name}">${name}&nbsp;:&nbsp;</label>
                            <input type="number" id="${div_id}-answer-${name}" step=${answer.step}>
                        </div>
                    `
                } else if (answer.type === 'radio') {
                    const choices = answer.choices.map(choice => `
                        <input type="radio" id="${div_id}-answer-${name}-${choice}" name="${div_id}-answer-${name}" value="${choice}">
                        <label for="${div_id}-answer-${name}-${choice}">${choice}</label>
                    `).join('&nbsp;&nbsp;')
                    answers_html += `
                        <div class="answer">
                            ${choices}
                        </div>
                    `
                }
            }
            
            answers.innerHTML = answers_html
        },
            
        setup_test_bank(id, params) {
            params = JSON.parse(params)
            const {labels, c_train, estimations, mode} = params
            
            const bank = document.getElementById(`${id}-bank`)

            const display_carac = c_train ? true : false
            const multiple_caracs = display_carac && Array.isArray(c_train[0])
            const caracNames = ['x', 'y', 'z']

            let caracColumnDefs
            let getCaracData
            if (multiple_caracs) {
                caracColumnDefs = caracNames.map((name, i) => ({headerName: `Caractéristique ${name}`, field: name, hide: c_train[0].length <= i}))
                getCaracData = (index) => {
                    const res = {}
                    for (let i = 0; i < c_train[index].length; i++) {
                        res[caracNames[i]] = Math.round(c_train[index][i] * 100) / 100
                    }
                    return res
                }
            } else {
                caracColumnDefs = [{headerName: 'Caractéristique x', field: 'x', hide: !display_carac}]
                if (display_carac) {
                    getCaracData = (index) => ({x: Math.round(c_train[index] * 100) / 100})
                } else {
                    getCaracData = () => ({})
                }
            }

            const display_estims = estimations ? true : false

            const select = (index) => {
                window.mathadata.run_python(`get_data(index=${index})`, (data) => {
                    window.mathadata.affichage(`${id}-selected`, data, {mode})
                })
            }

            const config = {
                columnDefs: [
                    {headerName: 'N°', field: 'index'},
                    {headerName: 'Vraie Réponse', field: 'r'},
                    ...caracColumnDefs,
                    {headerName: 'Estimation', field: 'r^', hide: !display_estims},
                    {headerName: 'Statut', field: 'status', hide: !display_estims},
                ],
                rowData: labels.map((label, index) => {
                    return {
                        index: index + 1,
                        r: label,
                        ...getCaracData(index),
                        'r^': display_estims ? estimations[index] : null,
                        status: display_estims ? (label === estimations[index] ? 'Vrai' : 'Faux') : null,
                    }
                }),
                defaultColDef: {
                    flex: 1,
                    minWidth: 100,
                },
                rowSelection: 'single',
                onSelectionChanged: ({api}) => {
                    const selected = api.getSelectedRows()
                    if (selected.length > 0) {
                        select(selected[0].index - 1)
                    }
                },

                // Custom navigation to select the row when using up/down arrow keys
                navigateToNextCell: params => {
                    const suggestedNextCell = params.nextCellPosition;

                    const KEY_UP = 'ArrowUp';
                    const KEY_DOWN = 'ArrowDown';

                    const noUpOrDownKey = params.key !== KEY_DOWN && params.key !== KEY_UP;
                    if (noUpOrDownKey) {
                        return suggestedNextCell;
                    }

                    const nodeToSelect = params.api.getDisplayedRowAtIndex(suggestedNextCell.rowIndex);
                    if (nodeToSelect) {
                        nodeToSelect.setSelected(true);
                    }

                    return suggestedNextCell;
                },
            }

            if (display_estims) {
                config.rowClassRules = {
                    'bank-row-misclassed': ({data}) => {return data.r !== data['r^']},
                };
            }
            
            agGrid.createGrid(bank, config)
            
            setTimeout(() => select(0), 500)
        },

        tracer_erreur(id, t_values, scores_array) {
            const config = {
                type: 'scatter',
                data: {
                    labels: t_values,  // Assuming t_values is already a JavaScript array
                    datasets: [{
                        label: "Erreur d'entrainement",
                        data: scores_array,  // Assuming scores_array is already a JavaScript array
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,  // Thicker cross points
                        pointStyle: 'cross',
                        pointRadius: 6,  // Larger cross points
                    }]
                },
                options: {
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Seuil t'  // Label for x-axis
                            },
                            min: Math.min(...t_values) - 2,
                            max: Math.max(...t_values) + 2,
                            ticks: {
                                stepSize: 2,
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Erreur f(t)',
                            },
                            min: 0,
                            max: 100,
                        }
                    },
                    plugins: {
                        legend: {
                            display: false,  // Remove the blue rectangle at the top
                        },
                    }
                }
            };

            window.mathadata.create_chart(id, config)
        },

        exercice_droite_carac(id, params) {
            params = JSON.parse(params)
            const {c_train, labels} = params

            mathadata.tracer_droite_carac(`${id}-chart`, params)
            const chart = window.mathadata.charts[`${id}-chart`]
            
            for (let i = 0; i < i_exercice_droite_carac; i++) {
                const datasetIndex = labels[i]
                chart.data.datasets[datasetIndex].data.push({x: c_train[i], y: 0})
            }

            chart.options.plugins.tooltip.callbacks.title = (context) => {
                if (context.length === 0 || context[0].datasetIndex >= 2) {
                    return undefined
                }
                
                let dataname = window.mathadata.challenge.strings.dataname
                dataname = dataname.charAt(0).toUpperCase() + dataname.slice(1)
                
                return `${dataname} n°${context[0].dataIndex + 1}`;    
            }

            function setStatusMessage(msg) {
                const status = document.getElementById(`${id}-status`)
                status.innerHTML = msg
            }

            if (i_exercice_droite_carac < c_train.length) {
                setStatusMessage(`Cliquez sur la droite au bon endroit pour placer ${mathadata.data('le')} n°${i_exercice_droite_carac + 1}.`)
                chart.options.onClick = (e) => {
                    const canvasPosition = Chart.helpers.getRelativePosition(e, chart);
                    const dataX = chart.scales.x.getValueForPixel(canvasPosition.x);

                    if (Math.abs(dataX - c_train[i_exercice_droite_carac]) < 1) {
                        const datasetIndex = labels[i_exercice_droite_carac]
                        chart.data.datasets[datasetIndex].data.push({x: c_train[i_exercice_droite_carac], y: 0});
                        i_exercice_droite_carac++
                        if (i_exercice_droite_carac == c_train.length) {
                            chart.options.onClick = null;
                            setStatusMessage(`Bravo, vous avez placé tous les points ! Exécutez la cellule suivante pour passer à la suite.`)
                            localStorage.setItem('exercice_droite_carac_ok', 'true')
                            window.mathadata.run_python(`set_exercice_droite_carac_ok()`)
                        } else {
                            setStatusMessage(`Bravo, vous pouvez placer ${mathadata.data('le')} suivant${mathadata.e_fem()} (n°${i_exercice_droite_carac + 1}) !`)
                        }
                        chart.update()
                    } else {
                        setStatusMessage(`${mathadata.data('le', {uppercase: true})} ne va pas ici sur la droite. Tu as placé un point à l'abscisse ${dataX.toFixed(2)} alors que ${mathadata.data('le')} a une caractéristique x = ${c_train[i_exercice_droite_carac].toFixed(2)}`)
                    }
                }
            } else {
                setStatusMessage(`Bravo, vous avez placé tous les points ! Exécutez la cellule suivante pour passer à la suite.`)
            }

            chart.update()
        },

        tracer_droite_carac(id, params) {
            const {c_train, labels, t} = params
            const max = Math.ceil(Math.max(...c_train) + 1)
            const min = Math.min(0, Math.floor(Math.min(...c_train) - 1))

            const config = {
                type: 'scatter',
                data: {
                    datasets: [
                    {
                        label: `${mathadata.data('', {plural: true, uppercase: true})} de ${mathadata.challenge.classes[0]}`,
                        data: [],
                        pointRadius: 4,
                        pointHoverRadius: 8,
                        pointBackgroundColor: window.mathadata.classColors[0],
                    },
                    {
                        label: `${mathadata.data('', {plural: true, uppercase: true})} de ${mathadata.challenge.classes[1]}`,
                        data: [],
                        pointRadius: 4,
                        pointHoverRadius: 8,
                        pointBackgroundColor: window.mathadata.classColors[1],
                    },
                    {
                        type: 'line',
                        label: 'Droite des réels',
                        data: [{x: min, y: 0}, {x: max, y: 0}],  
                        backgroundColor: 'rgb(75, 192, 192)',
                        borderColor: 'rgb(75, 192, 192)',
                        fill: false,
                        borderWidth: 1,
                        pointRadius: 0,
                        pointHitRadius: 0,
                    },
                    ]
                },
                options: {
                    aspectRatio: 5,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Caractéristique x',
                            },
                            min,
                            max,
                            ticks: {
                                stepSize: 1,
                            },
                            grid: {
                                display: true,
                            }
                        },
                        y: {
                            display: false,
                        },
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label(context) {
                                    return `Caractéristique x : ${context.parsed.x.toFixed(2)}`;
                                },
                            },
                            enabled: false, // Enable the default tooltip
                            position: 'nearest',
                            external: (context) => {
                                mathadata.dataTooltip(context, id)
                            },
                        }
                    },
                }
            }

            if (t != undefined) {
                config.data.datasets.push({
                    label: 'Seuil t',
                    data: [{x: t, y: 0}],
                    pointStyle: 'line',
                    rotation: 90,
                    borderWidth: 2,
                    pointRadius: 12,
                    backgroundColor: 'rgba(255, 0, 0, 0.2)',
                    borderColor: 'rgba(255, 0, 0, 1)',
                })
            }
            
            mathadata.create_chart(id, config)
        },
        
        getOrCreateTooltip(chart, id) {
            let tooltipEl = chart.canvas.parentNode.querySelector('div');

            if (!tooltipEl) {
                tooltipEl = document.createElement('div');
                tooltipEl.id = `${id}-tooltip`;
                tooltipEl.style.background = 'rgba(0, 0, 0, 0.7)';
                tooltipEl.style.borderRadius = '3px';
                tooltipEl.style.color = 'white';
                tooltipEl.style.opacity = 1;
                tooltipEl.style.pointerEvents = 'none';
                tooltipEl.style.position = 'absolute';
                tooltipEl.style.transform = 'translate(-50%, 0)';
                tooltipEl.style.transition = 'opacity .3s ease';
                tooltipEl.style.display = 'flex';
                tooltipEl.style.flexDirection = 'column';
                tooltipEl.style.gap = '5px';

                tooltipEl.innerHTML = `
                    <div id="${id}-tooltip-infos"></div>
                    <div id="${id}-tooltip-loading" style="width: 100%; text-align: center;">Chargement ${mathadata.data('du')}...</div>
                    <div id="${id}-tooltip-data" style="width: 100%;"></div>
                `

                chart.canvas.parentNode.appendChild(tooltipEl);
            }

            return tooltipEl;
        },
        
        dataTooltip(context, id) {
            // Tooltip Element
            const {chart, tooltip} = context;
            if (!tooltip?.dataPoints?.length) {
                return;
            }
            
            const {dataIndex, datasetIndex} = tooltip.dataPoints[0];
            const tooltipEl = mathadata.getOrCreateTooltip(chart, id);

            if (datasetIndex >= 2) {
                tooltipEl.style.opacity = 0;
                tooltip.options.enabled = true;
                return;
            }
            
            tooltip.options.enabled = false;

            // Hide if no tooltip
            if (tooltip.opacity === 0) {
                tooltipEl.style.opacity = 0;
                return;
            }

            const {xAlign, x, caretX} = tooltip;

            // Set Text
            if (tooltip.body) {
                const titleLines = tooltip.title || [];
                const bodyLines = tooltip.body.map(b => b.lines);
                const tooltipInfos = document.getElementById(`${id}-tooltip-infos`);

                tooltipInfos.innerHTML = '';

                titleLines.forEach(title => {
                    const span = document.createElement('span');
                    span.innerText = title;
                    span.style.fontWeight = 'bold';
                    tooltipInfos.appendChild(span);
                });

                bodyLines.forEach((body, i) => {
                    const colors = tooltip.labelColors[i];
                    const div = document.createElement('div');

                    const span = document.createElement('span');
                    span.style.background = colors.backgroundColor;
                    span.style.borderColor = colors.borderColor;
                    span.style.borderWidth = '2px';
                    span.style.marginRight = '10px';
                    span.style.height = '10px';
                    span.style.width = '10px';
                    span.style.display = 'inline-block';

                    const text = document.createTextNode(body);

                    div.appendChild(span);
                    div.appendChild(text);
                    tooltipInfos.appendChild(div);
                });

                const loadingDiv = document.getElementById(`${id}-tooltip-loading`);
                const dataDiv = document.getElementById(`${id}-tooltip-data`);

                loadingDiv.style.visibility = 'visible';
                dataDiv.style.visibility = 'hidden';
                
                let dataClass = mathadata.challenge.classes[datasetIndex]
                if (typeof dataClass === 'string') {
                    dataClass = `'${dataClass}'`
                }
                
                window.mathadata.run_python(`get_data(index=${dataIndex}, dataClass=${dataClass})`, (data) => {
                    window.mathadata.affichage(`${id}-tooltip-data`, data);
                    loadingDiv.style.visibility = 'hidden';
                    dataDiv.style.visibility = 'visible';
                });
            }

            const {offsetLeft: positionX, offsetTop: positionY} = chart.canvas;

            // Display, position, and set styles for font
            tooltipEl.style.opacity = 1;
            tooltipEl.style.left = positionX + tooltip.caretX + 'px';
            let translateX = '0';
            if (tooltip.xAlign === 'center') {
                translateX = '-50%';
            } else if (tooltip.xAlign === 'right') {
                translateX = '-100%';
            }
            
            tooltipEl.style.top = positionY + tooltip.caretY + 'px';
            let translateY = '0';
            if (tooltip.yAlign === 'center') {
                translateY = '-50%';
            } else if (tooltip.yAlign === 'bottom') {
                translateY = '-100%';
            }

            tooltipEl.style.transform = `translate(${translateX}, ${translateY})`;
                
            tooltipEl.style.font = tooltip.options.bodyFont.string;
            tooltipEl.style.padding = tooltip.options.padding + 'px ' + tooltip.options.padding + 'px';

            // Set caret Position
            tooltipEl.classList.remove('above', 'below', 'no-transform');
            if (tooltip.yAlign) {
                tooltipEl.classList.add(tooltip.yAlign);
            } else {
                tooltipEl.classList.add('no-transform');
            }
        },

        // variables in mathadata object
        charts: {},
        exercises: {},
        classColors: ['rgba(80,80,255,0.5)', 'rgba(255, 165, 0, 0.5)'],
        classColorCodes: ['80,80,255', '255, 165, 0'],
        centroidColorCodes: ['0,0,100', '255,100,0'],
    }

    window.mathadata = mathadata;

    if (localStorage.getItem('exercice_droite_carac_ok') === 'true') {
        window.mathadata.run_python(`set_exercice_droite_carac_ok()`)
        i_exercice_droite_carac = 10
    }
""")

run_js(f"""
    window.mathadata.files_url = '{files_url}';
""")

def create_sidebox():
    js_code = f"""
    
    let sidebox = document.getElementById('sidebox');
    if (sidebox !== null) {{
        sidebox.remove();
    }}
    
    sidebox = document.createElement('div');
    sidebox.id = 'sidebox';
    sidebox.style.left = '-20vw';
    
    sidebox.innerHTML = `
        <div class="sidebox-main">
            <div class="sidebox-header">
                <h3>Erreur</h3>
            </div>
            <div style="display: flex; flex-wrap: wrap; justify-content: space-around; width: 100%; margin-top: 2rem">
                <div class="sidebox-section">
                    <h4>Meilleur</h4>
                    <svg xmlns="http://www.w3.org/2000/svg" width=40 height=40 viewBox="0 0 512 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#B197FC" d="M4.1 38.2C1.4 34.2 0 29.4 0 24.6C0 11 11 0 24.6 0H133.9c11.2 0 21.7 5.9 27.4 15.5l68.5 114.1c-48.2 6.1-91.3 28.6-123.4 61.9L4.1 38.2zm503.7 0L405.6 191.5c-32.1-33.3-75.2-55.8-123.4-61.9L350.7 15.5C356.5 5.9 366.9 0 378.1 0H487.4C501 0 512 11 512 24.6c0 4.8-1.4 9.6-4.1 13.6zM80 336a176 176 0 1 1 352 0A176 176 0 1 1 80 336zm184.4-94.9c-3.4-7-13.3-7-16.8 0l-22.4 45.4c-1.4 2.8-4 4.7-7 5.1L168 298.9c-7.7 1.1-10.7 10.5-5.2 16l36.3 35.4c2.2 2.2 3.2 5.2 2.7 8.3l-8.6 49.9c-1.3 7.6 6.7 13.5 13.6 9.9l44.8-23.6c2.7-1.4 6-1.4 8.7 0l44.8 23.6c6.9 3.6 14.9-2.2 13.6-9.9l-8.6-49.9c-.5-3 .5-6.1 2.7-8.3l36.3-35.4c5.6-5.4 2.5-14.8-5.2-16l-50.1-7.3c-3-.4-5.7-2.4-7-5.1l-22.4-45.4z"/></svg>
                    <span id="highscore" class="score">{score_str(highscore) if highscore is not None else "..."}</span>
                </div>
                <div class="sidebox-section">
                    <h4>Session</h4>
                    <svg xmlns="http://www.w3.org/2000/svg" width=40 height=40 viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#63E6BE" d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512H418.3c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304H178.3z"/></svg>
                    <span id="session-score" class="score">...</span>
                </div>
            </div>
        </div>
    `;

    let collapseButton = document.createElement('div');
    collapseButton.className = 'sidebox-collapse-button';

    const buttonIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    buttonIcon.setAttribute('class', 'sidebox-button-icon');
    buttonIcon.setAttribute('width', '20');
    buttonIcon.setAttribute('height', '20');
    buttonIcon.setAttribute('viewBox', '0 0 320 512');
    buttonIcon.setAttribute('fill', 'white');

    buttonIcon.innerHTML = `
        <path d="M310.6 233.4c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L242.7 256 73.4 86.6c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0l192 192z"/>
    `;

    collapseButton.appendChild(buttonIcon);

    window.mathadata.collapseSidebox = () => {{
        let sidebox = document.getElementById('sidebox');
        if (sidebox.style.left === '0px') {{
            sidebox.style.left = '-20vw';
            buttonIcon.style.transform = ''; 
        }} else {{
            sidebox.style.left = '0px';
            buttonIcon.style.transform = 'rotate(180deg)'; 
        }}
    }};
    
    collapseButton.addEventListener('click', () => {{
        window.mathadata.collapseSidebox();
    }});

    sidebox.appendChild(collapseButton);
    document.body.appendChild(sidebox);
    """    
    
    run_js(js_code)

def create_sos_box():
    js_code = f"""
    let sosBox = document.getElementById('sos-box');
    if (sosBox !== null) {{
        sosBox.remove();
    }}
    
    sosBox = document.createElement('div');
    sosBox.id = 'sos-box';
    
    sosBox.innerHTML = `
        <div class="sos-details">
            <h3>Besoin d'aide ?</h3>
            <p>Si votre notebook ne fonctionne pas correctement (un affichage semble bloqué, il ne se passe rien quand vous écécutez une cellule, etc.)</p>
            <ol>
                <li>Cliquez sur le boutton <img src="{files_url}/rerun_button.png" alt="restart" style="height: 2.5rem; aspect-ratio: auto;"/> en haut dans la barre d'outils</li>
                <li>Confirmez en cliquant sur <img src="{files_url}/relancer_executer_confirmation.png" alt="Relancer et exécuter toutes les cellules" style="height: 2.5rem; aspect-ratio: auto;"/></li>
            </ol>
        </div>
        <button class="sos-button" title="Besoin d'aide ?">SOS ?<br/>(Cliquer ici)</button>
    `;

    const button = sosBox.querySelector('.sos-button');
    const details = sosBox.querySelector('.sos-details');
    
    button.addEventListener('click', () => {{
        details.classList.toggle('visible');
    }});

    // Fermer les détails si on clique en dehors
    document.addEventListener('click', (event) => {{
        if (!sosBox.contains(event.target)) {{
            details.classList.remove('visible');
        }}
    }});

    document.body.appendChild(sosBox);
    """
    
    run_js(js_code)

# Exécuté à l'import du notebook
create_sidebox()
create_sos_box()

def update_score():
    if highscore is None and session_score is None:
        return 
    js_code = ""
    if highscore is not None:
        js_code += f"document.getElementById('highscore').innerText = '{score_str(highscore)}';"
    if session_score is not None:
        js_code += f"document.getElementById('session-score').innerText = '{score_str(session_score)}';"

    # Pop out the sidebox if the score is updated
    js_code += "window.mathadata?.collapseSidebox();"

    run_js(js_code)


def set_step(number):
    colors = ['#2E8B57', '#87CEEB', '#4169E1', '#DC143C']
    names = ['Algorithme Fainéant', 'Algorithme Moyenne', 'Optimisation' , 'Faire mieux']
    
    run_js(f"""
        const pageElement = document.getElementById('notebook');
        pageElement.style.transition = 'background-color 0.5s ease';
        pageElement.style.backgroundColor = '{colors[number]}';
        pageElement.style.position = 'relative';

        let card = document.getElementById('step-card');
        if (!card) {{
            card = document.createElement('div');
            card.id = 'step-card';
            card.innerHTML = `
                <h3>Étape <span id="step-number">{number + 1}</span> / 4</h3>
                <p id="step-name" style="font-size: 1.5rem;">{names[number]}</p>
            `
            pageElement.appendChild(card);
        }} else {{
            document.getElementById('step-number').innerText = {number} + 1;
            document.getElementById('step-name').innerText = '{names[number]}';
        }}
    """)

### VALIDATION CLASSES ###

class _MathadataValidate():
    counter = 0
    last_call_time = None
    
    def __init__(self, success=None, function_validation=None, on_success=None, tips=None, *args, **kwargs):
        self.trials = 0
        self.success = success
        self.function_validation = function_validation
        self.child_on_success = on_success
        self.tips = tips
        if self.tips is not None:
            self.tips.reverse() # Display the last tip first

    def __call__(self):
        self.last_call_time = time.time()

        # Set question number at first trial
        if self.trials == 0:
            _MathadataValidate.counter += 1
            self.question_number = _MathadataValidate.counter
            self.first_call_time = time.time()

        self.trials += 1
        errors = []
        answers = {}

        res = None
        try:
            res = self.validate(errors, answers)
        except Exception as e:
            errors.append("Il y a eu une erreur dans la validation de la question. Vérifie que ta réponse est écrite correctement")
            if debug:
                raise e

        if res is None:
            res = len(errors) == 0

        self.send_analytics_event(res, answers)

        if len(errors) > 0:
            for error in errors:
                print_error(error)

        if not res and self.tips is not None:
            for tip in self.tips: 
                if 'trials' in tip and self.trials < tip['trials']:
                    continue
                if 'seconds' in tip and (self.first_call_time is None or time.time() - self.first_call_time < tip['seconds']):
                    continue

                if 'tip' in tip:
                    display(HTML(f'''
                        <div class="tip">
                            <p>💡 {tip['tip']}</p>
                        </div>
                    '''))

                if 'print_solution' in tip and tip['print_solution'] == True:
                    display(HTML(f'''
                        <div class="tip">
                            <p>💡 Voici la solution :</p>
                            {"<br/>".join(self.get_variables_str())}
                        </div>
                    '''))
                
                if 'validate' in tip and tip['validate'] == True:
                    display(HTML(f'''
                        <div class="tip">
                            <p>🔓 Tu peux passer cette question et continuer</p>
                        </div>
                    '''))
                    res = True
                
                break

        if has_variable('superuser') and get_variable('superuser') == True:
            res = True

        if res:
            self.on_success(answers) 

        return res

    def send_analytics_event(self, res, answers):
        if session_id is not None:
            try:
                answers_json = json.dumps(answers)
            except TypeError as e:
                answers = {}

            http_request(analytics_endpoint + '/event', 'POST', {
                'session_id': session_id,
                'question_number': self.question_number,
                'is_correct': res,
                'answer': answers,
            })

    def validate(self, errors, answers):
        if self.function_validation is not None:
            return self.function_validation(errors, answers)
        return True

    def on_success(self, answers):
        if self.success is not None:
            if self.success:
                print(self.success)
        else:
            print("Bravo, c'est la bonne réponse !")
        if self.child_on_success is not None:
            self.child_on_success(answers)

    def get_variables_str(self):
        pass

    def trial_count(self):
        return self.trials

if sequence:

    @validationclass
    class MathadataValidate(_MathadataValidate, Validate):
        
        def __init__(self, *args, **kwargs):
            _MathadataValidate.__init__(self, *args, **kwargs)
            Validate.__init__(self)
else:
    MathadataValidate = _MathadataValidate

def check_variable(errors, name, val, expected):
    if expected is None: # Variable is optional
        res = True
    elif isinstance(expected, dict):
        res = compare(val, expected['value'])
    else:
        res = compare(val, expected)

    if not res:
        check_errors(errors, name, val, expected)

    return res

def compare(val, expected):
    try:
        if isinstance(expected, dict):
            if 'is' in expected:
                return val == expected
            elif 'min' in expected and 'max' in expected:
                return val >= expected['min'] and val <= expected['max']
            elif 'min' in expected:
                return val >= expected['min']
            elif 'max' in expected:
                return val <= expected['max']
            elif 'in' in expected:
                return val in expected['in']
            else:
                raise ValueError(f"Malformed validation class : comparing {val} to {expected}")
        elif isinstance(val, (list, np.ndarray, tuple)):
            if not isinstance(expected, (list, np.ndarray, tuple)):
                return False
            if len(val) != len(expected):
                return False
            for i in range(len(val)):
                if not compare(val[i], expected[i]):
                    return False
            return True
        elif isinstance(val, (float, np.float64)):
            return math.isclose(val, expected, rel_tol=1e-9)
        else:
            return val == expected
    except Exception as e:
        return False

def check_errors(errors, name, val, expected):
    if isinstance(expected, dict) and 'errors' in expected:
        for error in expected['errors']:
            match_error = compare(val, error['value'])
            if match_error and 'if' in error:
                errors.append(error['if'])
                return
            if not match_error and 'else' in error:
                errors.append(error['else'])
                return

    # Default error message
    errors.append(f"{name} n'a pas la bonne valeur.")
 
class MathadataValidateVariables(MathadataValidate):
    def __init__(self, name_and_values=None, get_names_and_values=None, *args, **kwargs):
        if name_and_values is None and get_names_and_values is None:
            raise ValueError("You must provide either name_and_values or get_names_and_values")
        
        self.name_and_values = name_and_values
        self.get_name_and_values = get_names_and_values
        super().__init__(*args, **kwargs)

    def check_undefined_variables(self, errors):
        undefined_variables = [name for name in self.name_and_values if not has_variable(name) or get_variable(name) is Ellipsis]
        undefined_variables_str = ", ".join(undefined_variables)

        if len(undefined_variables) == 1:
            errors.append(f"As-tu bien remplacé les ... ? La variable {undefined_variables_str} n'a pas été définie.")
        elif len(undefined_variables) > 1:
            errors.append(f"As-tu bien remplacé les ... ? Les variables {undefined_variables_str} n'ont pas été définies.")

        return undefined_variables
    
    def check_variables(self, errors):
        for name in self.name_and_values:
            val = get_variable(name)
            expected = self.name_and_values[name]
            check_variable(errors, name, val, expected)

    def validate(self, errors, answers):
        if self.get_name_and_values is not None:
            self.name_and_values = self.get_name_and_values()
            
        for name in self.name_and_values:
            if not has_variable(name):
                answers[name] = None
            else:
                answers[name] = get_variable(name)

        undefined_variables = self.check_undefined_variables(errors)
        if len(undefined_variables) == 0:
            if self.function_validation is not None:
                return self.function_validation(errors, answers)
            else:
                self.check_variables(errors)

        return len(errors) == 0
    
    def get_variables_str(self):
        res = []
        for name in self.name_and_values:
            expected = self.name_and_values[name]
            if expected is None:
                continue
            elif isinstance(expected, dict):
                expected = expected['value']

            if isinstance(expected, dict):
                if 'is' in expected:
                    solution = expected['is']
                elif 'min' in expected and 'max' in expected:
                    solution = f"entre {expected['min']} et {expected['max']}"
                elif 'min' in expected:
                    solution = f"supérieur ou égal à {expected['min']}"
                elif 'max' in expected:
                    solution = f"inférieur ou égal à {expected['max']}"
                elif 'in' in expected:
                    solution = f"l'une de ces valeurs : {', '.join(expected['in'])}"
                else:
                    raise ValueError(f"Malformed validation class")
            else:
                solution = expected
            res.append(f"{name} : {solution}")
        return res


class MathadataValidateFunction(MathadataValidate):
    def __init__(self, function_name, test_set=[], expected=[], expected_function=None, success="Bravo, ta fonction est correcte", *args, **kwargs):
        super().__init__(success=success, *args, **kwargs)
        self.function_name = function_name
        self.test_set = test_set
        self.expected = expected
        self.expected_function = expected_function
        
    def validate(self, errors, answers):
        if not has_variable(self.function_name):
            errors.append(f"La fonction {self.function_name} n'est pas définie. Tu dois avoir une ligne de code qui commence par 'def {self.function_name}(...):'")
            return False
        
        func = get_variable(self.function_name)

        if callable(self.test_set):
            test_set = self.test_set()
        else:
            test_set = self.test_set
        
        if self.expected_function is not None:
            expected = []
            for i in range(len(test_set)):
                if not isinstance(test_set[i], tuple):
                    expected.append(self.expected_function(test_set[i]))
                else:
                    expected.append(self.expected_function(*test_set[i]))
        elif callable(self.expected):
            expected = self.expected()
        else:
            expected = self.expected
        
        for i in range(len(test_set)):
            try:
                if not isinstance(test_set[i], tuple):
                    res = func(test_set[i])
                else:
                    res = func(*test_set[i])
                if not compare(res, expected[i]):
                    errors.append("Pour les paramètres suivant :")
                    errors.append(f"{test_set[i]}")
                    errors.append(f"Ta fonction a renvoyé {res} au lieu de {expected[i]}.")
                    return False
            except Exception as e:
                errors.append(f"Ta fonction a fait une erreur pendant le test :")
                if debug:
                    raise e
                errors.append(str(e))
                return False

        return True
        

class ValidateScoreThresold(MathadataValidate):

    def __init__(self, *args, **kwargs):
        super().__init__(success="")

    def validate(self, errors, answers):

        
        if not has_variable('classification'):
            print_error("La fonction classification n'a pas été définie.")
            return False
        if not has_variable('caracteristique'):
            print_error("La fonction caracteristique n'a pas été définie.")
            return False
        if not has_variable('t'):
            print_error("Le seuil t n'a pas été défini.")
            return False
    
        caracteristique = get_variable('caracteristique')
        classification = get_variable('classification')
        t = get_variable('t')
        def algorithme(d):
            x = caracteristique(d)
            return classification(x, t)
        
        if not has_variable('erreur_10'):
            errors.append("La variable erreur_10 n'a pas été définie.")
            return False
        
        e_train_10 = get_variable('erreur_10')
        nb_errors = np.count_nonzero(np.array([algorithme(d) for d in challenge.d_train[0:10]]) != challenge.r_train[0:10]) 
        if nb_errors * 10 == e_train_10:
            print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname_plural']}, soit {e_train_10}% d'erreur")
            return True
        else:
            if e_train_10 == nb_errors:
                errors.append(f"Ce n'est pas la bonne valeur. Pour passer du nombre d'erreurs sur 10 {challenge.strings['dataname_plural']} au pourcentage, tu dois multiplier par 10 !")
            elif e_train_10 < 0 or e_train_10 > 100:
                errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
            else:
                errors.append("Ce n'est pas la bonne valeur. Compare ta liste de prédictions avec les vraies valeurs pour trouver le pourcentage d'erreur.")
            return False

if sequence:

    @validationclass
    class MathadataValidateVariables(MathadataValidateVariables):
        pass

    @validationclass
    class MathadataValidateFunction(MathadataValidateFunction):
        pass

    @validationclass
    class ValidateScoreThresold(ValidateScoreThresold):
        pass

# Instances de validation communes

def validation_func_score_fixed(errors, answers):     
    score_10 = answers['erreur_10']
    algorithme = get_algorithme_func(error="La fonction algorithme n'existe plus. Revenez en arrière et réexecutez la cellule avec 'def algorithme(d): ...'")
    
    estimations = [algorithme(d) for d in challenge.d_train[0:10]]
    nb_errors = np.sum(estimations != challenge.r_train[0:10])
    
    if not isinstance(score_10, int):
        if score_10 == float(nb_errors*10):
            print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname_plural']}, soit {score_10}% d'erreur")
            return True
        else:
            errors.append("La variable erreur_10 doit être un entier. Attention, écrivez uniquement le nombre sans le %.")
            return False

    if score_10 == nb_errors * 10:
        print(f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname_plural']}, soit {score_10}% d'erreur")
        return True
    
    if score_10 == nb_errors:
        errors.append("Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreur et non le pourcentage d'erreur.")
    elif score_10 < 0 or score_10 > 100:
        errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
    else:
        errors.append(f"Ce n'est pas la bonne valeur. Note la réponse donnée par ton algorithme pour chaque {challenge.strings['dataname']} et compte le nombre de différences avec les bonnes réponses.")
    
    return False

exercice_droite_carac_ok = False
def set_exercice_droite_carac_ok():
    global exercice_droite_carac_ok
    exercice_droite_carac_ok = True

def validate_exercice_droite_carac(errors, answers):
    if exercice_droite_carac_ok:
        return True
    else:
        errors.append("Réponds d'abord à la question ci-dessus en plaçant les points sur la droite.")
        return False

def get_validate_seuil():
    data = challenge.d_train[0:10]
    caracteristique = get_variable('caracteristique')
    caracs = [caracteristique(d) for d in data]
    return {
        't': {
            'value': {
                'min': min(caracs),
                'max': max(caracs),
            }
        }
    }

def validate_optimized_threshold(errors, answers):
    t = answers['t']
    
best_t = None
def get_validate_seuil_optimized():
    return {
        't': best_t,
    }



validation_execution_algo_fixe = MathadataValidate(success="")
validation_execution = MathadataValidate(success="")
validation_execution_calcul_score = MathadataValidate(success="")
validation_question_score_fixe = MathadataValidateVariables({
    'erreur_10': None
}, function_validation=validation_func_score_fixed, success="")
validation_score_fixe = MathadataValidate(success="")

validation_execution_affichage_classif = MathadataValidate(success="")
validation_exercice_droite_carac = MathadataValidate(function_validation=validate_exercice_droite_carac)
validation_question_ordre_caracteristique = MathadataValidateVariables(get_names_and_values=lambda: {
    'r_petite_caracteristique': {
        'value': challenge.r_petite_caracteristique,
        'errors': [
            {
                'value': {
                    'in': challenge.classes
                },
                'else': f"Tu dois répondre {challenge.classes[0]} ou {challenge.classes[1]}"
            },
        ]
    },
    'r_grande_caracteristique': {
        'value': challenge.r_grande_caracteristique,
        'errors': [
            {
                'value': {
                    'in': challenge.classes
                },
                'else': f"Tu dois répondre {challenge.classes[0]} ou {challenge.classes[1]}"
            },
        ]
    }
})
    
validation_question_seuil = MathadataValidateVariables(
    get_names_and_values=get_validate_seuil,
    success="Ton seuil est correct ! Il n'est pas forcément optimal, on verra dans la suite comment l'optimiser."
)
validation_execution_classif = MathadataValidate(success="")
validation_execution_affichage_score = MathadataValidate(success="")
validation_question_score_seuil = ValidateScoreThresold()
validation_score_carac = MathadataValidate(success="")
validation_execution_graph_erreur = MathadataValidate(success="")
validation_question_seuil_optimise = MathadataValidateVariables(
    get_names_and_values=get_validate_seuil_optimized,
    success="Bravo, c'est la bonne réponse ! Ton seuil est maintenant optimal"
)

validation_question_nombre = MathadataValidateVariables(get_names_and_values=lambda: {
    f'nombre_{challenge.strings["classes"][1]}': {
        'value': np.sum(challenge.r_train[0:10] == challenge.classes[1]),
        'errors': [
            {
                'value': np.sum(challenge.r_train[0:10] == challenge.classes[1]),
                'else' :"As-tu bien regardé les 10 premières lignes du tableau ? "
            },
        ]
    }
    })

validation_question_nombre_total =MathadataValidateVariables(get_names_and_values=lambda: {
    f'nombre_total_{challenge.strings["dataname_plural"]}': {
        'value': len(challenge.d_train),
        'errors': [
            {'value':len(challenge.d_train),
             'else': "Ce n'est pas la bonne réponse. Le tableau est intéractif."
            },
        ]
    }
    }, on_success=lambda answers: set_step(0))

# Defini pour éviter l'erreur python par defaut si l'élève mets reponse = chat
chat = 0
cat = 0

validation_question_faineant = MathadataValidateVariables(get_names_and_values=lambda: {
    'Reponse_Image_A': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Image_A n'a pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    },
    'Reponse_Image_B': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Image_B n'as pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    },
    'Reponse_Image_C': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Image_C n'as pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    }
}, success="", on_success=lambda answers: print(f"Bravo, {ac_fem('quelle', 'quel')} que soit {data('le')} l'algorithme fainéant répond toujours {challenge.classes[0]} !"))

validation_custom = MathadataValidate(success="Bravo, c'est un excellent taux d'erreur ! Tu peux passer à la suite pour essayer de faire encore mieux.")
validation_code_eleve = MathadataValidate(success="Bravo, c'est un excellent taxu d'erreur ! Tu as fini ce notebook.")

### --- Config matplotlib ---

# Capytale
# - largeur max du notebook : 1140px
# - unité figsize = dpi px = 100px -> 9.6 max pour que ça tienne sans scroll


# default
plt.rcParams['figure.dpi'] = 100

figw_full = 960 / plt.rcParams['figure.dpi']
plt.rcParams["figure.figsize"] = [figw_full, figw_full * 3 / 4] # par défaut toute la largeur, aspect ratio 4/3


### --- Common util functions ---

def print_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def has_variable(name):
    return hasattr(__main__, name) and get_variable(name) is not None and get_variable(name) is not Ellipsis

def get_variable(name):
    return getattr(__main__, name)

### --- Print utils with challenge variables ---

def data(article, plural=False, uppercase=False):
    res = ""
    if article == "de":
        if challenge.strings['contraction']:
            res += "d'"
        else:
            res += "de "
    elif article == "un":
        if challenge.strings['feminin']:
            res += "une "
        else:
            res += "un "
    elif article == "du":
        if challenge.strings['contraction']:
            res += "de l'"
        elif challenge.strings['feminin']:
            res += "de la "
        else:
            res += "du "
    elif article == "le":
        if challenge.strings['contraction']:
            res += "l'"
        elif challenge.strings['feminin']:
            res += "la "
        else:
            res += "le "
    
    if plural:
        res += challenge.strings['dataname_plural']
    else:
        res += challenge.strings['dataname']

    if uppercase:
        res = res.capitalize()
        
    return res

def ac_fem(fem, mas):
    if challenge.strings['feminin']:
        return fem
    else:
        return mas
    
def e_fem():
    return ac_fem("e", "")

### ONGOING ###

exercises = {}
def create_exercise(div_id, config):
    exercises[div_id] = config
    config_json = json.dumps(config)
    js = f"""
        const config = `{config_json}`;
        window.mathadata.create_exercise('{div_id}', JSON.parse(config));
    """
    run_js(js)

def submit_exercise(div_id, params_json):
    if div_id not in exercises:
        print_error(f"Exercice {div_id} non trouvé")
        return
 
    params = json.loads(params_json)
    answers = params['answers']
    config = exercises[div_id]['questions'][params['step']]

    errors = []
    for answer in answers:
        expected = config['answers'][answer['name']]
        if expected['type'] == 'number':
            val = float(answer['value'])
        else:
            val = answer['value']

        check_variable(errors, answer['name'], val, expected)

    is_correct = len(errors) == 0

    if is_correct:
        MathadataValidate()() # Validate import cell -> doesn't work
    
    res = {
        'is_correct': is_correct,
        'errors': errors
    }

    print(res)
    
    return json.dumps(res)

def generer_exercices_droites():
    questions = []
    min_m = -10
    max_m = 10
    min_p = -8
    max_p = 8

    for i in range(0, 5): # même m ou même p
        rand = np.random.randint(0, 2)
        if rand == 0: # même m
            m1 = np.random.randint(-10, 10) / 2
            m2 = m1
            p1 = np.random.randint(-8, 8)
            p2 = p1
            while p2 == p1:
                p2 = np.random.randint(-8, 8)
        else: # même p
            p1 = np.random.randint(-8, 8)
            p2 = p1
            m1 = np.random.randint(-10, 10) / 2
            m2 = m1
            while m2 == m1:
                m2 = np.random.randint(-10, 10) / 2
        
        questions.append({
            'question': 'Ces deux droites ont-elles le même paramètre m ou le même paramètre p ?',
            'chart': {
                'type': 'line',
                'data': {
                    'datasets': [
                        {
                            'data': [{'x': -10, 'y': m1 * -10 + p1}, {'x': 10, 'y': m1 * 10 + p1}],
                            'label': 'd1',
                        },
                        {
                            'data': [{'x': -10, 'y': m2 * -10 + p2}, {'x': 10, 'y': m2 * 10 + p2}],
                            'label': 'd2',
                        },
                    ]
                },
            },
            'answers': {
                'param': {
                    'type' : 'radio',
                    'choices': ['m', 'p'],
                    'value': 'm' if rand == 0 else 'p',
                }
            }
        })

    for i in range(0, 3): # quelle droite a le plus grand coefficient directeur m ?
        m1 = np.random.randint(-10, 10) / 2
        m2 = m1
        while m2 == m1:
            m2 = np.random.randint(-10, 10) / 2
        p1 = np.random.randint(-8, 8)
        p2 = np.random.randint(-8, 8)
        questions.append({
            'question': 'Quelle droite a le plus grand coefficient directeur m ?',
            'chart': {
                'type': 'line',
                'data': {
                    'datasets': [
                        {
                            'data': [{'x': -10, 'y': m1 * -10 + p1}, {'x': 10, 'y': m1 * 10 + p1}],
                            'label': 'd1',
                        },
                        {
                            'data': [{'x': -10, 'y': m2 * -10 + p2}, {'x': 10, 'y': m2 * 10 + p2}],
                            'label': 'd2',
                        },
                    ]
                },
            },
            'answers': {
                'droite': {
                    'type': 'radio',
                    'choices': ['d1', 'd2'],
                    'value': 'd1' if m1 > m2 else 'd2',
                }
            }
        })

    for i in range(0, 2): # donnez les paramètres de la droite
        m = np.random.randint(-10, 10) / 2.0
        p = np.random.randint(-8, 8)
        
        x_9 = int((9 - p) / m) if m != 0 else 9
        x_m9 = int((-9 - p) / m) if m != 0 else -9
        min_x = max(-9, min(x_9, x_m9))
        max_x = min(9, max(x_9, x_m9))
        
        x1 = np.random.randint(min_x, max_x)
        x2 = x1
        while x2 == x1:
            x2 = np.random.randint(min_x, max_x)
            
        questions.append({
            'question': 'Donnez les paramètres m et p de la droite passant par les points A et B.',
            'chart': {
                'data': {
                    'labels': ['A', 'B'],
                    'datasets': [
                        {
                            'type': 'scatter',
                            'data': [{'x': x1, 'y': m * x1 + p}, {'x': x2, 'y': m * x2 + p}],
                        },
                    ]
                },
            },
            'answers': {
                'm': {
                    'type': 'number',
                    'step': 0.5,
                    'value': m,
                },
                'p': {
                    'type': 'number',
                    'value': p,
                }
            }
        })
        

    create_exercise('exercice', {
        'chart': {
            'type': 'line',
            'options': {
                'scales': {
                    'x': {
                        'type': 'linear',
                        'position': 'center',
                        'min': -10,
                        'max': 10,
                        'ticks': {
                            'stepSize': 1,
                        }
                    },
                    'y': {
                        'type': 'linear',
                        'position': 'center',
                        'min': -10,
                        'max': 10,
                        'ticks': {
                            'stepSize': 1,
                        }
                    }
                }
            }
        },
        'questions': questions,
    })

def pass_breakpoint():
    Validate()()

run_js('''
/*
window.mathadata.observed_ids = {}
window.mathadata.add_observer = function(id, func) {
    window.mathadata.observed_ids[id] = func
}

const observer = new MutationObserver((mutationsList, observer) => {
    for (const mutation of mutationsList) {
        for (const node of mutation.addedNodes) {
            if (node.nodeType === 1 && node.id && node.id in window.mathadata.observed_ids) {
                console.log(`Element with ID #${node.id} appeared!`);
                window.mathadata.observed_ids[node.id]()
            }
        }
    }
});

observer.observe(document.body, { childList: true, subtree: true });
*/

window.mathadata.create_qcm = function(id, config) {
    if (typeof config === 'string') {
        config = JSON.parse(config);
    }
    
    //mathadata.add_observer(id, () => {
        const container = document.getElementById(id);
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        
        container.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 20px; text-align: center;">${config.question}</div>
            <div style="display: flex; gap: 25px; flex-wrap: wrap; justify-content: center;">
                ${config.choices.map(choice => {
                    return `
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <input type="radio" name="${id}-qcm" value="${choice}" id="${id}-qcm-${choice}">
                            <label for="${id}-qcm-${choice}">${choice}</label>
                        </div>
                    `
                }).join('\\n')}
            </div>
            <button class="${id}-qcm-validate-button" style="margin: auto; margin-top: 10px; margin-bottom: 10px; padding: 5px 10px; border-radius: 5px; background-color: #007bff; color: white; border: none; cursor: pointer;">Valider</button>
            <div class="${id}-qcm-status-text" style="margin-top: 10px; margin-bottom: 10px; text-align: center;"></div>
        `
        
        const validateButton = container.querySelector('.' + id + '-qcm-validate-button');
        validateButton.addEventListener('click', () => {
            const selectedValue = container.querySelector('input[name="' + id + '-qcm"]:checked')?.value;
            const statusText = container.querySelector('.' + id + '-qcm-status-text');

            if (selectedValue === undefined) {
                statusText.textContent = "Veuillez sélectionner une réponse.";
                statusText.style.color = 'red';
            } else if (selectedValue === config.answer) {
                statusText.textContent = 'Bonne réponse !';
                statusText.style.color = 'green';
                mathadata.pass_breakpoint();
            } else {
                statusText.textContent = "Ce n'est pas la bonne réponse. Essaie encore !";
                statusText.style.color = 'red';
            }
        });
    //});
}
''')

def create_qcm(div_id, config):
    display(HTML(f'''
        <div id="{div_id}"></div>
    '''))
    run_js(f'''
        setTimeout(() => window.mathadata.create_qcm('{div_id}', '{json.dumps(config)}'), 500)
    ''')

def qcm_test():
    create_qcm('qcm', {
        'question': 'Quel est le plus grand nombre ?',
        'choices': ['1', '2', '3'],
        'answer': '3',
    })
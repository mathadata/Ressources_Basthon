import os
import sys
import matplotlib.pyplot as plt
import requests
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
subdirectories = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if
                  os.path.isdir(os.path.join(base_directory, d))]

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
        self.classes = [0, 1]
        self.r_petite_caracteristique = 0
        self.r_grande_caracteristique = 1
        self.strings = {
            'dataname': {
                'nom': "donnée",
                'pluriel': "données",
                'feminin': True,
                'contraction': False,
            },
            'classes': [
                {
                    'nom': "classe1",
                    'pluriel': "classe1",
                    'feminin': True,
                    'contraction': False,
                    'nom_alt': "donnée de classe1",
                    'pluriel_alt': "données de classe1",
                    'feminin_alt': True,
                },
                {
                    'nom': "classe2",
                    'pluriel': "classe2",
                    'feminin': True,
                    'contraction': False,
                    'nom_alt': "donnée de classe2",
                    'pluriel_alt': "données de classe2",
                    'feminin_alt': True,
                }
            ],
            'train_size': "1000"
        }

    def affichage_banque(self, carac=None, mode=1, showPredictions=False, estimations=None):
        id = uuid.uuid4().hex

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

        run_js(
            f"mathadata.add_observer('{id}', () => window.mathadata.setup_test_bank('{id}', '{json.dumps(params, cls=NpEncoder)}'))");

        display(HTML(f'''
            <div id="{id}" style="display: flex; height: 500px; gap: 2rem;">
                <div id="{id}-bank" class="ag-theme-quartz" style="flex: 1; min-width: 300px;"></div>
                <div style="flex: 1; display: flex; justify-content: center; align-items: center;" id="{id}-selected"></div>
            </div>
        '''))

    def get_data_internal(self, index=None, dataClass=None, random=False):
        d = None
        if index is not None:
            if dataClass == None:
                d = self.d_train[index]
                label = self.r_train[index]
            else:
                class_index = self.classes.index(dataClass)
                d = self.d_train_by_class[class_index][index]
                label = dataClass
        if d is None:
            if random:
                index = np.random.randint(0, len(self.d_train))
                d = self.d_train[index]
                label = self.r_train[index]
            else:
                d = self.d
                # On suppose que d est de la première classe
                label = self.classes[0]
        
        return (d, label)
    
    def get_data(self, *args, **kwargs):
        (d, label) = self.get_data_internal(*args, **kwargs)
        return json.dumps(d, cls=NpEncoder)

    def get_data_and_label(self, *args, **kwargs):
        if current_algo is None:
            return {
                'data': None,
                'label': None
            }

        (d, label) = self.get_data_internal(*args, **kwargs)
        r_est = current_algo(d)
        return json.dumps({
            'data': d,
            'label': label,
            'estimation': r_est,
        }, cls=NpEncoder)

    def import_js_scripts(self):
        pass

    def affichage_html(self, d=None):
        id = uuid.uuid4().hex

        if d is None:
            d = self.d

        run_js(
            f"mathadata.add_observer('{id}', () => window.mathadata.affichage('{id}', {json.dumps(d, cls=NpEncoder)}))")

        display(HTML(f'<div id="{id}" style="min-height: 300px; min-width: 300px;"></div>'))


def init_challenge(challenge_instance):
    global challenge
    challenge = challenge_instance

    challenge.d_train_by_class = [challenge.d_train[challenge.r_train == k] for k in challenge.classes]

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


def get_data_and_label(*args, **kwargs):
    return challenge.get_data_and_label(*args, **kwargs)


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
                'drag-data-plugin': 'https://cdn.jsdelivr.net/npm/chartjs-plugin-dragdata@latest/dist/chartjs-plugin-dragdata.min',
                'confetti': 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min',
                'jsxgraph': 'https://cdn.jsdelivr.net/npm/jsxgraph@1.12.0/distrib/jsxgraphcore.min',
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

        require(['ag-grid-community', 'chartjs','drag-data-plugin', 'confetti', 'jsxgraph'], function(agGrid, Chart, DragData, Confetti, JXG) {
            window.agGrid = agGrid;
            //window.JXG = JXG;
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

    Validate()()  # Validate import cell


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
                    body.append(key, Blob.new([files[key].getvalue()], {'type': 'text/csv'}))
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
    # analytics_endpoint = "https://dev.mathadata.fr/api/notebooks"

analytics_endpoint = mathadata_endpoint + "/notebooks"

### Utilitaires requêtes HTTP ###

if sequence:
    mathadata_url = "https://mathadata.fr"
else:
    mathadata_url = "https://dev.mathadata.fr"
    # mathadata_url = "http://localhost:3000"

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
    c_train_par_population = [c_train[r_train == k] for k in classes]
    return c_train_par_population


def affichage_dix_caracteristique(predictions=False):
    challenge.affichage_dix()
    df = pd.DataFrame()
    df['$r$ (classe)'] = challenge.r_train[0:10]
    caracteristique = get_variable('caracteristique')
    df['$k$ (caracteristique)'] = [caracteristique(d) for d in challenge.d_train[0:10]]
    if predictions:
        df[r'$\hat{r}$ (prediction)'] = '?'
    df.index += 1

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
            print_error(
                "Vous avez remplacé autre chose que les ... . Revenez en arrière avec le raccourci clavier Ctrl+Z pour annuler vos modifications.")
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
    return erreur_train_optim(calcul_caracteristiques(d_train, caracteristique), r_train, x, classification)


# Calculer l'estimation et l'erreur a partir du tableau de caractéristique des images :
def erreur_train_optim(k_d_train, r_train, x, classification):
    # Vectorize the classification function if it's not already vectorized
    r_train_est = np.vectorize(classification)(k_d_train, x)

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
    t_min = int(k_d_train_sliced.min()) - 1
    t_max = int(k_d_train_sliced.max()) + 1

    # Vectorize the erreur_train function to apply it over an array of t_values
    vec_erreur_train = np.vectorize(
        lambda t: 100 * erreur_train_optim(k_d_train_sliced, r_train_sliced, t, func_classif))

    # Calculer pas_t pour avoir environ 50 points:
    pas_t = int((t_max - t_min) / 50) + 1
    # Create a range of t values using numpy's arange function
    t_values = np.arange(t_min, t_max, pas_t)

    # Apply the vectorized function to all t_values
    scores_array = vec_erreur_train(t_values)

    return (t_values, scores_array)


def calculer_score(algorithme, cb=None, caracteristique=None, banque=True, animation=True):
    try:
        r_prediction_train = get_estimations(challenge.d_train, algorithme=algorithme)
        score = np.mean(r_prediction_train != challenge.r_train)

        # Lancer l'animation de classification si demandée
        if animation:
            # global registered_calculer_score_cb
            # registered_calculer_score_cb = cb
            animation_classification_score(r_prediction_train)
        else:
            print("Calcul du pourcentage d'erreur en cours...")
            if banque:
                affichage_banque(carac=caracteristique, showPredictions=True, estimations=r_prediction_train)
            set_score(score)
            print(f"Pourcentage d'erreur : {score * 100:.2f}%")

        if cb is not None:
            cb(score)

    except Exception as e:
        print_error("Il y a eu un problème lors du calcul de l'erreur. Vérifie ta réponse.")
        if debug:
            raise(e)


current_algo = None


def run_algorithme(d_json):
    if current_algo is None:
        return None
    d = json.loads(d_json)
    if type(d) == list:
        d = np.array(d)
    res = current_algo(d)
    return json.dumps({
        'result': res
    }, cls=NpEncoder)

def test_algorithme(algorithme=None):
    """Test l'algorithme donnée par donnée avec l'interface animation calcul score"""
    global current_algo

    if algorithme is None:
        if current_algo is None:
            print_error("L'algorithme n'est pas encore complet et prêt à être utilisé.")
            return
        else:
            algorithme = current_algo
    else:
        current_algo = algorithme

    id = uuid.uuid4().hex

    run_js(f"mathadata.add_observer('{id}', () => window.mathadata.test_algorithme('{id}'))")

    # Container HTML
    display(HTML(f'''
    <div id="{id}">
        <div style="display: flex; justify-content: center; align-items: center; gap: 1rem; margin: 2rem 0;">
            <button id="{id}-button-random" class="mathadata-button mathadata-button--primary" style="flex: 1; max-width: 300px;">
                Tester avec {data('un', alt=True)} aléatoire
            </button>
            <button id="{id}-button-generated" class="mathadata-button mathadata-button--secondary" style="flex: 1; max-width: 300px;">
                Tester avec votre {data('', alt=True)}
            </button>
        </div>

        <div id="{id}-animation-container" class="animation-container"></div>

        <!-- Modal pour la génération de données -->
        <div id="{id}-modal" class="mathadata-modal-overlay" style="display: none;">
            <div class="mathadata-modal">
                <div class="mathadata-modal-header">
                    <h3 class="mathadata-modal-title">Générer votre {data('', alt=True)}</h3>
                    <button id="{id}-modal-close" class="mathadata-modal-close">×</button>
                </div>
                <div class="mathadata-modal-body">
                    <div id="{id}-data-gen"></div>
                </div>
                <div class="mathadata-modal-footer">
                    <button id="{id}-button-validate" class="mathadata-button mathadata-button--success">
                        ✓ Valider et tester
                    </button>
                </div>
            </div>
        </div>
    </div>
    '''))

def test_algorithme_faineant():
    """Test l'algorithme paresseux avec l'interface animation calcul score"""
    test_algorithme(algorithme=algo_faineant)

def test_algorithme_ref():
    test_algorithme(algorithme=algo_carac)


def animation_classification_score(estimations):
    """Lance l'animation de classification pour le calcul de score"""
    id = uuid.uuid4().hex

    params = {
        'labels': challenge.r_train,
        'estimations': estimations,
    }

    display(HTML(f'''
        <div id="{id}" style="margin: 20px 0;">
            <h3 style="text-align: center; margin-bottom: 15px;">
                Calcul du Pourcentage d'Erreur...
            </h3>
            <div id="{id}-animation-container" class="animation-container"></div>
        </div>

        <script>
            window.mathadata.animateClassification(
                '{id}-animation-container',
                '{json.dumps(params, cls=NpEncoder)}',
            );
        </script>
    '''))


### --- Config matplotlib ---

# Capytale
# - largeur max du notebook : 1140px
# - unité figsize = dpi px = 100px -> 9.6 max pour que ça tienne sans scroll


# default
plt.rcParams['figure.dpi'] = 100

figw_full = 960 / plt.rcParams['figure.dpi']
plt.rcParams["figure.figsize"] = [figw_full, figw_full * 3 / 4]  # par défaut toute la largeur, aspect ratio 4/3


### --- Common util functions ---

def print_error(*args, **kwargs):
    for msg in args:
        pretty_print_error(msg)


def has_variable(name):
    return hasattr(__main__, name) and get_variable(name) is not None and get_variable(name) is not Ellipsis


def get_variable(name):
    return getattr(__main__, name)


### --- Print utils with challenge variables ---

def nom(name_dict, article=None, plural=False, alt=False, court=False, uppercase=False):
    """
    Generic function that returns a French noun with optional article.

    Parameters:
        - name_dict: dictionary with noun information (see structure below)
        - article: one of None, 'de', 'du', 'le', 'un'
        - plural: whether to use the plural form
        - uppercase: whether to capitalize the first letter
        - alt: whether to use alternate forms (nom_alt, genre_alt, contraction_alt)
        - court: whether to use short forms (nom_court, pluriel_court)

    name_dict structure:
        {
            "nom": "singular form",
            "pluriel": "plural form",  # optional, defaults to nom + "s"
            "feminin": True/False, # optional, defaults to True
            "contraction": True/False,  # optional, defaults to False

            # Optional variants:
            "nom_court": "short singular",
            "pluriel_court": "short plural",
            "nom_alt": "alternate singular",
            "pluriel_alt": "alternate plural",
            "feminin_alt": True/False,
            "contraction_alt": True/False,
        }
    """
    if not isinstance(name_dict, dict):
        error = f"ERREUR: Appel invalide à la fonction 'nom' : {name_dict}"
        print(error)
        return error

    # Determine gender, possibly from alt entries
    if alt and "feminin_alt" in name_dict:
        feminine = name_dict["feminin_alt"]
    else:
        feminine = name_dict.get("feminin", False)

    # Determine contraction setting, preferring alt if requested
    if alt and "contraction_alt" in name_dict:
        contraction = name_dict.get("contraction_alt", False)
    else:
        contraction = name_dict.get("contraction", False)

    # Select full forms, preferring alt if requested
    base_singular = name_dict.get("nom_alt") if alt and "nom_alt" in name_dict else name_dict.get("nom", "")
    base_plural = name_dict.get("pluriel_alt") if alt and "pluriel_alt" in name_dict else name_dict.get("pluriel", (
        base_singular + "s" if base_singular else ""))

    # Select short or full
    if court:
        singular = name_dict.get("nom_court", base_singular)
        plural_form = name_dict.get("pluriel_court", base_plural)
    else:
        singular = base_singular
        plural_form = base_plural

    # Choose singular vs plural
    word = plural_form if plural else singular

    # Prepend article if needed
    if article:
        if article == "de":
            art = "d'" if contraction else "de "
        elif article == "un":
            art = "une " if feminine else "un "
        elif article == "du":
            if contraction:
                art = "de l'"
            else:
                art = "de la " if feminine else "du "
        elif article == "le":
            if contraction:
                art = "l'"
            else:
                art = "la " if feminine else "le "
        else:
            art = ""
        word = art + word

    # Apply capitalization if requested
    if uppercase and word:
        word = word[0].upper() + word[1:]

    return word


def data(*args, **kwargs):
    """
    Backward compatibility function that uses global variables.
    Uses the generic nom() function internally.
    """

    return nom(challenge.strings['dataname'], *args, **kwargs)


def classe(index, *args, **kwargs):
    """
    Returns a class name with optional article.

    Parameters:
        - index: index in the global 'classes' array (0-based) OR the class value itself
        - article: one of None, 'de', 'du', 'le', 'un'
        - plural: whether to use the plural form
        - uppercase: whether to capitalize the first letter
        - alt: whether to use alternate forms
        - court: whether to use short forms

    Global variables expected:
        - classes: list of class dictionaries following the nom() structure

    Example:
        classes = [
            {
                "nom": "2",
                "pluriel": "2",
                "genre": "m",
                "nom_alt": "deux",
                "pluriel_alt": "deux",
                "genre_alt": "m"
            },
            {
                "nom": "7",
                "pluriel": "7",
                "genre": "m",
                "nom_alt": "sept",
                "pluriel_alt": "sept",
                "genre_alt": "m"
            }
        ]
    """

    return nom(challenge.strings['classes'][index], *args, **kwargs)


def ac_fem(fem, mas):
    """
    Permet d'accorder en fonction du genre de la dénomination de la donnée
    fem : version si féminin
    mas : version si masculin
    """
    if challenge.strings['dataname'].get('feminin', False):
        return fem
    else:
        return mas


def e_fem():
    return ac_fem("e", "")


# Fonctions trame notebook générique

def algo_faineant(d):
    return challenge.classes[0]


def algo_carac(d):
    caracteristique = challenge.caracteristique
    t = get_variable('t')
    classification = get_variable('classification')
    x = caracteristique(d)
    return classification(x, t)


def algo_carac_custom(d):
    caracteristique = challenge.caracteristique_custom
    t = get_variable('t')
    classification = get_variable('classification')
    x = caracteristique(d)
    return classification(x, t)


def calculer_score_etape_1(animation=True):
    def cb(score):
        validation_score_fixe()

    calculer_score(algo_faineant, cb=cb, animation=animation)

def calculer_score_carac():
    def cb(score):
        validation_score_carac()

    calculer_score(algo_carac, cb=cb, caracteristique=caracteristique, animation=False)


def calculer_score_custom():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable(
            'r_grande_caracteristique'):
        print_error('Remplacez tous les ... par vos paramètres.')
        return

    def cb(score):
        res = challenge.cb_custom_score(score)
        if res:
            set_step_algo_carac_custom(answers=None)
            validation_custom()

    calculer_score(algo_carac_custom, caracteristique=challenge.caracteristique_custom, cb=cb, animation=False)


def calculer_score_code_eleve():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable(
            'r_grande_caracteristique'):
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
        if score < 0.04 or (has_variable('superuser') and get_variable('superuser') == True):
            validation_code_eleve()
        else:
            print_error("Essaie de trouver une zone qui fait moins de 5% d'erreur.")

    calculer_score(algorithme, caracteristique=get_variable('caracteristique'))


def calculer_score_code_free():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable(
            'r_grande_caracteristique'):
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

    validation_code_free()

    calculer_score(algorithme, caracteristique=get_variable('caracteristique'))


def calculer_score_code_etendue():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable(
            'r_grande_caracteristique'):
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

    validation_code_etendue()

    calculer_score(algorithme, caracteristique=get_variable('caracteristique'))


def calculer_score_code_moyenne_ligne():
    if not has_variable('t') or not has_variable('r_petite_caracteristique') or not has_variable(
            'r_grande_caracteristique'):
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

    validation_code_moyenne_ligne()

    calculer_score(algorithme, caracteristique=get_variable('caracteristique'))


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
        run_js(
            f'mathadata.add_observer("{id}", () => window.mathadata.tracer_erreur("{id}", {t_values.tolist()}, {scores_array.tolist()}))')
        display(HTML(f'''
            <canvas id="{id}"/>
        '''))
    else:
        run_js(f'window.mathadata.tracer_erreur("{id}", {t_values.tolist()}, {scores_array.tolist()})')


def update_graph_erreur(id="graph_custom", func_carac=None):
    if func_carac is None:
        func_carac = challenge.caracteristique_custom

    [t_values, scores_array] = get_erreur_plot(func_carac)
    run_js(f'window.mathadata.tracer_erreur("{id}", {t_values.tolist()}, {scores_array.tolist()})')


def exercice_droite_carac():
    id = uuid.uuid4().hex

    size = 10
    set = challenge.d_train[0:size]
    c_train = compute_c_train(challenge.caracteristique, set)
    params = {
        'c_train': c_train,
        'labels': [0 if r == challenge.classes[0] else 1 for r in challenge.r_train[0:size]],
    }

    run_js(
        f"mathadata.add_observer('{id}', () => window.mathadata.exercice_droite_carac('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
        <div id="{id}">
            <canvas id="{id}-chart"></canvas>
            <p id="{id}-status"></p>
        </div>
    '''))

    
def affichage_10_droite():
    id = uuid.uuid4().hex

    size = 10
    set = challenge.d_train[0:size]
    c_train = compute_c_train(challenge.caracteristique, set)

    t = get_variable('t')

    params = {
        'c_train': c_train,
        'labels': [0 if r == challenge.classes[0] else 1 for r in challenge.r_train[0:size]],
        't': t,
        'afficherPoints': True,
    }

    run_js(
        f"mathadata.add_observer('{id}', () => window.mathadata.tracer_droite_carac('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
        <canvas id="{id}"></canvas>
    '''))


def calculer_score_seuil_optimise():
    if not validation_question_seuil_optimise():
        return

    t = get_variable('t')
    caracteristique = get_variable('caracteristique')
    classification = get_variable('classification')

    def algorithme(d):
        x = caracteristique(d)
        return classification(x, t)

    calculer_score(algorithme)


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
    run_js(f'window.mathadata.updateScore({score});')


def submit(csv_content, challenge_id=116, method=None, parameters=None, cb=None):
    if (capytale_id is None):
        return

    def internal_cb(data):
        if data is not None and isinstance(data, dict) and 'score' in data:
            set_score(data['score'])

            if cb is not None:
                cb(data['score'])
        else:
            print(
                'Il y a eu un problème lors du calcul de l\'erreur. Ce n\'est probablement pas de ta faute, réessaye dans quelques instants.')
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
/* Override ce style par défaut du notebook qui est bizarre */
.CodeMirror pre.CodeMirror-line {
    z-index: 1;
}

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
    text-align: center;
}

.sidebox-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
    flex: 1;
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

.error {
    background-color: lightred; /* Light red background */
    padding: 15px; /* Increased padding */
    border-radius: 5px;
    border-left: 5px solid #F23431; /*light-red left border */
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

.mathadata-stepbar__container {
  position: fixed;
  top: 40%;
  transform: translateY(-50%);
  right: var(--stepbar-right, 16px);
  width: 240px; /* élargi pour accueillir labels + cercles */
  display: flex;
  flex-direction: column;
  gap: 10px;
  z-index: 9999;
  align-items: center;
}
.mathadata-stepbar__backdrop {
  width: 240px; /* match container width */
  background: #fff;
  border-top-left-radius: 20px;
  border-bottom-left-radius: 20px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  align-items: stretch; /* laisser les lignes prendre toute la largeur */
  padding: 20px 0; /* increased vertical padding around items for greater height */
}
.mathadata-stepbar__items {
  display: flex;
  flex-direction: column;
  align-items: stretch; /* chaque ligne sur toute la largeur */
  gap: 14px; /* increased spacing between boxes */
  width: 100%;
}
.mathadata-stepbar__itemRow {
  display: flex;
  align-items: center;
  justify-content: space-between; 
  gap: 8px;
  width: 100%;
  padding: 0 12px; /* marge intérieure pour coller le label à gauche sans toucher les bords */
}

.mathadata-stepbar__labelHard {
  font-size: 16px;
  color: #909090;
  user-select: none;
  white-space: normal; 
  text-overflow: clip;
  text-align: left; /* aligne le texte à gauche */
  flex: 1 1 auto; /* prend toute la place restante */
  max-width: unset; /* pas de limite artificielle, géré par flex */
  line-height: 1.2;
}

.mathadata-stepbar__itemRow--active .mathadata-stepbar__labelHard {
  font-weight: bold;   /* étape active: gras */
}

.mathadata-stepbar__itemRow--unlocked .mathadata-stepbar__labelHard {
  color: black; /* étape déverrouillée */
}

.mathadata-stepbar__item {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  flex: 0 0 56px; /* fixe la largeur dans le layout flex */
  aspect-ratio: 1 / 1; /* garantit la forme circulaire */
  box-sizing: border-box;
  border-radius: 50%;
  background: #fff;
  color: #222;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  font-weight: 600;
  cursor: pointer;
  transition: transform .12s ease, box-shadow .12s ease, background .12s ease;
  user-select: none;
}
.mathadata-stepbar__item:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,0.12); }
.mathadata-stepbar__item--active {
  box-shadow: 0 0 0 2px rgba(0,0,0,0.35), 0 6px 16px rgba(0,0,0,0.12);
}

.mathadata-stepbar__item--disabled { cursor: not-allowed; }
.mathadata-stepbar__item--disabled:hover { transform: none; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }

.animation-score-data-item {
    position: absolute;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    z-index: 5;
}

.animation-container {
    position: relative;
    width: 100%;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 2px solid #333;
    border-radius: 15px;
    overflow: hidden;
    margin: 20px auto;
}

.mathadata-button {
    padding: 12px 24px;
    font-size: 16px;
    font-weight: 500;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.mathadata-button:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

.mathadata-button:active:not(:disabled) {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.mathadata-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.mathadata-button--primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.mathadata-button--primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #5568d3 0%, #633d8a 100%);
}

.mathadata-button--secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.mathadata-button--secondary:hover:not(:disabled) {
    background: linear-gradient(135deg, #d97ce0 0%, #db4a5d 100%);
}

.mathadata-button--success {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
}

.mathadata-button--success:hover:not(:disabled) {
    background: linear-gradient(135deg, #3d91e3 0%, #00d4e3 100%);
}

.mathadata-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
    animation: fadeIn 0.2s ease;
}

.mathadata-modal {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    max-width: 90vw;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: slideIn 0.3s ease;
}

.mathadata-modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #f0f0f0;
}

.mathadata-modal-title {
    font-size: 24px;
    font-weight: 600;
    color: #333;
    margin: 0;
}

.mathadata-modal-close {
    background: none;
    border: none;
    font-size: 28px;
    cursor: pointer;
    color: #999;
    transition: color 0.2s ease;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.mathadata-modal-close:hover {
    color: #333;
}

.mathadata-modal-body {
    margin-bottom: 1.5rem;
}

.mathadata-modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
"""

run_js(f"""
    let style = document.getElementById('mathadata-style');
    if (style) {{
        style.remove();
    }}
    style = document.createElement('style');
    style.id = 'mathadata-style';
    style.innerHTML = `{styles}`;
    document.head.appendChild(style);

    let jsxgraphCSS = document.getElementById('jsxgraph-css');
    if (!jsxgraphCSS) {{
        jsxgraphCSS = document.createElement('link');
        jsxgraphCSS.id = 'jsxgraph-css';
        jsxgraphCSS.rel = 'stylesheet';
        jsxgraphCSS.href = 'https://cdn.jsdelivr.net/npm/jsxgraph@1.12.0/distrib/jsxgraph.min.css';
        document.head.appendChild(jsxgraphCSS);
    }}
""")

# Fonctions utilitaires JS
run_js("""
    // Cleanup si rechargement du notebook
    if (window.mathadata) {
        window.mathadata.mutationObserver?.disconnect();
    }
       
       
    // Redirection des logs pyodide dans la console
    if (window.pyodide) {
        // rediriger stdout
        window.pyodide.setStdout({
            batched: (msg) => {
            // msg est une ligne ou fragment de sortie
                console.log("[PYODIDE STDOUT]", msg);
            }
        });
        // rediriger stderr
        window.pyodide.setStderr({
            batched: (msg) => {
                console.error("[PYODIDE STDERR]", msg);
            }
        });
    } else {
        console.warn("pyodide n'est pas accessible dans ce contexte");
    }

    function chartjs_title(context) {
        return context[0]?.dataset?.label
    }
    
    function chartjs_label(context) {
        return `abscisse: ${context.label || context.parsed?.x.toFixed(2)} , ordonnée: ${context.parsed?.y.toFixed(2)}`;
    }

    let i_exercice_droite_carac = 0
    const mathadata = {

        nom(name_dict, params) {
            const { article, plural, alt, court, uppercase } = params || {};

            if (!name_dict || typeof name_dict !== 'object') {
                const error = `ERREUR: Appel invalide à la fonction 'nom' : ${name_dict}`;
                console.error(error);
                return error;
            }

            // Determine gender, possibly from alt entries
            const feminine = (alt && name_dict.feminin_alt !== undefined)
                ? name_dict.feminin_alt
                : (name_dict.feminin || false);

            // Determine contraction setting, preferring alt if requested
            const contraction = (alt && name_dict.contraction_alt !== undefined)
                ? name_dict.contraction_alt
                : (name_dict.contraction || false);

            // Select full forms, preferring alt if requested
            const base_singular = (alt && name_dict.nom_alt) ? name_dict.nom_alt : (name_dict.nom || "");
            const base_plural = (alt && name_dict.pluriel_alt)
                ? name_dict.pluriel_alt
                : (name_dict.pluriel || (base_singular + "s"));

            // Select short or full
            let singular, plural_form;
            if (court) {
                singular = name_dict.nom_court || base_singular;
                plural_form = name_dict.pluriel_court || base_plural;
            } else {
                singular = base_singular;
                plural_form = base_plural;
            }

            // Choose singular vs plural
            let word = plural ? plural_form : singular;

            // Prepend article if needed
            if (article) {
                let art = "";
                if (article === "de") {
                    art = contraction ? "d'" : "de ";
                } else if (article === "un") {
                    art = feminine ? "une " : "un ";
                } else if (article === "du") {
                    art = contraction ? "de l'" : (feminine ? "de la " : "du ");
                } else if (article === "le") {
                    art = contraction ? "l'" : (feminine ? "la " : "le ");
                } else if (article === "aux") {
                    art = contraction ? "aux " : "aux ";
                }
                word = art + word;
            }

            // Apply capitalization if requested
            if (uppercase && word) {
                word = word.charAt(0).toUpperCase() + word.slice(1);
            }

            return word;
        },

        data(params) {
            return window.mathadata.nom(window.mathadata.challenge.strings.dataname, params);
        },

        classe(index, params) {
            const classes = window.mathadata.challenge.strings.classes;
            if (!classes || index < 0 || index >= classes.length) {
                const error = `ERREUR: Index de classe invalide : ${index}`;
                console.error(error);
                return error;
            }
            return window.mathadata.nom(classes[index], params);
        },

        ac_fem(fem, mas) {
            const dataname = window.mathadata.challenge.strings.dataname;
            if (dataname && dataname.feminin) {
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

        run_python_async(python) {
            return new Promise((resolve, reject) => {
                this.run_python(python, (result) => {
                    resolve(result)
                })
            })
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
                wrapper.style.maxHeight = 'min(60vh, 600px)';
                wrapper.style.maxWidth = '100%';
                wrapper.style.aspectRatio = config.options.aspectRatio;
                wrapper.style.display = 'flex';
                wrapper.style.justifyContent = 'center';
                wrapper.style.alignItems = 'center';
                wrapper.style.backgroundColor = 'white';
                //wrapper.style.position = 'relative';
                wrapper.style.overflow = 'visible'; // Permet au tooltip de déborder

                // move inside wrapper
                const canvas = document.getElementById(div_id)
                canvas.style.width = '100%'
                canvas.style.maxHeight = '100%'
                canvas.parentNode.insertBefore(wrapper, canvas)
                wrapper.appendChild(canvas)

                const ctx = canvas.getContext('2d');

                let retries = 0;
                function createChart() {
                    if (window.Chart === undefined) {
                        if (retries === 0) {
                            console.log('Chart.js not loaded yet, retrying...');
                            // Création d'un élément Chart vide pour éviter les erreurs
                            window.mathadata.charts[div_id] = new Proxy({}, {
                                get: function(target, prop) {
                                    console.warn(`Chart.js is not loaded yet. Attempted to access property '${prop}' on chart '${div_id}'.`);
                                    if (prop === 'update') {
                                        return function() {};
                                    } else {
                                        return undefined;
                                    }
                                }
                            })
                        }
                        
                        if (retries < 60) {
                            retries++;
                            setTimeout(createChart, 500);
                        } else {
                            console.error('Chart.js failed to load after multiple attempts.');
                        }
                    } else {
                        window.mathadata.charts[div_id] = new Chart(ctx, config);
                    }
                } 
                createChart();
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
        
        display_tip(parent, text) {
            const tipContainer = document.createElement('div')
            tipContainer.className = 'tip'
            tipContainer.innerHTML = text
            parent.appendChild(tipContainer)
        },
        
        add_tip(parentId, tip, answers) {
            const parent = document.getElementById(parentId)
            if (!parent) {
                return
            }

            if (tip.tip) {
                window.mathadata.display_tip(parent, `💡 ${tip.tip}`)
            }

            if (tip.print_solution && answers) {
                window.mathadata.display_tip(parent, `📝 Voici la solution :<br/>${answers.join('<br/>')}`)
            }

            if (tip.validate) {
                window.mathadata.display_tip(parent, `🔓 Tu peux passer cette question et continuer`)
                mathadata.pass_breakpoint()
            }
        },

        setup_tips(id, params) {
            params = JSON.parse(params)
            const {tips, first_call_time, trials, answers} = params

            const now = Date.now()
            const time_since_first_call_s = now / 1000 - first_call_time // python uses seconds, js uses milliseconds
            
            for (const tip of tips) {
                if ('trials' in tip && trials < tip.trials) {
                    continue
                }

                if (!('seconds' in tip) || tip.seconds <= time_since_first_call_s) {
                    window.mathadata.add_tip(id, tip, answers) 
                } else {
                    setTimeout(() => {
                        window.mathadata.add_tip(id, tip, answers)
                    }, (tip.seconds - time_since_first_call_s) * 1000)
                }
            }
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

                // show first row selected at start
                onFirstDataRendered({api}) {
                    // select first row
                    const firstRowNode = api.getDisplayedRowAtIndex(0);
                    if (firstRowNode) {
                        firstRowNode.setSelected(true);
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
            
            let retries = 0
            function displayGrid() {
                retries++
                if (window.agGrid === undefined) {
                    if (retries < 60) { // wait max 30s
                        setTimeout(displayGrid, 500)
                    } else {
                        console.error('ag-Grid not loaded')
                    }
                } else {
                    agGrid.createGrid(bank, config)
                }
            }
            displayGrid()
        },


        test_algorithme(id, params) {
            let getGeneratedData;
            if (mathadata.interface_data_gen !== undefined) {
                getGeneratedData = mathadata.interface_data_gen(`${id}-data-gen`);
            }

            // Setup de l'animation avec closure encapsulée
            const containerId = `${id}-animation-container`;
            const animator = mathadata.setupAnimationAlgo(containerId, {
                showTotalInCounter: false,
                keepElementCount: 20,
                showErrors: false,
            });

            const buttonRandom = document.getElementById(`${id}-button-random`);
            const buttonGenerated = document.getElementById(`${id}-button-generated`);
            const modal = document.getElementById(`${id}-modal`);
            const modalClose = document.getElementById(`${id}-modal-close`);
            const buttonValidate = document.getElementById(`${id}-button-validate`);

            // Fonctions pour gérer la modal
            const openModal = () => {
                modal.style.display = 'flex';
                document.body.style.overflow = 'hidden'; // Empêcher le scroll
            };

            const closeModal = () => {
                modal.style.display = 'none';
                document.body.style.overflow = ''; // Restaurer le scroll
            };

            // Fermer la modal au clic sur l'overlay
            modal.onclick = (e) => {
                if (e.target === modal) {
                    closeModal();
                }
            };

            // Fermer la modal au clic sur le bouton X
            if (modalClose) {
                modalClose.onclick = closeModal;
            }

            // Fermer la modal avec Escape
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && modal.style.display === 'flex') {
                    closeModal();
                }
            });

            function run_test(params) {
                const { data, label, estimation } = params;

                // Désactiver les boutons pendant l'animation
                buttonRandom.disabled = true;
                if (buttonGenerated) buttonGenerated.disabled = true;
                if (buttonValidate) buttonValidate.disabled = true;

                animator.animateData({ data, label, estimation, animDuration: 2400 })
                    .then(() => {
                        // Réactiver les boutons
                        buttonRandom.disabled = false;
                        if (buttonGenerated) buttonGenerated.disabled = false;
                        if (buttonValidate) buttonValidate.disabled = false;
                    })
                    .catch(err => {
                        console.error('Animation error:', err);
                        // Réactiver les boutons même en cas d'erreur
                        buttonRandom.disabled = false;
                        if (buttonGenerated) buttonGenerated.disabled = false;
                        if (buttonValidate) buttonValidate.disabled = false;
                    });
            }

            buttonRandom.onclick = () => {
                if (animator.isAnimating()) return;

                mathadata.run_python('get_data_and_label(random=True)', (result) => {
                    run_test(result);
                });
            };

            if (getGeneratedData) {
                buttonGenerated.onclick = () => {
                    if (animator.isAnimating()) return;
                    openModal();
                };

                buttonValidate.onclick = () => {
                    if (animator.isAnimating()) return;

                    const data = getGeneratedData();

                    // Fermer la modal
                    closeModal();

                    mathadata.run_python(`run_algorithme('${JSON.stringify(data)}')`, (res) => {
                        // Pour les données générées, pas de vrai label connu
                        run_test({
                            data,
                            label: res.result,  // On met l'estimation comme label pour éviter l'erreur
                            estimation: res.result
                        });
                    });
                };
            } else {
                buttonGenerated.style.display = 'none';
            }
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
                                text: 'Pourcentage d\\'erreur f(t)',
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

            /* erreur recursion scriptable
            chart.options.plugins.legend = chart.options.plugins.legend || {}
            chart.options.plugins.legend.labels = chart.options.plugins.legend.labels || {}
            chart.options.plugins.legend.labels.filter = function(legendItem, data) {
                console.log(legendItem.text)
                return legendItem.text !== 'Erreur';
            }
            */
            
            for (let i = 0; i < i_exercice_droite_carac; i++) {
                const datasetIndex = labels[i]
                chart.data.datasets[datasetIndex].data.push({x: c_train[i], y: 0})
            }

            chart.options.plugins.tooltip.callbacks.title = (context) => {
                if (context.length === 0 || context[0].datasetIndex >= 2) {
                    return undefined
                }
                
                let dataname = window.mathadata.challenge.strings.dataname.nom
                dataname = dataname.charAt(0).toUpperCase() + dataname.slice(1)
                
                return `${dataname} n°${context[0].dataIndex + 1}`;    
            }

            function setStatusMessage(msg, color) {
                const status = document.getElementById(`${id}-status`)
                status.innerHTML = msg
                status.style.color = color || 'black'
            }

            // Variable pour gérer l'animation d'erreur en cours
            let currentErrorInterval = null;
            let currentErrorDatasetIndex = null;

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
                            setStatusMessage(`Bravo, vous avez placé tous les points ! Exécutez la cellule suivante pour passer à la suite.`, 'green')
                            localStorage.setItem('exercice_droite_carac_ok', 'true')
                            window.mathadata.run_python(`set_exercice_droite_carac_ok()`)
                        } else {
                            setStatusMessage(`Bravo, vous pouvez placer ${mathadata.data({article: 'le'})} suivant${mathadata.e_fem()} (n°${i_exercice_droite_carac + 1}) !`)
                        }
                        chart.update()
                    } else {
                        setStatusMessage(`${mathadata.data({article: 'le', uppercase: true})} n'est pas à la bonne position sur la droite. Tu as placé un point à l'abscisse ${dataX.toFixed(2)} alors que ${mathadata.data({article: 'le', alt: true})} a une caractéristique x = ${c_train[i_exercice_droite_carac].toFixed(2)}`, 'red')

                        // Nettoyer toute animation d'erreur en cours
                        if (currentErrorInterval !== null) {
                            clearInterval(currentErrorInterval);
                            // Supprimer le dataset d'erreur précédent s'il existe encore
                            if (currentErrorDatasetIndex !== null) {
                                const indexToRemove = chart.data.datasets.findIndex(ds => ds.label === 'Erreur');
                                if (indexToRemove !== -1) {
                                    chart.data.datasets.splice(indexToRemove, 1);
                                }
                            }
                        }

                        // Ajouter un dataset temporaire pour le point d'erreur
                        const errorDatasetIndex = chart.data.datasets.length;
                        chart.data.datasets.push({
                            label: 'Erreur',
                            data: [{x: dataX, y: 0}],
                            pointRadius: 4,
                            pointBackgroundColor: 'rgba(255, 0, 0, 1)', // Opacité complète au départ
                        });
                        chart.update();

                        // Stocker l'index du dataset d'erreur
                        currentErrorDatasetIndex = errorDatasetIndex;

                        // Faire clignoter le point en alternant l'opacité
                        let blinkCount = 0;
                        let isVisible = true;
                        currentErrorInterval = setInterval(() => {
                            const errorDataset = chart.data.datasets[errorDatasetIndex];
                            if (errorDataset) {
                                // Alterner l'opacité entre visible et presque invisible
                                isVisible = !isVisible;
                                errorDataset.pointBackgroundColor = isVisible ? 'rgba(255, 0, 0, 1)' : 'rgba(255, 0, 0, 0.1)';
                                errorDataset.pointBorderColor = isVisible ? 'rgb(200, 0, 0)' : 'rgba(200, 0, 0, 0.1)';
                                chart.update();
                                blinkCount++;

                                if (blinkCount >= 8) { // 4 clignotements complets
                                    clearInterval(currentErrorInterval);
                                    currentErrorInterval = null;
                                    currentErrorDatasetIndex = null;
                                    // Supprimer le dataset d'erreur après le clignotement
                                    const indexToRemove = chart.data.datasets.findIndex(ds => ds.label === 'Erreur');
                                    if (indexToRemove !== -1) {
                                        chart.data.datasets.splice(indexToRemove, 1);
                                    }
                                    chart.update();
                                }
                            }
                        }, 500); // Clignoter toutes les 200ms
                    }
                }
            } else {
                setStatusMessage(`Bravo, vous avez placé tous les points ! Exécutez la cellule suivante pour passer à la suite.`, 'green')
            }

            chart.update()
        },

        tracer_droite_carac(id, params) {
            if (typeof params === 'string') {
                params = JSON.parse(params)
            }
            
            const {c_train, labels, t, afficherPoints} = params
            const max = Math.ceil(Math.max(...c_train) + 1)
            const min = Math.min(0, Math.floor(Math.min(...c_train) - 1))

            const config = {
                type: 'scatter',
                data: {
                    datasets: [
                    {
                        label: mathadata.classe(0, {plural: true, uppercase: true, alt: true}),
                        data: [],
                        pointRadius: 4,
                        pointHoverRadius: 8,
                        pointBackgroundColor: window.mathadata.classColors[0],
                    },
                    {
                        label: mathadata.classe(1, {plural: true, uppercase: true, alt: true}),
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
            
            if (afficherPoints) {
                c_train.forEach((x, i) => {
                    config.data.datasets[labels[i]].data.push({x, y: 0});
                })
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
                tooltipEl.style.maxWidth = '200px';

                tooltipEl.innerHTML = `
                    <div id="${id}-tooltip-infos"></div>
                    <div id="${id}-tooltip-loading" style="width: 100%; text-align: center;">Chargement ${mathadata.data({article: 'du', alt: true})}...</div>
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

        setupClassificationDisplay(containerId, options = {}) {
            const {
                totalItems = 0,
                showTotalInCounter = true,  // Si false, affiche juste "x" au lieu de "x/n"
                showErrors = false,
            } = options;

            const container = document.getElementById(containerId);
            const classes = mathadata.challenge.classes;
            const classLabels = mathadata.challenge.strings.classes;

            const color1 = mathadata.classColorCodes[0].split(',').map(x => parseInt(x.trim()));
            const color2 = mathadata.classColorCodes[1].split(',').map(x => parseInt(x.trim()));

            // Dimensions et positions - toutes les constantes pour l'animation
            const layout = {
                // Zone d'animation
                zoneHeight: 300,
                algoboxHeight: 140,

                // Boîtes de classes
                boxWidth: 160,
                boxHeight: 120,
                boxRight: 20,
                box1Top: 30,
                box2Top: 170,
            };

            const counterText = showTotalInCounter
                ? `<span id="${containerId}-count" style="font-weight: bold;">0</span>/${totalItems}`
                : `<span id="${containerId}-count" style="font-weight: bold;">0</span>`;

            container.innerHTML = `
                <!-- Compteurs en haut -->
                <div style="display: flex; justify-content: center; padding: 10px 0;">
                    <div style="background: linear-gradient(135deg, rgba(0,0,0,0.8), rgba(30,30,30,0.8)); color: white; padding: 12px 20px; border-radius: 8px; font-size: 13px; font-family: monospace; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);">
                        ${showErrors ? `<div style="font-size: 18px; font-weight: bold; color: #ff4757; margin-bottom: 8px;">
                            <span id="${containerId}-error-rate">0.0</span>% d'erreur
                        </div>` : ''}
                        <div style="margin-bottom: 4px; font-size: 12px; opacity: 0.9;">${mathadata.data({plural: true, alt: true, uppercase: true})}: ${counterText}</div>
                        <div style="color: rgb(${color1[0]}, ${color1[1]}, ${color1[2]}); margin-bottom: 2px; font-size: 11px; opacity: 0.85;">${classLabels[0].nom}: <span id="${containerId}-c1-count" style="font-weight: bold;">0</span></div>
                        <div style="color: rgb(${color2[0]}, ${color2[1]}, ${color2[2]}); margin-bottom: 2px; font-size: 11px; opacity: 0.85;">${classLabels[1].nom}: <span id="${containerId}-c2-count" style="font-weight: bold;">0</span></div>
                        ${showErrors ? `<div style="color: #ff4757; font-size: 11px; opacity: 0.85;">
                            Erreurs: <span id="${containerId}-error-count" style="font-weight: bold;">0</span>
                        </div>` : ''}
                    </div>
                </div>

                <!-- Zone d'animation au milieu -->
                <div id="${containerId}-animation-zone" style="position: relative; height: ${layout.zoneHeight}px; margin: 10px 0; display: flex; justify-content: center; align-items: center;">
                    <!-- Algo Box (centrée) -->
                    <div id="${containerId}-algo-box" style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); width: ${layout.algoboxHeight}px; height: ${layout.algoboxHeight}px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; text-align: center; line-height: ${layout.algoboxHeight}px; font-weight: bold; font-size: 20px; z-index: 10; box-shadow: 0 8px 20px rgba(0,0,0,0.4); transition: opacity 300ms ease;">ALGO</div>

                    <!-- Boîte Classe 1 (alignée à droite, en haut) -->
                    <div style="position: absolute; top: ${layout.box1Top}px; right: ${layout.boxRight}px; width: ${layout.boxWidth}px; height: ${layout.boxHeight}px; background: linear-gradient(135deg, rgba(${color1[0]}, ${color1[1]}, ${color1[2]}, 0.15), rgba(${color1[0]}, ${color1[1]}, ${color1[2]}, 0.25)); border: 3px solid rgb(${color1[0]}, ${color1[1]}, ${color1[2]}); border-radius: 12px; box-shadow: 0 6px 16px rgba(${color1[0]}, ${color1[1]}, ${color1[2]}, 0.4);" id="${containerId}-c1">
                        <div style="position: absolute; top: -18px; left: 50%; transform: translateX(-50%); background: linear-gradient(135deg, rgb(${color1[0]}, ${color1[1]}, ${color1[2]}), rgba(${color1[0]}, ${color1[1]}, ${color1[2]}, 0.8)); color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">${classLabels[0].nom}</div>
                    </div>

                    <!-- Boîte Classe 2 (alignée à droite, en bas) -->
                    <div style="position: absolute; top: ${layout.box2Top}px; right: ${layout.boxRight}px; width: ${layout.boxWidth}px; height: ${layout.boxHeight}px; background: linear-gradient(135deg, rgba(${color2[0]}, ${color2[1]}, ${color2[2]}, 0.15), rgba(${color2[0]}, ${color2[1]}, ${color2[2]}, 0.25)); border: 3px solid rgb(${color2[0]}, ${color2[1]}, ${color2[2]}); border-radius: 12px; box-shadow: 0 6px 16px rgba(${color2[0]}, ${color2[1]}, ${color2[2]}, 0.4);" id="${containerId}-c2">
                        <div style="position: absolute; top: -18px; left: 50%; transform: translateX(-50%); background: linear-gradient(135deg, rgb(${color2[0]}, ${color2[1]}, ${color2[2]}), rgba(${color2[0]}, ${color2[1]}, ${color2[2]}, 0.8)); color: white; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">${classLabels[1].nom}</div>
                    </div>
                </div>
            `;

            const animationZone = document.getElementById(`${containerId}-animation-zone`);
            const algoBox = document.getElementById(`${containerId}-algo-box`);

            // Retourner un contexte avec toutes les références nécessaires ET le layout
            return {
                container,
                animationZone,
                algoBox,
                classes,
                classLabels,
                containerId,
                layout  // Partager les constantes de layout
            };
        },

        updateClassificationCounters(containerId, counters) {
            const { count, c1Count, c2Count, errorCount } = counters;
            const errorRate = count > 0 ? ((errorCount / count) * 100).toFixed(1) : '0.0';

            const countEl = document.getElementById(`${containerId}-count`);
            const c1CountEl = document.getElementById(`${containerId}-c1-count`);
            const c2CountEl = document.getElementById(`${containerId}-c2-count`);
            const errorCountEl = document.getElementById(`${containerId}-error-count`);
            const errorRateEl = document.getElementById(`${containerId}-error-rate`);

            if (countEl) countEl.textContent = count;
            if (c1CountEl) c1CountEl.textContent = c1Count;
            if (c2CountEl) c2CountEl.textContent = c2Count;
            if (errorCountEl) errorCountEl.textContent = errorCount;
            if (errorRateEl) errorRateEl.textContent = errorRate;
        },

        /**
         * Configure l'affichage et retourne une closure pour animer des données
         * La closure encapsule l'état (compteurs, lastElements) pour chaque instance
         */
        setupAnimationAlgo(containerId, options = {}) {
            const {
                showTotalInCounter = false,
                keepElementCount = 20,
                totalItems = 0,
                showErrors = false,
            } = options;

            // Setup de l'affichage
            const context = mathadata.setupClassificationDisplay(containerId, {
                totalItems,
                showTotalInCounter,
                showErrors,
            });

            const classes = mathadata.challenge.classes;

            // État encapsulé dans la closure (privé à cette instance)
            const state = {
                counters: {
                    count: 0,
                    c1Count: 0,
                    c2Count: 0,
                    errorCount: 0
                },
                lastElements: {
                    [classes[0]]: [],
                    [classes[1]]: []
                },
                isAnimating: false
            };

            /**
             * Fonction pour animer une donnée
             * @param {Object} params - { data, label, estimation, animDuration?, showErrors? }
             * @returns Promise qui se résout quand l'animation est terminée
             */
            const animateData = (params) => {
                const { data, label, estimation, animDuration = 2400 } = params;

                return new Promise((resolve, reject) => {
                    if (state.isAnimating) {
                        reject(new Error('Animation already running'));
                        return;
                    }

                    state.isAnimating = true;

                    mathadata.animateSingleDataItem({ context, data, label, estimation, animDuration, showErrors })
                        .then(({ dataElement, isError }) => {
                            // Mettre à jour les compteurs
                            state.counters.count++;
                            const classIndex = classes.indexOf(estimation);
                            if (classIndex === 0) state.counters.c1Count++;
                            else state.counters.c2Count++;
                            if (isError && showErrors) state.counters.errorCount++;

                            mathadata.updateClassificationCounters(containerId, state.counters);

                            // Gérer la queue des lastElements
                            state.lastElements[estimation].push(dataElement);
                            if (state.lastElements[estimation].length > keepElementCount) {
                                const toRemove = state.lastElements[estimation].shift();
                                toRemove.remove();
                            }

                            state.isAnimating = false;
                            resolve({ dataElement, isError, counters: state.counters });
                        })
                        .catch(err => {
                            state.isAnimating = false;
                            reject(err);
                        });
                });
            };

            /**
             * Fonction pour vérifier si une animation est en cours
             */
            const isAnimating = () => state.isAnimating;

            /**
             * Fonction pour obtenir les compteurs actuels
             */
            const getCounters = () => ({ ...state.counters });

            /**
             * Fonction pour réinitialiser l'état
             */
            const reset = () => {
                state.counters = { count: 0, c1Count: 0, c2Count: 0, errorCount: 0 };
                state.lastElements = {
                    [classes[0]]: [],
                    [classes[1]]: []
                };
                state.isAnimating = false;
                mathadata.updateClassificationCounters(containerId, state.counters);
                // Nettoyer tous les éléments visuels
                context.animationZone.querySelectorAll('.animation-score-data-item').forEach(el => el.remove());
            };

            // Retourner l'API publique
            return {
                animateData,
                isAnimating,
                getCounters,
                reset,
                context
            };
        },

        animateSingleDataItem(params) {
            const { context, data, label, estimation, animDuration = 2400, showErrors = true } = params;

            return new Promise((resolve) => {
                const { animationZone, algoBox, classes, containerId, layout } = context;
                const isError = label !== estimation;

                // Durée de chaque phase (1/3 du total)
                const phaseTime = animDuration / 3;

                const dataElement = document.createElement('div');
                dataElement.className = 'animation-score-data-item';
                dataElement.id = Math.random().toString(36).substr(2, 9);

                // Position initiale à gauche
                const zoneWidth = animationZone.offsetWidth;
                const zoneHeight = animationZone.offsetHeight;

                const dataHeight = layout.algoboxHeight;
                const halfHeight = dataHeight / 2;

                dataElement.style.left = '-150px';
                dataElement.style.top = (zoneHeight / 2 - halfHeight) + 'px';
                dataElement.style.height = dataHeight + 'px';
                // Ne pas forcer la largeur pour laisser le canvas s'adapter
                // Mais définir une largeur minimale pour la visibilité
                dataElement.style.minWidth = dataHeight + 'px';
                dataElement.style.transition = `all ${phaseTime}ms ease`;

                animationZone.appendChild(dataElement);
                mathadata.affichage(dataElement.id, data);

                // Attendre que le DOM soit mis à jour pour obtenir la vraie largeur du contenu
                requestAnimationFrame(() => {
                    // Obtenir la largeur réelle après le rendu
                    const dataWidth = dataElement.offsetWidth;
                    const halfWidth = dataWidth / 2;

                    // Phase 1: Gauche → Algo (centre)
                    dataElement.style.left = (zoneWidth / 2 - halfWidth) + 'px';
                    dataElement.style.top = (zoneHeight / 2 - halfHeight) + 'px';

                    // Phase 2: Pause dans l'algo (1/3 du temps)
                    setTimeout(() => {
                        algoBox.style.opacity = '0.3';
                    }, phaseTime);

                    // Phase 3: Algo → Boîte (dernier 1/3 du temps)
                    setTimeout(() => {
                        // Restaurer l'opacité de la boîte algo
                        algoBox.style.opacity = '1';

                        // Surligner en rouge si erreur
                        if (isError && showErrors) {
                            dataElement.style.boxShadow = '0 0 20px 5px rgba(255, 71, 87, 0.8)';
                            dataElement.style.border = '2px solid #ff4757';
                        }

                        // Utiliser les constantes de layout pour calculer les centres des boîtes
                        const boxCenterRight = layout.boxRight + layout.boxWidth / 2;
                        const boxCenterTop = (estimation === classes[0] ? layout.box1Top : layout.box2Top) + layout.boxHeight / 2;

                        console.log(boxCenterRight, boxCenterTop);

                        // Décalage aléatoire par rapport au centre
                        const offsetX = (Math.random() - 0.5) * 30;
                        const offsetY = (Math.random() - 0.5) * 20;
                        const randomRotate = (Math.random() - 0.5) * 30; // Rotation aléatoire entre -15 et +15 degrés

                        const finalScale = 0.3;

                        // Positionner le centre du dataElement sur le centre de la boîte (+ décalage aléatoire)
                        // IMPORTANT: left/top s'appliquent AVANT le scale, donc on utilise les dimensions non-scalées
                        // boxCenterRight = distance du centre de la boîte depuis le bord droit
                        dataElement.style.left = (zoneWidth - boxCenterRight - offsetX - dataWidth / 2) + 'px';
                        dataElement.style.top = (boxCenterTop + offsetY - dataHeight / 2) + 'px';
                        dataElement.style.transform = `scale(${finalScale}) rotate(${randomRotate}deg)`;
                        dataElement.style.opacity = '1';
                    }, phaseTime * 2);

                    // Résolution de la Promise à la fin de l'animation
                    setTimeout(() => {
                        resolve({ dataElement, isError });
                    }, animDuration);
                });
            });
        },

        animateClassification(containerId, params) {
            params = JSON.parse(params);
            const { labels, estimations } = params;

            const totalItems = labels.length;

            const animator = mathadata.setupAnimationAlgo(containerId, {
                keepElementCount: 20,
                showTotalInCounter: true,
                totalItems,
                showErrors: true,
            });

            const { animateData, context } = animator;
            const { container, classes } = context;

            // État pour la boucle
            let i = 0;
            let isAnimationRunning = true;
            let isAnimationEnding = false;

            // Configuration de l'accélération
            const initialAnimDuration = 3000;
            const minAnimDuration = 500;
            const accelerationFactor = 1.15;

            function processData() {
                if (!isAnimationRunning || i >= totalItems) return;

                const animDuration = Math.max(minAnimDuration, initialAnimDuration / Math.pow(accelerationFactor, i));
                if (animDuration === minAnimDuration && !isAnimationEnding) {
                    isAnimationEnding = true;
                    endAnimation();
                }

                const currentLabel = labels[i];
                const currentEstimation = estimations[i];

                // Charger la donnée et animer
                mathadata.run_python(`get_data(index=${i})`, (data) => {
                    animateData({
                        data,
                        label: currentLabel,
                        estimation: currentEstimation,
                        animDuration
                    })
                        .then(() => {
                            // Passer à l'élément suivant
                            i++;
                            processData();
                        }).catch(err => {
                            console.error('Animation error:', err);
                            isAnimationEnding = true;
                            endAnimation();
                        });
                });
            }

            function endAnimation() {
                const fadeTime = 2000; // 2 secondes pour le fondu
                
                // Fait apparaitre un overlay noir au dessus
                const overlay = document.createElement('div');
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 0, 0)';
                overlay.style.opacity = '0';
                overlay.style.zIndex = '100';
                overlay.style.transition = `opacity ${fadeTime}ms ease`;

                container.appendChild(overlay);
                requestAnimationFrame(() => {
                    overlay.style.opacity = '1';

                    setTimeout(() => {
                        // Stopper l'animation quand le fond noir est complètement opaque
                        isAnimationRunning = false;
                        
                        // Attendre la fin de la dernière animation
                        setTimeout(() => {
                            // Afficher les compteurs finaux
                            let c0Count = 0, c1Count = 0, errorCount = 0;
                            for (let j = 0; j < estimations.length; j++) {
                                if (estimations[j] === classes[0]) c0Count++;
                                else c1Count++;
                                if (estimations[j] !== labels[j]) errorCount++;
                            }

                            mathadata.updateClassificationCounters(containerId, {
                                count: totalItems,
                                c1Count: c0Count,
                                c2Count: c1Count,
                                errorCount: errorCount,
                            });
                            
                            // Enlever le fond noir pour montrer l'état final
                            overlay.style.opacity = '0';

                            // Appeler le callback python après la fin du fondu
                            setTimeout(() => {
                                const score = errorCount / totalItems;
                                mathadata.updateScore(score);
                            }, fadeTime);
                        }, minAnimDuration * 1.5);

                    }, fadeTime);
                });
            }

            processData();
        },

        // variables in mathadata object
        charts: {},
        exercises: {},
        classColors: ['rgba(80,80,255,0.5)', 'rgba(255, 150, 0, 0.8)'],
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
                <h3>Calcul du taux d'erreur de l'algorithme</h3>
            </div>
            <div style="display: flex; justify-content: center; width: 100%; margin-top: 2rem">
                <div class="sidebox-section">
                    <h4 style="text-align: center;">Meilleur score:</h4>
                    <div style="display: flex; flex-direction: column; align-items: center; gap: 1rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width=40 height=40 viewBox="0 0 512 512"><!--!Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path fill="#B197FC" d="M4.1 38.2C1.4 34.2 0 29.4 0 24.6C0 11 11 0 24.6 0H133.9c11.2 0 21.7 5.9 27.4 15.5l68.5 114.1c-48.2 6.1-91.3 28.6-123.4 61.9L4.1 38.2zm503.7 0L405.6 191.5c-32.1-33.3-75.2-55.8-123.4-61.9L350.7 15.5C356.5 5.9 366.9 0 378.1 0H487.4C501 0 512 11 512 24.6c0 4.8-1.4 9.6-4.1 13.6zM80 336a176 176 0 1 1 352 0A176 176 0 1 1 80 336zm184.4-94.9c-3.4-7-13.3-7-16.8 0l-22.4 45.4c-1.4 2.8-4 4.7-7 5.1L168 298.9c-7.7 1.1-10.7 10.5-5.2 16l36.3 35.4c2.2 2.2 3.2 5.2 2.7 8.3l-8.6 49.9c-1.3 7.6 6.7 13.5 13.6 9.9l44.8-23.6c2.7-1.4 6-1.4 8.7 0l44.8 23.6c6.9 3.6 14.9-2.2 13.6-9.9l-8.6-49.9c-.5-3 .5-6.1 2.7-8.3l36.3-35.4c5.6-5.4 2.5-14.8-5.2-16l-50.1-7.3c-3-.4-5.7-2.4-7-5.1l-22.4-45.4z"/></svg>
                        <span id="highscore" class="score">...</span>
                    </div>
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

    // Variable pour stocker le highscore
    window.mathadata.highscore = null;

    window.mathadata.updateScore = (newScore) => {{
        if (newScore === null || newScore === undefined) return;

        // Mettre à jour le highscore si le nouveau score est meilleur (plus bas)
        if (window.mathadata.highscore === null || newScore < window.mathadata.highscore) {{
            window.mathadata.highscore = newScore;
            const percent = (newScore * 100).toFixed(1);
            document.getElementById('highscore').innerText = percent + '%';

            // Pop out the sidebox if the score is updated
            window.mathadata?.collapseSidebox();
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


steps = {
    'bdd': {
        'name': 'Présentation des données',
        'color': 'rgb(250,243,8)',
    },
    'depart': {
        'name': 'Algorithme de départ',
        'color': 'rgb(62,178,136)',
    },
    'data': {
        'name': "La donnée pour l'ordinateur",
        'color': 'rgb(244,167,198)',
    },
    'carac': {
        'name': 'Calcul de la caractéristique',
        'color': 'rgb(241,132,18)',
    },
    'classif': {
        'name': 'Classification',
        'color': 'rgb(230,29,73)',
    },
    # 'optim': {
    #     'name': 'Optimisation',
    #     'color': 'rgb(244,167,198)',
    # },
    'custom': {
        'name': 'Votre propre caractéristique',
        'color': 'rgb(20,129,173)',
    },
    # 'bacasable': {
    #     'name': 'Zone libre',
    #     'color': 'rgb(117,94,224)',
    # },
}

### PRETTY PRINT FUNCTIONS ###
import html

_BASE_STYLE = (
    "padding:12px 14px;"
    "border-radius:8px;"
    "border:1px solid;"
    "margin:8px 0;"
    "font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
)


def pretty_print_error(error_message):
    msg = error_message
    style = _BASE_STYLE + "background:#fde8e8;border-color:#f5b5b5;color:#7a1c1c;"
    display(HTML(f'<div style="{style}">❌ {msg}</div>'))


def pretty_print_success(success_message):
    msg = success_message
    style = _BASE_STYLE + "background:#e8f9f1;border-color:#a6e7c9;color:#14532d;"
    display(HTML(f'<div style="{style}">✅ {msg}</div>'))


### VALIDATION CLASSES ###

class _MathadataValidate():
    counter = 0
    last_call_time = None

    def __init__(self, success=None, function_validation=None, on_success=None, tips=None, get_tips=None, *args,
                 **kwargs):
        self.trials = 0
        self.success = success
        self.function_validation = function_validation
        self.child_on_success = on_success
        self.tips = tips
        self.get_tips = get_tips

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
            errors.append(
                "Il y a eu une erreur dans la validation de la question. Vérifie que ta réponse est écrite correctement")
            if debug:
                raise e

        if res is None:
            res = len(errors) == 0

        self.send_analytics_event(res, answers)

        if len(errors) > 0:
            for error in errors:
                pretty_print_error(error)

        if not res and (self.tips is not None or self.get_tips is not None):
            if self.get_tips is not None:
                self.tips = self.get_tips()

            id = uuid.uuid4().hex

            params = {
                'first_call_time': self.first_call_time,
                'tips': self.tips,
                'trials': self.trials,
                'answers': self.get_variables_str(),
            }
            run_js(f'''
                    mathadata.add_observer('{id}-tips', () => window.mathadata.setup_tips('{id}-tips', `{json.dumps(params, cls=NpEncoder)}`));
            ''')

            display(HTML(f'''<div id="{id}-tips"></div>'''))

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
                pretty_print_success(self.success)
        else:
            pretty_print_success("Bravo, c'est la bonne réponse !")
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
    if expected is None:  # Variable is optional
        res = True
    elif isinstance(expected, dict):
        # Check type first if specified
        if 'type' in expected:
            if not check_type(errors, name, val, expected['type']):
                return False

        # Then check value if specified
        if 'value' in expected:
            res = compare(val, expected['value'])
        else:
            res = True
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


# Types custom - utilise le système générique en remplaçant le type par ses paramètres
CUSTOM_TYPE_ALIASES = {
    'point': {
        'type': tuple,
        'element_type': (int, float),
        'length': 2,
        'name': 'un point',
        'example': '(3, 4)'
    },
    'vecteur': {
        'type': tuple,
        'element_type': (int, float),
        'length': 2,
        'name': 'un vecteur',
        'example': '(1, 2)'
    }
}


def replace_custom_types(type_spec):
    """
    Normalize a type specification, expanding custom type aliases.
    """
    # Handle string custom types
    if isinstance(type_spec, str) and type_spec in CUSTOM_TYPE_ALIASES:
        # Create a copy of the alias dict
        return dict(CUSTOM_TYPE_ALIASES[type_spec])

    # Handle dict with custom type
    if isinstance(type_spec, dict) and 'type' in type_spec and \
            isinstance(type_spec['type'], str) and type_spec['type'] in CUSTOM_TYPE_ALIASES:
        # Create a copy of the alias dict
        result = dict(CUSTOM_TYPE_ALIASES[type_spec['type']])
        # Pour pouvoir surcharger les paramètres par défaut de l'alias : ex {'type': 'vecteur', 'length': 3}
        for key in ['element_type', 'length']:
            if key in type_spec:
                result[key] = type_spec[key]
        return result

    return type_spec


def parse_type_spec(type_spec):
    """
    Parse a type specification and return (base_types, element_type, length).
    base_types is always a tuple, element_type is None or a tuple, length is None or int.
    """
    # Normalize first to handle custom types
    type_spec = replace_custom_types(type_spec)

    if isinstance(type_spec, dict) and 'type' in type_spec:
        base_type = type_spec['type']
        base_types = base_type if isinstance(base_type, tuple) else (base_type,)

        element_type = type_spec.get('element_type', None)
        if element_type is not None and not isinstance(element_type, tuple):
            element_type = (element_type,)

        length = type_spec.get('length', None)

        return base_types, element_type, length

    if isinstance(type_spec, tuple):
        return type_spec, None, None,

    return (type_spec,), None, None,


def get_type_str(type_spec):
    """
    Get a human-readable name for a type specification, adapted for students.
    """
    type_names = {
        'int': 'un nombre entier',
        'float': 'un nombre décimal',
        'str': 'un texte',
        'list': 'une liste',
        'tuple': 'une liste entre parenthèses',
        'bool': 'True ou False',
    }

    # Si le nom est déjà précisé, on le garde
    if isinstance(type_spec, dict) and 'name' in type_spec:
        return type_spec['name']

    base_types, element_types, length = parse_type_spec(type_spec)
    base_names = [type_names.get(getattr(t, '__name__', str(t))) for t in base_types]

    if len(base_names) == 1:
        base_str = base_names[0]
    elif len(base_names) == 2:
        base_str = f"{base_names[0]} ou {base_names[1]}"
    else:
        base_str = ', '.join(base_names[:-1]) + f" ou {base_names[-1]}"

    if length is not None:
        return f"{base_str} de {length} éléments"

    return base_str


def get_type_example(type_spec):
    """
    Get an example of correct syntax for a type specification.
    """
    examples = {
        'int': '42',
        'float': '3.14',
        'str': '"exemple"',
        'bool': 'True',
    }

    # Si l'exemple est déjà précisé, on le garde
    if isinstance(type_spec, dict) and 'example' in type_spec:
        return type_spec['example']

    base_types, element_types, length = parse_type_spec(type_spec)

    base_type = base_types[0]  # Use first type for example
    base_name = base_type.__name__ if hasattr(base_type, '__name__') else str(base_type)

    if element_types is not None:
        element_type = element_types[0]  # Use first type for example
        elem_name = element_type.__name__ if hasattr(element_type, '__name__') else str(element_type)
        elem_example = examples.get(elem_name, '...')

        if length is None:
            length = 2  # Default length for example

        # Generate example with exact length
        elems = ', '.join([elem_example] * length)
        if base_name == 'list':
            return f'[{elems}]'
        elif base_name == 'tuple':
            return f'({elems})'
    else:
        return examples.get(base_name, base_name)


def check_type(errors, name, val, type_spec):
    """
    Check if val matches the type specification and format errors if not.

    type_spec can be:
    - A basic type: int, float, str, list, tuple, etc.
    - A tuple of types: (int, float) for multiple acceptable types
    - A custom type string: 'point', 'vecteur'
    - A dict with 'type' and optionally 'element_type' and 'length':
      {'type': list, 'element_type': int}
      {'type': tuple, 'element_type': int, 'length': 3}
      {'type': tuple, 'element_type': (int, float)} for mixed types
      {'type': (list, tuple), 'element_type': int} for multiple container types
      {'type': 'point'} for custom types (expands to tuple spec)
    """
    base_types, element_types, length = parse_type_spec(type_spec)
    type_str = get_type_str(type_spec)
    example = get_type_example(type_spec)

    # Check base type
    if not any(isinstance(val, t) for t in base_types):
        errors.append(f"{name} doit être {type_str}. Exemple : {name} = {example}")
        return False

    # Check length for sequences
    if length is not None and hasattr(val, '__len__'):
        if len(val) != length:
            errors.append(
                f"{name} doit contenir {length} éléments. Elle en contient {len(val)} : {', '.join(repr(v) for v in val)}")
            # continue with other errors

    # Check element types for list/tuple
    if element_types is not None and hasattr(val, '__iter__'):
        for i, item in enumerate(val):
            if not any(isinstance(item, t) for t in element_types):
                # Create a type_spec just for the element types
                elem_type_str = get_type_str(element_types)
                errors.append(f"{name} doit être {type_str} avec {elem_type_str}. Exemple : {name} = {example}")
                return False

    return True


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
        undefined_variables = []
        for name in self.name_and_values:
            if not hasattr(__main__, name):
                undefined_variables.append(name)
                errors.append(f"La variable {name} n'a pas été définie. Ecris dans ta cellule : {name} = ta_reponse")
            else:
                var = get_variable(name)
                if var is Ellipsis or (hasattr(var, '__iter__') and any(v is Ellipsis for v in var)):
                    undefined_variables.append(name)
                    errors.append(f"Remplace les ... par ta réponse pour {name}.")

        return undefined_variables

    def check_variables(self, errors):
        for name in self.name_and_values:
            val = get_variable(name)
            expected = self.name_and_values[name]
            check_variable(errors, name, val, expected)

        return len(errors) == 0

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
            res = self.check_variables(errors)
            if res and self.function_validation is not None:
                return self.function_validation(errors, answers)

        return len(errors) == 0

    def get_variables_str(self):
        res = []
        for name in self.name_and_values:
            expected = self.name_and_values[name]
            if expected is None:
                continue
            elif isinstance(expected, dict):
                if 'value' in expected:
                    expected = expected['value']
                else:
                    continue

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
    def __init__(self, function_name, test_set=[], expected=[], expected_function=None,
                 success="Bravo, ta fonction est correcte", *args, **kwargs):
        super().__init__(success=success, *args, **kwargs)
        self.function_name = function_name
        self.test_set = test_set
        self.expected = expected
        self.expected_function = expected_function

    def validate(self, errors, answers):
        if not has_variable(self.function_name):
            errors.append(
                f"La fonction {self.function_name} n'est pas définie. Tu dois avoir une ligne de code qui commence par 'def {self.function_name}(...):'")
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
        nb_errors = np.count_nonzero(
            np.array([algorithme(d) for d in challenge.d_train[0:10]]) != challenge.r_train[0:10])
        if nb_errors * 10 == e_train_10:
            pretty_print_success(
                f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname']['pluriel']}, soit {e_train_10}% d'erreur")
            return True
        else:
            if e_train_10 == nb_errors:
                errors.append(
                    f"Ce n'est pas la bonne valeur. Pour passer du nombre d'erreurs sur 10 {challenge.strings['dataname']['pluriel']} au pourcentage, tu dois multiplier par 10 !")
            elif e_train_10 < 0 or e_train_10 > 100:
                errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
            else:
                errors.append(
                    "Ce n'est pas la bonne valeur. Compare ta liste de prédictions avec les vraies valeurs pour trouver le pourcentage d'erreur.")
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
    algorithme = get_algorithme_func(
        error="La fonction algorithme n'existe plus. Revenez en arrière et réexecutez la cellule avec 'def algorithme(d): ...'")

    estimations = [algorithme(d) for d in challenge.d_train[0:10]]
    nb_errors = np.sum(estimations != challenge.r_train[0:10])

    if not isinstance(score_10, int):
        if score_10 == float(nb_errors * 10):
            pretty_print_success(
                f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname']['pluriel']}, soit {score_10}% d'erreur")
            return True
        else:
            errors.append(
                "La variable erreur_10 doit être un entier. Attention, écrivez uniquement le nombre sans le %.")
            return False

    if score_10 == nb_errors * 10:
        pretty_print_success(
            f"Bravo, ton algorithme actuel a fait {nb_errors} erreurs sur les 10 {ac_fem('premières', 'premiers')} {challenge.strings['dataname']['pluriel']}, soit {score_10}% d'erreur")
        return True

    if score_10 == nb_errors:
        errors.append("Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreur et non le pourcentage d'erreur.")
    elif score_10 < 0 or score_10 > 100:
        errors.append("Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100.")
    else:
        errors.append(
            f"Ce n'est pas la bonne valeur. Note la réponse donnée par ton algorithme pour chaque {challenge.strings['dataname']['nom']} et compte le nombre de différences avec les bonnes réponses.")

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


def set_step_algo_faineant(answers):
    global current_algo
    current_algo = algo_faineant


def set_step_algo_carac(answers):
    global current_algo
    current_algo = algo_carac


def set_step_algo_carac_custom(answers):
    global current_algo
    current_algo = algo_carac_custom


validation_execution_algo_fixe = MathadataValidate(success="", on_success=set_step_algo_faineant)
validation_execution = MathadataValidate(success="")
validation_execution_calcul_score = MathadataValidate(success="")

validation_question_score_fixe = MathadataValidateVariables({
    'erreur_10': None
}, function_validation=validation_func_score_fixed,
    tips=[
        {
            'seconds': 50,
            'tip': 'Compte dans le tableau le nombre d\'erreurs commmises. Il restera une opération mathématique à faire pour obtenir le pourcentage d\'erreur.'
        }, {
            'seconds': 100,
            'tip': 'Il faut diviser par 10 pour otebnir la proportion d\'erreur parmi les 10 premières valeurs.'
        }], success="")
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
validation_execution_classif = MathadataValidate(success="", on_success=set_step_algo_carac)


def affichage_seuil():
    t = get_variable('t')
    if t is None or t is Ellipsis:
        print_error("Le seuil n'a pas été défini. Tu dois d'abord définir la variable t.")
        return False
    else:
        pretty_print_success(
            f"Le seuil que tu as choisi est {t}. Tu peux maintenant calculer le score de ton algorithme sur les 10 premières lignes du tableau.")
        return True


validation_execution_affichage_10_droite = MathadataValidate(success="")

validation_question_score_seuil = ValidateScoreThresold()
validation_score_carac = MathadataValidate(success="")
validation_execution_graph_erreur = MathadataValidate(success="")
validation_question_seuil_optimise = MathadataValidateVariables(
    get_names_and_values=get_validate_seuil_optimized,
    success="Bravo, c'est la bonne réponse ! Ton seuil est maintenant optimal"
)

validation_question_nombre = MathadataValidateVariables(get_names_and_values=lambda: {
    f'nombre_{challenge.classes[1]}': {
        'value': np.sum(challenge.r_train[0:10] == challenge.classes[1]),
        'errors': [
            {
                'value': np.sum(challenge.r_train[0:10] == challenge.classes[1]),
                'else': "As-tu bien regardé les 10 premières lignes du tableau ? "
            },
        ]
    }
})

validation_question_nombre_total = MathadataValidateVariables(get_names_and_values=lambda: {
    f'nombre_total_{data(alt=True, plural=True)}': {
        'value': len(challenge.d_train),
        'errors': [
            {'value': len(challenge.d_train),
             'else': "Ce n'est pas la bonne réponse. Le tableau est intéractif."
             },
        ]
    }
}, tips=[
    {
        'seconds': 30,
        'tip': 'Après avoir cliqué dans le tableau intéractif tu peux utiliser les flêches du clavier pour naviguer dans les lignes du tableau. Tu peux aussi utiliser la molette de la souris pour faire défiler le tableau.'
    },

    {
        'seconds': 60,
        'trials': 4,

        'print_solution': True,
        'validate': True
    }
])

validation_execution_test_algorithme_faineant = MathadataValidate(success="", on_success=lambda a: print(f"Testez l'algorithme sur plusieurs {data(alt=True, plural=True)} avant de passer à la suite."))
validation_execution_test_algorithme_ref = MathadataValidate(success="", on_success=lambda a: print(f"Testez l'algorithme sur plusieurs {data(alt=True, plural=True)} avant de passer à la suite."))

# Defini pour éviter l'erreur python par defaut si l'élève mets reponse = chat
chat = 0
cat = 0

validation_question_faineant = MathadataValidateVariables(get_names_and_values=lambda: {
    'Reponse_Donnee_A': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Donnee_A n'a pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    },
    'Reponse_Donnee_B': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Donnee_B n'as pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    },
    'Reponse_Donnee_C': {
        'value': challenge.classes[0],
        'errors': [
            {
                'value': challenge.classes[0],
                'else': "Reponse_Donnee_C n'as pas la bonne valeur. As-tu bien lu ce que fait l'algorithme fainéant ?"
            },
        ]
    }
}, get_tips=lambda: [{
    'seconds': 30,
    'tip': f"L'algorithme fainéant ne réfléchit pas. Il répond toujours la même chose peu importe {data('le', alt=True)}. Relis le texte au dessus pour voir ce qu'il répond."
}],
                                                          success="", on_success=lambda answers: pretty_print_success(
        f"Bravo, {ac_fem('quelle', 'quel')} que soit {data('le', alt=True)} l'algorithme fainéant répond toujours {challenge.classes[0]} !"))

validation_custom = MathadataValidate(
    success="Bravo, c'est un excellent taux d'erreur ! Tu peux passer à la suite pour essayer de faire encore mieux.")
validation_code_eleve = MathadataValidate(success="Bravo, c'est un excellent taux d'erreur ! Tu as fini ce notebook.")

validation_code_free = MathadataValidate(success="Bravo")

validation_code_etendue = MathadataValidate(
    success="Bravo, tu as fini ce notebook ! Tu peux maintenant passer à la suite pour découvrir d'autres algorithmes de classification.")

validation_code_moyenne_ligne = MathadataValidate(
    success="Bravo, tu as fini ce notebook ! Tu peux maintenant passer à la suite pour découvrir d'autres algorithmes de classification.")

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
        MathadataValidate()()  # Validate import cell -> doesn't work

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

    for i in range(0, 5):  # même m ou même p
        rand = np.random.randint(0, 2)
        if rand == 0:  # même m
            m1 = np.random.randint(-10, 10) / 2
            m2 = m1
            p1 = np.random.randint(-8, 8)
            p2 = p1
            while p2 == p1:
                p2 = np.random.randint(-8, 8)
        else:  # même p
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
                    'type': 'radio',
                    'choices': ['m', 'p'],
                    'value': 'm' if rand == 0 else 'p',
                }
            }
        })

    for i in range(0, 3):  # quelle droite a le plus grand coefficient directeur m ?
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

    for i in range(0, 2):  # donnez les paramètres de la droite
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
    if sequence:
        Validate()()


run_js('''
// Système d'observation d'apparition d'éléments dans le DOM. Utilisé pour programmer l'exécution d'une fonction JS quand un élément avec un ID donné apparaît, par exemple lorsque le code Python crée un div container avec cet ID.
window.mathadata.observed_ids = {}
window.mathadata.add_observer = function(id, func) {
    if (document.getElementById(id)) {
        console.log(`Element with ID #${id} already exists! Calling callback immediately.`);
        func();
        return;
    }
    
    window.mathadata.observed_ids[id] = {
        callback() {
            // Appelle la fonction programmée puis supprime l'observateur
            func()
            delete window.mathadata.observed_ids[id];
        }
    }
    console.log(`Observer added for ID #${id}`);
}

window.mathadata.mutationObserver = new MutationObserver((mutationsList, observer) => {
    for (const mutation of mutationsList) {
        for (const node of mutation.addedNodes) {
            if (node.nodeType !== Node.ELEMENT_NODE) {
                continue; // Ignorer les nœuds qui ne sont pas des éléments
            }
            
            if (node.id) {
                if (node.id in window.mathadata.observed_ids) {
                    console.log(`Element with ID #${node.id} appeared!`);
                    window.mathadata.observed_ids[node.id].callback()
                }
            }

            // Vérifier dans les enfants
            if (node.querySelectorAll) {
                const observedIds = Object.keys(window.mathadata.observed_ids);
                const selector = observedIds.map(id => `#${CSS.escape(id)}`).join(', ');
                if (selector) {
                    const elements = node.querySelectorAll(selector);
                    if (elements.length > 0) {
                        elements.forEach(el => {
                            console.log(`Element with ID #${el.id} appeared!`);
                            window.mathadata.observed_ids[el.id].callback()
                        });
                    }
                }
            }
        }
    }
});

window.mathadata.mutationObserver.observe(document.body, { childList: true, subtree: true });

window.mathadata.create_qcm = function(id, configInput) {
    if (typeof configInput === 'string') {
        configInput = JSON.parse(configInput);
    }

    let configs;
    let multipleQuestions = true;

    // Normaliser en tableau pour supporter une ou plusieurs questions
    if (!Array.isArray(configInput)) {
        if (configInput.questions !== undefined) {
            configs = configInput.questions;
            if (configs.length <= 1) {
                multipleQuestions = false;
            }
        } else {
            configs = [configInput];
            multipleQuestions = false;
        }
    } else {
        configs = configInput;
        configInput = { questions: configs };
    }

    // Fonction de hashing simple pour la config
    function hashConfig(config) {
        // Créer une représentation déterministe de la config pour le hash
        const configToHash = {
            questions: config.questions || [config],
        };
        const str = JSON.stringify(configToHash);

        // Simple hash function (DJB2)
        let hash = 5381;
        for (let i = 0; i < str.length; i++) {
            hash = ((hash << 5) + hash) + str.charCodeAt(i);
        }
        return 'qcm_' + (hash >>> 0).toString(36);
    }

    // Générer le hash de cette config
    const configHash = hashConfig(configInput);

    // Vérifier si ce QCM a déjà été réussi
    const completedQCMs = JSON.parse(localStorage.getItem('mathadata_completed_qcms') || '{}');
    const isAlreadyCompleted = completedQCMs[configHash] === true;

    const container = document.getElementById(id);
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.alignItems = 'center';

    let htmlContent = '';

    configs.forEach((config, index) => {
        // Déterminer les bonnes réponses pour savoir si checkbox ou radio
        let correctAnswers = [];
        if (config.answers !== undefined) {
            correctAnswers = config.answers;
        } else if (config.answers_indexes !== undefined) {
            correctAnswers = config.answers_indexes.map(idx => config.choices[idx]);
        } else if (config.answer !== undefined) {
            correctAnswers = [config.answer];
        } else if (config.answer_index !== undefined) {
            correctAnswers = [config.choices[config.answer_index]];
        }

        const isMultipleAnswers = correctAnswers.length > 1;
        const inputType = isMultipleAnswers ? 'checkbox' : 'radio';

        htmlContent += `
        <div class="qcm-question-block" data-index="${index}" style="width: 100%; display: flex; flex-direction: column; align-items: center;">
            <div style="font-weight: bold; margin-bottom: 10px; text-align: center;">${config.question_html || config.question}</div>
            ${isMultipleAnswers ? '<div style="font-style: italic; margin-bottom: 10px; color: #666;">(Plusieurs réponses possibles)</div>' : ''}
            <div style="display: flex; column-gap: 25px; row-gap: 1rem; flex-wrap: wrap; justify-content: center; ${config.multiline ? 'flex-direction: column;' : ''}">
                ${config.choices.map((choice, choiceIdx) => {
                    // Utilisation d'un name unique par question : ${id}-qcm-${index}
                    return `
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <input type="${inputType}" name="${id}-qcm-${index}" value="${choice}" id="${id}-qcm-${index}-${choiceIdx}">
                            <label for="${id}-qcm-${index}-${choiceIdx}" style="margin-bottom: 0;">${choice}</label>
                        </div>
                    `
                }).join('\\n')}
            </div>
            <div id="${id}-qcm-status-${index}" style="margin-top: 5px; text-align: center; min-height: 2rem;"></div>
        </div>
        `;
    });

    container.innerHTML = `
        <div style="display: flex; flex-direction: column; gap: 30px; align-items: center;">
            ${htmlContent}
        </div>
        <button id="${id}-qcm-validate-button" style="margin: auto; margin-top: 20px; margin-bottom: 10px; padding: 5px 10px; border-radius: 5px; background-color: #007bff; color: white; border: none; cursor: pointer;">Valider</button>
        <div id="${id}-qcm-global-status" style="margin-top: 10px; margin-bottom: 10px; text-align: center; font-weight: bold;"></div>
    `;

    const validateButton = document.getElementById(id + '-qcm-validate-button');

    validateButton.addEventListener('click', () => {
        let allCorrect = true;
        let allAnswered = true;

        configs.forEach((config, index) => {
            // Re-calcul des bonnes réponses pour validation
            let correctAnswers = [];
            if (config.answers !== undefined) {
                correctAnswers = config.answers;
            } else if (config.answers_indexes !== undefined) {
                correctAnswers = config.answers_indexes.map(idx => config.choices[idx]);
            } else if (config.answer !== undefined) {
                correctAnswers = [config.answer];
            } else if (config.answer_index !== undefined) {
                correctAnswers = [config.choices[config.answer_index]];
            }

            const isMultipleAnswers = correctAnswers.length > 1;
            const statusText = document.getElementById(`${id}-qcm-status-${index}`);
            
            // Récupérer les réponses cochées pour CETTE question
            const selectedValues = Array.from(container.querySelectorAll(`input[name="${id}-qcm-${index}"]:checked`))
                .map(input => input.value);

            if (selectedValues.length === 0) {
                allAnswered = false;
                statusText.style.color = 'orange';
                if (isMultipleAnswers) {
                    statusText.textContent = "Veuillez sélectionner au moins une réponse.";
                } else {
                    statusText.textContent = "Veuillez sélectionner une réponse.";
                }
                return;
            }

            let questionCorrect = false;
            if (isMultipleAnswers) {
                const hasWrongAnswer = selectedValues.some(val => !correctAnswers.includes(val));
                const hasAllCorrectAnswers = correctAnswers.every(ans => selectedValues.includes(ans));

                if (hasWrongAnswer) {
                    statusText.textContent = "Il y a au moins une mauvaise réponse dans ta sélection. Essaie encore !";
                    statusText.style.color = 'red';
                } else if (!hasAllCorrectAnswers) {
                    statusText.textContent = "Tu n'as pas trouvé toutes les réponses valides. Continue à chercher !";
                    statusText.style.color = 'orange';
                } else {
                    statusText.textContent = 'Bravo ! Tu as trouvé toutes les bonnes réponses !';
                    statusText.style.color = 'green';
                    questionCorrect = true;
                }
            } else {
                if (correctAnswers.includes(selectedValues[0])) {
                    questionCorrect = true;
                    statusText.textContent = 'Bonne réponse !';
                    statusText.style.color = 'green';
                } else {
                    statusText.textContent = "Ce n'est pas la bonne réponse. Essaie encore !";
                    statusText.style.color = 'red';
                }
            }

            if (!questionCorrect) {
                allCorrect = false;
            }
        });

        const globalStatus = document.getElementById(`${id}-qcm-global-status`);
        if (allCorrect && allAnswered) {
            if (configInput.success) {
                globalStatus.textContent = configInput.success;
            } else {
                globalStatus.textContent = 'Bravo ! Vous pouvez passer à la suite.';
            }
            globalStatus.style.color = 'green';

            // Sauvegarder dans le localStorage que ce QCM a été réussi
            completedQCMs[configHash] = true;
            localStorage.setItem('mathadata_completed_qcms', JSON.stringify(completedQCMs));

            mathadata.pass_breakpoint();
        } else if (multipleQuestions) {
            globalStatus.textContent = "Répondez correctement à toutes les questions pour passer à la suite.";
            globalStatus.style.color = 'red';
        }
    });

    // Si le QCM a déjà été complété, afficher un message et passer le breakpoint automatiquement
    if (isAlreadyCompleted) {
        const globalStatus = document.getElementById(`${id}-qcm-global-status`);
        globalStatus.textContent = '✓ Vous avez déjà réussi ce QCM précédemment. Vous pouvez continuer.';
        globalStatus.style.color = 'green';
        globalStatus.style.fontStyle = 'italic';
        mathadata.pass_breakpoint();
    } else if (configInput.superuser) {
        mathadata.pass_breakpoint();
    }
}
''')


def create_qcm(elements, qcm_id=None):
    if qcm_id is None:
        qcm_id = uuid.uuid4().hex

    if has_variable('superuser') and get_variable('superuser') == True:
        if isinstance(elements, dict):
            elements['superuser'] = True
        elif isinstance(elements, list):
            elements = {
                'questions': elements,
                'superuser': True
            }

    # Use json.dumps with ensure_ascii=False to properly handle LaTeX
    config_json = json.dumps(elements, ensure_ascii=False)
    # Escape backslashes for JavaScript template string
    config_json = config_json.replace('\\', '\\\\')

    run_js(f'''
        mathadata.add_observer('{qcm_id}', () => window.mathadata.create_qcm('{qcm_id}', `{json.dumps(elements)}`))
        mathadata.add_observer('{qcm_id}', () => window.mathadata.create_qcm('{qcm_id}', `{config_json}`))
    ''')
    display(HTML(f'''
        <div id="{qcm_id}"></div>
    '''))


def qcm_test():
    create_qcm({
        'question': 'Quel est le plus grand nombre ?',
        'choices': ['1', '2', '3'],
        'answer': '3',
    })


custom_carac_free_exemple_code = """
    # Ajouter votre code ici
    
    
    
    return ...
"""

etendue_carac_exemple_code = """
    # Exemple pour utiliser l'étendue comme caractéristique.
    # Complètez le code en remplaçant les ... et en utilisant les fonction min et max
    
    minimum = min(d)
    maximum = ...
    etendue = ...
    
    
    return etendue
"""


def validate_caracteristique_libre(errors, answers):
    """
    Validation de la caractéristique libre.
    La caractéristique doit être un nombre.
    """
    caracteristique = answers['caracteristique']
    for d in challenge.d_train[0:5]:
        if not isinstance(caracteristique(d), float) and not isinstance(caracteristique(d), int):
            errors.append("La caractéristique doit être un nombre. Ta fonction ne semble pas renvoyer un nombre.")
            return False
    return True


def on_success_histogramme(answers):
    if has_variable('afficher_histogramme'):
        get_variable('afficher_histogramme')(legend=True, caracteristique=get_variable('caracteristique'))


validation_caracteristique_libre_et_affichage = MathadataValidateVariables(name_and_values={'caracteristique': None},
                                                                           function_validation=validate_caracteristique_libre,
                                                                           success="Ta fonction renvoie bien un nombre. Testons ta proposition",
                                                                           on_success=on_success_histogramme)

# Stepbar (barre d'étapes)

run_js('''
  const anchorPrefix = 'debut-etape-';

  window.mathadata.stepbar = {
    rootId: null,
    steps: [],
    stepNames: [],
    colors: [],
    current: 1,
    container: null,
    lastActivatedIndex: 0,
    observer: null,
    mutationObserver: null,
    // Convertit un nom d'étape en index (1-indexé)
    nameToIndex(name) {
      const idx = this.stepNames.indexOf(name);
      return idx >= 0 ? idx + 1 : null;
    },
    // Convertit un index (1-indexé) en nom d'étape
    indexToName(idx) {
      return this.stepNames[idx - 1] || null;
    },
    // Active visuellement l'étape idx et les précédentes si pas encore activée (appelé depuis create_step_anchor)
    activate(idx){
        if (!this.container) return;
        if (this.lastActivatedIndex >= idx) return;

        for (let i = 1; i <= idx; i++) {
            const el = this.container.querySelector(`.mathadata-stepbar__item[data-step-index="${i}"]`);
            const color = Array.isArray(this.colors) && this.colors[i - 1] ? this.colors[i - 1] : null;
            if (el && color) {
                el.style.background = color;
                el.style.color = '#fff';
                el.style.borderColor = 'transparent';
                el.classList.remove('mathadata-stepbar__item--disabled');
                el.parentElement.classList.add('mathadata-stepbar__itemRow--unlocked');
            }
        }
        this.lastActivatedIndex = idx;
        const color = Array.isArray(this.colors) && this.colors[idx - 1] ? this.colors[idx - 1] : null;

        // Changer le fond de la page
        const pageElement = document.getElementById('notebook');
        if (pageElement && color) {
          pageElement.style.transition = 'background-color 0.5s ease';
          pageElement.style.backgroundColor = color;
          pageElement.style.position = 'relative';
        }
    },
    // Initialise l'observateur pour les divs invisibles
    initObserver(){
      console.log('[Stepbar] Initializing observers');

      // IntersectionObserver pour détecter la visibilité
      if (this.observer) {
        this.observer.disconnect();
      }

      this.observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const stepName = entry.target.id.replace(anchorPrefix, '');
            const stepIndex = this.nameToIndex(stepName);
            console.log('[Stepbar] Trigger visible:', stepName, '(index:', stepIndex + ')');
            if (stepIndex) {
              // Première fois qu'on passe cette étape : on l'active (débloque)
              if (this.lastActivatedIndex < stepIndex) {
                console.log('[Stepbar] First time seeing step', stepName, '- activating');
                this.activate(stepIndex);
                // Animation confettis (sauf première étape)
                if (stepIndex > 1 && typeof window.mathadata.fireSideCannons === 'function') {
                  window.mathadata.fireSideCannons();
                }
              }
              // Toujours mettre à jour le current
              this.setCurrent(stepIndex);
            }
          }
        });
      }, {
        threshold: 0.1,
        rootMargin: '0px'
      });

      // MutationObserver pour détecter les nouvelles divs ajoutées au DOM
      if (this.mutationObserver) {
        this.mutationObserver.disconnect();
      }

      this.mutationObserver = new MutationObserver((mutations) => {
        mutations.forEach(mutation => {
          mutation.addedNodes.forEach(node => {
            if (node.nodeType === 1) { // ELEMENT_NODE
              // Vérifier le noeud lui-même
              if (node.id && node.id.startsWith(anchorPrefix)) {
                const stepName = node.id.replace(anchorPrefix, '');
                console.log('[Stepbar] New anchor detected:', stepName, '- adding to observer');
                this.observeElement(node);
              }
              // Vérifier dans les enfants
              if (node.querySelectorAll) {
                const anchors = node.querySelectorAll(`[id^="${anchorPrefix}"]`);
                if (anchors.length > 0) {
                  console.log('[Stepbar] Found', anchors.length, 'anchor(s) in children');
                  anchors.forEach(el => {
                    const stepName = el.id.replace(anchorPrefix, '');
                    console.log('[Stepbar] New anchor detected (child):', stepName, '- adding to observer');
                    this.observeElement(el);
                  });
                }
              }
            }
          });
        });
      });

      // Observer le body pour détecter les nouvelles cellules markdown
      this.mutationObserver.observe(document.body, {
        childList: true,
        subtree: true
      });

      // Observer les éléments existants
      this.observeExistingElements();
    },

    // Observe un élément spécifique
    observeElement(el){
        this.observer.observe(el);
    },
    // Observer tous les éléments existants
    observeExistingElements(){
      const anchors = document.querySelectorAll(`[id^="${anchorPrefix}"]`);
      console.log('[Stepbar] Found', anchors.length, 'existing anchor(s)');
      anchors.forEach(el => {
        const stepName = el.id.replace(anchorPrefix, '');
        console.log('[Stepbar] Observing existing anchor:', stepName);
        this.observeElement(el);
      });
    },
    // Rafraîchit l'observateur (utile si de nouvelles divs sont ajoutées)
    refreshObserver(){
      this.initObserver();
    },
    setCurrent(idx){
      this.current = idx;
      if (!this.container) return;
      // Toggle classe active sur le cercle
      this.container.querySelectorAll('.mathadata-stepbar__item').forEach(el => {
        el.classList.toggle('mathadata-stepbar__item--active', Number(el.getAttribute('data-step-index')) === idx);
      });
      // Toggle classe active sur la ligne (pour styliser le label)
      this.container.querySelectorAll('.mathadata-stepbar__itemRow').forEach(row => {
        const item = row.querySelector('.mathadata-stepbar__item');
        if (!item) return;
        const stepIndex = Number(item.getAttribute('data-step-index'));
        row.classList.toggle('mathadata-stepbar__itemRow--active', stepIndex === idx);
      });
    },
    goto(idx){
      const stepName = this.indexToName(idx);
      if (!stepName) return;
      const anchor = document.getElementById(`${anchorPrefix}${stepName}`);
      if (anchor) {
        anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    },
    toggle(){
      if (!this.container) return;
      const computedRight = window.getComputedStyle(this.container).right;
      const isCollapsed = parseInt(computedRight) < 0;
      this.container.style.right = isCollapsed ? '0px' : '-240px';

      // Faire pivoter l'icône du bouton
      if (this.collapseButton) {
        const icon = this.collapseButton.querySelector('svg');
        if (icon) {
          icon.style.transform = isCollapsed ? '' : 'rotate(180deg)';
        }
      }
    },
    createItem(idx, label){
      const btn = document.createElement('div');
      btn.className = 'mathadata-stepbar__item';
      btn.setAttribute('data-step-index', String(idx));
      btn.textContent = String(idx);

      // Toujours grisé par défaut (sera coloré lors de l'activation)
      btn.style.backgroundColor = '#d9d9d9';
      btn.style.color = '#777';
      btn.classList.add('mathadata-stepbar__item--disabled');

      btn.addEventListener('click', () => {
        // Bloque la navigation si la step n'est pas activée
        if (this.lastActivatedIndex < idx) return;
        this.goto(idx);
      });

      // Conteneur ligne: label (gauche) + cercle (droite)
      const row = document.createElement('div');
      row.className = 'mathadata-stepbar__itemRow';
      const hardLabel = document.createElement('div');
      hardLabel.className = 'mathadata-stepbar__labelHard';
      hardLabel.textContent = label;
      row.appendChild(hardLabel);
      row.appendChild(btn);
      return row;
    }
  };

  window.mathadata.init_stepbar = (config) => {
    console.log(config)
    if (typeof config === 'string') { config = JSON.parse(config); }

    // Stocker les données avant de créer le DOM
    const steps = Array.isArray(config.steps) ? config.steps : [];
    const stepNames = Array.isArray(config.stepNames) ? config.stepNames : [];
    const colors = Array.isArray(config.colors) ? config.colors : [];
    const current = Number(config.current || 1);

    // Nettoyer l'ancienne stepbar si elle existe
    const oldStepbar = document.getElementById('stepbar');
    if (oldStepbar) oldStepbar.remove();

    // Créer le container
    const stepbar = document.createElement('div');
    stepbar.id = 'stepbar';
    stepbar.className = 'mathadata-stepbar__container';
    stepbar.style.setProperty('--stepbar-right', `${config.position_right_px || 16}px`);

    const backdrop = document.createElement('div');
    backdrop.className = 'mathadata-stepbar__backdrop';
    const itemsWrap = document.createElement('div');
    itemsWrap.className = 'mathadata-stepbar__items';
    backdrop.appendChild(itemsWrap);

    // Stocker les données dans stepbar (doit être fait avant createItem)
    window.mathadata.stepbar.steps = steps;
    window.mathadata.stepbar.stepNames = stepNames;
    window.mathadata.stepbar.colors = colors;
    window.mathadata.stepbar.current = current;
    window.mathadata.stepbar.container = stepbar;
    window.mathadata.stepbar.rootId = 'stepbar';

    // Créer les items
    steps.forEach((label, i) => {
      const idx = i + 1;
      const item = window.mathadata.stepbar.createItem(idx, label);
      itemsWrap.appendChild(item);
    });

    // Créer le bouton de collapse
    const collapseButton = document.createElement('div');
    collapseButton.className = 'mathadata-stepbar__collapseButton';
    collapseButton.style.cssText = `
      position: absolute;
      left: -30px;
      top: 50%;
      transform: translateY(-50%);
      width: 30px;
      height: 60px;
      background: rgba(255, 255, 255, 0.9);
      border-radius: 8px 0 0 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: -2px 0 8px rgba(0,0,0,0.1);
      transition: background 0.2s;
    `;

    const buttonIcon = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    buttonIcon.setAttribute('width', '12');
    buttonIcon.setAttribute('height', '12');
    buttonIcon.setAttribute('viewBox', '0 0 320 512');
    buttonIcon.setAttribute('fill', '#666');
    buttonIcon.style.transition = 'transform 0.3s ease';
    buttonIcon.innerHTML = '<path d="M310.6 233.4c12.5 12.5 12.5 32.8 0 45.3l-192 192c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L242.7 256 73.4 86.6c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0l192 192z"/>';

    collapseButton.appendChild(buttonIcon);
    collapseButton.addEventListener('click', () => window.mathadata.stepbar.toggle());

    collapseButton.addEventListener('mouseenter', () => {
      collapseButton.style.background = 'rgba(255, 255, 255, 1)';
    });
    collapseButton.addEventListener('mouseleave', () => {
      collapseButton.style.background = 'rgba(255, 255, 255, 0.9)';
    });

    backdrop.appendChild(collapseButton);
    stepbar.appendChild(backdrop);
    document.body.appendChild(stepbar);

    // Ajouter la transition au container
    stepbar.style.transition = 'right 0.3s ease';
    stepbar.style.right = '0px';

    // Stocker la référence au bouton
    window.mathadata.stepbar.collapseButton = collapseButton;

    // Initialiser l'observateur pour les divs invisibles
    window.mathadata.stepbar.initObserver();
  };

  mathadata.fireSideCannons = function() {
    var end = Date.now() + 1000; // 2 seconds
    var colors = ["#a786ff", "#fd8bbc", "#eca184", "#f8deb1"];
    function frame() {
        if (Date.now() > end) return;
        if (window.confetti) {
            window.confetti({
                particleCount: 4,
                angle: 60,
                origin: { x: 0, y: 0.5 },
                colors: colors,
            });
            window.confetti({
                particleCount: 4,
                angle: 120,
                origin: { x: 1, y: 0.5 },
                colors: colors,
            });
        }
        setTimeout(frame, 50);
    }
    requestAnimationFrame(frame);
  }
''')


def init_stepbar():
    """
    Initialise une barre latérale verticale d'étapes (fixée à droite).

    - steps: liste de labels d'étapes (ex: ["Étape 1", "Étape 2", ...])
    - current_step_index: 1-indexé, étape active au chargement
    - position_right_px: marge droite (px)
    """
    steps_labels = []
    steps_keys = []
    steps_colors = []
    for key, step in steps.items():
        steps_labels.append(step['name'])
        steps_keys.append(key)
        steps_colors.append(step['color'])

    params = {
        'steps': steps_labels,
        'stepNames': steps_keys,
        'colors': steps_colors,
        'current': 1,
        'position_right_px': 24,
    }
    run_js(f"window.mathadata.init_stepbar(`{json.dumps(params, cls=NpEncoder)}`)")


def toggle_stepbar():
    """Affiche ou cache la stepbar."""
    run_js("window.mathadata && window.mathadata.stepbar && window.mathadata.stepbar.toggle()")


init_stepbar()
# Change la couleur au démarage
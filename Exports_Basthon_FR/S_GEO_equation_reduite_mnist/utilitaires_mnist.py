# coding=utf-8

###### VERSION 2/7 #####

# Import des librairies utilisées dans le notebook
import pickle
from io import BytesIO
from zipfile import ZipFile

import matplotlib.patches as mpatches

import utilitaires_common as common
from utilitaires_common import *

# from scipy.spatial import Voronoi, voronoi_plot_2d

strings = {
    "dataname": {
        "nom": "image",
        "pluriel": "images",
        "feminin": True,
        "contraction": True,
    },
    "classes": [
        {
            "nom": "2",
            "pluriel": "2",
            "feminin": False,
            "contraction": False,
            "nom_alt": "image de 2",
            "pluriel_alt": "images de 2",
            "feminin_alt": True,
            "contraction_alt": True,
        },
        {
            "nom": "7",
            "pluriel": "7",
            "feminin": False,
            "contraction": False,
            "nom_alt": "image de 7",
            "pluriel_alt": "images de 7",
            "feminin_alt": True,
            "contraction_alt": True,
        }
    ],
    "r_petite_caracteristique": 7,
    "r_grande_caracteristique": 2,
    "train_size": "6 000",
    "objectif_score_droite": 10,
    "objectif_score_droite_custom": 8,
    "pt_retourprobleme": "(40; 20)"
}

### --- IMPORT DES DONNÉES ---
# Téléchargement et extraction des inputs contenus dans l'archive zip

inputs_zip_urls = [f"{files_url}/mnist.zip",
                   "https://raw.githubusercontent.com/mathadata/data_challenges_lycee/main/mnist.zip"]
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
    pretty_print_error(
        f"Erreur du chargement des données. Essayez de recharger la page, relancer le notebook en suivant les instructions SOS ou sur un autre navigateur internet.")
    pretty_print_error(
        f'Si le problème persiste, transmettez cette erreur à votre professeur pour qu\'il nous l\'envoie :\n\n{error}')
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

print(f"données chargées")

# VERSION 2/7 : 
classes = [2, 7]

N = len(d_train)

d_train_par_population = [d_train[r_train == k] for k in classes]

d = d_train[10, :, :].copy()
d2 = d_train[2, :, :].copy()

# 10 images en tableau format 3x3
d_train_simple = [
    [[195, 195, 195], [0, 190, 1], [190, 0, 0]],
    [[255, 76, 132], [18, 240, 59], [101, 37, 200]],
    [[64, 128, 249], [92, 11, 175], [220, 58, 143]],
    [[7, 99, 187], [142, 33, 255], [68, 214, 121]],
    [[85, 19, 236], [194, 129, 72], [150, 223, 41]],
    [[203, 54, 116], [25, 177, 88], [244, 61, 135]],
    [[39, 212, 97], [181, 14, 250], [73, 160, 226]],
    [[109, 42, 190], [5, 167, 251], [138, 81, 219]],
    [[120, 231, 53], [146, 27, 205], [66, 156, 248]],
    [[178, 36, 141], [222, 63, 109], [8, 199, 255]]
]
d_simple = np.array(d_train_simple[2])
# Noms de variables pour la question 'fainéant
chat = 'chat'
Chat = 'chat'
cat = 'cat'


class Mnist(common.Challenge):
    def __init__(self):
        super().__init__()
        self.id = 'mnist'
        self.strings = strings
        self.d_train = d_train
        self.r_train = r_train
        self.d = d
        self.d2 = d2
        self.classes = classes
        self.r_petite_caracteristique = strings['r_petite_caracteristique']
        self.r_grande_caracteristique = strings['r_grande_caracteristique']
        self.custom_zone = None
        self.custom_zones = None

        # STATS
        # Moyenne histogramme
        self.carac_explanation = f"C'est la bonne réponse ! Les images de 7 ont souvent moins de pixels blancs que les images de 2. C'est pourquoi leur caractéristique, leur moyenne, est souvent plus petite."

        # GEO
        # Tracer 10 points
        self.dataset_10_points = d_train[20:30]
        self.labels_10_points = r_train[20:30]
        self.droite_10_points = {
            'm': 0.5,
            'p': 20,
        }

        # GEO
        # Tracer 20 points
        self.dataset_20_points = d_train[10:30]
        self.labels_20_points = r_train[10:30]
        self.droite_20_points = {
            'm': 0.5,
            'p': 20,
        }

        # Droite produit scalaire
        # Pour les versions question_normal
        self.M_retourprobleme = (40, 20)

        # Objectif de score avec les 2 caracs de référence pour passer à la suite
        self.objectif_score_droite = strings['objectif_score_droite']
        self.objectif_score_droite_custom = strings['objectif_score_droite_custom']

        # Centroides
        self.dataset_10_centroides = d_train[30:40]
        self.labels_10_centroides = r_train[30:40]

    def affichage_dix(self, d=d_train, a=None, b=None, zones=None, y=r_train, n=10, axes=False):
        if zones is None:
            zones = []
        global r_train
        fig, ax = plt.subplots(1, n, figsize=(figw_full, figw_full / n + 1))

        # Cachez les axes des subplots
        for j in range(n):
            if axes:
                ax[j].set_xticks(np.arange(0, 28, 5))
                ax[j].set_yticks(np.arange(0, 28, 5))
                ax[j].xaxis.tick_top()
                ax[j].tick_params(axis='both', which='major', labelsize=7)

            else:
                ax[j].axis('off')
            imshow(ax[j], d[j])
            if y is not None:
                ax[j].set_title('$r = $' + str(y[j]), y=0, size=16, pad=-20)

            outline_selected(ax[j], a, b)
            for zone in zones:
                outline_selected(ax[j], zone[0], zone[1], zoneName=zone[2])

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.2 if axes else 0.05, hspace=0)
        plt.show()
        plt.close()

    def display_custom_selection(self, id):
        run_js(f'''
            setTimeout(() => {{
                window.mathadata.setup_zone_selection('{id}', '{json.dumps(self.d_train[4:8].tolist())}')
                const gif = document.createElement('img')
                gif.id = 'selection-gif'
                gif.src = '{files_url}/gif_explication_zone.gif'
                gif.alt = 'Cliquez sur les images pour sélectionner une zone'
                gif.style = 'width: 25%; height: auto; margin-inline: auto;'
                const container = document.getElementById('{id}-selection')
                container.appendChild(gif)
            }}, 500)
        ''')

    def display_custom_selection_2d(self, id):
        run_js(f'''
            setTimeout(() => {{
                window.mathadata.setup_zone_selection_2d('{id}', '{json.dumps(self.d_train[4:8].tolist())}')
                const gif = document.createElement('img')
                gif.id = 'selection-gif'
                gif.src = '{files_url}/gif_explication_zone.gif'
                gif.alt = 'Cliquez sur les images pour sélectionner une zone'
                gif.style = 'width: 25%; height: auto; margin-inline: auto;'
                const container = document.getElementById('{id}')
                container.appendChild(gif)
            }}, 500)
        ''')

    def caracteristique(self, d):
        return moyenne(d)

    def caracteristique_custom(self, d):
        if self.custom_zone is None:
            return 0

        return moyenne_zone(d, self.custom_zone[0], self.custom_zone[1])

    def deux_caracteristiques(self, d):
        zone_1 = [(0, 0), (13, 27)]
        zone_2 = [(14, 0), (27, 27)]

        k1 = moyenne_zone(d, zone_1[0], zone_1[1])
        k2 = moyenne_zone(d, zone_2[0], zone_2[1])
        return (k1, k2)

    def deux_caracteristiques_custom(self, d):
        if self.custom_zones is None:
            return (0, 0)

        if self.custom_zones[0] is None or self.custom_zones[0][0] is None or self.custom_zones[0][0][0] is None:
            k1 = 0
        else:
            k1 = moyenne_zone(d, self.custom_zones[0][0], self.custom_zones[0][1])

        if self.custom_zones[1] is None or self.custom_zones[1][0] is None or self.custom_zones[1][0][0] is None:
            k2 = 0
        else:
            k2 = moyenne_zone(d, self.custom_zones[1][0], self.custom_zones[1][1])

        return (k1, k2)

    def cb_custom_score(self, score):
        if score < 0.1 or (has_variable('superuser') and get_variable('superuser') == True):
            return True
        elif score < 0.3:
            pretty_print_success(
                "Bravo, ta zone choisie pour calculer la moyenne est meilleure que faire la moyenne de toute l'image. Améliore encore ta zone pour faire moins de 10% d'erreur et passer à la suite.")
        else:
            print_error(
                "Modifie ta zone pour faire moins de 10% d'erreur et passer à la suite. Cherche une zone où l'image est différente si c'est un 2 ou un 7")

        return False

    def affichage_2_cara(self, A1=None, B1=None, A2=None, B2=None, displayPoints=False, titre1="", titre2=""):
        """Fonction qui affiche deux images côte à côté, avec un rectangle rouge délimité par les points A et B
            A : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
            B : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
        Si displayPoints est True, les points A et B sont affichés
        """

        image1 = self.d
        image2 = self.d2

        fig, ax = plt.subplots(1, 2, figsize=(figw_full, figw_full / 2))
        imshow(ax[0], image1)
        ax[0].set_title(titre1)
        ax[0].set_xticks(np.arange(0, 28, 5))
        ax[0].xaxis.tick_top()
        outline_selected(ax[0], A1, B1, displayPoints, nameA='A1', nameB='B1', color='red')
        outline_selected(ax[0], A2, B2, displayPoints, nameA='A2', nameB='B2', color='C0')

        imshow(ax[1], image2)
        ax[1].set_title(titre2)
        ax[1].set_xticks(np.arange(0, 28, 5))
        ax[1].xaxis.tick_top()
        outline_selected(ax[1], A1, B1, displayPoints, nameA='A1', nameB='B1', color='red')
        outline_selected(ax[1], A2, B2, displayPoints, nameA='A2', nameB='B2', color='C0')

        plt.show()
        plt.close()

        self.custom_zones = [(A1, B1), (A2, B2)]
        return


init_challenge(Mnist())


def imshow(ax, image, **kwargs):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255, extent=[0, 28, 28, 0], **kwargs)


def outline_selected(ax, a=None, b=None, displayPoints=False, zoneName=None, zoneNamePos='right', nameA='A', nameB='B',
                     color='red'):
    if a is not None:
        if b is None:
            b = a

        numero_ligne_debut = min(a[0], b[0])
        numero_ligne_fin = max(a[0], b[0])
        numero_colonne_debut = min(a[1], b[1])
        numero_colonne_fin = max(a[1], b[1])

        if numero_ligne_debut < 0 or numero_colonne_debut < 0 or numero_ligne_fin > 27 or numero_colonne_fin > 27:
            print_error("Les valeurs des index doivent être compris entre 0 et 27.")
            return

        padding = 0  # adjust this value as needed
        rect = mpatches.Rectangle((numero_colonne_debut + padding, numero_ligne_debut + padding),
                                  numero_colonne_fin - numero_colonne_debut + 1 - 2 * padding,
                                  numero_ligne_fin - numero_ligne_debut + 1 - 2 * padding,
                                  fill=False, edgecolor=color, lw=2)
        ax.add_patch(rect)

        if displayPoints and a != b:
            if a[1] <= b[1]:
                ha_a = 'right'
                ha_b = 'left'
                x_a = a[1]
                x_b = b[1] + 1
            else:
                ha_a = 'left'
                ha_b = 'right'
                x_a = a[1] + 1
                x_b = b[1]

            if a[0] <= b[0]:
                va_a = 'bottom'
                va_b = 'top'
                y_a = a[0]
                y_b = b[0] + 1
            else:
                va_a = 'top'
                va_b = 'bottom'
                y_a = a[0] + 1
                y_b = b[0]

            ax.text(x_a, y_a, nameA, ha=ha_a, va=va_a, color=color, fontsize=12, fontweight='bold')
            ax.text(x_b, y_b, nameB, ha=ha_b, va=va_b, color=color, fontsize=12, fontweight='bold')

        if zoneName is not None:
            if zoneNamePos == 'right':
                col = 30
                ha = 'left'
            elif zoneNamePos == 'left':
                col = -3
                ha = 'right'
            elif zoneNamePos == 'center':
                col = (numero_colonne_debut + numero_colonne_fin) / 2
                ha = 'center'
            else:
                raise ValueError("zoneNamePos doit valoir 'right', 'left' ou 'center'")

            ax.text(col,
                    (numero_ligne_debut + numero_ligne_fin) / 2,
                    zoneName, ha=ha, va='center', color=color, fontsize=12, fontweight='bold')


# Affichage d'une image
def affichage(image, a=None, b=None, displayPoints=False, titre=""):
    """Fonction qui affiche une image avec un rectangle rouge délimité par les points a et b
        a : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        b : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
    Si displayPoints est True, les points A et B sont affichés
    """
    if image.min().min() < 0 or image.max().max() > 255:
        print_error("fonction affichage : Les valeurs des pixels de l'image doivent être compris entre 0 et 255.")
        return

    fig, ax = plt.subplots(figsize=(figw_full / 2, figw_full / 2))
    imshow(ax, image)
    ax.set_title(titre)
    ax.set_xticks(np.arange(0, 28, 5))
    ax.xaxis.tick_top()
    ax.set_title(titre)
    ax.xaxis.set_label_position('top')
    outline_selected(ax, a, b, displayPoints)

    plt.show()
    plt.close()


def affichage_2_geo(display_k=False):
    zone_1 = [(0, 0), (13, 27)]
    zone_2 = [(14, 0), (27, 27)]
    deux_caracteristiques = common.challenge.deux_caracteristiques

    images = np.array([common.challenge.d2, common.challenge.d])
    fig, ax = plt.subplots(1, len(images), figsize=(figw_full * 0.8, figw_full * 0.8 / 2))
    c_train = np.array([np.array(deux_caracteristiques(d)) for d in images])

    for i in range(len(images)):
        imshow(ax[i], images[i])
        k = c_train[i]
        if i == 0:
            zoneNamePos = 'left'
        elif i == len(images) - 1:
            zoneNamePos = 'right'
        else:
            zoneNamePos = 'center'

        k1 = f'{k[0]:.2f}' if display_k else '?'
        k2 = f'{k[1]:.2f}' if display_k else '?'
        outline_selected(ax[i], zone_1[0], zone_1[1], zoneName=f'$x = {k1}$', zoneNamePos=zoneNamePos)
        outline_selected(ax[i], zone_2[0], zone_2[1], zoneName=f'$y = {k2}$', zoneNamePos=zoneNamePos)

    fig.subplots_adjust(left=0.15, right=0.85, top=1, bottom=0, wspace=0.2, hspace=0)

    plt.show()
    plt.close()

    reversed = c_train[::-1]

    if not display_k:
        df = pd.DataFrame(
            {'$x$': reversed[:, 0], '$y$': reversed[:, 1], '$r$': ['$r_1$ = 2 ou 7 ?', '$r_2$ = 2 ou 7 ?']})
        df.index += 1
        display(df)
        return


# Affichage d'une image sous forme de tableau
pd.set_option('display.max_rows', 28)
pd.set_option('display.max_columns', 28)
# Set the width of each column
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('colheader_justify', 'center')


def affichage_tableau(image, a=None, b=None):
    """Fonction qui affiche une image sous forme de tableau avec un rectangle rouge délimité par les points a et b
        a : tuple (ligne, colonne) représentant le coin en haut à gauche du rectangle
        b : tuple (ligne, colonne) représentant le coin en bas à droite du rectangle
        Si a et b ne sont pas fournis, la fonction affiche l'image sans rectangle rouge."""
    df = pd.DataFrame(image)
    if a is not None:
        if b is not None:
            numero_ligne_debut = min(a[0], b[0])
            numero_ligne_fin = max(a[0], b[0])
            numero_colonne_debut = min(a[1], b[1])
            numero_colonne_fin = max(a[1], b[1])
        else:
            numero_ligne_debut = a[0]
            numero_ligne_fin = a[0]
            numero_colonne_debut = a[1]
            numero_colonne_fin = a[1]

        if numero_ligne_debut < 0 or numero_colonne_debut < 0 or numero_ligne_fin > 27 or numero_colonne_fin > 27:
            print_error("Les valeurs des index doivent être compris entre 0 et 27.")
            return

        slice_ = (slice(numero_ligne_debut, numero_ligne_fin), slice(numero_colonne_debut, numero_colonne_fin))
        try:
            s = df.style.set_properties(**{'background-color': 'red'}, subset=slice_)
            display(s)
            return
        except:
            if b is not None:
                # return df.iloc[max(0, numero_ligne_debut - 1):min(len(image), numero_ligne_fin+2), max(0, numero_colonne_debut - 1):min(len(image), numero_colonne_fin+2)]
                display(df.iloc[numero_ligne_debut:numero_ligne_fin + 1, numero_colonne_debut:numero_colonne_fin + 1])
                return
    display(df)
    return


def affichage_image_et_pixels(image=None):
    if image is None:
        image = d

    id = uuid.uuid4().hex

    # Données pour affichage "dessin"
    params = {
        'd': image,
    }

    run_js(
        f"""mathadata.add_observer('{id}', () => mathadata.affichage_image_et_pixels('{id}', '{json.dumps(params, cls=NpEncoder)}'));""")

    # Création du conteneur parent avec 2 div
    display(
        HTML(f'''
        <div id="{id}" style="display:flex; gap: 20px; margin: 20px 0; justify-content: center;">
            <div id="{id}-data" style="width: 48%; aspect-ratio: 1;">
                <!-- Zone affichage -->
            </div>
            <div id="{id}-tab" style="width: 48%; aspect-ratio: 1; overflow: auto; background: white; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <!-- Zone tab -->
            </div>
        </div>
        ''')
    )


image_3x3 = np.array([
    [100, 200, 250],
    [0, 200, 0],
    [150, 0, 0]
])


def afficher_image_3x3():
    affichage_image_et_pixels(image_3x3)


# Moyenne
def moyenne(liste):
    arr = np.array(liste)
    return np.mean(arr)


def moyenne_zone(arr, a, b):
    if a is None or b is None:
        print_error("Les points A et B ne sont pas définis.")
        return 0

    numero_ligne_debut = min(a[0], b[0])
    numero_ligne_fin = max(a[0], b[0])
    numero_colonne_debut = min(a[1], b[1])
    numero_colonne_fin = max(a[1], b[1])
    return np.mean(arr[numero_ligne_debut:numero_ligne_fin + 1, numero_colonne_debut:numero_colonne_fin + 1])


def check_pixel_coordinates(coords, errors):
    if not isinstance(coords, tuple) or len(coords) != 2:
        errors.append("Les coordonnées du pixel doivent être entre parenthèses séparées par une virgule. Exemple :")
        errors.append("(0, 0)")
        return False
    if coords[0] is Ellipsis or coords[1] is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not isinstance(coords[0], int) or not isinstance(coords[1], int):
        errors.append("Les coordonnées du pixel doivent être des nombres entiers.")
        return False
    if coords[0] < 0 or coords[0] > 27 or coords[1] < 0 or coords[1] > 27:
        errors.append("Les coordonnées du pixel doivent être entre 0 et 27.")
        return False
    return True


def update_selected(A, B):
    common.challenge.custom_zone = (A, B)


def update_selected_2(A1, B1, A2, B2):
    common.challenge.custom_zones = [(A1, B1), (A2, B2)]


def affichage_trois(data=None):
    id = uuid.uuid4().hex

    # Création du conteneur parent avec 2 div
    display(
        HTML(f'''
        <div id="{id}-parent" style="display:flex; gap: 20px; margin: 20px 0;">
            <div id="{id}-affichage" style="flex: 1; height: 100px;">
                <!-- Zone affichage -->
            </div>
            <div id="{id}-tab" style="flex: 1; overflow:auto; max-height:400px;">
                <!-- Zone tab -->
            </div>
        </div>
        ''')
    )

    # Par défaut, prendre une image 3x3 factice
    if data is None:
        data = np.array(d_train_simple[0])

    # Données pour affichage "dessin"
    data_json = json.dumps(data.tolist())
    params_json = json.dumps({'with_selection': False})

    # --- Conversion du tableau en HTML pour affichage dans {id}-tab ---
    df = pd.DataFrame(data)
    table_html = df.to_html(border=1, justify="center")  # tableau HTML natif (inspiré de affichage_tableau)
    table_html_js = json.dumps(table_html)  # safe pour injection JS

    run_js(f"""
        // Récupération des conteneurs
        const affichageContainer = document.getElementById("{id}-affichage");
        const tabContainer = document.getElementById("{id}-tab");

        // Appel fonction mathadata côté JS pour affichage graphique
        const params = {params_json};
        const data = {data_json};
        mathadata.affichage("{id}-affichage", data, params);

        // Injection du tableau HTML côté Python dans la div tab
        tabContainer.innerHTML = {table_html_js};
    """)


def qcm_carac_moyenne():
    id = "qcm-carac-moyenne-question-html"

    image_indexes = [958, 2015]
    carac1 = common.challenge.caracteristique(common.challenge.d_train[image_indexes[0]])
    carac2 = common.challenge.caracteristique(common.challenge.d_train[image_indexes[1]])
    diff = carac1 - carac2
    
    if abs(diff) < 5: # Au cas où l'ordre du dataset change, correction générique
        answer_index = 3  # On ne peut pas savoir
    elif diff > 0:
        # Image A a plus grande caractéristique
        answer_index = 0 
    else:
        # Image B a plus grande caractéristique
        answer_index = 1
        # Même caractéristique

    run_js(f"""mathadata.add_observer('{id}', () => {{
        console.log('{json.dumps(common.challenge.d, cls=NpEncoder)}');
        mathadata.affichage("{id}-qcm-image-A", {json.dumps(common.challenge.d_train[image_indexes[0]], cls=NpEncoder)});
        mathadata.affichage("{id}-qcm-image-B", {json.dumps(common.challenge.d_train[image_indexes[1]], cls=NpEncoder)});
    }})""")

    html = f"""
        <div id="{id}" style="display: flex; flex-direction: column; gap: 2rem; justify-content: center; align-items: center;">
            <p>Parmi les deux images suivantes, laquelle a la plus grande caractéristique ?</p>
            <div style="display: flex; gap: 4rem; justify-content: center; align-items: center;">
                <div style="text-align: center;">
                    <h4 style="margin-bottom: 1rem;">Image A</h4>
                    <div id="{id}-qcm-image-A" style="width: 250px; aspect-ratio: 1;"></div>
                </div>
                <div style="text-align: center;">
                    <h4 style="margin-bottom: 1rem;">Image B</h4>
                    <div id="{id}-qcm-image-B" style="width: 250px; aspect-ratio: 1;"></div>
                </div>
            </div>
        </div> 
    """
    
    create_qcm({
        'question_html': html,
        'choices': ["L'image A", "L'image B", "Les deux images ont la même caractéristique", "On ne peut pas savoir"],
        'answer_index': answer_index,
        'multiline': True,
        'success': "En effet, l'image B a plus de pixels blancs donc une moyenne plus haute.",
    })

# JS

styles = '''
    .image-table {
        width: 100%;
        height: 100%;
        margin: 0 !important;
        border-width: 0;
        border-collapse: collapse;
        border-spacing: 0;
        table-layout: fixed;
    }

    .image-table tr {
        height: calc(100% / 28);
    }
    
    .image-table td, .image-table th {
        width: min(calc(1 / 28 * 100%), 30px);
        aspect-ratio: 1;
        padding: 0;
        margin: 0;
        border: none;
        position: relative;
    }

    .image-table-numbers td, .image-table-numbers th {
        text-align: center;
        vertical-align: middle;
        overflow: hidden;
        border: 1px solid #e8e8e8;
        box-sizing: border-box;
        padding: 0;
        line-height: 1;
        white-space: nowrap;
    }

    .image-table-numbers th {
        background: #f5f5f5;
        font-weight: bold;
        border: 1px solid #ccc !important;
    }

    .image-table-selection td:hover {
        outline: 1px solid red;
    }
    
    .custom-images {
        display: flex;
        gap: 1rem;
        width: 100%;

    }
'''

run_js(f"""
    let style = document.getElementById('mathadata-style-mnist');
    if (style !== null) {{
        style.remove();
    }}

    style = document.createElement('style');
    style.id = 'mathadata-style-mnist';
    style.innerHTML = `{styles}`;
    document.head.appendChild(style);
""")

run_js(r"""
let isSelecting = false;

let zone1 = false;
let zone2 = false;
let zones = [[null, null], [null, null]];

document.addEventListener('mouseup', () => {
    isSelecting = false;
});

function toggleCellSelection(cell, select, zone) {
    if (select) {
        const div = document.createElement('div');
        div.style.width = '100%';
        div.style.height = '100%';
        div.style.backgroundColor = zone === 2 ? 'blue' : 'red';
        div.style.opacity = '0.5';
        div.style.position = 'absolute';
        div.style.top = '0';
        div.style.left = '0';
        div.style.pointerEvents = 'none';
        cell.appendChild(div);
    } else {
        cell.innerHTML = '';
    }
}

function clearSelection() {
    window.mathadata.image_tables.forEach(table => {
        const selectedCells = table.querySelectorAll('td');
        selectedCells.forEach(cell => toggleCellSelection(cell, false));
    });
}

let exec_python = null;

function getZoneIndexes(zone) {
    const [start, end] = zone;
    if (start === null || end === null) {
        return [null, null, null, null];
    }
    
    const startRowIndex = start.parentElement.rowIndex;
    const startColIndex = start.cellIndex;
    const endRowIndex = end.parentElement.rowIndex;
    const endColIndex = end.cellIndex;

    const minRowIndex = Math.min(startRowIndex, endRowIndex);
    const maxRowIndex = Math.max(startRowIndex, endRowIndex);
    const minColIndex = Math.min(startColIndex, endColIndex);
    const maxColIndex = Math.max(startColIndex, endColIndex);
    
    return [minRowIndex, minColIndex, maxRowIndex, maxColIndex];
}

function selectCells() {
    const [minRowIndex, minColIndex, maxRowIndex, maxColIndex] = getZoneIndexes(zones[0]);
    const [minRowIndex2, minColIndex2, maxRowIndex2, maxColIndex2] = getZoneIndexes(zones[1]);

    clearSelection();

    if (minRowIndex !== null) {
        for (const table of window.mathadata.image_tables) {
            for (let i = minRowIndex; i <= maxRowIndex; i++) {
                for (let j = minColIndex; j <= maxColIndex; j++) {
                    const cell = table.rows[i].cells[j];
                    toggleCellSelection(cell, true, 1);
                }
            }
        }
    }

    if ((zone1 || zone2) && minRowIndex2 !== null) {
        for (const table of window.mathadata.image_tables) {
            for (let i = minRowIndex2; i <= maxRowIndex2; i++) {
                for (let j = minColIndex2; j <= maxColIndex2; j++) {
                    const cell = table.rows[i].cells[j];
                    toggleCellSelection(cell, true, 2);
                }
            }
        }
    }
    
    if (minRowIndex === null || ((zone1 || zone2) && minRowIndex2 === null)) {
        return
    }

    if (exec_python) {
        clearTimeout(exec_python);
    }

    exec_python = setTimeout(() => {
        document.getElementById('selection-gif')?.remove();
        let python;
        if (zone1 || zone2) {
            python = `update_selected_2((${minRowIndex}, ${minColIndex}), (${maxRowIndex}, ${maxColIndex}), (${minRowIndex2}, ${minColIndex2}), (${maxRowIndex2}, ${maxColIndex2}))`;
        } else {
            python = `update_selected((${minRowIndex}, ${minColIndex}), (${maxRowIndex}, ${maxColIndex}))`;
        }

        window.mathadata.run_python(python, () => {
            window.mathadata.on_custom_update();
        });
    }, 200);
}


window.mathadata.affichage = (id, matrix, params) => {
    if (params) {
        if (typeof params === 'string') {
            params = JSON.parse(params)
        }
    } else {
        params = {}
    }
    
    const { with_selection } = params;

    const container = document.getElementById(id)
    container.style.aspectRatio = '1'
    container.innerHTML = '';
    
    const table_id = id + '-table'
    const table = document.createElement('table')
    container.appendChild(table)
    table.id = table_id
    table.classList.add('image-table')
    table.innerHTML = '';
    for (let row = 0; row < matrix.length; row++) {
        const tr = document.createElement('tr');
        for (let col = 0; col < matrix[row].length; col++) {
            const td = document.createElement('td');
            const value = matrix[row][col];
            td.style.backgroundColor = `rgb(${value}, ${value}, ${value})`;
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
    
    if (with_selection) {
        table.classList.add('image-table-selection');

        table.addEventListener('mousedown', (event) => {
            const target = event.target;
            if (target.tagName === 'TD') {
                isSelecting = true;
                if (zone2) {
                    zones[1][0] = target;
                    zones[1][1] = target;
                } else {
                    zones[0][0] = target;
                    zones[0][1] = target;
                }
                selectCells();
            }
        });

        table.addEventListener('mouseover', (event) => {
            if (isSelecting) {
                const target = event.target;
                if (target.tagName === 'TD') {
                    if (zone2) {
                        zones[1][1] = target;
                    } else {
                        zones[0][1] = target;
                    }
                    selectCells();
                }
            }
        });

        table.addEventListener('mouseup', () => {
            isSelecting = false;
        });

        return table
    }
}

// affiche l'image avec les valeurs des pixels
function affichage_tableau(id, matrix) {
    const container = document.getElementById(id)
    container.style.overflow = 'hidden'
    container.style.display = 'flex'
    container.style.alignItems = 'center'
    container.style.justifyContent = 'center'
    container.innerHTML = '';

    const table = document.createElement('table')
    table.classList.add('image-table')
    table.classList.add('image-table-numbers')
    table.style.width = '100%'
    table.style.height = '100%'

    // Calculer la taille de police en fonction du nombre de lignes/colonnes
    // Plus il y a de cellules, plus la police doit être petite
    const numCols = matrix[0].length + 1; // +1 pour la colonne header
    const numRows = matrix.length + 1; // +1 pour la ligne header
    const cellSize = Math.min(100 / numCols, 100 / numRows); // pourcentage
    // Taille de police proportionnelle à la taille de cellule
    const fontSize = Math.max(6, Math.min(14, cellSize * 0.4)); // entre 6px et 14px
    table.style.fontSize = fontSize + 'px';

    container.appendChild(table)

    const thead = document.createElement('thead');
    const corner_th = document.createElement('th');
    corner_th.innerText = 'n°';
    thead.appendChild(corner_th);
    for (let col = 0; col < matrix[0].length; col++) {
        const th = document.createElement('th');
        th.innerText = col + 1;
        thead.appendChild(th);
    }

    table.appendChild(thead);
    for (let row = 0; row < matrix.length; row++) {
        const tr = document.createElement('tr');
        const th = document.createElement('th');
        th.innerText = row + 1;
        tr.appendChild(th);

        for (let col = 0; col < matrix[row].length; col++) {
            const td = document.createElement('td');
            const value = matrix[row][col];
            td.innerText = value;
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }
}
    

window.mathadata.affichage_image_et_pixels = (id, params) => {
    if (typeof params === 'string') {
        params = JSON.parse(params)
    }
    
    const { d } = params;
    mathadata.affichage(id + '-data', d);
    affichage_tableau(id + '-tab', d);
}

window.mathadata.animation_moyenne = async (id, data) => {
    const container = document.getElementById(id);
    const table = container.querySelector('table');
    const cells = Array.from(table.querySelectorAll('td'));

    if (!data) {
        data = await mathadata.run_python_async('get_data(0)');
    }

    // Rendre le container position relative pour l'overlay
    const originalPosition = container.style.position;
    container.style.position = 'relative';

    const totalPixels = 28 * 28;
    const flatData = data.flat();

    // Phase 0 : Afficher toutes les valeurs d'un coup en fadeIn
    cells.forEach((cell, i) => {
        const value = flatData[i];
        cell.style.fontSize = '6px';
        cell.style.fontWeight = 'bold';
        cell.style.textAlign = 'center';
        cell.style.color = value > 127 ? 'rgba(0, 0, 0, 0)' : 'rgba(255, 255, 255, 0)';
        cell.style.transition = 'color 1000ms ease';
        cell.innerText = value;
    });

    requestAnimationFrame(() => {
        cells.forEach((cell, i) => {
            const value = flatData[i];
            cell.style.color = value > 127 ? 'rgba(0, 0, 0, 1)' : 'rgba(255, 255, 255, 1)';
        });
    });

    // Pause pour voir toutes les valeurs
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Créer le conteneur d'affichage avec somme à gauche et count à droite
    const displayContainer = document.createElement('div');
    displayContainer.style.position = 'absolute';
    displayContainer.style.left = '50%';
    displayContainer.style.top = '50%';
    displayContainer.style.transform = 'translate(-50%, -50%)';
    displayContainer.style.display = 'flex';
    displayContainer.style.alignItems = 'center';
    displayContainer.style.gap = '20px';
    displayContainer.style.fontSize = '28px';
    displayContainer.style.fontWeight = 'bold';
    displayContainer.style.color = '#fff';
    displayContainer.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    displayContainer.style.padding = '20px 30px';
    displayContainer.style.borderRadius = '12px';
    displayContainer.style.zIndex = '1000';
    container.appendChild(displayContainer);

    // Somme à gauche
    const sumDisplay = document.createElement('div');
    sumDisplay.textContent = '0';
    displayContainer.appendChild(sumDisplay);

    // Zone pour le symbole diviser (initialement vide)
    const dividerDisplay = document.createElement('div');
    dividerDisplay.style.fontSize = '32px';
    dividerDisplay.style.opacity = '0';
    displayContainer.appendChild(dividerDisplay);

    // Compteur à droite
    const countDisplay = document.createElement('div');
    countDisplay.textContent = '0';
    displayContainer.appendChild(countDisplay);

    let sum = 0;
    let count = 0;
    const initialDelay = 15;  // Début: 15ms par pixel
    const minDelay = 0.5;     // Fin: 0.5ms par pixel
    const accelerationFactor = 1.08;

    // Phase 1 : Parcours rapide avec "+" qui tombent
    for (let i = 0; i < cells.length; i++) {
        const cell = cells[i];
        const value = flatData[i];

        // Calculer le délai avec accélération
        const delay = Math.max(minDelay, initialDelay / Math.pow(accelerationFactor, i / 50));

        // Créer le symbole "+" qui pop
        const plusSign = document.createElement('div');
        plusSign.textContent = '+';
        plusSign.style.position = 'absolute';
        plusSign.style.fontSize = '14px';
        plusSign.style.fontWeight = 'bold';
        plusSign.style.color = '#ff4444';
        plusSign.style.pointerEvents = 'none';
        plusSign.style.top = '50%';
        plusSign.style.left = '50%';
        plusSign.style.transform = 'translate(-50%, -50%) scale(0)';
        plusSign.style.transition = 'transform 200ms ease-out, opacity 400ms ease-out';
        plusSign.style.zIndex = '10';
        plusSign.style.opacity = '1';
        cell.appendChild(plusSign);

        // Animation de pop
        requestAnimationFrame(() => {
            plusSign.style.transform = 'translate(-50%, -50%) scale(1.5)';
        });

        // Ajouter à la somme et au compteur
        sum += value;
        count++;
        sumDisplay.textContent = Math.round(sum);
        countDisplay.textContent = count;

        // Faire disparaître en fondu après 200ms
        setTimeout(() => {
            plusSign.style.opacity = '0';
            plusSign.style.transform = 'translate(-50%, -50%) scale(1)';
            // Supprimer après le fade out
            setTimeout(() => plusSign.remove(), 400);
        }, 200);

        await new Promise(resolve => setTimeout(resolve, delay));
    }

    // Phase 2 : Faire apparaître le symbole diviser
    await new Promise(resolve => setTimeout(resolve, 400));
    dividerDisplay.textContent = '÷';
    dividerDisplay.style.transition = 'opacity 400ms ease';
    dividerDisplay.style.opacity = '1';

    await new Promise(resolve => setTimeout(resolve, 800));

    // Phase 3 : Transformer en résultat final
    const moyenne = sum / totalPixels;
    const moyenneColor = Math.round(moyenne);

    // Masquer le layout somme/count et afficher le résultat
    displayContainer.style.backgroundColor = `rgb(${moyenneColor}, ${moyenneColor}, ${moyenneColor})`;
    displayContainer.style.color = moyenneColor > 127 ? '#000' : '#fff';
    displayContainer.style.flexDirection = 'column';
    displayContainer.style.gap = '10px';
    sumDisplay.textContent = 'Moyenne';
    sumDisplay.style.fontSize = '18px';
    dividerDisplay.style.display = 'none';
    countDisplay.textContent = moyenne.toFixed(1);
    countDisplay.style.fontSize = '32px';

    await new Promise(resolve => setTimeout(resolve, 1200));

    // Nettoyer
    displayContainer.remove();

    // Enlever toutes les valeurs affichées
    cells.forEach(cell => {
        cell.innerText = '';
    });
    container.style.position = originalPosition;
}

window.mathadata.setup_zone_selection = (id, matrixes) => {
    matrixes = JSON.parse(matrixes)
    
    const container = document.getElementById(id)
    container.innerHTML = `
        <div id="${id}-selection">
            <div class="custom-images" id="${id}-images"></div>
        </div>
    `;
    
    const imagesContainer = document.getElementById(`${id}-images`)
    window.mathadata.image_tables = [];
    for (let i = 0; i < matrixes.length; i++) {
        const table = document.createElement('div')
        table.style.flex = '1'
        table.style.aspectRatio = '1'
        table.id = `${id}-image-${i}`
        imagesContainer.appendChild(table)
        window.mathadata.image_tables.push(window.mathadata.affichage(`${id}-image-${i}`, matrixes[i], {with_selection: true}))
    }

    selectCells(); // To show zones if reexecute
}

window.mathadata.setup_zone_selection_2d = (id, matrixes) => {
    matrixes = JSON.parse(matrixes)
    
    const container = document.getElementById(id)
    container.innerHTML = `
         <style>
            /* Styles pour le conteneur du Toggle */
            .slider-container {
            margin-top: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            }

            /* Cache le checkbox */
            .toggle-checkbox {
            display: none;
            }

            /* Label du bouton de toggle */
            .toggle-label {
            position: relative;
            width: 120px;
            height: 40px;
            background-color: red;
            border-radius: 20px;
            cursor: pointer;
            transition: background-color 0.3s;
            }

            /* Cercle du toggle */
            .toggle-circle {
            position: absolute;
            top: 3px;
            left: 3px;
            width: 34px;
            height: 34px;
            background-color: white;
            border-radius: 50%;
            transition: transform 0.3s ease;
            }

            /* Lorsque le toggle est activé */
            .toggle-checkbox:checked + .toggle-label .toggle-circle {
            transform: translateX(80px);
            }

            .toggle-checkbox:checked + .toggle-label {
            background-color: blue; /* Couleur verte quand activé */
            }

            /* Styles pour les zones */
            .custom-images {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
            gap: 2rem;
            }

            .toggle-checkbox:checked ~ .custom-images .zone1-content {
            display: none;
            }

            .toggle-checkbox:checked ~ .custom-images .zone2-content {
            display: block;
            }

            /* Styles des boutons de sélection de zone */
            .zone-toggle-button {
            padding: 10px 20px;
            background-color: transparent;
            border: none;
            font-size: 1rem;
            cursor: pointer;
            transition: color 0.3s;
            }

            /* Bouton actif */
            .zone-toggle-button.active {
            font-weight: bold;
            color: '#007BFF';
            }
        </style>

         <div id="toggle-container">
    <div>
       <div style="display: flex; justify-content: center;">
       <label style="margin-top:3rem; margin-right:1rem" id="${id}-labelzone1">Zone x</label>
        <!-- Le Toggle Glissable -->
        <div class="slider-container">
            <input type="checkbox" id="toggle" class="toggle-checkbox" />
            <label for="toggle" class="toggle-label">
            <span class="toggle-circle"></span>
            </label>
        </div>
        <label style="margin-top:3rem; margin-left:1rem" id="${id}-labelzone2">Zone y</label>
        </div>
        <!-- Conteneur des images ou contenu spécifique à chaque zone -->
        <div class="custom-images" id="${id}-images">
            <!-- Par défaut, Zone 1 est affichée -->
        </div>

        
        </div>
    </div>
    `;

    const zone1Button = document.getElementById(`${id}-zone1`)
    const zone2Button = document.getElementById(`${id}-zone2`)
    const toggle = document.getElementById('toggle')

    zone1 = true
    zone2 = false
    toggle.addEventListener('change', () => {
        zone1 = !toggle.checked;
        zone2 = toggle.checked;
    })
    
    const imagesContainer = document.getElementById(`${id}-images`)
    window.mathadata.image_tables = [];
    for (let i = 0; i < matrixes.length; i++) {
        const table = document.createElement('div')
        table.style.flex = '1'
        table.style.aspectRatio = '1'
        table.id = `${id}-image-${i}`
        imagesContainer.appendChild(table)
        window.mathadata.image_tables.push(window.mathadata.affichage(`${id}-image-${i}`, matrixes[i], {with_selection: true}))
    }

    selectCells(); // To show zones if reexecute
}
       
// Fonction pour vérifier si le tableau est entièrement noir (tout effacé)
function isBoardBlack(board) {
    let boardBlack = true;
    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board[i].length; j++) {
            if (board[i][j] !== 0) {
                boardBlack = false;
            }
        }
        if (!boardBlack) break;
    }
    return boardBlack;
};

// Fonction pour afficher le dessin (fonction principale)
window.mathadata.interface_data_gen = (id) => {
    const container = document.getElementById(id);
    const table_id = id + '-table';

    container.innerHTML = `
        <div>
            <table id="${table_id}" class="image-table" style="height: 300px; aspect-ratio: 1;"></table>
        </div>
        <div style="display: flex; flex-direction: column; gap: 1.5rem; align-items: center; justify-content: center; height: 100%">
            <p style="font-size: 13px; color: #666; text-align: center; max-width: 280px; margin: 0; line-height: 1.5;">
                Cliquez sur la zone noir et déplacez la souris pour dessiner
            </p>

            <!-- Switch Dessin/Gomme -->
            <div style="display: flex; flex-direction: column; gap: 0.5rem; align-items: center;">
                <label style="font-size: 14px; font-weight: 500; color: #333;">Mode</label>
                <div style="display: flex; align-items: center; gap: 0.5rem; background: #f5f5f5; padding: 0.5rem 0.75rem; border-radius: 8px;">
                    <span id="${id}-mode-label-draw" style="font-size: 13px; font-weight: 600; color: #667eea;">Dessin</span>
                    <label style="position: relative; display: inline-block; width: 50px; height: 26px; margin: 0;">
                        <input type="checkbox" id="${id}-gomme" style="opacity: 0; width: 0; height: 0;">
                        <span id="${id}-switch-bg" style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #667eea; transition: 0.3s; border-radius: 26px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);"></span>
                        <span id="${id}-switch-btn" style="position: absolute; content: ''; height: 20px; width: 20px; left: 3px; bottom: 3px; background-color: white; transition: 0.3s; border-radius: 50%; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></span>
                    </label>
                    <span id="${id}-mode-label-erase" style="font-size: 13px; color: #999;">Gomme</span>
                </div>
            </div>

            <!-- Bouton Effacer tout -->
            <button id="${id}-reset" class="mathadata-button" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white; padding: 12px 24px; font-size: 14px;">
                Tout effacer
            </button>
        </div>
    `;

    const table = document.getElementById(table_id);
    const toggleGomme = document.getElementById(`${id}-gomme`);
    const resetButton = document.getElementById(`${id}-reset`);
    const modeLabelDraw = document.getElementById(`${id}-mode-label-draw`);
    const modeLabelErase = document.getElementById(`${id}-mode-label-erase`);
    const switchBg = document.getElementById(`${id}-switch-bg`);
    const switchBtn = document.getElementById(`${id}-switch-btn`);

    // Style du container (restauré de l'original)
    container.style = 'display:flex; justify-content: space-around; align-items: center; gap: 2rem;';

    // Gérer le changement de mode visuel
    toggleGomme.addEventListener('change', () => {
        if (toggleGomme.checked) {
            // Mode gomme
            modeLabelDraw.style.color = '#999';
            modeLabelDraw.style.fontWeight = '400';
            modeLabelErase.style.color = '#ff6b6b';
            modeLabelErase.style.fontWeight = '600';
            // Changer la couleur du switch
            switchBg.style.backgroundColor = '#ff6b6b';
            switchBtn.style.left = '27px';
        } else {
            // Mode dessin
            modeLabelDraw.style.color = '#667eea';
            modeLabelDraw.style.fontWeight = '600';
            modeLabelErase.style.color = '#999';
            modeLabelErase.style.fontWeight = '400';
            // Changer la couleur du switch
            switchBg.style.backgroundColor = '#667eea';
            switchBtn.style.left = '3px';
        }
    });

    // Création de la table avec des cellules initialement noires
    for (let row = 0; row < 28; row++) {
        const tr = document.createElement('tr');
        for (let col = 0; col < 28; col++) {
            const td = document.createElement('td');
            td.style.backgroundColor = 'rgb(0, 0, 0)';
            tr.appendChild(td);
        }
        table.appendChild(tr);
    }

    let isSelecting = false; // Variable pour savoir si on est en train de sélectionner avec la souris

    // Fonction pour changer la couleur d'une cellule et de ses voisins
    function changerCouleur(cell) {
        // Fonction pour obtenir la couleur de fond actuelle d'une cellule
        function getCellBackgroundColor(cell) {
            return window.getComputedStyle(cell).backgroundColor;
        }

        // Fonction pour éclaircir une couleur (ajouter une nuance de blanc)
        function lightenColor(color, increment) {
            const rgb = color.match(/\d+/g).map(Number); // Récupérer les valeurs RGB
            const maxWhite = 255;
            // Appliquer l'incrément pour éclaircir la couleur sans dépasser 255
            let newColor = rgb.map(value => Math.min(value + increment, maxWhite)); 
            return `rgb(${newColor[0]}, ${newColor[1]}, ${newColor[2]})`;
        }

        // Vérifier si la gomme est activée
        const isGommeActive = toggleGomme.checked;

        // Si la gomme est activée, on réinitialise la couleur à blanc pour la cellule
        if (isGommeActive) {
            cell.style.backgroundColor = 'rgb(0, 0, 0)'; // Effacer la cellule en blanc
        } else {
            cell.style.backgroundColor = 'rgb(255, 255, 255)'; // Colorier la cellule en blanc total
       // Récupérer l'index de la ligne et de la colonne de la cellule cible
        const rowIndex = cell.parentNode.rowIndex;
        const colIndex = cell.cellIndex;

        // Définir les voisins (haut, bas, gauche, droite, et diagonales)
        const voisins = [
            [-1, 0], [1, 0], [0, -1], [0, 1], // Voisins directs
            [-1, -1], [-1, 1], [1, -1], [1, 1] // Voisins diagonaux
        ];

        // Parcourir les voisins et appliquer la nuance de blanc
        voisins.forEach(([dx, dy]) => {
            const voisinRow = rowIndex + dx;
            const voisinCol = colIndex + dy;

            // Vérifier si les indices des voisins sont valides
            if (voisinRow >= 0 && voisinRow < cell.parentNode.parentNode.rows.length &&
                voisinCol >= 0 && voisinCol < cell.parentNode.children.length) {
                
                const voisinCell = cell.parentNode.parentNode.rows[voisinRow].cells[voisinCol];

                // Récupérer la couleur de fond actuelle du voisin
                let voisinColor = getCellBackgroundColor(voisinCell);

                // Éclaircir la couleur du voisin si nécessaire
                voisinCell.style.backgroundColor = lightenColor(voisinColor, 40); // Éclaircir de 40 (ajuster si nécessaire)
            }
        });
        }

        
    }

    // Attacher les événements à la table
    table.addEventListener('mousedown', (event) => {
        const target = event.target;
        if (target.tagName === 'TD') {
            isSelecting = true;
            changerCouleur(target);
        }
    });

    table.addEventListener('mouseover', (event) => {
        if (isSelecting) {
            const target = event.target;
            if (target.tagName === 'TD') {
                changerCouleur(target);
            }
        }
    });

    table.addEventListener('mouseup', () => {
        isSelecting = false; // Arrêter de dessiner
    });

    resetButton.addEventListener('click', () => {
        const cells = document.getElementById(id).querySelectorAll('td');
        cells.forEach(cell => cell.style.backgroundColor = 'rgb(0, 0, 0)');
    });

    // renvoie une fonction qui permet d'obtenir la donnée générée
    return function() {
        let dessin = [];

        for (let i = 0; i < 28; i++) {
            let row = [];
            for (let j = 0; j < 28; j++) {
                const cell = table.rows[i].cells[j];
                const cellBackgroundColor = cell.style.backgroundColor.match(/\d+/g).map(Number)[0];
                row.push(cellBackgroundColor);
            }
            dessin.push(row);
        }

        return dessin;
    };
};
""")


### Validation cellules ###

def validate_algo(errors, answers):
    r_prediction = get_estimations(d_test)
    # check that all answers are 2 or 7
    if not np.all(np.logical_or(r_prediction == 2, r_prediction == 7)):
        non_matching = np.where(np.logical_and(r_prediction != 2, r_prediction != 7))
        first_non_matching = r_prediction[non_matching][0]
        if first_non_matching is Ellipsis:
            errors.append("Votre n'avez pas remplacé les ... dans la fonction algorithme")
        else:
            errors.append("Votre algorithme a répondu autre chose que 2 ou 7 : " + str(first_non_matching))
        return False

    return True


def question_pixel():
    value = 210
    create_qcm({
        'question_html': f"A votre avis, quelle est la valeur associée à ce pixel ?<br/><div style='width: 60px; height: 60px; background-color: rgb({value}, {value}, {value}); margin: 10px auto; border: 1px solid black;'></div>",
        'choices': ['10', '110', str(value)],
        'answer': str(value)
    })


validation_execution_affichage = MathadataValidate(success="")

validation_question_nb_pixels = MathadataValidateVariables({
    'nb_pixels': d.shape[0] * d.shape[1],
}, tips=[{
    'seconds': 10,
    'tip': "Pour connaître le nombre de pixels dans une image, il faut multiplier le nombre de lignes par le nombre de colonnes."
}, {
    'seconds': 30,
    'trials': 2,
    'tip': "Les images ont 28 lignes et 28 colonnes."
}],
    success="Bravo, les images ont 28 lignes et 28 colonnes, donc elles contiennent 28 X 28 = 784 pixels.")

validation_execution_afficher_image_3x3 = MathadataValidate(success="")
validation_question_moyenne_3x3 = MathadataValidateVariables({
    'x': np.mean(image_3x3),
}, tips=[{
    'tip': "Pour calculer la moyenne des pixels, il faut additionner les valeurs de tous les pixels puis diviser par le nombre total de pixels."
}, {
    'seconds': 60,
    'trials': 2,
    'tip': f"La somme des valeurs des pixels de l'image est {np.sum(image_3x3)}."
}],
    success=f"Bravo, la caractéristique vaut {np.sum(image_3x3)} / {np.size(image_3x3)} = {np.mean(image_3x3)}")

validation_question_moyenne = MathadataValidateVariables({'moyenne_zone_4pixels': np.mean(d[14:16, 15:17])},
                                                         success="Bravo, la moyenne vaut en effet (142 + 154 + 0 + 0) / 4 = 74")
validation_question_moyenne_triple = MathadataValidateVariables({'moyenne_pixels': np.mean(d_simple)},
                                                                success="Bravo, la moyenne vaut en effet ((64 + 128 + 249) + (92 + 11 + 175) + (220 + 58 + 143)) / 9 = 142")


# Geometrie

def on_success_2_caracteristiques(answers):
    affichage_2_geo(display_k=True)


validation_execution_2_caracteristiques = MathadataValidate(success="")
validation_question_2_caracteristiques = MathadataValidateVariables({
    'r1': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r1 n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    },
    'r2': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "r2 n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    }
},
    success="C'est la bonne réponse. L'image de 7 a presque la même moyenne sur la moitié haute et la moitié basse. L'image de 2 a une moyenne plus élevée sur la moitié basse car il y a plus de pixels blancs.",
    on_success=on_success_2_caracteristiques)

validation_execution_def_caracteristiques_ripou = MathadataValidate(success="")

# Notebook histogramme

validation_question_hist_2 = MathadataValidateVariables({
    'nombre_2': {
        'value': 46,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 350
                },
                'else': "nombre_2 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 2 avec une caractéristique entre 20 et 22 ?"
            }
        ]
    },
    'nombre_7': {
        'value': 238,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 350
                },
                'else': "nombre_7 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 7 avec une caractéristique entre 20 et 22 ?"
            }
        ]
    },
})

validation_question_hist_3 = MathadataValidateVariables({
    'nombre_2_inf_16': {
        'value': 13,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 350,
                },
                'else': "nombre_2_inf_16 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 2 avec une caractéristique inférieure à 16 ?"
            }
        ]
    },
    'nombre_7_inf_16': {
        'value': 62,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 400
                },
                'else': "nombre_7_inf_16 n'a pas la bonne valeur. As-tu bien remplacé les ... par le nombre d'image de 7 avec une caractéristique inférieure à 16 ?"
            }
        ]
    },
}, tips=[
    {
        'seconds': 30,
        'tip': 'En passant la souris sur les barres de l\'histogramme, tu peux voir le nombre d\'images qui ont une caractéristique dans l\'intervalle correspondant.'
    }
])

validation_affichage_trois = MathadataValidate(success="")
# la fonction validation_question_affichage_trois doit verifier si l'utilisateur a bien saisi la bonne moyenne
validation_question_affichage_trois = MathadataValidateVariables({
    'moyenne_pixels_simple': {
        'value': ((195 * 3) + (0 + 190 + 1) + (190 + 0 + 0)) / 9,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 255
                },
                'else': "moyenne_pixels_simple n'a pas la bonne valeur. As-tu bien remplacé les ... par la moyenne des pixels de l'image ?"
            }
        ]
    }
})

# Stockage valeur zones custom proposés
A_2 = (7, 2)  # <- coordonnées du point A1
B_2 = (9, 25)  # <- coordonnées du point B1
A_1 = (14, 2)  # <- coordonnées du point A2
B_1 = (23, 10)  # <- coordonnées du point B2


def affichage_zones_custom_2_cara(A1, B1, A2, B2):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)


def caracteristique_ligne_correction(d):
    # Sélection de la ligne 12 et 14
    ligne1 = d[11, :]
    ligne2 = d[13, :]
    # Calcul de la somme des pixels de cette ligne
    somme_pixels = 0
    nombre_pixels = 0
    for pixel in ligne1:
        somme_pixels += pixel
        nombre_pixels += 1
    for pixel in ligne2:
        somme_pixels += pixel
        nombre_pixels += 1
    # Calcul de la moyenne
    moyenne_des_pixels = somme_pixels / nombre_pixels

    return moyenne_des_pixels


def on_success_affichage_histograme(answers):
    if has_variable('afficher_histogramme'):
        get_variable('afficher_histogramme')(legend=True, caracteristique=get_variable('caracteristique'))


validation_caracteristique_ligne_et_affichage = MathadataValidateFunction(
    'caracteristique',
    test_set=lambda: common.challenge.d_train[0:100],
    expected=lambda: [caracteristique_ligne_correction(d) for d in common.challenge.d_train[0:100]],
    on_success=on_success_affichage_histograme
)

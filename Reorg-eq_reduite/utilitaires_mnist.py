# coding=utf-8

###### VERSION 2/7 #####

# Import des librairies utilisées dans le notebook
import pickle
from io import BytesIO
from zipfile import ZipFile

import matplotlib.patches as mpatches

import utilitaires_common as common
from utilitaires_common import *

strings = {
    "dataname": {
        "nom": "image",
        "pluriel": "images",
        "feminin": True,
        "contraction": True,
    },
    "labelname": "Chiffre",
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
    "train_size": "5 036",
    "train_size_approx": "5 000",
    "objectif_score_droite": 12,
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
    d_train_full = pickle.load(f)
    d_train = []
    d_train_test = []
    if len(d_train_full) > 0:
        split_index = int(2/3 * len(d_train_full))
        d_train = d_train_full[:split_index]
        d_train_test = d_train_full[split_index:]
    

with open(r_train_path, 'rb') as f:
    r_train_full = pickle.load(f)
    r_train = []
    r_train_test = []
    if len(r_train_full) > 0:
        split_index = int(2/3 * len(r_train_full))
        r_train = r_train_full[:split_index]
        r_train_test = r_train_full[split_index:]

pretty_print_success('Le chargement de la base de données s\'est déroulé à merveille ! Tu peux poursuivre en déroulant la page.')

# VERSION 2/7 : 
classes = [2, 7]

N = len(d_train)

d_train_par_population = [d_train[r_train == k] for k in classes]

d = d_train[10, :, :].copy()
d2 = d_train[2, :, :].copy()

# Modifier l'image 23 qui sert de référence pour les exo
d_train[23][6, 7] = 0
d_train[23][6, 8] = 0
d_train[23][7, 8] = 255
d_train[23][7, 6] = 0
d_train[23][7, 9] = 0
d_train[23][9, 8] = 0

# Modifier l'image 9
d_train[9][6, 10] = 240
d_train[9][20, 21] = 200

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

# Variables pour validation histogrammes zones (permettre A et B sans guillemets)
A = 'A'
B = 'B'


class Mnist(common.Challenge):
    def __init__(self):
        super().__init__()
        self.id = 'mnist'
        self.strings = strings
        self.d_train = d_train
        self.r_train = r_train
        self.r_train_test = r_train_test
        self.d_train_test = d_train_test
        self.d = d
        self.d2 = d2
        self.classes = classes
        self.r_petite_caracteristique = strings['r_petite_caracteristique']
        self.r_grande_caracteristique = strings['r_grande_caracteristique']
        self.custom_zone = None
        self.custom_zones = None

        # Imges de ref pour les question calcul caracteristique
        self.ids_images_ref = (9, 23, 24)

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

        # GEO zones de références pour les deux caractéristiques
        self.zone_1_ref = [(4, 4), (11, 11)]
        self.zone_2_ref = [(16, 18), (22, 25)]

        self.zone_1_exo = [(6,8), (9,10)]
        self.zone_2_exo = [(19, 18), (21, 22)]

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
            mathadata.add_observer('{id}', () => {{
                window.mathadata.setup_zone_selection('{id}', '{json.dumps(self.d_train[4:8].tolist())}')
                const gif = document.createElement('img')
                gif.id = 'selection-gif'
                gif.src = '{files_url}/gif_explication_zone.gif'
                gif.alt = 'Cliquez sur les images pour sélectionner une zone'
                gif.style = 'width: 25%; height: auto; margin-inline: auto;'
                const container = document.getElementById('{id}-selection')
                container.appendChild(gif)
            }});
        ''')

    def display_custom_selection_2d(self, id):
        run_js(f'''
            mathadata.add_observer('{id}', () => {{
                window.mathadata.setup_zone_selection_2d('{id}', '{json.dumps(self.d_train[4:8].tolist())}')
                const gif = document.createElement('img')
                gif.id = 'selection-gif'
                gif.src = '{files_url}/gif_explication_zone.gif'
                gif.alt = 'Cliquez sur les images pour sélectionner une zone'
                gif.style = 'width: 25%; height: auto; margin-inline: auto;'
                const container = document.getElementById('{id}')
                container.appendChild(gif)
            }});
        ''')

    def caracteristique(self, d):
        return moyenne(d)

    def caracteristique_custom(self, d):
        if self.custom_zone is None:
            return 0

        return moyenne_zone(d, self.custom_zone[0], self.custom_zone[1])

    def deux_caracteristiques(self, d):
        # Nouvelle caractéristique de réference : deux rectangles suffisamment petits pour qu'on puisse compte les pixels blancs à l'intérieur, et qui donnent un erreur min de 10% environ
        zone_1 = self.zone_1_ref
        zone_2 = self.zone_2_ref

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
        <div id="{id}" style="display:flex; gap: 20px; margin: 20px 0; width: 100%; align-items: flex-start;">
            <div id="{id}-data" style="flex: 35 1 0; aspect-ratio: 1; min-width: 0;">
                <!-- Zone affichage -->
            </div>
            <div id="{id}-tab" style="flex: 65 1 0; aspect-ratio: 1; min-width: 0; overflow: hidden; background: white; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
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


def afficher_deux_exemples_zones(
    id1=None,
    id2=None,
    id3=None,
    show_points=False,
    save_png=False,
    png_scale=2,
    save_png_path=None,
    zones=None,
    show_zones=True,
    show_zone_x=True,
    show_zone_y=True,
    crop=0,
    seuil_texte=130,
    fontsize=8,
    grid_gap_px=1,
    max_width_px=440,
    show_values=True,
    zone_colors=("red", "blue"),
    zone_lw=6,
    point_names=("A", "B"),
    legends=True,
    legend_offset_px=8,
    legend_font_px=24,
    legend_background=False,
    troisieme=False,
    **kwargs,
):
    """
    Affiche 2 images MNIST côte à côte (JS), exactement comme :
        afficher_image_pixels(seuillage_0_200_250(d_train[i]), show_zones=True, zones=zones)

    - Seuillage : 0/200/250 (mêmes seuils que dans le notebook export)
    - Affichage : niveaux de gris + valeur dans chaque pixel
    - Zones : rectangles (rouge puis bleu)
    - Option (show_points=True) : sous chaque image, afficher le point caractéristique (moyenne sur les zones)

    Par défaut, si le challenge expose `ids_images_ref`, alors ces images sont utilisées en priorité.
    Si `troisieme=True`, une 3e image de référence (si disponible) est affichée.
    """

    # Alias (compat) : ancien paramètre `show_point`
    if "show_point" in kwargs:
        show_points = kwargs.pop("show_point")

    # Choix d'images par défaut (priorité) : `ids_images_ref` défini dans le challenge.
    ids_images_ref = getattr(common.challenge, "ids_images_ref", None)
    if ids_images_ref is not None:
        try:
            ids_images_ref = tuple(ids_images_ref)
        except Exception:
            ids_images_ref = None

    if ids_images_ref is not None and (id1 is None or id2 is None):
        if id1 is None and len(ids_images_ref) >= 1:
            id1 = int(ids_images_ref[0])
        if id2 is None and len(ids_images_ref) >= 2:
            id2 = int(ids_images_ref[1])
        if id3 is None and len(ids_images_ref) >= 3:
            id3 = int(ids_images_ref[2])

    # Fallback : 1 image de chaque classe (si possible)
    if id1 is None or id2 is None:
        try:
            r = common.challenge.r_train
            classes = common.challenge.classes
            if id1 is None:
                id1 = int(np.where(r == classes[0])[0][0])
            if id2 is None:
                id2 = int(np.where(r == classes[1])[0][0])
        except Exception:
            if id1 is None:
                id1 = 0
            if id2 is None:
                id2 = 1

    zone_1_ref = getattr(common.challenge, "zone_1_exo", None)
    zone_2_ref = getattr(common.challenge, "zone_2_exo", None)

    if zones is None:
        zones = [zone_1_ref, zone_2_ref]
    else:
        # Autoriser l'utilisateur à ne donner qu'UNE zone : on complète avec la zone de référence manquante.
        try:
            zones = list(zones)
        except Exception:
            zones = [zones]

        if len(zones) == 1:
            zones = [zones[0], zone_2_ref]
        elif len(zones) >= 2:
            zones = [zones[0], zones[1]]

    display_id = uuid.uuid4().hex

    ids = [int(id1), int(id2)]
    if troisieme:
        if id3 is not None:
            ids.append(int(id3))
        elif ids_images_ref is not None and len(ids_images_ref) >= 3:
            ids.append(int(ids_images_ref[2]))

    params = {
        "ids": ids,
        # Compat : JS accepte `show_points` et `show_point`
        "show_points": bool(show_points),
        "show_point": bool(show_points),
        "save_png": bool(save_png),
        "png_scale": float(png_scale) if png_scale is not None else 2.0,
        "save_png_path": save_png_path,
        "show_zones": bool(show_zones),
        "show_zone_x": bool(show_zone_x),
        "show_zone_y": bool(show_zone_y),
        "zones": zones,
        "crop": int(crop) if crop is not None else 0,
        "legends": (
            [
                {"left": ["x<sub>2</sub>", "y<sub>2</sub>"], "right": [None, None]},
                {"left": [None, None], "right": ["x<sub>7</sub>", "y<sub>7</sub>"]},
            ]
            if legends is True
            else legends
        ),
        "legend_offset_px": float(legend_offset_px),
        "legend_font_px": float(legend_font_px),
        "legend_background": bool(legend_background),
        "seuil_texte": float(seuil_texte),
        "fontsize": float(fontsize),
        "grid_gap_px": int(grid_gap_px),
        "max_width_px": float(max_width_px) if max_width_px is not None else None,
        "show_values": bool(show_values),
        "zone_colors": list(zone_colors) if zone_colors is not None else ["red", "blue"],
        "zone_lw": float(zone_lw),
        "point_names": list(point_names) if point_names is not None else ["A", "B"],
    }

    run_js(
        f"mathadata.add_observer('{display_id}', () => window.mathadata.afficher_deux_exemples_zones('{display_id}', '{json.dumps(params, cls=NpEncoder)}'))"
    )
    display(HTML(f'<div id="{display_id}" class="mathadata-mnist-exemples-zones"></div>'))


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


def _mathadata_save_png_base64(path, b64_png):
    """
    Sauvegarde un PNG fourni en base64 dans le système de fichiers côté Python.

    Remarques :
    - Cela ne marche que si l'environnement Python a accès à un vrai FS (Jupyter local, etc.).
    - Dans certains environnements (Basthon/JupyterLite/Capytale), l'écriture peut être limitée ou éphémère.
    """
    if not path:
        return
    try:
        import base64
        import os

        data = base64.b64decode(b64_png.encode("ascii"))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        print(f"PNG enregistré : {path}")
    except Exception as e:
        print_error(f"Impossible d'enregistrer le PNG dans '{path}' : {e}")


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

def qcm_1_cara_mnist():
    create_qcm({
        "question": "Quelle image a la plus grande caractéristique x ?",
        "choices": [
            "l'image de 2",
            "l'image de 7",
        ],
        "answer": "l'image de 7",
    })

def qcm_contre_exemple_mnist():
    create_qcm({
        "question": "Pour la 3eme image ci-dessus, le rectangle rouge contient ...",
        "choices": [
            "moins de pixels blancs que sur l'image de 7, la caractéristique sera donc plus petite.",
            "plus de pixels blancs que sur l'image de 7, la caractéristique sera donc plus grande.",
        ],
        "answer": "plus de pixels blancs que sur l'image de 7, la caractéristique sera donc plus grande.",
    })

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

    .mathadata-mnist-exemples-zones {
        width: 100%;
    }

    .mathadata-mnist-exemples-zones-grid {
        display: flex;
        gap: 2rem;
        justify-content: center;
        align-items: flex-start;
        flex-wrap: wrap;
        width: 100%;
        margin: 1rem 0;
    }

    /* Variante : forcer toutes les images sur une seule ligne (utile si on en affiche 3) */
    .mathadata-mnist-exemples-zones-grid.mathadata-mnist-exemples-zones-grid--one-line {
        flex-wrap: nowrap;
        justify-content: flex-start;
        overflow-x: auto;
        overflow-y: hidden;
        -webkit-overflow-scrolling: touch;
    }

    .mathadata-mnist-exemples-zones-grid.mathadata-mnist-exemples-zones-grid--one-line .mathadata-mnist-exemples-zones-item {
        max-width: none;
        min-width: 0;
    }

    .mathadata-mnist-exemples-zones-item {
        display: flex;
        flex-direction: column;
        gap: 0.6rem;
        align-items: center;
        max-width: 420px;
    }

    .mathadata-mnist-exemples-zones-title {
        font-family: sans-serif;
        font-weight: 600;
        text-align: center;
    }

    .mathadata-mnist-exemples-zones-point {
        font-family: sans-serif;
        text-align: center;
        font-size: 0.95rem;
        line-height: 1.2;
    }

    .mathadata-mnist-pixelgrid {
        display: grid;
        gap: 1px;
        padding: 1px;
        background: #b3b3b3;
        border-radius: 10px;
        position: relative;
        user-select: none;
    }

    .mathadata-mnist-pixelcell {
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        line-height: 1;
        white-space: nowrap;
        overflow: hidden;
    }

	    .mathadata-mnist-zonerect {
	        position: absolute;
	        pointer-events: none;
	        box-sizing: border-box;
	        border-radius: 6px;
	    }

	    .mathadata-mnist-zonelabel {
	        position: absolute;
	        pointer-events: none;
	        font-family: sans-serif;
	        font-weight: 700;
	        white-space: nowrap;
	        line-height: 1;
	        padding: 2px 6px;
	        border-radius: 999px;
	        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
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

const zone1CheckpointId = 'mnist_zone_custom';
const zone2CheckpointId = 'mnist_zones_custom_2d';

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

            mathadata.checkpoints.save(zone2CheckpointId, {
                zone1: {
                    A: [minRowIndex, minColIndex],
                    B: [maxRowIndex, maxColIndex]
                },
                zone2: {
                    A: [minRowIndex2, minColIndex2],
                    B: [maxRowIndex2, maxColIndex2]
                }
            });
        } else {
            python = `update_selected((${minRowIndex}, ${minColIndex}), (${maxRowIndex}, ${maxColIndex}))`;

            mathadata.checkpoints.save(zone1CheckpointId, {
                A: [minRowIndex, minColIndex],
                B: [maxRowIndex, maxColIndex]
            });
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
    const container = document.getElementById(id);
    container.style.overflow = 'hidden';
    container.style.display = 'flex';
    container.style.alignItems = 'center';
    container.style.justifyContent = 'center';
    container.style.padding = '0';
    container.innerHTML = '';

    const table = document.createElement('table');
    table.style.borderCollapse = 'collapse';
    table.style.tableLayout = 'fixed';
    table.style.width = '100%';
    table.style.height = '100%';
    table.style.fontFamily = 'monospace';

    // Dimensionnement typographique
    const numCols = matrix[0].length + 1;
    const numRows = matrix.length + 1;
    const rect = container.getBoundingClientRect();
    const cellPx = Math.min(rect.width / numCols, rect.height / numRows);
    const hasRect = Number.isFinite(cellPx) && cellPx > 0;
    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

    const bodyFontPx = hasRect ? clamp(Math.floor(cellPx * 0.45), 6, 11) : 6;
    const headerFontPx = hasRect ? clamp(Math.floor(cellPx * 0.65), bodyFontPx, 14) : 8;
    table.style.fontSize = bodyFontPx + 'px';

    container.appendChild(table);

    /* ===== EN-TÊTE COLONNES ===== */
    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');

    const corner = document.createElement('th');
    corner.innerText = '';
    styleHeader(corner, headerFontPx);
    headRow.appendChild(corner);

    for (let col = 0; col < matrix[0].length; col++) {
        const th = document.createElement('th');
        th.innerText = col+1;
        styleHeader(th, headerFontPx);
        headRow.appendChild(th);
    }

    thead.appendChild(headRow);
    table.appendChild(thead);

    /* ===== CORPS ===== */
    const tbody = document.createElement('tbody');

    for (let row = 0; row < matrix.length; row++) {
        const tr = document.createElement('tr');

        // En-tête ligne
        const th = document.createElement('th');
        th.innerText = row+1;
        styleHeader(th, headerFontPx);
        tr.appendChild(th);

        for (let col = 0; col < matrix[row].length; col++) {
            const td = document.createElement('td');
            td.innerText = matrix[row][col];
            td.style.border = '1px solid #ccc';
            td.style.textAlign = 'center';
            td.style.padding = '0';
            td.style.lineHeight = '1';
            td.style.whiteSpace = 'nowrap';
            td.style.overflow = 'hidden';
            tr.appendChild(td);
        }

        tbody.appendChild(tr);
    }

    table.appendChild(tbody);
}

window.mathadata.affichage_tableau = affichage_tableau;

/* ===== STYLE STRUCTUREL ===== */
function styleHeader(cell, headerFontPx) {
    cell.style.backgroundColor = '#f0f0f0';
    cell.style.border = '1px solid #000';
    cell.style.fontWeight = 'bold';
    cell.style.textAlign = 'center';
    cell.style.padding = '0';
    cell.style.lineHeight = '1';
    cell.style.whiteSpace = 'nowrap';
    cell.style.fontSize = (headerFontPx !== undefined ? headerFontPx + 'px' : '1.4em');
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

    // Charger la zone sauvegardée si elle existe

    const savedZone = window.mathadata.checkpoints.get(zone1CheckpointId);
    if (savedZone && savedZone.A?.every(v => v !== null) && savedZone.B?.every(v => v !== null)) {
        // Restaurer la zone sauvegardée
        const firstTable = window.mathadata.image_tables[0];
        if (firstTable && firstTable.rows) {
            const startCell = firstTable.rows[savedZone.A[0]]?.cells[savedZone.A[1]];
            const endCell = firstTable.rows[savedZone.B[0]]?.cells[savedZone.B[1]];
            if (startCell && endCell) {
                zones[0] = [startCell, endCell];
            }
        }
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

    // Charger les zones sauvegardées si elles existent
    const savedZones = window.mathadata.checkpoints.get(zone2CheckpointId);
    if (savedZones) {
        const firstTable = window.mathadata.image_tables[0];
        if (firstTable && firstTable.rows) {

            if (savedZones.zone1.A?.every(v => v !== null) && savedZones.zone1.B?.every(v => v !== null)) {
                // Restaurer zone 1
                const start1 = firstTable.rows[savedZones.zone1.A[0]]?.cells[savedZones.zone1.A[1]];
                const end1 = firstTable.rows[savedZones.zone1.B[0]]?.cells[savedZones.zone1.B[1]];
                if (start1 && end1) {
                    zones[0] = [start1, end1];
                }
            }
            
            if (savedZones.zone2.A?.every(v => v !== null) && savedZones.zone2.B?.every(v => v !== null)) {
                // Restaurer zone 2
                const start2 = firstTable.rows[savedZones.zone2.A[0]]?.cells[savedZones.zone2.A[1]];
                const end2 = firstTable.rows[savedZones.zone2.B[0]]?.cells[savedZones.zone2.B[1]];
                if (start2 && end2) {
                    zones[1] = [start2, end2];
                }
            }
        }
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
        'question_html': f"À ton avis, quelle est la valeur associée à ce pixel ?<br/><div style='width: 60px; height: 60px; background-color: rgb({value}, {value}, {value}); margin: 10px auto; border: 1px solid black;'></div>",
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


# -------------------------------------------------------------------------
# NOUVEAUX EXERCICES : COMPRÉHENSION DES ZONES (4x3)
# -------------------------------------------------------------------------

# Données simplifiées 4x3
img_2_4x3 = np.array([
    [0, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [255, 255, 255]
])

img_7_4x3 = np.array([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [0, 255, 0]
])

def setup_interactive_line_js():
    # Injection du code JS nécessaire pour la droite interactive simplifiée (version Chart.js "Native")
    run_js("""
    window.mathadata.setup_simple_line = (id, targets, labels, callback_python) => {
        const chartId = id + '-chart';
        
        // Configuration pour tracer_droite_carac
        // On veut tracer l'axe, mais pas afficher les points tout de suite
        const params = {
            c_train: targets, // Les valeurs cibles (juste pour définir min/max axe)
            labels: [0, 1],   // Classe 0 (2) et Classe 1 (7)
            afficherPoints: false
        };

        // Utiliser la fonction existante pour initialiser le graphique (même style que l'exercice officiel)
        window.mathadata.tracer_droite_carac(chartId, params);
        const chart = window.mathadata.charts[chartId];
        
        // État du jeu
        let currentTargetIndex = 0; // 0 pour Image 2, 1 pour Image 7
        const statusEl = document.getElementById(id + '-feedback');
        
        // Messages
        const updateStatus = () => {
            if (currentTargetIndex >= targets.length) {
                statusEl.textContent = "Bravo ! C'est correct.";
                statusEl.style.color = "green";
                // Désactiver l'interaction
                chart.options.onClick = null;
                // Validation Python
                window.mathadata.run_python(callback_python + '()');
            } else {
                const currentLabel = labels[currentTargetIndex];
                statusEl.textContent = `Clique sur l'axe pour placer la moyenne pour l'${currentLabel}.`;
                statusEl.style.color = "#333";
            }
        };
        
        updateStatus();

        // Interaction Click
        chart.options.onClick = (e) => {
            const canvasPosition = Chart.helpers.getRelativePosition(e, chart);
            const dataX = chart.scales.x.getValueForPixel(canvasPosition.x);
            
            const target = targets[currentTargetIndex];
            
            // Tolérance de 5 unités (comme avant)
            if (Math.abs(dataX - target) < 5) {
                // Succès : Ajouter le point au dataset
                // Dataset 0 = Classe 0 (Bleu/2), Dataset 1 = Classe 1 (Orange/7)
                // On suppose que targets[0] est pour classe 0 et targets[1] pour classe 1
                const datasetIndex = currentTargetIndex; 
                
                chart.data.datasets[datasetIndex].data.push({x: target, y: 0});
                chart.update();
                
                currentTargetIndex++;
                updateStatus();
            } else {
                // Erreur
                statusEl.textContent = `Ce n'est pas la bonne moyenne (tu as cliqué sur ${dataX.toFixed(1)}). Recalcule et réessaie !`;
                statusEl.style.color = "red";
                
                // Petit feedback visuel temporaire (point rouge erreur)
                // On peut réutiliser la logique d'erreur de l'exercice officiel ou faire simple
                const errorDatasetIndex = chart.data.datasets.length; // Nouveau dataset temporaire
                chart.data.datasets.push({
                    label: 'Erreur',
                    data: [{x: dataX, y: 0}],
                    pointRadius: 4,
                    pointBackgroundColor: 'red',
                    pointBorderColor: 'red'
                });
                chart.update();
                
                setTimeout(() => {
                    chart.data.datasets.splice(errorDatasetIndex, 1);
                    chart.update();
                }, 1000);
            }
        };
    };
    """)

exo_moy_globale_ok = False
exo_moy_zone_ok = False

def set_exo_moy_globale_ok():
    global exo_moy_globale_ok
    exo_moy_globale_ok = True

def set_exo_moy_zone_ok():
    global exo_moy_zone_ok
    exo_moy_zone_ok = True

def _afficher_exercice_interactif(id_prefix, titre, zone_coords=None, callback_ok="set_exo_moy_globale_ok"):
    # On génère un ID unique à chaque exécution pour éviter le bug du canvas blanc (conflit Chart.js)
    id = f"{id_prefix}_{uuid.uuid4().hex}"

    # Calcul des moyennes cibles
    if zone_coords:
        # Zone spécifique
        # zone_coords = (row_start, row_end, col_start, col_end) (inclusif/exclusif style python slice ?)
        # Disons indices inclusifs pour CSS, slicing python exclusif pour le calcul
        r1, r2, c1, c2 = zone_coords
        targets = [np.mean(img_2_4x3[r1:r2, c1:c2]), np.mean(img_7_4x3[r1:r2, c1:c2])]
        zone_css = {
            'r_min': r1, 'r_max': r2-1, # indices inclusifs pour JS
            'c_min': c1, 'c_max': c2-1
        }
    else:
        # Global
        targets = [np.mean(img_2_4x3), np.mean(img_7_4x3)]
        zone_css = None

    setup_interactive_line_js()

    instruction_zone = "de la zone rouge" if zone_coords else "de toute l'image"

    display(HTML(f'''
    <div id="{id}" style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: #f9f9f9;">
        <h3 style="margin-top:0; text-align: center;">{titre}</h3>
        <p style="text-align: center; margin-bottom: 20px; font-size: 1.1em;">
            Calcule la moyenne des pixels {instruction_zone} et place le résultat sur l'axe ci-dessous.
        </p>
        <div style="display: flex; gap: 40px; justify-content: center; margin: 20px 0;">
            <div style="text-align: center;">
                <strong>Image 2</strong>
                <div style="display: flex; gap: 10px; margin-top: 10px; justify-content: center;">
                    <div id="{id}-tab-0" style="width: 180px; aspect-ratio: 1;"></div>
                    <div id="{id}-img-0" style="width: 180px; aspect-ratio: 1;"></div>
                </div>
            </div>
            <div style="text-align: center;">
                <strong>Image 7</strong>
                <div style="display: flex; gap: 10px; margin-top: 10px; justify-content: center;">
                    <div id="{id}-tab-1" style="width: 180px; aspect-ratio: 1;"></div>
                    <div id="{id}-img-1" style="width: 180px; aspect-ratio: 1;"></div>
                </div>
            </div>
        </div>
        
        <div style="background: white; border: 1px solid #ccc; border-radius: 4px; padding: 10px;">
            <canvas id="{id}-chart" style="width: 100%; height: 100px;"></canvas>
        </div>
        <p id="{id}-feedback" style="min-height: 20px; margin-top: 10px; text-align: center; font-weight: bold;"></p>
    </div>
    '''))
    
    run_js(f"""
        // Affichage des images
        const img2 = {json.dumps(img_2_4x3.tolist())};
        const img7 = {json.dumps(img_7_4x3.tolist())};
        
        // Timeout pour s'assurer que le DOM est prêt et affichage_tableau dispo
        setTimeout(() => {{
            // Affichage Tableaux
            if (window.mathadata.affichage_tableau) {{
                window.mathadata.affichage_tableau('{id}-tab-0', img2);
                window.mathadata.affichage_tableau('{id}-tab-1', img7);
            }}

            // Affichage Images Graphiques
            // On utilise window.mathadata.affichage définie plus haut
            if (window.mathadata.affichage) {{
                window.mathadata.affichage('{id}-img-0', img2, {{}});
                window.mathadata.affichage('{id}-img-1', img7, {{}});
            }}
            
            // Application de la zone rouge si nécessaire
            {f'''
            const zone = {json.dumps(zone_css)};
            [0, 1].forEach(idx => {{
                // STYLE POUR LE TABLEAU
                const containerTab = document.getElementById('{id}-tab-' + idx);
                
                const styleZoneTab = () => {{
                    const table = containerTab.querySelector('table');
                    if (!table) return;
                    
                    for(let r=0; r<table.rows.length; r++) {{
                        for(let c=0; c<table.rows[r].cells.length; c++) {{
                            // table.rows inclut les headers ! (affichage_tableau lignes 950+)
                            // row 0 = headers colonnes
                            // col 0 = headers lignes
                            // donc indices données sont r-1, c-1
                            
                            const dataR = r - 1;
                            const dataC = c - 1;
                            const cell = table.rows[r].cells[c];
                            
                            // Réinitialiser
                            cell.style.boxShadow = '';
                            cell.style.border = '';
                            
                            if (dataR >= zone.r_min && dataR <= zone.r_max &&
                                dataC >= zone.c_min && dataC <= zone.c_max) {{
                                
                                const borders = [];
                                if (dataR === zone.r_min) borders.push('2px solid red'); else borders.push('0');
                                if (dataR === zone.r_max) borders.push('2px solid red'); else borders.push('0');
                                if (dataC === zone.c_min) borders.push('2px solid red'); else borders.push('0');
                                if (dataC === zone.c_max) borders.push('2px solid red'); else borders.push('0');
                                
                                cell.style.borderTop = borders[0];
                                cell.style.borderBottom = borders[1];
                                cell.style.borderLeft = borders[2];
                                cell.style.borderRight = borders[3];
                            }}
                        }}
                    }}
                }};
                setTimeout(styleZoneTab, 50);

                // STYLE POUR L'IMAGE GRAPHIQUE
                // window.mathadata.affichage crée aussi une table avec la classe .image-table
                const containerImg = document.getElementById('{id}-img-' + idx);
                const styleZoneImg = () => {{
                    const table = containerImg.querySelector('table');
                    if (!table) return;
                     // Cette table n'a PAS de headers (générée par affichage() ligne 874)
                    for(let r=0; r<table.rows.length; r++) {{
                        for(let c=0; c<table.rows[r].cells.length; c++) {{
                            const cell = table.rows[r].cells[c];
                            
                            // Réinitialiser
                            cell.style.boxShadow = '';
                            cell.style.border = '';
                            
                            if (r >= zone.r_min && r <= zone.r_max &&
                                c >= zone.c_min && c <= zone.c_max) {{
                                
                                const borders = [];
                                if (r === zone.r_min) borders.push('2px solid red'); else borders.push('0');
                                if (r === zone.r_max) borders.push('2px solid red'); else borders.push('0');
                                if (c === zone.c_min) borders.push('2px solid red'); else borders.push('0');
                                if (c === zone.c_max) borders.push('2px solid red'); else borders.push('0');
                                
                                cell.style.borderTop = borders[0];
                                cell.style.borderBottom = borders[1];
                                cell.style.borderLeft = borders[2];
                                cell.style.borderRight = borders[3];
                            }}
                        }}
                    }}
                }};
                setTimeout(styleZoneImg, 50);

            }});
            ''' if zone_coords else ''}
            
            // Setup de la ligne interactive
            window.mathadata.setup_simple_line('{id}', {json.dumps(targets)}, ['image 2', 'image 7'], '{callback_ok}');
            
        }}, 100);
    """)

def exercice_moyenne_globale():
    _afficher_exercice_interactif('exo_moy_globale', "Moyenne Globale", callback_ok="set_exo_moy_globale_ok")

def exercice_moyenne_zone():
    # Zone bas gauche (2x2) : Lignes 2 et 3, Colonnes 0 et 1
    # Slice Python [2:4, 0:2]
    _afficher_exercice_interactif('exo_moy_zone', "Moyenne Zone", zone_coords=(2, 4, 0, 2), callback_ok="set_exo_moy_zone_ok")

def validate_exo_moy_globale(errors, answers):
    if exo_moy_globale_ok:
        return True
    else:
        errors.append("Place correctement les points sur l'axe (moins de 5 d'écart).")
        return False

def validate_exo_moy_zone(errors, answers):
    if exo_moy_zone_ok:
        return True
    else:
        errors.append("Place correctement les points sur l'axe (moins de 5 d'écart).")
        return False

validation_exercice_moyenne_globale = MathadataValidate(function_validation=validate_exo_moy_globale, success="Bravo ! Tu as bien calculé la moyenne globale.")
validation_exercice_moyenne_zone = MathadataValidate(function_validation=validate_exo_moy_zone, success="Bravo ! Tu vois que la moyenne de la zone permet de mieux séparer les deux images (plus grand écart).")


# Sélection de 2 images de 2 et 2 images de 7 pour l'affichage bas
def _get_samples_2_and_7(n_per_class=2):
    idx_2 = np.where(r_train == 2)[0][22    :22+n_per_class]
    idx_7 = np.where(r_train == 7)[0][:n_per_class]
    return np.concatenate([idx_2, idx_7])


def _compute_zone_means(images, zone):
    # zone: ((r_min, c_min), (r_max, c_max))
    (r0, c0), (r1, c1) = zone
    return [float(np.mean(img[r0:r1 + 1, c0:c1 + 1])) for img in images]


def exercice_zones_comparaison():
    """
    Affiche deux zones candidates (A et B) et 4 images d'exemple (2x classe 2, 2x classe 7).
    - A (rouge) : zone 5x4 en bas à gauche
    - B (bleu) : zone 5x4 en haut à droite
    Interaction :
    - Cliquer sur A ou B applique la zone correspondante sur les 4 images du bas (par défaut A).
    - Affiche 2 histogrammes : Histo 1 (zone B), Histo 2 (zone A), classés par classe (2 vs 7).
    - Sur la 1ère image de la rangée du bas, une flèche indique la moyenne dans la zone active.
    """
    # Zones (coordonnées inclusives) — nouvelle taille 6x10
    # A (rouge) : bas-gauche, hauteur 6, largeur 10
    
    zone_A = ((17, 4), (22, 13))
    # B (bleue) : haut-droite, hauteur 6, largeur 10
    
    zone_B = ((5, 14), (10, 23))

    # Échantillons
    sample_idx = _get_samples_2_and_7()
    images = d_train[sample_idx]
    labels = r_train[sample_idx]  # valeurs 2 ou 7

    # Moyennes par zone
    means_A = _compute_zone_means(images, zone_A)
    means_B = _compute_zone_means(images, zone_B)

    # Préparation des données d'histogramme (format data displayHisto : {bin: [c0, c1]})
    def build_data(means, labels):
        data = {}
        for m, lab in zip(means, labels):
            k = int(m // 2) * 2
            if k not in data:
                data[k] = [0, 0]
            if lab == 2:
                data[k][0] += 1
            else:
                data[k][1] += 1
        return data

    data_histo_B = build_data(means_B, labels)  # Histo 1 -> zone B (bleue)
    data_histo_A = build_data(means_A, labels)  # Histo 2 -> zone A (rouge)

    # Paramètres JS
    params = {
        "images": images.tolist(),
        "labels": labels.tolist(),
        "zoneA": zone_A,
        "zoneB": zone_B,
        "meansA": means_A,
        "meansB": means_B,
        "dataHistoA": data_histo_A,
        "dataHistoB": data_histo_B,
    }

    id = uuid.uuid4().hex

    display(HTML(f"""
    <div id="{id}" style="display: flex; flex-direction: column; gap: 16px;">
        <!-- Top row A / B -->
        <div style="display: flex; gap: 24px; justify-content: center;">
            <div id="{id}-A" class="zone-selector selected-zone" data-zone="A" style="cursor: pointer; text-align:center;">
                <div style="margin-bottom: 4px; font-weight: 700;">Zone A (rouge)</div>
                <div id="{id}-A-canvas" style="width: 140px; aspect-ratio: 1;"></div>
            </div>
            <div id="{id}-B" class="zone-selector" data-zone="B" style="cursor: pointer; text-align:center;">
                <div style="margin-bottom: 4px; font-weight: 700;">Zone B (bleue)</div>
                <div id="{id}-B-canvas" style="width: 140px; aspect-ratio: 1;"></div>
            </div>
        </div>

        <!-- Bottom images -->
        <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            <div id="{id}-img-0" class="img-box" style="position: relative; width: 120px; aspect-ratio: 1;"></div>
            <div id="{id}-img-1" class="img-box" style="position: relative; width: 120px; aspect-ratio: 1;"></div>
            <div id="{id}-img-2" class="img-box" style="position: relative; width: 120px; aspect-ratio: 1;"></div>
            <div id="{id}-img-3" class="img-box" style="position: relative; width: 120px; aspect-ratio: 1;"></div>
        </div>

        <!-- Histograms -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 32px;">
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h4 style="margin: 0 0 8px 0; text-align: center;">Histogramme 1</h4>
                <div style="width: 100%; height: 250px;">
                    <canvas id="{id}-histo-b" style="width: 100%; height: 100%;"></canvas>
                </div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: center;">
                <h4 style="margin: 0 0 8px 0; text-align: center;">Histogramme 2</h4>
                <div style="width: 100%; height: 250px;">
                    <canvas id="{id}-histo-a" style="width: 100%; height: 100%;"></canvas>
                </div>
            </div>
        </div>
    </div>
    """))

    # JS pour affichage et interaction
    run_js(f"""
    setTimeout(() => {{
        const rootId = "{id}";
        const data = {json.dumps(params, cls=NpEncoder)};

        // Fonction d'affichage locale robuste (remplace displayHisto global)
        const localDisplayHisto = (div_id, data, params) => {{
            try {{
                // Gestion permissive du parsing JSON
                if (typeof data === 'string') data = JSON.parse(data);
                if (typeof params === 'string') params = JSON.parse(params);
            }} catch (e) {{
                console.warn('[mathadata] localDisplayHisto: parse error', e);
                data = {{}};
                params = {{}};
            }}

            const canvas = document.getElementById(div_id);
            if (!canvas) {{
                console.warn('[mathadata] localDisplayHisto: canvas introuvable', div_id);
                return;
            }}
            
            if (typeof Chart === 'undefined') {{
                // Si Chart.js n'est pas encore chargé, on réessaie plus tard
                console.warn('[mathadata] localDisplayHisto: Chart.js non chargé, nouvel essai dans 500ms');
                setTimeout(() => localDisplayHisto(div_id, data, params), 500);
                return;
            }}

            // Destruction du chart existant s'il y en a un (évite les superpositions)
            const existingChart = Chart.getChart(canvas);
            if (existingChart) existingChart.destroy();

            const entries = Object.entries(data);
            const data_1 = entries.map(([key, v]) => ({{ x: parseInt(key) + 1, y: Array.isArray(v) ? (v[0] || 0) : 0 }}));
            const data_2 = entries.map(([key, v]) => ({{ x: parseInt(key) + 1, y: Array.isArray(v) ? (v[1] || 0) : 0 }}));
            
            // Calculer min/max dès maintenant pour éviter les ticks excessifs
            const keys = entries.map(([k]) => parseInt(k) + 1);
            const xMin = keys.length > 0 ? Math.min(...keys) : 0;
            const xMax = keys.length > 0 ? Math.max(...keys) : 10;
            const xStep = 10;
            const xMinRounded = Math.floor(xMin / xStep) * xStep;
            const xMaxRounded = Math.ceil(xMax / xStep) * xStep;
            
            // Création du graphique
            const chart = new Chart(canvas.getContext('2d'), {{
                type: 'bar',
                data: {{
                    datasets: [
                        {{
                            label: 'Classe 2',
                            data: data_1,
                            backgroundColor: 'rgba(80,140,255,0.8)',
                            borderColor: 'rgba(80,140,255,1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Classe 7',
                            data: data_2,
                            backgroundColor: 'rgba(255,170,80,0.8)',
                            borderColor: 'rgba(255,170,80,1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{
                            type: 'linear',
                            stacked: true,
                            offset: false,
                            grid: {{ offset: false }},
                            min: xMinRounded,
                            max: xMaxRounded,
                            ticks: {{
                                stepSize: xStep
                            }},
                            title: {{ display: true, text: 'Caractéristique (moyenne zone)' }}
                        }},
                        y: {{
                            beginAtZero: true,
                            min: 0,
                            max: 8,
                            ticks: {{ stepSize: 2 }},
                            title: {{ display: true, text: 'Nombre' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ display: true }}
                    }}
                }}
            }});
            
            // Enregistrement global pour accès ultérieur (tick updates)
            if (!window.mathadata.charts) window.mathadata.charts = {{}};
            window.mathadata.charts[div_id] = chart; // On utilise div_id comme clé unique
        }};

        // Helpers pour dessiner la zone (bordures)
        function applyZone(table, zone, color) {{
            if (!table) return;
            for (let r = 0; r < table.rows.length; r++) {{
                for (let c = 0; c < table.rows[r].cells.length; c++) {{
                    const cell = table.rows[r].cells[c];
                    cell.style.boxShadow = '';
                    cell.style.border = '';
                    if (r >= zone[0][0] && r <= zone[1][0] && c >= zone[0][1] && c <= zone[1][1]) {{
                        const bt = (r === zone[0][0]) ? `2px solid ${{color}}` : '0';
                        const bb = (r === zone[1][0]) ? `2px solid ${{color}}` : '0';
                        const bl = (c === zone[0][1]) ? `2px solid ${{color}}` : '0';
                        const br = (c === zone[1][1]) ? `2px solid ${{color}}` : '0';
                        cell.style.borderTop = bt;
                        cell.style.borderBottom = bb;
                        cell.style.borderLeft = bl;
                        cell.style.borderRight = br;
                    }}
                }}
            }}
        }}

        // Affichage des deux zones A/B (images noires)
        function renderZoneCanvas(targetId, zone, color) {{
            const container = document.getElementById(targetId);
            if (!container || !window.mathadata.affichage) return;
            const zeroImg = Array.from({{length: 28}}, () => Array(28).fill(0));
            window.mathadata.affichage(targetId, zeroImg, {{}});
            const table = container.querySelector('table');
            applyZone(table, zone, color);
        }}

        // Affichage des 4 images exemples
        function renderImages(activeZoneKey) {{
            const zone = activeZoneKey === 'A' ? data.zoneA : data.zoneB;
            const color = activeZoneKey === 'A' ? 'red' : 'blue';
            for (let i = 0; i < data.images.length; i++) {{
                const container = document.getElementById(`${{rootId}}-img-${{i}}`);
                if (!container || !window.mathadata.affichage) continue;
                window.mathadata.affichage(`${{rootId}}-img-${{i}}`, data.images[i], {{}});
                const table = container.querySelector('table');
                applyZone(table, zone, color);

                // Flèche + valeur sur la 1ère image (idx 0) pour la zone active
                const existing = container.querySelector('.mean-badge');
                if (existing) existing.remove();
                if (i === 0) {{
                    const meanVal = activeZoneKey === 'A' ? data.meansA[0] : data.meansB[0];
                    const badge = document.createElement('div');
                    badge.className = 'mean-badge';
                    badge.style.position = 'absolute';
                    badge.style.top = activeZoneKey === 'A' ? '70%' : '5%';
                    badge.style.left = activeZoneKey === 'A' ? '5%' : '55%';
                    badge.style.background = '#000';
                    badge.style.color = '#fff';
                    badge.style.padding = '2px 6px';
                    badge.style.borderRadius = '4px';
                    badge.style.fontSize = '11px';
                    badge.style.boxShadow = '0 1px 4px rgba(0,0,0,0.3)';
                    badge.textContent = `↘ moyenne = ${{meanVal.toFixed(1)}}`;
                    container.appendChild(badge);
                }}
            }}
        }}

        // Histos (B puis A)
        function renderHistos() {{
            const paramsA = {{ with_legend: true, with_axes_legend: true }};
            const paramsB = {{ with_legend: true, with_axes_legend: true }};
            
            // Appel direct à notre fonction locale (avec timeout intégré si Chart manquant)
            localDisplayHisto(`${{rootId}}-histo-b`, data.dataHistoB, paramsB);
            localDisplayHisto(`${{rootId}}-histo-a`, data.dataHistoA, paramsA);

            // Ajuste les ticks : abscisses par pas de 4, ordonnées par pas de 2 (FIXE)
            // On encapsule dans un setTimeout pour s'assurer que les charts sont créés
            setTimeout(() => {{
                try {{
                    const charts = window.mathadata.charts || {{}};
                    const keysB = Object.keys(data.dataHistoB || {{}}).map(k => parseInt(k) + 1);
                    const keysA = Object.keys(data.dataHistoA || {{}}).map(k => parseInt(k) + 1);
                    
                    // Plage globale pour les deux histogrammes
                    const allKeys = [...keysB, ...keysA];
                    const xMinGlobal = Math.min(...allKeys, 0);
                    const xMaxGlobal = Math.max(...allKeys, 0);
                    const xStep = 10;
                    const xMinRounded = Math.floor(xMinGlobal / xStep) * xStep;
                    const xMaxRounded = Math.ceil(xMaxGlobal / xStep) * xStep;
                    
                    ['histo-b', 'histo-a'].forEach(key => {{
                        const chart = charts[`${{rootId}}-${{key}}`];
                        if (!chart || !chart.options || !chart.options.scales) return;
                        const allowed = key === 'histo-b' ? keysB : keysA;
                        
                        // Abscisses : même plage pour les deux, pas fixe de 4
                        if (chart.options.scales.x) {{
                            chart.options.scales.x.min = xMinRounded;
                            chart.options.scales.x.max = xMaxRounded;
                            chart.options.scales.x.ticks = {{
                                ...(chart.options.scales.x.ticks || {{}}),
                                stepSize: xStep
                            }};
                        }}
                        
                        // Ordonnées : fixe de 0 à 8, pas fixe de 2
                        if (chart.options.scales.y) {{
                            chart.options.scales.y.min = 0;
                            chart.options.scales.y.max = 8;
                            chart.options.scales.y.ticks = {{
                                ...(chart.options.scales.y.ticks || {{}}),
                                stepSize: 2
                            }};
                        }}
                        
                        chart.update();
                    }});
                }} catch (e) {{
                    console.error('[mathadata] tick adjust error', e);
                }}
            }}, 1000); // Délai pour laisser le temps aux charts de se créer (polling Chart.js dans localDisplayHisto)
        }}

        // Sélection visuelle A/B avec badge "Cliquez"
        function highlight(selected) {{
            ['A','B'].forEach(z => {{
                const el = document.getElementById(`${{rootId}}-${{z}}`);
                if (!el) return;
                el.classList.toggle('selected-zone', z === selected);
                
                // Style de base
                el.style.position = 'relative';
                el.style.cursor = 'pointer';
                el.style.transition = 'all 0.2s ease';
                el.style.borderRadius = '8px';
                el.style.padding = '8px';
                
                if (z === selected) {{
                    // Zone sélectionnée : bordure épaisse + ombre
                    el.style.border = z === 'A' ? '3px solid #ff0000' : '3px solid #0000ff';
                    el.style.boxShadow = '0 4px 12px rgba(0,0,0,0.25)';
                    el.style.backgroundColor = z === 'A' ? 'rgba(255, 0, 0, 0.05)' : 'rgba(0, 0, 255, 0.05)';
                    
                    // Supprimer le badge "Cliquez" si présent
                    const clickBadge = el.querySelector('.click-badge');
                    if (clickBadge) clickBadge.remove();
                    
                    // Ajouter badge "Sélectionné"
                    let selectedBadge = el.querySelector('.selected-badge');
                    if (!selectedBadge) {{
                        selectedBadge = document.createElement('div');
                        selectedBadge.className = 'selected-badge';
                        selectedBadge.style.position = 'absolute';
                        selectedBadge.style.top = '3px';
                        selectedBadge.style.right = '3px';
                        selectedBadge.style.background = z === 'A' ? '#ff0000' : '#0000ff';
                        selectedBadge.style.color = 'white';
                        selectedBadge.style.padding = '2.25px 6px';
                        selectedBadge.style.borderRadius = '9px';
                        selectedBadge.style.fontSize = '8.25px';
                        selectedBadge.style.fontWeight = 'bold';
                        selectedBadge.style.boxShadow = '0 1.5px 3px rgba(0,0,0,0.2)';
                        selectedBadge.textContent = '✓ Sélectionné';
                        el.appendChild(selectedBadge);
                    }}
                }} else {{
                    // Zone non sélectionnée : bordure fine + badge "Cliquez"
                    el.style.border = '2px solid #ccc';
                    el.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
                    el.style.backgroundColor = 'rgba(250, 250, 250, 0.8)';
                    
                    // Supprimer le badge "Sélectionné" si présent
                    const selectedBadge = el.querySelector('.selected-badge');
                    if (selectedBadge) selectedBadge.remove();
                    
                    // Ajouter badge "Cliquez"
                    let clickBadge = el.querySelector('.click-badge');
                    if (!clickBadge) {{
                        clickBadge = document.createElement('div');
                        clickBadge.className = 'click-badge';
                        clickBadge.style.position = 'absolute';
                        clickBadge.style.top = '3px';
                        clickBadge.style.right = '3px';
                        clickBadge.style.background = z === 'A' ? '#ff0000' : '#0000ff';
                        clickBadge.style.color = 'white';
                        clickBadge.style.padding = '3px 7.5px';
                        clickBadge.style.borderRadius = '9px';
                        clickBadge.style.fontSize = '8.25px';
                        clickBadge.style.fontWeight = '600';
                        clickBadge.style.boxShadow = z === 'A' ? '0 1.5px 4.5px rgba(255, 0, 0, 0.4)' : '0 1.5px 4.5px rgba(0, 0, 255, 0.4)';
                        clickBadge.style.animation = 'pulse 2s ease-in-out infinite';
                        clickBadge.textContent = 'Cliquez';
                        el.appendChild(clickBadge);
                    }}
                }}
            }});
        }}
        
        // Ajouter l'animation CSS pour le badge "Cliquez"
        if (!document.getElementById('mathadata-pulse-animation')) {{
            const style = document.createElement('style');
            style.id = 'mathadata-pulse-animation';
            style.textContent = `
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; transform: scale(1); }}
                    50% {{ opacity: 0.8; transform: scale(1.05); }}
                }}
            `;
            document.head.appendChild(style);
        }}

        // Initial render
        renderZoneCanvas(`${{rootId}}-A-canvas`, data.zoneA, 'red');
        renderZoneCanvas(`${{rootId}}-B-canvas`, data.zoneB, 'blue');
        renderImages('A'); // défaut A
        renderHistos();
        highlight('A');

        // Click handlers
        const zoneAEl = document.getElementById(`${{rootId}}-A`);
        const zoneBEl = document.getElementById(`${{rootId}}-B`);
        if (zoneAEl) zoneAEl.onclick = () => {{ renderImages('A'); highlight('A'); }};
        if (zoneBEl) zoneBEl.onclick = () => {{ renderImages('B'); highlight('B'); }};
    }}, 100);
    """)


def question_hist_zones_markdown():
    """Affiche la question sur l'association histogramme/zone avec mise en forme HTML."""
    display(HTML("""
<div class="admonition admonition-question">
<p class="admonition-title">Quel histogramme correspond à quelle zone ?</p>
<p>
Regarde les deux histogrammes affichés : l'un est associé à la zone B (bleue), l'autre à la zone A (rouge).
Indique dans la cellule suivante quel histogramme (1 ou 2) correspond à la zone A et lequel à la zone B.
</p>
</div>
"""))


def _normalize_hist_zone(val):
    """Normalise la valeur pour accepter A/B avec ou sans guillemets, insensible à la casse."""
    if val is Ellipsis:
        return None
    if val is None:
        return None
    if isinstance(val, str):
        normalized = val.strip().upper()
        return normalized if normalized else None
    # Si c'est une variable Python (sans guillemets), essayer de la convertir en string
    try:
        normalized = str(val).strip().upper()
        return normalized if normalized else None
    except:
        return None

def _validate_hist_zones(errors, answers):
    """Validation permissive qui normalise les valeurs avant comparaison."""
    h1_raw = answers.get('histogramme_1')
    h2_raw = answers.get('histogramme_2')
    
    # Vérifier que les variables sont définies
    if h1_raw is Ellipsis or h1_raw is None:
        errors.append("histogramme_1 n'est pas défini. Tu dois remplacer les ... par A ou B.")
        return False
    if h2_raw is Ellipsis or h2_raw is None:
        errors.append("histogramme_2 n'est pas défini. Tu dois remplacer les ... par A ou B.")
        return False
    
    # Normaliser les valeurs
    h1 = _normalize_hist_zone(h1_raw)
    h2 = _normalize_hist_zone(h2_raw)
    
    # Vérifier que la normalisation a fonctionné
    if h1 is None:
        errors.append("histogramme_1 n'a pas une valeur valide. Tu dois utiliser A ou B (avec ou sans guillemets).")
        return False
    if h2 is None:
        errors.append("histogramme_2 n'a pas une valeur valide. Tu dois utiliser A ou B (avec ou sans guillemets).")
        return False
    
    # Mettre à jour answers avec les valeurs normalisées
    answers['histogramme_1'] = h1
    answers['histogramme_2'] = h2
    
    # Vérifier que les valeurs sont dans ['A', 'B']
    if h1 not in ['A', 'B']:
        errors.append(f"histogramme_1 doit être A ou B, pas '{h1_raw}'. Tu peux utiliser A, B, 'A', 'B', \"A\" ou \"B\". Si tu vois une erreur 'NameError', c'est que tu as utilisé une lettre sans guillemets qui n'est pas définie (comme C, D, 7, 2, etc.). Utilise des guillemets : 'A' ou 'B'.")
        return False
    if h2 not in ['A', 'B']:
        errors.append(f"histogramme_2 doit être A ou B, pas '{h2_raw}'. Tu peux utiliser A, B, 'A', 'B', \"A\" ou \"B\". Si tu vois une erreur 'NameError', c'est que tu as utilisé une lettre sans guillemets qui n'est pas définie (comme C, D, 7, 2, etc.). Utilise des guillemets : 'A' ou 'B'.")
        return False
    
    # Vérifier que les deux valeurs ne sont pas identiques
    if h1 == h2:
        errors.append(f"Les deux histogrammes ne peuvent pas correspondre à la même zone. Tu as mis '{h1_raw}' pour les deux. L'un doit être A et l'autre B.")
        return False
    
    return True

validation_hist_zones = MathadataValidateVariables({
    'histogramme_1': None,  # On vérifie juste que c'est défini, la validation se fait dans function_validation
    'histogramme_2': None
}, function_validation=_validate_hist_zones, success="Bravo, tu as bien associé chaque histogramme à la bonne zone !")


# -------------------------------------------------------------------------
# JS : affichage 2 exemples + zones + valeurs pixels (MNIST)
# -------------------------------------------------------------------------

run_js(r"""
(function () {
    if (!window.mathadata) return;

    // Typeset MathJax (support v2/v3), sans planter si non disponible.
    // Retourne une Promise qui se résout après le rendu (utile quand on injecte du LaTeX dynamiquement).
    const typeset = (el) => {
        if (!el) return Promise.resolve(false);

        // MathJax v3 (typesetPromise)
        if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
            try {
                const startup = window.MathJax.startup?.promise;
                if (startup && typeof startup.then === "function") {
                    return startup
                        .then(() => window.MathJax.typesetPromise([el]))
                        .then(() => true)
                        .catch(() => false);
                }
                return window.MathJax.typesetPromise([el]).then(() => true).catch(() => false);
            } catch (e) {
                return Promise.resolve(false);
            }
        }

        // MathJax v2 (Hub.Queue)
        if (window.MathJax && window.MathJax.Hub && typeof window.MathJax.Hub.Queue === "function") {
            return new Promise((resolve) => {
                try {
                    window.MathJax.Hub.Queue(["Typeset", window.MathJax.Hub, el], () => resolve(true));
                } catch (e) {
                    resolve(false);
                }
            });
        }

        return Promise.resolve(false);
    };

    // ---------------------------------------------------------------------
    // Spécification notebook : seuillage_0_200_250(img)
    // - 0 <= v < 180   -> 0
    // - 180 <= v < 220 -> 200
    // - 220 <= v <=255 -> 250
    // + clip dans [0;255]
    //
    // Exception (dev) : v == 240 reste 240 (pour permettre de “marquer” un pixel sans qu'il soit ramené à 250).
    // ---------------------------------------------------------------------
    window.mathadata.mnist_seuillage_0_200_250 = (img) => {
        if (!Array.isArray(img) || img.length === 0 || !Array.isArray(img[0])) {
            return img;
        }

        const out = new Array(img.length);
        for (let r = 0; r < img.length; r++) {
            const row = img[r];
            const oRow = new Array(row.length);
            for (let c = 0; c < row.length; c++) {
                let v = Number(row[c]);
                if (!Number.isFinite(v)) v = 0;
                v = Math.max(0, Math.min(255, v));

                if (v < 180) oRow[c] = 0;
                else if (v < 220) oRow[c] = 200;
                else if (v === 240) oRow[c] = 240;
                else oRow[c] = 250;
            }
            out[r] = oRow;
        }
        return out;
    };

    // Moyenne des pixels dans une zone rectangulaire (inclusif), comme moyenne_zone Python.
    window.mathadata.mnist_mean_zone = (img, zone) => {
        if (!Array.isArray(img) || img.length === 0 || !Array.isArray(img[0])) return null;
        if (!Array.isArray(zone) || zone.length !== 2) return null;

        const A = zone[0];
        const B = zone[1];
        if (!Array.isArray(A) || !Array.isArray(B) || A.length !== 2 || B.length !== 2) return null;

        const h = img.length;
        const w = img[0].length;

        const r0 = Math.round(Number(A[0]));
        const c0 = Math.round(Number(A[1]));
        const r1 = Math.round(Number(B[0]));
        const c1 = Math.round(Number(B[1]));

        let rmin = Math.max(0, Math.min(r0, r1));
        let rmax = Math.min(h - 1, Math.max(r0, r1));
        let cmin = Math.max(0, Math.min(c0, c1));
        let cmax = Math.min(w - 1, Math.max(c0, c1));

        let sum = 0;
        let count = 0;
        for (let r = rmin; r <= rmax; r++) {
            const row = img[r];
            for (let c = cmin; c <= cmax; c++) {
                const v = Number(row[c]);
                if (!Number.isFinite(v)) continue;
                sum += v;
                count += 1;
            }
        }

        return count > 0 ? (sum / count) : null;
    };

    // Format "max 1 décimale"
    const round1 = (v) => Math.round((v + Number.EPSILON) * 10) / 10;
    const format1 = (v) => {
        if (v === null || v === undefined) return "—";
        if (typeof v !== "number" || !Number.isFinite(v)) return "—";
        const r = round1(v);
        return String(r);
    };

    const drawZoneRect = (grid, zone, color, zoneLw, cellSize, gapPx, padPx) => {
        if (!grid || !zone) return;
        const A = zone[0];
        const B = zone[1];
        if (!A || !B) return;

        const r0 = Math.round(Number(A[0]));
        const c0 = Math.round(Number(A[1]));
        const r1 = Math.round(Number(B[0]));
        const c1 = Math.round(Number(B[1]));
        if (![r0, c0, r1, c1].every(Number.isFinite)) return;

        const rmin = Math.min(r0, r1);
        const rmax = Math.max(r0, r1);
        const cmin = Math.min(c0, c1);
        const cmax = Math.max(c0, c1);

        const rect = document.createElement("div");
        rect.className = "mathadata-mnist-zonerect";

        const lwPx = Math.max(1, Math.round(Number(zoneLw || 6) * (cellSize / 16)));

        rect.style.border = `${lwPx}px solid ${color || "red"}`;
        // IMPORTANT : le `border` est dessiné *à l'intérieur* de la boîte.
        // Pour ne pas masquer les valeurs des pixels situés sur le bord de la zone,
        // on agrandit la boîte de `lwPx` tout autour :
        // - l'intérieur du rectangle (après border) correspond exactement à la zone
        // - le trait est donc "à l'extérieur" de la zone.
        const left = padPx + cmin * (cellSize + gapPx);
        const top = padPx + rmin * (cellSize + gapPx);
        const width = (cmax - cmin + 1) * cellSize + (cmax - cmin) * gapPx;
        const height = (rmax - rmin + 1) * cellSize + (rmax - rmin) * gapPx;

        rect.style.left = `${left - lwPx}px`;
        rect.style.top = `${top - lwPx}px`;
        rect.style.width = `${width + 2 * lwPx}px`;
        rect.style.height = `${height + 2 * lwPx}px`;

        grid.appendChild(rect);
    };

    // ---------------------------------------------------------------------
    // Spécification notebook : afficher_image_pixels(img, seuil_texte=130, fontsize=10, show_zones, zones, zone_colors, zone_lw)
    // Version JS : rendu en grille HTML (rapide) + rectangles de zones (rouge/bleu).
    // ---------------------------------------------------------------------
    window.mathadata.mnist_afficher_image_pixels = (containerId, img, options) => {
        const container = (typeof containerId === "string") ? document.getElementById(containerId) : containerId;
        if (!container) return;

        const opts = options || {};
        const seuilTexte = Number.isFinite(Number(opts.seuil_texte)) ? Number(opts.seuil_texte) : 130;
        const fontsize = Number.isFinite(Number(opts.fontsize)) ? Number(opts.fontsize) : 10;
        const showValues = (opts.show_values === undefined) ? true : Boolean(opts.show_values);
        const showZones = Boolean(opts.show_zones);
        const showZoneX = (opts.show_zone_x === undefined) ? true : Boolean(opts.show_zone_x);
        const showZoneY = (opts.show_zone_y === undefined) ? true : Boolean(opts.show_zone_y);
        const zones = Array.isArray(opts.zones) ? opts.zones : null;
        const zoneColors = Array.isArray(opts.zone_colors) ? opts.zone_colors : ["red", "blue"];
        const zoneLw = Number.isFinite(Number(opts.zone_lw)) ? Number(opts.zone_lw) : 6;

        container.innerHTML = "";

        if (!Array.isArray(img) || img.length === 0 || !Array.isArray(img[0])) {
            container.textContent = "Image invalide (attendu: tableau 2D).";
            return;
        }

        const h = img.length;
        const w = img[0].length;

        // Paramètres visuels proches du notebook
        const gapPxRaw = Number(opts.grid_gap_px);
        const gapPx = Number.isFinite(gapPxRaw) ? Math.max(0, Math.min(8, Math.round(gapPxRaw))) : 1;
        const padPx = gapPx;
        const maxWidthPxRaw = opts.max_width_px;
        const maxWidthPxNum = Number(maxWidthPxRaw);
        const maxWidthPx = (
            maxWidthPxRaw === null ||
            maxWidthPxRaw === undefined ||
            !Number.isFinite(maxWidthPxNum) ||
            maxWidthPxNum <= 0
        ) ? 360 : maxWidthPxNum;

        const available = Math.max(1, maxWidthPx - 2 * padPx - (w - 1) * gapPx);
        const minCellPx = showValues ? 8 : 4;
        const cellSize = Math.max(minCellPx, Math.min(22, Math.floor(available / w)));
        const fontPx = showValues ? Math.max(4, Math.min(Math.floor(fontsize), cellSize - 1)) : 0;

        const grid = document.createElement("div");
        grid.className = "mathadata-mnist-pixelgrid";
        grid.style.gap = `${gapPx}px`;
        grid.style.padding = `${padPx}px`;
        grid.style.gridTemplateColumns = `repeat(${w}, ${cellSize}px)`;
        grid.style.gridAutoRows = `${cellSize}px`;

        const frag = document.createDocumentFragment();
        for (let r = 0; r < h; r++) {
            const row = img[r];
            for (let c = 0; c < w; c++) {
                const raw = Number(row[c]);
                const v = Number.isFinite(raw) ? raw : 0;
                const displayVal = Math.round(v);
                const g = Math.max(0, Math.min(255, displayVal));

                const cell = document.createElement("div");
                cell.className = "mathadata-mnist-pixelcell";
                cell.style.backgroundColor = `rgb(${g}, ${g}, ${g})`;
                cell.style.color = (displayVal >= seuilTexte) ? "black" : "white";
                cell.style.fontSize = showValues ? `${fontPx}px` : "0px";
                cell.textContent = showValues ? String(displayVal) : "";

                frag.appendChild(cell);
            }
        }
        grid.appendChild(frag);
        container.appendChild(grid);

	        if (showZones && zones) {
	            if (showZoneX && zones.length > 0) drawZoneRect(grid, zones[0], zoneColors[0] || "red", zoneLw, cellSize, gapPx, padPx);
	            if (showZoneY && zones.length > 1) drawZoneRect(grid, zones[1], zoneColors[1] || "blue", zoneLw, cellSize, gapPx, padPx);
	        }

	        // Légendes (optionnel) alignées sur le centre vertical des zones
	        const legend = opts?.legend;
	        const legendEnabled = legend && (Array.isArray(legend.left) || Array.isArray(legend.right));
	        if (legendEnabled && zones && zones.length > 0) {
	            const offsetPx = Number.isFinite(Number(legend.offset_px)) ? Number(legend.offset_px) : 8;
	            const fontPxLegend = Number.isFinite(Number(legend.font_px)) ? Number(legend.font_px) : 14;
	            const withBg = (legend.background === undefined) ? true : Boolean(legend.background);

	            const zoneCenterY = (zone) => {
	                if (!Array.isArray(zone) || zone.length !== 2) return null;
	                const A = zone[0];
	                const B = zone[1];
	                if (!Array.isArray(A) || !Array.isArray(B) || A.length !== 2 || B.length !== 2) return null;

	                const r0 = Number(A[0]);
	                const r1 = Number(B[0]);
	                if (!Number.isFinite(r0) || !Number.isFinite(r1)) return null;

	                const rmin = Math.min(r0, r1);
	                const rmax = Math.max(r0, r1);
	                const top = padPx + rmin * (cellSize + gapPx);
	                const height = (rmax - rmin + 1) * cellSize + (rmax - rmin) * gapPx;
	                return top + height / 2;
	            };

	            const makeLabel = (html, side, color, topPx) => {
	                if (!html) return;
	                const el = document.createElement("div");
	                el.className = "mathadata-mnist-zonelabel";
	                el.innerHTML = String(html);
	                el.style.fontSize = `${fontPxLegend}px`;
	                el.style.color = color || "#111";
	                el.style.top = `${topPx}px`;

	                if (withBg) {
	                    el.style.background = "rgba(255,255,255,0.92)";
	                } else {
	                    el.style.background = "transparent";
	                    el.style.boxShadow = "none";
	                    el.style.padding = "0";
	                }

	                if (side === "left") {
	                    el.style.left = `-${offsetPx}px`;
	                    el.style.transform = "translate(-100%, -50%)";
	                } else {
	                    el.style.right = `-${offsetPx}px`;
	                    el.style.transform = "translate(100%, -50%)";
	                }

	                grid.appendChild(el);
	            };

	            const leftLabels = Array.isArray(legend.left) ? legend.left : [];
	            const rightLabels = Array.isArray(legend.right) ? legend.right : [];

	            for (let zi = 0; zi < zones.length; zi++) {
	                if (zi === 0 && !showZoneX) continue;
	                if (zi === 1 && !showZoneY) continue;
	                const cy = zoneCenterY(zones[zi]);
	                if (cy === null) continue;
	                const color = (Array.isArray(zoneColors) && zoneColors[zi]) ? zoneColors[zi] : undefined;
	                makeLabel(leftLabels[zi], "left", color, cy);
	                makeLabel(rightLabels[zi], "right", color, cy);
	            }
	        }
	    };

    // ---------------------------------------------------------------------
    // Export PNG (screenshot DOM) via html2canvas (chargé à la demande)
    // ---------------------------------------------------------------------
    window.mathadata._ensure_html2canvas = window.mathadata._ensure_html2canvas || (async () => {
        if (window.html2canvas) return window.html2canvas;

        const loadScript = (src) => new Promise((resolve, reject) => {
            const s = document.createElement("script");
            s.src = src;
            s.async = true;
            s.onload = () => resolve();
            s.onerror = () => reject(new Error("Impossible de charger html2canvas"));
            document.head.appendChild(s);
        });

        try {
            await loadScript("https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js");
        } catch (e) {
            console.error(e);
            return null;
        }

        return window.html2canvas || null;
    });

    window.mathadata._save_element_png = window.mathadata._save_element_png || (async (element, filename) => {
        if (!element) return false;
        const h2c = await window.mathadata._ensure_html2canvas();
        if (!h2c) return false;

        const stripIds = (node) => {
            if (!node || node.nodeType !== 1) return;
            if (node.hasAttribute && node.hasAttribute("id")) node.removeAttribute("id");
            if (node.children) Array.from(node.children).forEach(stripIds);
        };

        const clone = element.cloneNode(true);
        stripIds(clone);
        clone.querySelectorAll && clone.querySelectorAll(".mathadata-noexport").forEach(el => el.remove());

        // Forcer l'affichage complet (pas de scroll interne) pour la capture
        const grid = clone.querySelector && clone.querySelector(".mathadata-mnist-exemples-zones-grid");
        if (grid) {
            grid.style.overflow = "visible";
            grid.style.maxWidth = "none";
        }

        clone.style.position = "fixed";
        clone.style.left = "-10000px";
        clone.style.top = "0";
        clone.style.background = "#fff";
        clone.style.maxWidth = "none";
        clone.style.width = `${Math.max(element.scrollWidth || 0, element.clientWidth || 0)}px`;

        document.body.appendChild(clone);
        try {
            await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

            const w = Math.max(clone.scrollWidth || 0, clone.clientWidth || 0);
            const h = Math.max(clone.scrollHeight || 0, clone.clientHeight || 0);

            const canvas = await h2c(clone, {
                backgroundColor: "#ffffff",
                scale: 2,
                width: w || undefined,
                height: h || undefined,
            });

            const blob = await new Promise((resolve) => {
                if (canvas.toBlob) {
                    canvas.toBlob((b) => resolve(b), "image/png");
                } else {
                    // Fallback ancien navigateur
                    const dataUrl = canvas.toDataURL("image/png");
                    const bin = atob(dataUrl.split(",")[1]);
                    const arr = new Uint8Array(bin.length);
                    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
                    resolve(new Blob([arr], { type: "image/png" }));
                }
            });

            if (!blob) return false;

            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = filename || "mnist_exemples_zones.png";
            document.body.appendChild(a);
            a.click();
            a.remove();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
            return true;
        } finally {
            clone.remove();
        }
    });

    // ---------------------------------------------------------------------
    // Affichage demandé : 2 images côte à côte + zones + (option) point caractéristique.
    // ---------------------------------------------------------------------
	    window.mathadata.afficher_deux_exemples_zones = async (id, params) => {
        try {
            if (typeof params === "string") params = JSON.parse(params);
        } catch (e) {
            console.error("Paramètres invalides", e);
            return;
        }

	        const root = document.getElementById(id);
	        if (!root) return;

	        const ids = Array.isArray(params?.ids) ? params.ids : [0, 1];
	        const nImgs = ids.length;
	        const showPoint = Boolean((params?.show_points !== undefined) ? params.show_points : params?.show_point);
	        const showZones = Boolean(params?.show_zones);
	        const zones = Array.isArray(params?.zones) ? params.zones : null;
	        const showZoneX = (params?.show_zone_x === undefined) ? true : Boolean(params.show_zone_x);
	        const showZoneY = (params?.show_zone_y === undefined) ? true : Boolean(params.show_zone_y);
	        const crop = Number.isFinite(Number(params?.crop)) ? Math.max(0, Math.floor(Number(params.crop))) : 0;
	        const seuilTexte = params?.seuil_texte;
	        const fontsize = params?.fontsize;
	        const zoneColors = params?.zone_colors;
	        const zoneLw = params?.zone_lw;
	        const pointNames = Array.isArray(params?.point_names) ? params.point_names : ["A", "B"];
	        const gridGapPx = params?.grid_gap_px;
	        const maxWidthPx = params?.max_width_px;
	        const showValues = (params?.show_values === undefined) ? true : Boolean(params.show_values);
	        const legends = Array.isArray(params?.legends) ? params.legends : null;
	        const legendOffsetPx = params?.legend_offset_px;
	        const legendFontPx = params?.legend_font_px;
	        const legendBackground = params?.legend_background;
	        const savePng = Boolean(params?.save_png);
	        const pngScaleRaw = Number(params?.png_scale);
	        const pngScale = (Number.isFinite(pngScaleRaw) && pngScaleRaw > 0) ? Math.min(8, pngScaleRaw) : 2;
	        const savePngPath = (params?.save_png_path !== undefined && params?.save_png_path !== null) ? String(params.save_png_path) : null;

	        const itemsHtml = ids.map((_, i) => `
	            <div class="mathadata-mnist-exemples-zones-item">
	                <div id="${id}-img-${i}"></div>
	                <div id="${id}-pt-${i}" class="mathadata-mnist-exemples-zones-point" style="${showPoint ? "" : "display:none;"}"></div>
	            </div>
	        `).join("");
	        const gridClass = (nImgs >= 3)
	            ? "mathadata-mnist-exemples-zones-grid mathadata-mnist-exemples-zones-grid--one-line"
	            : "mathadata-mnist-exemples-zones-grid";
	        root.innerHTML = `<div class="${gridClass}">${itemsHtml}</div>`;
	        const effectiveMaxWidthPx = maxWidthPx;

	        const fetchImg = async (index) => {
	            const safeIndex = Number.isFinite(Number(index)) ? Number(index) : 0;
	            return await window.mathadata.run_python_async(`get_data(index=${safeIndex})`);
	        };

	        const images = await Promise.all(ids.map((idx) => fetchImg(idx)));
	        const cropImg = (img, c) => {
	            if (!c || c <= 0) return img;
	            if (!Array.isArray(img) || img.length === 0 || !Array.isArray(img[0])) return img;
	            const h = img.length;
	            const w = img[0].length;
	            const maxCrop = Math.floor(Math.min(h, w) / 2) - 1;
	            if (c > maxCrop) c = Math.max(0, maxCrop);
	            if (c <= 0) return img;
	            return img.slice(c, h - c).map(row => Array.isArray(row) ? row.slice(c, w - c) : row);
	        };
	        const shiftZone = (zone, c, h, w) => {
	            if (!c || c <= 0) return zone;
	            if (!Array.isArray(zone) || zone.length !== 2) return null;
	            const A = zone[0], B = zone[1];
	            if (!Array.isArray(A) || !Array.isArray(B) || A.length !== 2 || B.length !== 2) return null;
	            const r0 = Number(A[0]) - c, c0 = Number(A[1]) - c;
	            const r1 = Number(B[0]) - c, c1 = Number(B[1]) - c;
	            if (![r0, c0, r1, c1].every(Number.isFinite)) return null;
	            const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
	            const rr0 = clamp(Math.round(r0), 0, h - 1);
	            const rr1 = clamp(Math.round(r1), 0, h - 1);
	            const cc0 = clamp(Math.round(c0), 0, w - 1);
	            const cc1 = clamp(Math.round(c1), 0, w - 1);
	            return [[rr0, cc0], [rr1, cc1]];
	        };

	        const thresholded = images.map(img => cropImg(window.mathadata.mnist_seuillage_0_200_250(img), crop));

	        let zonesAdj = zones;
	        const tRef = thresholded[0];
	        if (crop > 0 && Array.isArray(zonesAdj) && Array.isArray(tRef) && tRef.length > 0 && Array.isArray(tRef[0])) {
	            const h = tRef.length;
	            const w = tRef[0].length;
	            zonesAdj = zonesAdj.map(z => shiftZone(z, crop, h, w));
	        }

	        const optsBase = {
	            seuil_texte: seuilTexte,
	            fontsize,
	            show_values: showValues,
	            show_zones: showZones,
	            show_zone_x: showZoneX,
	            show_zone_y: showZoneY,
	            zones: zonesAdj,
	            zone_colors: zoneColors,
	            zone_lw: zoneLw,
	            max_width_px: effectiveMaxWidthPx,
	            grid_gap_px: gridGapPx,
	        };

	        const patchLegend = (legend) => {
	            if (!legend) return null;
	            const out = Object.assign({}, legend);
	            if (legendOffsetPx !== undefined) out.offset_px = legendOffsetPx;
	            if (legendFontPx !== undefined) out.font_px = legendFontPx;
	            if (legendBackground !== undefined) out.background = legendBackground;
	            return out;
	        };

	        thresholded.forEach((tImg, i) => {
	            const legend = (legends && legends.length > i) ? patchLegend(legends[i]) : null;
	            const opts = Object.assign({}, optsBase, { legend });
	            window.mathadata.mnist_afficher_image_pixels(`${id}-img-${i}`, tImg, opts);
	        });

        const escapeHtml = (s) => String(s)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");

	        if (showPoint && zonesAdj && zonesAdj.length >= 2) {
	            const defaultNames = ["A", "B", "C", "D"];
	            thresholded.forEach((tImg, i) => {
	                const x = window.mathadata.mnist_mean_zone(tImg, zonesAdj[0]);
	                const y = window.mathadata.mnist_mean_zone(tImg, zonesAdj[1]);

	                const pt = document.getElementById(`${id}-pt-${i}`);
	                if (!pt) return;

	                const name = escapeHtml(pointNames[i] || defaultNames[i] || `P${i + 1}`);
	                pt.innerHTML = `${name}(<span class="mathadata-carac-x">${escapeHtml(format1(x))}</span>&nbsp;;&nbsp;<span class="mathadata-carac-y">${escapeHtml(format1(y))}</span>)`;
	            });

	            // (Optionnel) Si MathJax est dispo, on peut typeset les blocs.
	            // Ici on affiche déjà une version HTML (toujours lisible, même sans MathJax).
	        }

	        if (savePng) {
	            const safeIds = ids.map(x => String(x).replace(/[^0-9A-Za-z_-]/g, "")).join("_");
	            const filename = `mnist_exemples_zones_${safeIds || "export"}.png`;

	            const htmlToText = (html) => {
	                if (!html) return "";
	                const tmp = document.createElement("div");
	                tmp.innerHTML = String(html);
	                let text = (tmp.textContent || "").trim();
	                // Remplacer chiffres en indice par unicode (x2 -> x₂) si présent
	                text = text.replace(/([0-9])/g, (d) => ({
	                    "0":"₀","1":"₁","2":"₂","3":"₃","4":"₄","5":"₅","6":"₆","7":"₇","8":"₈","9":"₉"
	                }[d] || d));
	                return text;
	            };

	            const downloadCanvasPng = async (canvas, fname) => {
	                if (!canvas) return { ok: false, blob: null };
	                const blob = await new Promise((resolve) => {
	                    if (canvas.toBlob) {
	                        canvas.toBlob((b) => resolve(b), "image/png");
	                    } else {
	                        const dataUrl = canvas.toDataURL("image/png");
	                        const bin = atob(dataUrl.split(",")[1]);
	                        const arr = new Uint8Array(bin.length);
	                        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
	                        resolve(new Blob([arr], { type: "image/png" }));
	                    }
	                });
	                if (!blob) return { ok: false, blob: null };
	                const url = URL.createObjectURL(blob);
	                const a = document.createElement("a");
	                a.href = url;
	                a.download = fname || "mnist_exemples_zones.png";
	                document.body.appendChild(a);
	                a.click();
	                a.remove();
	                setTimeout(() => URL.revokeObjectURL(url), 1000);
	                return { ok: true, blob };
	            };

    const maybeSavePngToPython = async (blob, path) => {
        if (!path) return true;
        if (!window.mathadata || typeof window.mathadata.run_python_async !== "function") {
            console.warn("[save_png] run_python_async indisponible, impossible d'enregistrer côté Python.");
            return false;
        }
        try {
            const dataUrl = await new Promise((resolve, reject) => {
                const r = new FileReader();
                r.onload = () => resolve(String(r.result || ""));
                r.onerror = () => reject(new Error("FileReader error"));
                r.readAsDataURL(blob);
            });
            const b64 = dataUrl.split(",")[1] || "";
            if (!b64) return false;
            const py = `from challenges.mnist.utilitaires import _mathadata_save_png_base64 as _s; _s(${JSON.stringify(String(path))}, ${JSON.stringify(String(b64))})`;
            await window.mathadata.run_python_async(py);
            return true;
        } catch (e) {
            console.error("[save_png] save to python failed", e);
            return false;
        }
    };

	            const exportViaCanvas = async () => {
	                try {
	                    const imgs = thresholded;
	                    if (!Array.isArray(imgs) || imgs.length === 0) return false;

	                    const n = imgs.length;
	                    const outerPad = 16;
	                    const gapBetween = 24;
	                    const pointFontPx = 18;
	                    const pointLineH = 26;

	                    const zonesForExport = zonesAdj;
	                    const zColors = Array.isArray(zoneColors) ? zoneColors : ["red", "blue"];
	                    const zLw = Number.isFinite(Number(zoneLw)) ? Number(zoneLw) : 6;

	                    // Mesure de marges pour légendes
	                    const legendFont = Number.isFinite(Number(legendFontPx)) ? Number(legendFontPx) : 14;
	                    const legendOff = Number.isFinite(Number(legendOffsetPx)) ? Number(legendOffsetPx) : 8;
	                    const legendBg = (legendBackground === undefined) ? true : Boolean(legendBackground);

	                    const perLayouts = imgs.map((img, i) => {
	                        const h = img.length;
	                        const w = img[0].length;

	                        const gapPxRaw = Number(gridGapPx);
	                        const gapPx = Number.isFinite(gapPxRaw) ? Math.max(0, Math.min(8, Math.round(gapPxRaw))) : 1;
	                        const padPx = gapPx;

	                        const maxWraw = effectiveMaxWidthPx;
	                        const maxWnum = Number(maxWraw);
	                        const maxW = (maxWraw === null || maxWraw === undefined || !Number.isFinite(maxWnum) || maxWnum <= 0) ? 360 : maxWnum;

	                        const available = Math.max(1, maxW - 2 * padPx - (w - 1) * gapPx);
	                        const minCellPx = showValues ? 8 : 4;
	                        const cellSize = Math.max(minCellPx, Math.min(22, Math.floor(available / w)));

	                        const fsNum = Number(fontsize);
	                        const fs = Number.isFinite(fsNum) ? fsNum : 10;
	                        const fontPx = showValues ? Math.max(4, Math.min(Math.floor(fs), cellSize - 1)) : 0;

	                        const gridW = 2 * padPx + w * cellSize + (w - 1) * gapPx;
	                        const gridH = 2 * padPx + h * cellSize + (h - 1) * gapPx;

	                        const legend = (legends && legends.length > i) ? patchLegend(legends[i]) : null;
	                        const leftLabels = legend && Array.isArray(legend.left) ? legend.left : [];
	                        const rightLabels = legend && Array.isArray(legend.right) ? legend.right : [];

	                        const labelsToMeasure = [];
	                        for (let zi = 0; zi < 2; zi++) {
	                            if (zi === 0 && !showZoneX) continue;
	                            if (zi === 1 && !showZoneY) continue;
	                            if (leftLabels[zi]) labelsToMeasure.push(htmlToText(leftLabels[zi]));
	                            if (rightLabels[zi]) labelsToMeasure.push(htmlToText(rightLabels[zi]));
	                        }

	                        return {
	                            w, h, gapPx, padPx, cellSize, fontPx, gridW, gridH,
	                            legend, leftLabels, rightLabels, labelsToMeasure,
	                            legendFont, legendOff, legendBg,
	                        };
	                    });

	                    // Canvas temporaire de mesure
	                    const meas = document.createElement("canvas");
	                    const mctx = meas.getContext("2d");
	                    if (!mctx) return false;
	                    mctx.font = `700 ${legendFont}px sans-serif`;

	                    perLayouts.forEach((lay) => {
	                        let leftW = 0, rightW = 0;
	                        const left = lay.legend && Array.isArray(lay.leftLabels) ? lay.leftLabels : [];
	                        const right = lay.legend && Array.isArray(lay.rightLabels) ? lay.rightLabels : [];
	                        for (let zi = 0; zi < 2; zi++) {
	                            if (zi === 0 && !showZoneX) continue;
	                            if (zi === 1 && !showZoneY) continue;
	                            const lt = htmlToText(left[zi]);
	                            const rt = htmlToText(right[zi]);
	                            if (lt) leftW = Math.max(leftW, mctx.measureText(lt).width);
	                            if (rt) rightW = Math.max(rightW, mctx.measureText(rt).width);
	                        }
	                        lay.marginLeft = leftW ? (leftW + legendOff + 10) : 0;
	                        lay.marginRight = rightW ? (rightW + legendOff + 10) : 0;
	                    });

	                    const blockWidths = perLayouts.map(lay => lay.marginLeft + lay.gridW + lay.marginRight);
	                    const maxBlockH = Math.max(...perLayouts.map(lay => lay.gridH + (showPoint ? pointLineH : 0)));

	                    const canvasW = outerPad * 2 + blockWidths.reduce((a, b) => a + b, 0) + gapBetween * (n - 1);
	                    const canvasH = outerPad * 2 + maxBlockH;

	                    const canvas = document.createElement("canvas");
	                    canvas.width = Math.ceil(canvasW * pngScale);
	                    canvas.height = Math.ceil(canvasH * pngScale);
	                    const ctx = canvas.getContext("2d");
	                    if (!ctx) return false;
	                    ctx.setTransform(pngScale, 0, 0, pngScale, 0, 0);
	                    ctx.imageSmoothingEnabled = false;

	                    ctx.fillStyle = "#fff";
	                    ctx.fillRect(0, 0, canvas.width, canvas.height);

	                    const seuilNum = Number(seuilTexte);
	                    const seuil = Number.isFinite(seuilNum) ? seuilNum : 130;

	                    const drawLegendBox = (x, y, text, color) => {
	                        if (!text) return;
	                        ctx.font = `700 ${legendFont}px sans-serif`;
	                        const metrics = ctx.measureText(text);
	                        const padX = 6;
	                        const padY = 4;
	                        const w = metrics.width + padX * 2;
	                        const h = legendFont + padY * 2;
	                        if (legendBg) {
	                            ctx.fillStyle = "rgba(255,255,255,0.92)";
	                            ctx.strokeStyle = "rgba(0,0,0,0.08)";
	                            ctx.lineWidth = 1;
	                            ctx.beginPath();
	                            ctx.roundRect ? ctx.roundRect(x, y - h / 2, w, h, 10) : ctx.rect(x, y - h / 2, w, h);
	                            ctx.fill();
	                            ctx.stroke();
	                        }
	                        ctx.fillStyle = color || "#111";
	                        ctx.textAlign = "left";
	                        ctx.textBaseline = "middle";
	                        ctx.fillText(text, x + (legendBg ? padX : 0), y);
	                    };

	                    const drawPointText = (cx, y, name, xVal, yVal) => {
	                        const left = `${name}(`;
	                        const mid = " ; ";
	                        const right = ")";
	                        const xStr = String(format1(xVal));
	                        const yStr = String(format1(yVal));

	                        ctx.font = `600 ${pointFontPx}px sans-serif`;
	                        const wLeft = ctx.measureText(left).width;
	                        const wX = ctx.measureText(xStr).width;
	                        const wMid = ctx.measureText(mid).width;
	                        const wY = ctx.measureText(yStr).width;
	                        const wRight = ctx.measureText(right).width;
	                        const total = wLeft + wX + wMid + wY + wRight;

	                        let x = cx - total / 2;
	                        ctx.textAlign = "left";
	                        ctx.textBaseline = "middle";

	                        ctx.fillStyle = "#111";
	                        ctx.fillText(left, x, y);
	                        x += wLeft;

	                        ctx.fillStyle = "#d64545";
	                        ctx.fillText(xStr, x, y);
	                        x += wX;

	                        ctx.fillStyle = "#111";
	                        ctx.fillText(mid, x, y);
	                        x += wMid;

	                        ctx.fillStyle = "#2f6fed";
	                        ctx.fillText(yStr, x, y);
	                        x += wY;

	                        ctx.fillStyle = "#111";
	                        ctx.fillText(right, x, y);
	                    };

	                    let xCursor = outerPad;
	                    for (let i = 0; i < n; i++) {
	                        const img = imgs[i];
	                        const lay = perLayouts[i];
	                        const gridX = xCursor + lay.marginLeft;
	                        const gridY = outerPad;

	                        // Fond du "cadre" (pour ressembler au widget)
	                        ctx.fillStyle = "#b3b3b3";
	                        ctx.fillRect(gridX, gridY, lay.gridW, lay.gridH);
	                        ctx.fillStyle = "#ffffff";
	                        ctx.fillRect(gridX, gridY, lay.gridW, lay.gridH);

	                        // Pixels
	                        ctx.textAlign = "center";
	                        ctx.textBaseline = "middle";
	                        ctx.font = `${lay.fontPx}px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace`;

	                        for (let r = 0; r < lay.h; r++) {
	                            const row = img[r];
	                            for (let c = 0; c < lay.w; c++) {
	                                const raw = Number(row[c]);
	                                const v = Number.isFinite(raw) ? raw : 0;
	                                const displayVal = Math.round(v);
	                                const g = Math.max(0, Math.min(255, displayVal));
	                                const x0 = gridX + lay.padPx + c * (lay.cellSize + lay.gapPx);
	                                const y0 = gridY + lay.padPx + r * (lay.cellSize + lay.gapPx);

	                                ctx.fillStyle = `rgb(${g},${g},${g})`;
	                                ctx.fillRect(x0, y0, lay.cellSize, lay.cellSize);

	                                if (showValues && lay.fontPx > 0) {
	                                    ctx.fillStyle = (displayVal >= seuil) ? "#000" : "#fff";
	                                    ctx.fillText(String(displayVal), x0 + lay.cellSize / 2, y0 + lay.cellSize / 2);
	                                }
	                            }
	                        }

	                        // Zones
	                        if (showZones && zonesForExport && Array.isArray(zonesForExport)) {
	                            const drawZone = (zone, color) => {
	                                if (!zone || !Array.isArray(zone) || zone.length !== 2) return;
	                                const A = zone[0], B = zone[1];
	                                if (!A || !B) return;
	                                const r0 = Math.round(Number(A[0]));
	                                const c0 = Math.round(Number(A[1]));
	                                const r1 = Math.round(Number(B[0]));
	                                const c1 = Math.round(Number(B[1]));
	                                if (![r0, c0, r1, c1].every(Number.isFinite)) return;

	                                const rmin = Math.min(r0, r1);
	                                const rmax = Math.max(r0, r1);
	                                const cmin = Math.min(c0, c1);
	                                const cmax = Math.max(c0, c1);
	                                const left = gridX + lay.padPx + cmin * (lay.cellSize + lay.gapPx);
	                                const top = gridY + lay.padPx + rmin * (lay.cellSize + lay.gapPx);
	                                const width = (cmax - cmin + 1) * lay.cellSize + (cmax - cmin) * lay.gapPx;
	                                const height = (rmax - rmin + 1) * lay.cellSize + (rmax - rmin) * lay.gapPx;
	                                const lwPx = Math.max(1, Math.round(Number(zLw || 6) * (lay.cellSize / 16)));

	                                ctx.strokeStyle = color || "red";
	                                ctx.lineWidth = lwPx;
	                                ctx.strokeRect(left - lwPx / 2, top - lwPx / 2, width + lwPx, height + lwPx);

	                                return { top, height };
	                            };

	                            let rect0 = null, rect1 = null;
	                            if (showZoneX && zonesForExport.length > 0) rect0 = drawZone(zonesForExport[0], zColors[0] || "red");
	                            if (showZoneY && zonesForExport.length > 1) rect1 = drawZone(zonesForExport[1], zColors[1] || "blue");

	                            // Légendes
	                            const legend = lay.legend;
	                            if (legend) {
	                                const leftLabels = Array.isArray(legend.left) ? legend.left : [];
	                                const rightLabels = Array.isArray(legend.right) ? legend.right : [];

	                                const zoneCenters = [];
	                                if (rect0) zoneCenters[0] = rect0.top + rect0.height / 2;
	                                if (rect1) zoneCenters[1] = rect1.top + rect1.height / 2;

	                                if (showZoneX && zoneCenters[0] !== undefined) {
	                                    const lt = htmlToText(leftLabels[0]);
	                                    const rt = htmlToText(rightLabels[0]);
	                                    if (lt) {
	                                        const tw = ctx.measureText(lt).width;
	                                        drawLegendBox(gridX - legendOff - tw - 10, zoneCenters[0], lt, zColors[0] || "#111");
	                                    }
	                                    if (rt) {
	                                        drawLegendBox(gridX + lay.gridW + legendOff, zoneCenters[0], rt, zColors[0] || "#111");
	                                    }
	                                }
	                                if (showZoneY && zoneCenters[1] !== undefined) {
	                                    const lt = htmlToText(leftLabels[1]);
	                                    const rt = htmlToText(rightLabels[1]);
	                                    if (lt) {
	                                        const tw = ctx.measureText(lt).width;
	                                        drawLegendBox(gridX - legendOff - tw - 10, zoneCenters[1], lt, zColors[1] || "#111");
	                                    }
	                                    if (rt) {
	                                        drawLegendBox(gridX + lay.gridW + legendOff, zoneCenters[1], rt, zColors[1] || "#111");
	                                    }
	                                }
	                            }
	                        }

	                        // Point caractéristique (sous l'image)
	                        if (showPoint && zonesForExport && zonesForExport.length >= 2) {
	                            const xVal = window.mathadata.mnist_mean_zone(img, zonesForExport[0]);
	                            const yVal = window.mathadata.mnist_mean_zone(img, zonesForExport[1]);
	                            const defaultNames = ["A", "B", "C", "D"];
	                            const name = String(pointNames[i] || defaultNames[i] || `P${i + 1}`);
	                            const centerX = gridX + lay.gridW / 2;
	                            const y = gridY + lay.gridH + pointLineH / 2;
	                            drawPointText(centerX, y, name, xVal, yVal);
	                        }

	                        xCursor += (lay.marginLeft + lay.gridW + lay.marginRight);
	                        if (i < n - 1) xCursor += gapBetween;
	                    }

	                    const dl = await downloadCanvasPng(canvas, filename);
	                    if (dl && dl.ok && dl.blob) {
	                        await maybeSavePngToPython(dl.blob, savePngPath);
	                        return true;
	                    }
	                    return false;
	                } catch (e) {
	                    console.error("[save_png] exportViaCanvas error", e);
	                    return false;
	                }
	            };

	            const createToolbar = () => {
	                const toolbar = document.createElement("div");
	                toolbar.className = "mathadata-noexport";
	                toolbar.style.display = "flex";
	                toolbar.style.justifyContent = "center";
	                toolbar.style.gap = "0.5rem";
	                toolbar.style.marginTop = "0.5rem";

	                const btn = document.createElement("button");
	                btn.type = "button";
	                btn.textContent = "Télécharger en PNG";
	                btn.style.fontFamily = "sans-serif";
	                btn.style.fontSize = "14px";
	                btn.style.padding = "6px 10px";
	                btn.style.borderRadius = "8px";
	                btn.style.border = "1px solid #ddd";
	                btn.style.background = "#fff";
	                btn.style.cursor = "pointer";

	                btn.addEventListener("click", async () => {
	                    // On privilégie l'export via Canvas (pas de dépendance réseau).
	                    const okCanvas = await exportViaCanvas();
	                    if (okCanvas) return;

	                    // Fallback : screenshot DOM via html2canvas (si dispo)
	                    const okDom = await window.mathadata._save_element_png(root, filename);
	                    if (!okDom) console.warn("[save_png] Export PNG échoué (html2canvas indisponible ?).");
	                });

	                toolbar.appendChild(btn);
	                root.appendChild(toolbar);
	            };

	            // Ajouter le bouton dans tous les cas (plus fiable que l'auto-download selon navigateur).
	            createToolbar();

	            // Tentative auto (peut être bloquée selon contexte)
	            await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
	            const okAuto = await exportViaCanvas();
	            if (!okAuto) {
	                console.warn("[save_png] Export auto échoué; clique sur 'Télécharger en PNG' ou vérifie la console.");
	            }
	        }
	    };
})();
""")

# coding=utf-8

###### VERSION 2/7 #####

# Import des librairies utilisées dans le notebook
import requests
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
#from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd

from utilitaires_common import *
import utilitaires_common as common

from IPython.display import display # Pour afficher des DataFrames avec display(df)

# Pour afficher les dataframe pandas en HTML
from IPython.display import display_html
from itertools import chain, cycle

# Import des strings pour lire l'export dans jupyter
if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))

### --- IMPORT DES DONNÉES ---
# Téléchargement et extraction des inputs contenus dans l'archive zip

print("Chargement de la base de donnée d'images en cours...")

inputs_zip_url = "https://dev.mathadata.fr/assets/fichiers_notebooks/mnist-papier.zip"
inputs_zip = requests.get(inputs_zip_url)
zf = ZipFile(BytesIO(inputs_zip.content))
zf.extractall()
zf.close()

chiffres = np.arange(10)
classes = chiffres

# # Inputs
with open('train_set.pickle', 'rb') as f:
    d_train = pickle.load(f).astype(int)

with open('test_set.pickle', 'rb') as f:
    d_test = pickle.load(f)
    
with open('train_labels.pickle', 'rb') as f:
    r_train = pickle.load(f)

N = len(d_train)

d = d_train[10,:,:].copy()
d2 = d_train[2,:,:].copy()

print("Images chargées !") 

# Noms de variables pour la question 'fainéant
chat = 'chat'
cat = 'cat'

class Mnist(common.Challenge):
    def __init__(self):
        super().__init__()
        self.d_train = d_train
        self.d_test = d_test
        self.r_train = r_train
        self.d = d
        self.d2 = d2
        self.classes = classes
        
    def affichage_dix(self, d=d_train, a=None, b=None, zones=[], y = r_train, n=10, axes=False):
        global r_train
        fig, ax = plt.subplots(1, n, figsize=(figw_full, figw_full / n + 1))
        
        # Cachez les axes des subplots
        for j in range(n):
            if axes:
                ax[j].set_xticks(np.arange(0,28,5))
                ax[j].set_yticks(np.arange(0,28,5))
                ax[j].xaxis.tick_top()
                ax[j].tick_params(axis='both', which='major', labelsize=7)

            else:
                ax[j].axis('off')
            imshow(ax[j], d[j])
            if y is not None:
                ax[j].set_title('$r = $'+str(y[j]), y=0, size=16, pad=-20)

            outline_selected(ax[j], a, b)
            for zone in zones:
                outline_selected(ax[j], zone[0], zone[1], zoneName=zone[2])
        
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.2 if axes else 0.05, hspace=0)
        plt.show()
        plt.close()

    def display_custom_selection(self, id):
        run_js(f'''
            setTimeout(() => {{
                window.mathadata.setup_zone_selection('{id}', '{json.dumps(self.d_train[0:8].tolist())}')
                const gif = document.createElement('img')
                gif.id = 'selection-gif'
                gif.src = '{files_url}/gif_explication_zone.gif'
                gif.alt = 'Cliquez sur les images pour sélectionner une zone'
                gif.style = 'position: absolute; top: calc(400px + 2rem); right: 0; left: 0; width: 25%; height: auto; margin-inline: auto;'
                const container = document.getElementById('{id}')
                container.style.position = 'relative'
                container.appendChild(gif)
            }}, 100)
        ''')

    def caracteristique(self, d):
        return moyenne(d)
    
    def caracteristique_custom(self, d):
        if self.custom_zone is None:
            return 0
        
        return moyenne_zone(d, self.custom_zone[0], self.custom_zone[1])

    def deux_caracteristiques(self, d):
        zone_1 = [(0,0), (13,27)]
        zone_2 = [(14,0), (27,27)]

        k1 = moyenne_zone(d, zone_1[0], zone_1[1])
        k2 = moyenne_zone(d, zone_2[0], zone_2[1])
        return (k1, k2)

    def deux_caracteristiques_custom(self, d):
        if self.custom_zones is None:
            return (0, 0)
        
        k1 = moyenne_zone(d, self.custom_zones[0][0], self.custom_zones[0][1])
        k2 = moyenne_zone(d, self.custom_zones[1][0], self.custom_zones[1][1])
        return (k1, k2)

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
        ax[0].set_xticks(np.arange(0,28,5))
        ax[0].xaxis.tick_top()
        outline_selected(ax[0], A1, B1, displayPoints, nameA='A1', nameB='B1', color='red')
        outline_selected(ax[0], A2, B2, displayPoints, nameA='A2', nameB='B2', color='C0')

        imshow(ax[1], image2)
        ax[1].set_title(titre2)
        ax[1].set_xticks(np.arange(0,28,5))
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

def outline_selected(ax, a=None, b=None, displayPoints=False, zoneName=None, zoneNamePos='right', nameA='A', nameB='B', color='red'):
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

    fig, ax = plt.subplots(figsize=(figw_full /2, figw_full /2))
    imshow(ax, image)
    ax.set_title(titre)
    ax.set_xticks(np.arange(0,28,5))
    ax.xaxis.tick_top()
    ax.set_title(titre)
    ax.xaxis.set_label_position('top') 
    outline_selected(ax, a, b, displayPoints)

    plt.show()
    plt.close()

def affichage_2_geo(display_k=False):
    zone_1 = [(0,0), (13,27)]
    zone_2 = [(14,0), (27,27)]
    deux_caracteristiques = common.challenge.deux_caracteristiques

    images = np.array([common.challenge.d2, common.challenge.d])
    fig, ax = plt.subplots(1, len(images), figsize=(figw_full, figw_full / 2))
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
        df = pd.DataFrame({'$x$': reversed[:,0], '$y$': reversed[:,1], '$r$': ['$r_1$ = 2 ou 7 ?', '$r_2$ = 2 ou 7 ?']})
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
                display(df.iloc[numero_ligne_debut:numero_ligne_fin+1, numero_colonne_debut:numero_colonne_fin+1])
                return 
    display(df)
    return

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
    return np.mean(arr[numero_ligne_debut:numero_ligne_fin+1, numero_colonne_debut:numero_colonne_fin+1])

 
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

def update_selected(A,B):
    common.challenge.custom_zone = (A,B)
    
# JS

styles = '''
    .image-table {
        aspect-ratio: 1;
        height: 100%;
        margin: auto !important;
        border-width: 0;
        margin: 0;
        border-collapse: collapse;
        border-spacing: 0;
        table-layout: fixed; /* Ensures uniform cell sizing */
    }
    
    .image-table tr {
        height: calc(1 / 28 * 100%);
    }
    
    .image-table td {
        width: calc(1 / 28 * 100%);
        aspect-ratio: 1;
        padding: 0;
        margin: 0;
        border: none;
    }

    .image-table-selection {
        background-color: red;
    }
    
    .image-table-selection td:hover {
        outline: 1px solid red;
    }
    
    .image-table-selection .td-selected {
        opacity: 0.3;
    }
    
    .custom-images {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        width: 100%;
        height: 400px;
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

run_js("""
let isSelecting = false;
let startCell = null;
let endCell = null;

document.addEventListener('mouseup', () => {
    isSelecting = false;
});

function toggleCellSelection(cell, select) {
    if (select) {
        cell.classList.add('td-selected');
    } else {
        cell.classList.remove('td-selected');
    }
}

function clearSelection() {
    window.mathadata.image_tables.forEach(table => {
        const selectedCells = table.querySelectorAll('.td-selected');
        selectedCells.forEach(cell => cell.classList.remove('td-selected'));
    });
}

let exec_python = null;
function selectCells(start, end) {
    const startRowIndex = start.parentElement.rowIndex;
    const startColIndex = start.cellIndex;
    const endRowIndex = end.parentElement.rowIndex;
    const endColIndex = end.cellIndex;

    const minRowIndex = Math.min(startRowIndex, endRowIndex);
    const maxRowIndex = Math.max(startRowIndex, endRowIndex);
    const minColIndex = Math.min(startColIndex, endColIndex);
    const maxColIndex = Math.max(startColIndex, endColIndex);

    for (const table of window.mathadata.image_tables) {
        for (let i = minRowIndex; i <= maxRowIndex; i++) {
            for (let j = minColIndex; j <= maxColIndex; j++) {
                const cell = table.rows[i].cells[j];
                toggleCellSelection(cell, true);
            }
        }
    }


    if (exec_python) {
        clearTimeout(exec_python);
    }

    exec_python = setTimeout(() => {
        document.getElementById('selection-gif')?.remove();
        window.mathadata.run_python(`update_selected((${minRowIndex}, ${minColIndex}), (${maxRowIndex}, ${maxColIndex}))`, () => {
            window.mathadata.on_custom_update()
        }) 
    }, 200);
}

function updateSelectionRange(start, end) {
    const startRowIndex = start.parentElement.rowIndex;
    const startColIndex = start.cellIndex;
    const endRowIndex = end.parentElement.rowIndex;
    const endColIndex = end.cellIndex;

    const table = start.parentElement.parentElement;
    startCell = table.rows[Math.min(startRowIndex, endRowIndex)].cells[Math.min(startColIndex, endColIndex)];
    endCell = table.rows[Math.max(startRowIndex, endRowIndex)].cells[Math.max(startColIndex, endColIndex)];
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
                startCell = target;
                endCell = target;
                clearSelection();
                selectCells(startCell, endCell);
            }
        });

        table.addEventListener('mouseover', (event) => {
            if (isSelecting) {
                const target = event.target;
                if (target.tagName === 'TD') {
                    endCell = target;
                    clearSelection();
                    selectCells(startCell, endCell);
                }
            }
        });

        table.addEventListener('mouseup', () => {
            isSelecting = false;
            updateSelectionRange(startCell, endCell);
        });

        return table
    }

}

window.mathadata.setup_zone_selection = (id, matrixes) => {
    matrixes = JSON.parse(matrixes)
    
    const container = document.getElementById(id)
    container.innerHTML = '';
    container.classList.add('custom-images')
    
    window.mathadata.image_tables = [];
    for (let i = 0; i < matrixes.length; i++) {
        const table = document.createElement('div')
        table.id = `${id}-image-${i}`
        container.appendChild(table)
        window.mathadata.image_tables.push(window.mathadata.affichage(`${id}-image-${i}`, matrixes[i], {with_selection: true}))
    }
}
""")
    
### Validation cellules ###

validation_execution_nombre_3 = MathadataValidateVariables({
    'Nombre_3': {
        'value': np.sum(r_train[:10] == 3),
        'errors': [
            {
                'value':np.sum(r_train[:10] == 3),
                'else' :"As-tu bien regardé les 10 premières lignes du tableau ? "
            },
        ]
    }
    })

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


def validate_pixel_noir(errors, answers):
    answers['pixel'] = int(d[17][15])
    if d[17][15] == 0:
        return True
    elif d[17][15] == 254:
        errors.append("Tu n'as pas changé la valeur du pixel, il vaut toujours 254")
    else:
        errors.append("Tu as bien changé la valeur mais ce n'est pas la bonne. Relis l'énoncé pour voir la valeur à donner pour un pixel noir.")
    return False
 
validation_execution_affichage = MathadataValidate(success="")
validation_question_pixel = MathadataValidateVariables({
    'pixel': {
        'value': int(d[18,15]),
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 255
                },
                'else': "Ta réponse n'est pas bonne. Les pixels peuvent uniquement avoir des valeurs entre 0 et 255."
            },
            {
                'value': int(d[15,18]),
                'if': "Attention ! Les coordonnées sont données en (ligne, colonne)."
            }
        ]
    }
})
validation_question_pixel_noir = MathadataValidate(success="Bravo, le pixel est devenu noir", function_validation=validate_pixel_noir)
validation_question_moyenne = MathadataValidateVariables({'moyenne_zone_4pixels': np.mean(d[14:16,15:17])}, success="Bravo, la moyenne vaut en effet (142 + 154 + 0 + 0) / 4 = 74")

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
}, success="C'est la bonne réponse. L'image de 7 a presque la même moyenne sur la moitié haute et la moitié basse. L'image de 2 a une moyenne plus élevée sur la moitié basse car il y a plus de pixels blancs.",
    on_success=on_success_2_caracteristiques)

validation_execution_def_caracteristiques_ripou = MathadataValidate(success="")


# Notebook histogramme

validation_question_hist_1 = MathadataValidateVariables({
    'r_histogramme_orange': {
        'value': common.challenge.classes[1],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"r_histogramme_orange n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    },
    'r_histogramme_bleu': {
        'value': common.challenge.classes[0],
        'errors': [
            {
                'value': {
                    'in': common.challenge.classes,
                },
                'else': f"r_histogramme_bleu n'a pas la bonne valeur. Tu dois répondre par {common.challenge.classes[0]} ou {common.challenge.classes[1]}."
            }
        ]
    }
}, success="C'est la bonne réponse ! Les images de 7 ont souvent moins de pixels blancs que les images de 2. C'est pourquoi leur caractéristique est souvent plus petite.")

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
        'value': 237,
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
})

validation_question_hist_seuil = MathadataValidateVariables({
    't': {
        'value': 34,
        'errors': [
            {
                'value': {
                    'max': 30,
                },
                'if': "Ton seuil t est trop bas. Regarde ou les 2 histogrammes se croisent pour trouver le meilleur seuil."
            },
            {
                'value': {
                    'min': 38,
                },
                'if': "Ton seuil t est trop haut. Regarde ou les 2 histogrammes se croisent pour trouver le meilleur seuil."
            },
            {
                'value': {
                    'min': 30,
                    'max': 38
                },
                'if': "Tu te rapproches mais ce n'est pas le meilleur seuil. Il doit y avoir plus de 7 que de 2 qui ont une caractéristique x inférieure ou égale à t et inversement pour x supérieur à t."
            }
        ]
        
    }
}, success="Bravo, ton seuil est maintenant optimal !")

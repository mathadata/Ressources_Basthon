from IPython.display import display, IFrame, clear_output # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys
import numpy as np
import time
#import ipywidgets as widgets
# import mplcursors

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common
if not sequence:
    from themes.geo.utilitaires import *
else:
    from utilitaires_geo import *


def tracer_6000_points():
    id = uuid.uuid4().hex

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=common.challenge.d_train, r_train=common.challenge.r_train)
    params = {
        'points': c_train_par_population,
        'additionalPoints': {
            'A': [27,60],
            'B': [55,32],
            'C': [65,61]
        },
        'hover': True,
    }

    run_js(f"mathadata.add_observer('{id}-chart', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
            <canvas id="{id}-chart"></canvas>
    '''))


def tracer_10_points_droite():
    data = common.challenge.dataset_10_centroides
    labels = common.challenge.labels_10_centroides

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=data, r_train=labels)

    id = uuid.uuid4().hex

    params = {
        'points': c_train_par_population,
        'centroides': {},
        'droite': {
          'avec_zones': True,
        },
        'drag': True,
    }

    run_js(f"mathadata.add_observer('{id}-chart', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
            <canvas id="{id}-chart"></canvas>
    '''))


def tracer_10_points_centroides():
    data = common.challenge.dataset_10_centroides
    labels = common.challenge.labels_10_centroides

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=data, r_train=labels)

    id = uuid.uuid4().hex

    params = {
        'points': c_train_par_population,
        'centroides': {},
        'hover': {
          'type': 'distance',
        }
    }

    run_js(f"mathadata.add_observer('{id}-chart', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'<canvas id="{id}-chart"></canvas>'))


def tracer_points_centroides(id=None, carac=None, droite=False, initial_hidden=False):
    if id is None:
        id = uuid.uuid4().hex
    score_div = f'<div id="{id}-score-container" style="text-align: center; font-weight: bold; font-size: 2rem;">Pourcentage d\'erreur : <span id="{id}-score">...</span></div>'

    if carac is None:
        carac = common.challenge.deux_caracteristiques

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    params = {
        'points': c_train_par_population,
        'centroides': {},
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'droite': droite,
        'compute_score': True,
    }

    run_js(f"mathadata.add_observer('{id}-container', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
        <div id="{id}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
            {droite and score_div}
            <canvas id="{id}-chart"></canvas>
        </div>
    '''))


    
def affichage_zones_custom(A1, B1, A2, B2):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    tracer_points_centroides(carac=common.challenge.deux_caracteristiques_custom,droite=True)

def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_centroides(id=id, carac=common.challenge.deux_caracteristiques_custom, droite=True, initial_hidden=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                mathadata.update_points('{id}', {{points}})

                const container = document.getElementById('{id}-container');
                if (container) {{
                    container.style.visibility = 'visible';
                }}
            }})
        }}
    ''')

def calculer_score_droite():
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite)

def calculer_score_custom_droite():
    calculer_score_droite_geo(custom=True, validate=10, error_msg="Continuez à chercher 2 zones pour avoir moins de 10% d'erreur.")

### Attempting jsxgraph
def _build_srcdoc(points, title, centroid_name='M',
                  width=600, height=500,
                  xmin=-3, xmax=7, ymin=-5, ymax=5):
    """
    Build HTML for an iframe srcdoc showing draggable points and midpoint/centroid.
    """
    # JS to create points
    create_pts = []
    for i,(x,y) in enumerate(points):
        create_pts.append(
            f"var P{i}=board.create('point',[{x},{y}],{{name:'{chr(65+i)}',size:4, snapToGrid: true, snapSizeX: 0.1, snapSizeY: 0.1}});"
        )
    # JS for centroid or midpoint
    n = len(points)
    if n==2:
        centroid_js = "var C=board.create('midpoint',[P0,P1],{name:'%s',size:4,face:'cross'});" % centroid_name
    else:
        sumx = '+'.join([f'P{i}.X()' for i in range(n)])
        sumy = '+'.join([f'P{i}.Y()' for i in range(n)])
        centroid_js = (
            "var C=board.create('point',["
            f"function(){{return({sumx})/{n}}},"
            f"function(){{return({sumy})/{n}}}],{{name:'{centroid_name}',size:4,face:'cross'}});"
        )
    pts_js = '\n          '.join(create_pts + [centroid_js])

    # HTML document
    html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
  <style>html,body{{margin:0;padding:0;overflow:hidden;font-family:sans-serif;}}
    #title{{text-align:center;font-weight:bold;margin:4px 0;}}
    #coords{{position:absolute;bottom:8px;left:50%;transform:translateX(-50%);
        background:rgba(0,0,0,0.6);color:#fff;padding:4px 8px;border-radius:4px;font-size:12px;}}

  </style>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraph.css"/>
  <script src="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraphcore.js"></script>
</head><body>
  <div id="title">{title}</div>
  <div id="board" style="width:{width}px;height:{height}px;"></div>
  <div id="coords">Coords</div>
  <script>
    setTimeout(function(){{
      var board=JXG.JSXGraph.initBoard('board',{{boundingbox:[{xmin},{ymax},{xmax},{ymin}],
        axis:true,showNavigation:false,showCopyright:false}});
      {pts_js}
      function upd(){{document.getElementById('coords').innerText=
            'Coordonnées du point moyen : ('+C.X().toFixed(2)+','+C.Y().toFixed(2)+')';}}
      board.on('update',upd);upd();
    }},50);
  </script>
</body></html>
"""
    # Escape quotes for embedding
    return html.replace('"','&quot;')

# Precompute srcdocs
_SRC_DOCS = {
    '2': _build_srcdoc([(-1,1),(2,-2)], "Point moyen pour 2 points", 'M'),
    '3': _build_srcdoc([(-1,1),(2,-2),(-1,-2)], "Point moyen pour 3 points", 'G'),
    '4': _build_srcdoc([(-1,1),(2,-2),(-1,-2),(2,1)], "Point moyen pour 4 points", 'G'),
}


def afficher_manip_points_moyens():
    """
    Display a dropdown to choose among 2/3/4-point canvases and show the selected iframe.
    """
    # Unique IDs for elements
    sel_id = 'sel_' + uuid.uuid4().hex
    reset_id = 'rst_' + uuid.uuid4().hex
    frame_ids = {n: 'frm_' + uuid.uuid4().hex for n in ['2','3','4']}

    # Prebuilt srcdoc strings
    src2 = _SRC_DOCS['2']
    src3 = _SRC_DOCS['3']
    src4 = _SRC_DOCS['4']

    # HTML with three hidden iframes and controls
    html = f'''
<div style="margin-bottom:8px;">
  <select id="{sel_id}">
    <option value="2">2 Points</option>
    <option value="3">3 Points</option>
    <option value="4">4 Points</option>
  </select>
  <button id="{reset_id}">Reinitialiser</button>
</div>
<div id="canvas_container">
  <iframe id="{frame_ids['2']}" srcdoc="{src2}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
  <iframe id="{frame_ids['3']}" srcdoc="{src3}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
  <iframe id="{frame_ids['4']}" srcdoc="{src4}" width="600" height="450" style="border:0;overflow:hidden;display:none;"></iframe>
</div>
<script>
(function() {{
  var sel = document.getElementById('{sel_id}');
  var btn = document.getElementById('{reset_id}');
  var frames = {{
    '2': '{frame_ids['2']}',
    '3': '{frame_ids['3']}',
    '4': '{frame_ids['4']}'
  }};
  function showCanvas(v) {{
    Object.keys(frames).forEach(function(k) {{
      var f = document.getElementById(frames[k]);
      f.style.display = (k === v ? 'block' : 'none');
    }});
  }}
  sel.addEventListener('change', function() {{ showCanvas(sel.value); }});
  btn.addEventListener('click', function() {{
    var f = document.getElementById(frames[sel.value]);
    // reload iframe
    f.srcdoc = f.srcdoc;
  }});
  // initial display
  showCanvas(sel.value);
}})();
</script>
'''
    display(HTML(html))


def afficher_manip_distance(fixed=False):
    """
    Render the interactive JSXGraph demo via an iframe(srcdoc) so that
    inline <script> tags aren't stripped by JupyterLab's sanitizer.
    """
    box_id   = f"jxgbox_{uuid.uuid4().hex}"
    js_fixed = 'true' if fixed else 'false'

    # Build the full HTML page for the iframe
    page = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Distance par rapport à deux points moyens</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraph.css" />
    <script src="https://cdn.jsdelivr.net/npm/jsxgraph/distrib/jsxgraphcore.js"></script>
    <style>
      html, body {{ margin:0; padding:0; overflow:hidden; font-family:sans-serif; }}
      #title {{ width:600px; text-align:center; font-size:16px; font-weight:bold; }}
    </style>
  </head>
  <body>
    <div id="title">Distance par rapport à deux points moyens</div>
    <div id="{box_id}" class="jxgbox" style="width:600px; height:600px;"></div>
    <script>
      // Initialize board
      var board = JXG.JSXGraph.initBoard('{box_id}', {{
        boundingbox:[-1,9,9,-1], axis:true, grid:true,
        showNavigation:false, showCopyright:false
      }});
      // Create points
      var C1 = board.create('point',[2,2],{{name:'A', color:'blue', size:4, fixed:true, label:{{fontSize:14,autoPosition:true}}}});
      var C2 = board.create('point',[7,7],{{name:'B', color:'orange', size:4, fixed:true, label:{{fontSize:14,autoPosition:true}}}});
      var P  = board.create('point',[6,5],{{name:'P', color:'green', size:3, fixed:{js_fixed}, snapToGrid:true, snapSizeX:0.5, snapSizeY:0.5, label:{{fontSize:14,autoPosition:true}}}});
      // Hypotenuse segments
      board.create('segment',[P,C1],{{color:'blue', strokeWidth:2}});
      board.create('segment',[P,C2],{{color:'orange', strokeWidth:2}});
      // Auxiliary corner points
      var P1_C1 = board.create('point',[function(){{return 2;}}, function(){{return P.Y();}}],{{visible:false}});
      var P1_C2 = board.create('point',[function(){{return 7;}}, function(){{return P.Y();}}],{{visible:false}});
      // Leg segments for C1 (blue)
      board.create('segment',[P,P1_C1],{{dash:2, color:'blue'}});
      board.create('segment',[P1_C1,C1],{{dash:2, color:'blue'}});
      // Leg segments for C2 (orange)
      board.create('segment',[P,P1_C2],{{dash:2, color:'orange'}});
      board.create('segment',[P1_C2,C2],{{dash:2, color:'orange'}});
      // Length labels
      board.create('text',[
        function(){{return (P.X()+C1.X())/2;}}, function(){{return (P.Y()+C1.Y())/2-0.3;}},
        function(){{return Math.hypot(P.X()-C1.X(), P.Y()-C1.Y()).toFixed(2);}}
      ],{{visible:false, fontSize:14, color:'blue', fontWeight:'bold'}});
      board.create('text',[
        function(){{return (P.X()+2)/2;}}, function(){{return P.Y()+0.2;}},
        function(){{return Math.abs(P.X()-2).toFixed(2);}}
      ],{{fontSize:12, color:'blue'}});
      board.create('text',[
        function(){{return 2-0.4;}}, function(){{return (P.Y()+2)/2;}}, 
        function(){{return Math.abs(P.Y()-2).toFixed(2);}}
      ],{{fontSize:12, color:'blue'}});
      board.create('text',[
        function(){{return (P.X()+C2.X())/2;}}, function(){{return (P.Y()+C2.Y())/2+0.3;}}, 
        function(){{return Math.hypot(P.X()-C2.X(), P.Y()-C2.Y()).toFixed(2);}}
      ],{{visible:false, fontSize:14, color:'orange', fontWeight:'bold'}});
      board.create('text',[
        function(){{return (P.X()+7)/2;}}, function(){{return P.Y()-0.2;}}, 
        function(){{return Math.abs(P.X()-7).toFixed(2);}}
      ],{{fontSize:12, color:'orange'}});
      board.create('text',[
        function(){{return 7+0.2;}}, function(){{return (P.Y()+7)/2;}}, 
        function(){{return Math.abs(P.Y()-7).toFixed(2);}}
      ],{{fontSize:12, color:'orange'}});
      // Legend
      board.create('text',[0.5, 8.85,
        'A : point moyen des 2<br/>B : point moyen des 7<br/>P : caractéristiques d\\'une image'
      ],{{
        anchorX:'left', anchorY:'top', fontSize:14,
        useMathJax:false, cssStyle:'text-align:left;'
      }});
    </script>
  </body>
</html>"""

    # Escape quotes for embedding in the srcdoc attribute
    srcdoc = page.replace('"', '&quot;')

    # Display the iframe
    iframe = (
        f'<iframe srcdoc="{srcdoc}" '
        'style="width:620px; height:620px; border:none;"></iframe>'
    )
    display(HTML(iframe))


### Validation

def function_validation_classes(errors, answers):
    classe_point_A = answers.get('classe_point_A')
    classe_point_B = answers.get('classe_point_B')
    classe_point_C = answers.get('classe_point_C')

    # Vérifier que les valeurs sont des entiers
    if not isinstance(classe_point_A, int) or not isinstance(classe_point_B, int) or not isinstance(classe_point_C, int):
        errors.append("Les classes des points doivent être des nombres entiers.")
        return False

    # Vérifier les valeurs attendues
    if classe_point_A != 2:
        errors.append("La classe du point A n'est pas correcte. Réessayez.")
    if classe_point_B != 7:
        errors.append("La classe du point B n'est pas correcte. Réessayez.")
    if classe_point_C != 2:
        errors.append("La classe du point C n'est pas correcte. Ce point n'est pas facile à classer. Il aurait bien pu s'agir d'un 2. Réessayez.")

    return len(errors) == 0

def function_validation_valeurs_deplacement_horizontal(errors, answers):
    x2 = answers.get('deplacement_horizontal_2')
    x3 = answers.get('deplacement_horizontal_3')
    x4 = answers.get('deplacement_horizontal_4')

    # Pour vérifier que les valeurs sont bien des floats ou int
    for var_name, value in [('deplacement_horizontal_2', x2), ('deplacement_horizontal_3', x3), ('deplacement_horizontal_4', x4)]:
        if not isinstance(value, (int, float)):
            errors.append(f"La valeur de {var_name} doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
            return False

    # Comparaison avec tolérance
    if abs(x2 - 1.5) > 1e-2:
        errors.append("deplacement_horizontal_2 n'est pas correct. Bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(x3 - 1) > 1e-2:
        errors.append("deplacement_horizontal_3 n'est pas correct. Pensez à cliquer sur 3 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(x4 - 0.75) > 1e-2:
        errors.append("deplacement_horizontal_4 n'est pas correct. Pensez à cliquer sur 4 Points en haut à gauche. Bougez le point B sur la figure et regardez de combien bouge M.")

    return len(errors) == 0

def function_validation_valeurs_deplacement_vertical(errors, answers):
    y2 = answers.get('deplacement_vertical_2')
    y3 = answers.get('deplacement_vertical_3')
    y4 = answers.get('deplacement_vertical_4')

    # Pour vérifier que les valeurs sont bien des floats ou int
    for var_name, value in [('deplacement_vertical_2', y2), ('deplacement_vertical_3', y3), ('deplacement_vertical_4', y4)]:
        if not isinstance(value, (int, float)):
            errors.append(f"La valeur de {var_name} doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
            return False

    # Comparaison avec tolérance
    if abs(y2 - 3) > 1e-2:
        errors.append("deplacement_vertical_2 n'est pas correct. Bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(y3 - 2) > 1e-2:
        errors.append("deplacement_vertical_3 n'est pas correct. Pensez à cliquer sur 3 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")
    if abs(y4 - 1.5) > 1e-2:
        errors.append("deplacement_vertical_4 n'est pas correct. Pensez à cliquer sur 4 Points en haut à gauche. Puis bougez le point B sur la figure et regardez de combien bouge M.")

    return len(errors) == 0

validation_question_score_droite = MathadataValidateVariables({
    'erreur_10': {
        'value': 20,
        'errors': [
            {
                'value': {
                    'min': 0,
                    'max': 100,
                },
                'else': "Ce n'est pas la bonne valeur. Le pourcentage d'erreur doit être compris entre 0 et 100."
            },
            {
                'value': 2,
                'if': "Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreurs et non le pourcentage d'erreur."
            }
        ]
    }
},
    success="C'est la bonne réponse. 2 points bleus sont plus proches du point moyen des 7 que de celui des 2 : il vont donc être classés \"7\". On a donc deux erreurs sur 10 points, ce qui fait 20%.",
)

validation_execution_caracteristiques_custom = MathadataValidate(success="")
# Définir les variables de validation
validation_question_classes = MathadataValidateVariables({
    'classe_point_A': 2,
    'classe_point_B': 7,
    'classe_point_C': 2
}, 
    function_validation=function_validation_classes,
    success="Bien joué ! Ce n'était pas facile de voir que le point C est un 2. Il aurait très bien pu être un 7.")

# Exécuter la validation
validation_execution_classes = MathadataValidate(success="Les classes des points sont correctes; tu peux passer à la suite.")
validation_deplacement_horizontal = MathadataValidateVariables(
    {
        'deplacement_horizontal_2': 1.5,
        'deplacement_horizontal_3': 1,
        'deplacement_horizontal_4': 0.75,
    },
    tips=[
        {
            'seconds':20,
            'tip': 'Bougez le point B sur la figure ci-dessus et regardez de combien bouge M.'
        },
        {
            'seconds':40,
            'tip': 'Pensez à changer le nombre de points en haut à gauche.'
        }
    ],
    function_validation=function_validation_valeurs_deplacement_horizontal
)

validation_deplacement_vertical = MathadataValidateVariables(
    {
        'deplacement_vertical_2': 3,
        'deplacement_vertical_3': 2,
        'deplacement_vertical_4': 1
    },
    tips=[
        {
            'seconds': 20,
            'tip': 'Bougez le point B sur la figure ci-dessus et regardez de combien bouge M.'
        },
        {
            'seconds':40,
            'tip': 'Pensez à changer le nombre de points en haut à gauche.'
        }
    ],
    function_validation=function_validation_valeurs_deplacement_vertical
)

validation_execution_afficher_manip_distance = MathadataValidate(success="")

### ATTENTION CHALLENGE DÉPENDANT - À FACTORISER 
validation_distance_point_moyen = MathadataValidateVariables({
    'AP': 5,
    'BP': {
        'value': {
            'min': 2.22,
            'max': 2.25,
            }
        },
    'classe_P': {
        'value': 7,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_P n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            }
        ]
    }
},
    success="Bien joué, tu peux passer à la suite")

validation_execution_10_points_droite = MathadataValidate(success="")

def function_validation_recherche_mediatrice(errors, answers):
    point_1 = answers.get('point_1')
    point_2 = answers.get('point_2')
    point_3 = answers.get('point_3')

    for var_name, value in [('point_1', point_1), ('point_2', point_2), ('point_3', point_3)]:
        if not isinstance(value, tuple):
            errors.append(f"La valeur de {var_name} doit être un couple de deux coordonnées. Ex (2, 3).")
            return False
        for coord in value:
            if not isinstance(coord, (int, float)):
                errors.append(f"La valeur de {var_name} doit être un couple de deux coordonnées. Ex (2, 3).")
                return False

    points = [point_1, point_2, point_3]
    points_name = ['point_1', 'point_2', 'point_3']
    for i in range(3):
        point = points[i]
        point_name = points_name[i]
        x, y = point[0], point[1]

        # on vérifie que les trois points sont différents
        if not (point_1 != point_2 and point_1 != point_3 and point_2 != point_3):
            errors.append('Les trois points doivent être différents')
            return False

        # On vérifie si le point est sur la droite médiatrice
        if x+y-9!=0:
            errors.append(f'le point {point_name} n\'est pas à la même distance de A et B')

    return len(errors) == 0

validation_recherche_mediatrice = MathadataValidateVariables({
    'point_1': None,
    'point_2': None,
    'point_3': None
}, function_validation=function_validation_recherche_mediatrice)

validation_execution_tracer_6000_points = MathadataValidate(success="")
validation_execution_afficher_manip_points_moyens = MathadataValidate(success="")
validation_execution_afficher_manip_distance_2 = MathadataValidate(success="")
validation_execution_tracer_points_centroides = MathadataValidate(success="")
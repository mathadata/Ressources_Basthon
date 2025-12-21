from utilitaires_common import *

# import mplcursors

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if not sequence:
    from themes.geo.utilitaires import *
    import themes.geo.utilitaires as geo
else:
    from utilitaires_geo import *
    import utilitaires_geo as geo


def tracer_20_points_droite(show_eq=False):
    afficher_separation_line(show_equation=show_eq)


def tracer_10_points_droite(dataset=common.challenge.dataset_10_points, labels=common.challenge.labels_10_points):
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques,
                                                      d_train=dataset, r_train=labels)

    display_id = uuid.uuid4().hex

    params = {
        'points': c_train_par_population,
        'droite': common.challenge.droite_10_points,
        'hover': True,
        'force_origin': True,
        'equation_hide': True
    }
    params['droite']['avec_zones'] = True
    params['droite']['mode'] = 'affine'

    run_js(
        f"mathadata.add_observer('{display_id}-chart', () => window.mathadata.tracer_points('{display_id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'''
        <canvas id="{display_id}-chart"></canvas>
    '''))


def tracer_points_droite_a_b_c():
    tracer_points_droite(directeur=True, values={'a': 4, 'b': -2, 'c': 10})


def tracer_points_droite_c():
    tracer_points_droite(directeur=True, inputs='c', values={'a': 2, 'b': -3, 'c': 0}, ranges={'c': {'min': -20, 'max': 50, 'step': 5}})
    
def tracer_points_droite_a_b():
    tracer_points_droite(directeur=True, inputs=('a', 'b'), values={'a': -6, 'b': 2, 'c': 5}, ranges={'a': {'min': -10, 'max': 10, 'step': 0.5}, 'b': {'min': -10, 'max': 10, 'step': 0.5}})

def tracer_points_droite(
    id_content=None,
    display_value="range",
    carac=None,
    initial_hidden=False,
    save=False,
    directeur=False,
    inputs=('a', 'b', 'c'),
    values={'a': 5, 'b': -20, 'c': 0},
    ranges=None,  # <-- NOUVEAU : dict optionnel { 'a': {...}, 'b': (...), ... }
):
    """
    ranges (optionnel) : dictionnaire qui peut contenir pour chaque paramètre :
      - dict: {'min': ..., 'max': ..., 'step': ...} (champs optionnels)
      - tuple/list: (min, max) ou (min, max, step)

    Ex:
      ranges={
        'a': {'min': -20, 'max': 20, 'step': 0.5},
        'b': (-30, 30, 1),
        'c': {'min': -200, 'max': 200}  # step par défaut conservé
      }
    """

    if id_content is None:
        id_content = uuid.uuid4().hex

    if carac is None:
        carac = common.challenge.deux_caracteristiques

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    # --- RANGES PAR DÉFAUT ---
    default_ranges = {
        'a': {'min': -5,  'max': 5,  'step': 0.1},
        'b': {'min': -5,  'max': 5,  'step': 0.1},
        'c': {'min': -10, 'max': 10, 'step': 1},
    }

    def _normalize_range(key: str):
        """Fusionne ranges[key] avec les defaults, sans casser si rien n'est fourni."""
        base = dict(default_ranges.get(key, {}))
        if not ranges or key not in ranges or ranges[key] is None:
            return base

        r = ranges[key]
        if isinstance(r, (tuple, list)):
            if len(r) >= 2:
                base['min'], base['max'] = r[0], r[1]
            if len(r) >= 3:
                base['step'] = r[2]
            return base

        if isinstance(r, dict):
            for field in ('min', 'max', 'step'):
                if field in r and r[field] is not None:
                    base[field] = r[field]
            return base

        # Si format inattendu, on ignore et on garde le défaut
        return base

    # build inputs dict from function parameter 'inputs'
    # => IMPORTANT : on garde les clés comme avant, mais on peut stocker un dict de config
    inputs_dict = {k: _normalize_range(k) for k in inputs}

    # Valeurs (évite KeyError si values ne contient pas toutes les clés)
    values = dict(values) if values else {}
    for k in inputs_dict.keys():
        values.setdefault(k, 0)

    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': display_value == "range",
        'save': save,
        'droite': {'mode': 'cartesienne'},
        'vecteurs': {'directeur': directeur},
        'inputs': inputs_dict,          # <-- on passe aussi les ranges ici (ne casse pas)
        'initial_values': values,
        'compute_score': True,
    }

    run_js(
        f"mathadata.add_observer('{id_content}-container', () => "
        f"window.mathadata.tracer_points('{id_content}', '{json.dumps(params, cls=NpEncoder)}'))"
    )

    def _input_html(key: str):
        spec = inputs_dict.get(key, {})
        val = values.get(key, 0)

        # On ne met min/max/step que si c'est un input "range" ou "number"
        attrs = ""
        if display_value in ("range", "number"):
            if 'min' in spec:
                attrs += f' min="{spec["min"]}"'
            if 'max' in spec:
                attrs += f' max="{spec["max"]}"'
            if 'step' in spec:
                attrs += f' step="{spec["step"]}"'

        return f"""
        <div>
            <label for="{id_content}-input-{key}" id="{id_content}-label-{key}">{key} = </label>
            <input type="{display_value}"{attrs} value="{val}" id="{id_content}-input-{key}">
        </div>
        """

    display(HTML(f'''
        <div id="{id_content}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
            <div id="{id_content}-score-container" style="text-align: center; font-weight: bold; font-size: 2rem;">
                Pourcentage d'erreur : <span id="{id_content}-score">...</span>
            </div>

            <canvas id="{id_content}-chart"></canvas>

            <div id="{id_content}-inputs"
                 style="display: flex; gap: 1rem; justify-content: center;
                        flex-direction: {'column' if display_value == "range" else 'row'};">
                { _input_html('a') if 'a' in inputs_dict else '' }
                { _input_html('b') if 'b' in inputs_dict else '' }
                { _input_html('c') if 'c' in inputs_dict else '' }
            </div>

            <button id="{id_content}-reset" style="margin-top:1rem;">Réinitialiser</button>
        </div>
    '''))

    run_js(f"""
    mathadata.add_observer('{id_content}-reset', () => {{
        document.getElementById('{id_content}-reset').addEventListener('click', () => {{
            const initialValues = {json.dumps(values, cls=NpEncoder)};
            Object.entries(initialValues).forEach(([key, val]) => {{
                const input = document.getElementById('{id_content}-input-' + key);
                if (input) {{
                    input.value = val;
                    input.dispatchEvent(new Event('input'));
                }}
            }});
        }});
    }});
    """)

u_schema = (2, 3)


def afficher_plusieurs_vecteurs(coefx=u_schema[0], coefy=u_schema[1]):
    """
    Affiche un vecteur u(a, b), la droite dont il est le vecteur directeur,
    et les vecteurs 2u, -u et (-a, b).

    Args:
        coefx: coordonnée x du vecteur u
        coefy: coordonnée y du vecteur u
    """
    box_id = f"jxgbox_{uuid.uuid4().hex}"
    equation = "3x - 2y -4 = 0"

    # Utiliser add_observer pour attendre que l'élément soit dans le DOM
    run_js(f"""
mathadata.add_observer('{box_id}', () => {{
  var board = JXG.JSXGraph.initBoard('{box_id}', {{
    boundingbox:[-8, 8, 8, -8],
    axis:true,
    grid:true,
    showNavigation:false,
    showCopyright:false
  }});

  // Origine
  var O = board.create('point', [0, 0], {{
    name:'O',
    size:3,
    fixed:true,
    color:'black',
    label:{{fontSize:16}}
  }});

  // === Droite avec équation ===
  var droite = board.create('line', [[0, -2], [{coefx}, {coefy - 2}]], {{
    strokeColor:'#666',
    strokeWidth:2,
    straightFirst:true,
    straightLast:true,
    dash:2,
    withLabel:false,
    name:'{equation}',
    label:{{fontSize:16, offset:[80, 30]}}
  }});

  // === Vecteur u (bleu) - sur la droite ===
  var vecU = board.create('arrow', [[0, 0], [{coefx}, {coefy}]], {{
    strokeColor:'#0000FF',
    strokeWidth:4,
    lastArrow:{{type:2, size:8}}
  }});
  var labelU = board.create('text', [{coefx + 0.5}, {coefy + 0.8}, 'u({coefx},{coefy})'], {{
    fontSize:16,
    color:'#0000FF',
    fixed:true
  }});

  // === Vecteur 2u (rouge) ===
  var vec2U = board.create('arrow', [[-6, -4], [{-6 + 2 * coefx}, {-4 + 2 * coefy}]], {{
    strokeColor:'#FF0000',
    strokeWidth:3,
    lastArrow:{{type:2, size:7}}
  }});
  var label2U = board.create('text', [{-6 + 2 * coefx + 0.5}, {-4 + 2 * coefy + 0.8}, '2u({2 * coefx},{2 * coefy})'], {{
    fontSize:16,
    color:'#FF0000',
    fixed:true
  }});

  // === Vecteur -u (vert) ===
  var vecUOpp = board.create('arrow', [[2, -2], [{-coefx + 2}, {-2 - coefy}]], {{
    strokeColor:'#00AA00',
    strokeWidth:3,
    lastArrow:{{type:2, size:7}}
  }});
  var labelUOpp = board.create('text', [{-coefx + 3}, {-2 - coefy}, '-u({-coefx},{-coefy})'], {{
    fontSize:16,
    color:'#00AA00',
    fixed:true
  }});

  // === Vecteur (-a, b) (orange) ===
  var vecUPerp = board.create('arrow', [[6, -4], [{6 - coefx}, {-4 + coefy}]], {{
    strokeColor:'#FF8800',
    strokeWidth:3,
    lastArrow:{{type:2, size:7}}
  }});
  var labelUPerp = board.create('text', [{6 - coefx + 0.5}, {-4 + coefy + 0.8}, 'v({-coefx},{coefy})'], {{
    fontSize:16,
    color:'#FF8800',
    fixed:true
  }});
}});
""")

    # Afficher le conteneur JSXGraph
    display(HTML(f'<div id="{box_id}" class="jxgbox" style="width:800px; height:600px;"></div>'))


def determinant_decouverte_exercice():
    # Paramètres de l'iframe GeoGebra :
    # - id/zcedkaag : identifiant de l'activité GeoGebra
    # - width/3000, height/800 : dimensions de l'iframe interne
    # - border/888888 : couleur de la bordure (hexadécimal)
    # - sfsb/true : Show Fullscreen Button (afficher le bouton plein écran)
    # - smb/false : Show Menu Bar (afficher la barre de menu)
    # - stb/false : Show Toolbar (afficher la barre d'outils)
    # - stbh/false : Show Toolbar Help (afficher l'aide de la barre d'outils)
    # - ai/false : Allow Input Bar (autoriser la barre de saisie)
    # - asb/false : Allow Style Bar (autoriser la barre de style)
    # - sri/false : Show Reset Icon (afficher l'icône de réinitialisation)
    # - rc/false : Right Click (activer le clic droit)
    # - ld/true : Language Direction (direction de la langue, true pour LTR)
    # - sdz/true : Show Drag Zooming (afficher le zoom par glissement)
    # - ctl/false : Show Construction Protocol (afficher le protocole de construction)
    display(HTML(
        """<iframe scrolling="no" title="Déterminant et droite" src="https://www.geogebra.org/material/iframe/id/zcedkaag/width/4000/height/800/border/888888/sfsb/true/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/true/sdz/true/ctl/false" width="3000px" height="800px" style="border:0px;"> </iframe>"""))


def lalambdada_exercice():
    # Paramètres de l'iframe GeoGebra :
    # - id/kkz6pyfe   : identifiant de l'activité GeoGebra
    # - width/3000, height/800 : dimensions de l'iframe interne
    # - border/888888 : couleur de la bordure (hexadécimal)
    # - sfsb/true : Show Fullscreen Button (afficher le bouton plein écran)
    # - smb/false : Show Menu Bar (afficher la barre de menu)
    # - stb/false : Show Toolbar (afficher la barre d'outils)
    # - stbh/false : Show Toolbar Help (afficher l'aide de la barre d'outils)
    # - ai/false : Allow Input Bar (autoriser la barre de saisie)
    # - asb/false : Allow Style Bar (autoriser la barre de style)
    # - sri/false : Show Reset Icon (afficher l'icône de réinitialisation)
    # - rc/false : Right Click (activer le clic droit)
    # - ld/true : Language Direction (direction de la langue, true pour LTR)
    # - sdz/true : Show Drag Zooming (afficher le zoom par glissement)
    # - ctl/false : Show Construction Protocol (afficher le protocole de construction)
    display(HTML(
        """<iframe scrolling="no" title="Déterminant et droite" src="https://www.geogebra.org/material/iframe/id/kkz6pyfe/width/4000/height/800/border/888888/sfsb/true/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/true/sdz/true/ctl/false" width="3000px" height="800px" style="border:0px;"> </iframe>"""))


def affichage_zones_custom(A1, B1, A2, B2):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    tracer_points_droite(display_value="number", carac=common.challenge.deux_caracteristiques_custom, save=False)


def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    # Utilise display_custom_selection_2d si elle existe, sinon display_custom_selection
    if hasattr(common.challenge, 'display_custom_selection_2d'):
        common.challenge.display_custom_selection_2d(id)
    else:
        common.challenge.display_custom_selection(id)

    tracer_points_droite(id_content=id, display_value="number", carac=common.challenge.deux_caracteristiques_custom,
                         initial_hidden=True, save=False)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                mathadata.update_points('{id}', {{points}})

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                document.getElementById('{id}-container').style.visibility = 'visible';
            }})
        }}
    ''')


# JS

def calculer_score_droite(animation=False):
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, animation=animation, banque=False)


def calculer_score_custom_droite():
    calculer_score_droite_geo(custom=True, validate=common.challenge.objectif_score_droite_custom,
                              error_msg="Continuez à chercher 2 zones pour avoir moins de " + str(
                                  common.challenge.objectif_score_droite_custom) + "% d'erreur. Pensez à changer les valeurs de m et p après avoir défini votre zone.")


# Variables globales pour les valeurs des sliders et taux d'erreur
slider_p_value = None
slider_m_value = None
error_score = None


def update_slider_values(p_val, m_val, e_val):
    """Fonction appelée par JavaScript pour mettre à jour les valeurs des sliders"""
    global slider_p_value, slider_m_value, error_score
    slider_p_value = p_val
    slider_m_value = m_val
    error_score = e_val


listener_js = """
<script>
window.addEventListener('message', function(event) {
    if (event.data.type === 'slider_values') {
        // Appeler la fonction Python
        mathadata.run_python(
            'update_slider_values(' + event.data.p + ', ' + event.data.m + ', ' + event.data.e + ')'
        );
    }
});
</script>
"""

dataset_20_points = {
    "blue_pts": [[2, 15], [5, 9], [10, 18], [15, 15], [8, 11], [12, 12], [18, 8], [22, 22], [25, 19], [20, 24]],
    "orange_pts": [[5, 1], [5, 7], [10, 8], [15, 5], [20, 6], [25, 15], [8, 6], [12, 9], [18, 12], [14, 11]]}


def afficher_separation_line(p=1, slope=1, show_equation=False, width=466, height=400, jsx_version="1.4.0",
                             dataset=None, labels=None):
    """
    JSXGraph iframe with:
      - parameter 'p' (intercept) instead of 'b'
      - optional sliders for p and m (color-coded)
      - slider single/two start positions:
          * single slider -> starts at (5, -5)
          * two sliders -> p at (5, -5) and m at (5, -10)
      - show_equation (bool) toggles the fixed equation box at (30,5) (default False)
      - XMIN = -2, YMAX = 46
      - legend located at (5,45)
      - dataset and labels: optional parameters to provide data points
    """
    display(HTML(listener_js))
    box_id = f"jxgbox_{uuid.uuid4().hex}"
    js_show_equation = 'true' if show_equation else 'false'

    # Use provided dataset and labels, or fall back to default points
    if dataset is not None and labels is not None:
        # Calculate points from dataset using the same method as other functions
        c_train_par_population = compute_c_train_by_class(
            fonction_caracteristique=common.challenge.deux_caracteristiques,
            d_train=dataset,
            r_train=labels
        )

        # Extract points for each class and convert NumPy arrays to Python lists
        if len(c_train_par_population) >= 2:
            # Convert NumPy arrays to Python lists with proper type conversion
            blue_pts = [[float(x), float(y)] for x, y in c_train_par_population[0]] if len(
                c_train_par_population[0]) > 0 else [[20.2, 29.1]]
            orange_pts = [[float(x), float(y)] for x, y in c_train_par_population[1]] if len(
                c_train_par_population[1]) > 0 else [[18.3, 18.2]]
        else:
            # Fallback to default if data is insufficient
            blue_pts = dataset_20_points["blue_pts"]
            orange_pts = dataset_20_points["orange_pts"]
    else:
        # Original example points
        blue_pts = dataset_20_points["blue_pts"]
        orange_pts = dataset_20_points["orange_pts"]

    js_blue = json.dumps(blue_pts)
    js_orange = json.dumps(orange_pts)
    error_score = False

    # Calculate dynamic boundaries to fit all points AND include origin (0,0)
    all_points = blue_pts + orange_pts
    if all_points:
        x_coords = [pt[0] for pt in all_points]
        y_coords = [pt[1] for pt in all_points]

        # Include origin in the data range calculation
        x_coords_with_origin = x_coords + [0]
        y_coords_with_origin = y_coords + [0]

        data_x_min = min(x_coords_with_origin)
        data_x_max = max(x_coords_with_origin)
        data_y_min = min(y_coords_with_origin)
        data_y_max = max(y_coords_with_origin)

        # Add padding around data points (5% of range, minimum 5 units)
        x_range = data_x_max - data_x_min
        y_range = data_y_max - data_y_min
        x_padding = max(x_range * 0.05, 5)
        y_padding = max(y_range * 0.05, 5)

        # Ensure origin is always visible with minimal left padding
        # Left margin reduced to just show the y-axis
        x_left_padding = max(x_range * 0.02, 1)  # Minimal left padding (2% or 1 unit)
        view_x_min = min(data_x_min - x_left_padding, -1)  # Reduced to -1 for minimal margin
        view_x_max = max(data_x_max + x_padding, 2)  # At least 2 to show origin clearly
        view_y_max = max(data_y_max + y_padding, 2)  # At least 2 to show origin clearly
        view_y_min = min(data_y_min - y_padding, -2, 0)

        # Rendre le repère orthonormé : ajuster X pour correspondre au ratio largeur/hauteur
        # On garde l'unité de l'ordonnée comme référence
        canvas_ratio = width / height  # Ratio largeur/hauteur du canvas
        y_range_units = view_y_max - view_y_min  # Plage Y en unités
        x_range_units_needed = y_range_units * canvas_ratio  # Plage X nécessaire pour orthonormé

        # Centrer la plage X autour des données tout en gardant l'orthonormé
        # S'assurer que l'origine (0) reste visible
        x_center = (view_x_min + view_x_max) / 2
        new_x_min = x_center - x_range_units_needed / 2
        new_x_max = x_center + x_range_units_needed / 2

        # Limiter l'axe des abscisses à -5 minimum du côté négatif
        x_min_limit = -5
        if new_x_min < x_min_limit:
            # Ajuster pour respecter la limite minimale
            new_x_min = x_min_limit
            new_x_max = new_x_min + x_range_units_needed

        # Si l'origine n'est pas visible, ajuster pour l'inclure
        if new_x_min > 0:
            # L'origine est à gauche, décaler vers la droite
            view_x_min = 0
            view_x_max = x_range_units_needed
        elif new_x_max < 0:
            # L'origine est à droite, décaler vers la gauche
            view_x_min = -x_range_units_needed
            view_x_max = 0
        else:
            # L'origine est déjà dans la plage
            view_x_min = new_x_min
            view_x_max = new_x_max

        # Appliquer la limite minimale finale (ne pas aller en dessous de -5)
        if view_x_min < x_min_limit:
            view_x_min = x_min_limit
            # Réajuster view_x_max pour maintenir l'orthonormé
            view_x_max = view_x_min + x_range_units_needed
    else:
        # Fallback values if no points - centered around origin
        # Calculate space needed for sliders

        view_x_min, view_x_max, view_y_max = -10, 50, 50
        view_y_min = -10

        # Rendre le repère orthonormé : ajuster X pour correspondre au ratio largeur/hauteur
        # On garde l'unité de l'ordonnée comme référence
        canvas_ratio = width / height  # Ratio largeur/hauteur du canvas
        y_range_units = view_y_max - view_y_min  # Plage Y en unités
        x_range_units_needed = y_range_units * canvas_ratio  # Plage X nécessaire pour orthonormé

        # Centrer la plage X autour de l'origine (cas fallback)
        # Limiter l'axe des abscisses à -5 minimum du côté négatif
        x_min_limit = -5
        view_x_min = -x_range_units_needed / 2
        view_x_max = x_range_units_needed / 2

        # Appliquer la limite minimale
        if view_x_min < x_min_limit:
            view_x_min = x_min_limit
            view_x_max = view_x_min + x_range_units_needed

    # Colors
    two_color = "#4C6EF5"  # blue for 2
    seven_color = "#F6C85F"  # orange for 7
    m_color = "#239E28"
    p_color = "#FF0000"
    css_url = f"https://cdn.jsdelivr.net/npm/jsxgraph@{jsx_version}/distrib/jsxgraph.css"
    js_url = f"https://cdn.jsdelivr.net/npm/jsxgraph@{jsx_version}/distrib/jsxgraphcore.js"

    # Variable pour éviter les backslash dans f-string (Python < 3.12)
    score_div = f'<div id="{box_id}-score-container" style="text-align: center; font-weight: bold; font-size: 1rem;">Pourcentage d{chr(39)}erreur : <span id="{box_id}-score">...</span></div>' if error_score else ""

    page = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8"/>
    <title>Séparation linéaire - JSXGraph</title>
    <link rel="stylesheet" href="{css_url}" />
    <script src="{js_url}"></script>
    <style>
      html, body {{ margin:0; padding:0; overflow:hidden; font-family:sans-serif; }}
      .title {{ width:100%; text-align:center; font-size:16px; font-weight:bold; padding:6px 0; }}
      .jxgbox {{ margin:auto; }}
      .main-container {{
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: center;
          gap: 20px;
          padding: 10px;
      }}
      .slope-box {{
          width: 250px;
          height: 245px; 
          border: 4px solid black;
          border-radius: 20px;
          background-color: white;
          display: flex;
          align-items: center;
          justify-content: center;
          position: relative;
      }}
    </style>
  </head>
  <body>
    {score_div}

    <div class="main-container">
        <div id="{box_id}" class="jxgbox" style="width:{width}px; height:{height}px;"></div>

        <div id="slope-box-container" class="slope-box" style="display: {'none'};">
            <svg id="slope-svg" width="100%" height="100%" viewBox="0 0 250 216" style="overflow: visible;">
                <line id="slope-base" stroke="black" stroke-width="4" stroke-linecap="round" />
                <line id="slope-height" stroke="{m_color}" stroke-width="4" stroke-linecap="round" />
                <line id="slope-hypo" stroke="purple" stroke-width="5" stroke-linecap="round" />
                <text id="text-base" text-anchor="middle" font-weight="bold" font-size="16">5</text>
                <text id="text-calc" text-anchor="start" font-weight="bold" font-size="16" fill="{m_color}"></text>
            </svg>
        </div>
    </div>
    <script>
      (function() {{
        if (!window.JXG) {{
          document.getElementById('{box_id}').innerHTML = '<div style="color:crimson;padding:10px;">JSXGraph failed to load.</div>';
          return;
        }}
        // Dynamic viewport calculated from data points
        var XMIN = {view_x_min}, XMAX = {view_x_max}, YMAX = {view_y_max};
        var YMIN = {view_y_min};
        var board = JXG.JSXGraph.initBoard('{box_id}', {{
          boundingbox: [XMIN, YMAX, XMAX, YMIN],
          axis: false,
          grid: false,
          showNavigation: false,
          showCopyright: false,
          selection: {{
            enabled: false
          }}
        }});
        // background corners (invisible)
        var topLeft     = board.create('point', [function(){{return XMIN;}}, function(){{return YMAX;}}], {{visible:false}});
        var topRight    = board.create('point', [function(){{return XMAX;}}, function(){{return YMAX;}}], {{visible:false}});
        var bottomLeft  = board.create('point', [function(){{return XMIN;}}, function(){{return YMIN;}}], {{visible:false}});
        var bottomRight = board.create('point', [function(){{return XMAX;}}, function(){{return YMIN;}}], {{visible:false}});
        // parameters from Python
        var p_fixed = {p};
        var m_fixed = {slope};
        var s_p = null;
        var s_m = null;
        var err_score = 100
        function currentP() {{ return s_p ? s_p.Value() : p_fixed; }}
        function currentM() {{ return s_m ? s_m.Value() : m_fixed; }}
        function currentE() {{ return err_score; }}
        // format number without unnecessary trailing zeros
        function formatNumber(num) {{
          return parseFloat(num.toFixed(2)).toString();
        }}
        // computeScore pour le taux d'erreur
        const computeScore = () => {{
              const n = {blue_pts}.length;

              // Points oranges incorrectement classés (au-dessus de la droite alors qu'ils devraient être en dessous)
              const errorsA = {orange_pts}.filter(([x, y]) => y >= currentM() * x + currentP()).length;

              // Points bleus incorrectement classés (en dessous de la droite alors qu'ils devraient être au-dessus)
              const errorsB = {blue_pts}.filter(([x, y]) => y <= currentM() * x + currentP()).length;

              const totalErrors = errorsA + errorsB;
              const totalPoints = 2 * n;

              const errorPercentage = Math.round((totalErrors / totalPoints) * 10000) / 100; // en %
              const scoreEl = document.getElementById("{box_id}-score");
              if (scoreEl) scoreEl.innerHTML = `${{errorPercentage}}%`;
              err_score = errorPercentage;
          }}
        // dynamic boundary points for shading
        var P_line_left  = board.create('point', [function(){{ return XMIN; }}, function(){{ return currentM()*XMIN + currentP(); }}], {{visible:false}});
        var P_line_right = board.create('point', [function(){{ return XMAX; }}, function(){{ return currentM()*XMAX + currentP(); }}], {{visible:false}});
        // shading polygons (below everything)
        var polyAbove = board.create('polygon', [topLeft, topRight, P_line_right, P_line_left], {{
          fillColor: '#EAF0FF', fillOpacity: 0.55, borders: false, layer: 0
        }});
        var polyBelow = board.create('polygon', [bottomLeft, bottomRight, P_line_right, P_line_left], {{
          fillColor: '#FFF3DB', fillOpacity: 0.55, borders: false, layer: 0
        }});
        // grid and axes above shading
        board.create('grid', [], {{strokeColor:'#DCDCDC', strokeWidth:1, strokeOpacity:0.6}});
        board.create('axis', [[0,0],[1,0]], {{ name: 'Caractéristique x', ticks: {{minorTicks:0}}, strokeWidth: 1.2, layer: 4 }});
        board.create('axis', [[0,0],[0,1]], {{ name: 'Caractéristique y', ticks: {{minorTicks:0,label: {{
            anchorX: 'right',  // Aligne le texte à droite du point d'ancrage
            offset: [-10, 0]   // Décale de 10 pixels vers la gauche
        }}}}, strokeWidth: 1.2, layer: 4 }});
        // create the line using two hidden points (so polygons auto-update)
        var A = board.create('point', [0, function(){{ return currentP(); }}], {{visible:false}});
        var B = board.create('point', [XMAX, function(){{ return currentM()*XMAX + currentP(); }}], {{visible:false}});
        var sep = board.create('line', [A, B], {{
          strokeColor:'#222', 
          strokeWidth:2, 
          layer:5,
          fixed: true,
          highlight: false,
          withLabel: false
        }});
        // optionally show the equation box at (30, 5)
        if ({js_show_equation}) {{
          // colored parameter texts inside the box (2 decimal) - positioned relative to XMAX to avoid wrapping
          var eq_x = XMAX - 12;
          var eq_y = 5;
          board.create('text', [eq_x, eq_y, function(){{
              var m = formatNumber(currentM());
              var p_val = currentP();
              var sign = p_val >= 0 ? "+" : "-";
              var p_abs = formatNumber(Math.abs(p_val));
              // Construction HTML pour un espacement naturel et dynamique
              return "y = <span style='color:{m_color}'>" + m + "</span> x <span style='color:{p_color}'>" + sign + " " + p_abs + "</span>";
          }}], {{fontSize:18, anchorX:'left', anchorY:'middle', layer:9, fixed: true}});
        }}
        // Points: draw after axes/grid so they are on top; color-coded to match parameters
        var blue = {js_blue};
        for (var i = 0; i < blue.length; i++) {{
          var p = blue[i];
          board.create('point', [p[0], p[1]], {{
            name:'', 
            shape:'+', 
            strokeColor:'{two_color}', 
            fillColor:'{two_color}', 
            size:3, 
            layer:10,
            fixed: true,
            highlight: false,
            withLabel: false
          }});
        }}
        var orange = {js_orange};
        for (var j = 0; j < orange.length; j++) {{
          var q = orange[j];
          board.create('point', [q[0], q[1]], {{
            name:'', 
            shape:'+', 
            strokeColor:'{seven_color}', 
            fillColor:'{seven_color}', 
            size:3, 
            layer:10,
            fixed: true,
            highlight: false,
            withLabel: false
          }});
        }}
        // Legend - position adaptively
        var legend_x = XMIN + (XMAX - XMIN) * 0.15;  // 15% from left
        var legend_y_top = YMAX - (YMAX - YMIN) * 0.05;  // 5% from top
        var legend_y_bottom = legend_y_top - 2;  // 2 units below (divisé par 2)

        // Rectangle pour "Images de 2"
        var rect_width = 1;  // Divisé par 2 (était 2)
        var rect_height = 1;  // Divisé par 2 (était 2)

        // Créer des points invisibles pour le rectangle "Images de 2"
        var rect2_p1 = board.create('point', [legend_x, legend_y_top - rect_height/2], {{visible:false}});
        var rect2_p2 = board.create('point', [legend_x + rect_width, legend_y_top - rect_height/2], {{visible:false}});
        var rect2_p3 = board.create('point', [legend_x + rect_width, legend_y_top + rect_height/2], {{visible:false}});
        var rect2_p4 = board.create('point', [legend_x, legend_y_top + rect_height/2], {{visible:false}});

        board.create('polygon', [rect2_p1, rect2_p2, rect2_p3, rect2_p4], {{
          fillColor: '{two_color}',
          fillOpacity: 1,
          borders: {{strokeColor: '{two_color}', strokeWidth: 1}},
          layer: 11,
          fixed: true,
          highlight: false,
          withLines: false,
          vertices: {{visible: false}}
        }});

        board.create('text',[legend_x + rect_width + 0.5, legend_y_top, "Images de 2"], {{
          fontSize:16, 
          color:'#000000', 
          layer:11, 
          fontWeight: 'bold',
          fixed: true,
          highlight: false
        }});

        // Créer des points invisibles pour le rectangle "Images de 7"
        var rect7_p1 = board.create('point', [legend_x, legend_y_bottom - rect_height/2], {{visible:false}});
        var rect7_p2 = board.create('point', [legend_x + rect_width, legend_y_bottom - rect_height/2], {{visible:false}});
        var rect7_p3 = board.create('point', [legend_x + rect_width, legend_y_bottom + rect_height/2], {{visible:false}});
        var rect7_p4 = board.create('point', [legend_x, legend_y_bottom + rect_height/2], {{visible:false}});

        board.create('polygon', [rect7_p1, rect7_p2, rect7_p3, rect7_p4], {{
          fillColor: '{seven_color}',
          fillOpacity: 1,
          borders: {{strokeColor: '{seven_color}', strokeWidth: 1}},
          layer: 11,
          fixed: true,
          highlight: false,
          withLines: false,
          vertices: {{visible: false}}
        }});

        board.create('text',[legend_x + rect_width + 0.5, legend_y_bottom, "Images de 7"], {{
          fontSize:16,
          color:'#000000',
          layer:11,
          fontWeight: 'bold',
          fixed: true,
          highlight: false
        }});

        // Fonction pour envoyer les valeurs des sliders à Python
        function sendSliderValues() {{
          var p_val = currentP();
          var m_val = currentM();
          var e_val = currentE();
          window.parent.postMessage({{type:'slider_values', p: p_val, m: m_val, e: e_val}}, '*');
        }}

        function updateSlopeViz() {{
            var m = currentM();
            var svg = document.getElementById('slope-svg');
            if (!svg) return;

            var baseLine = document.getElementById('slope-base');
            var heightLine = document.getElementById('slope-height');
            var hypoLine = document.getElementById('slope-hypo');
            var textBase = document.getElementById('text-base');
            var textCalc = document.getElementById('text-calc');

            var unit = 20; 
            var vizBase = 40; // Base visuelle en pixels

            // Calcul vertical
            // Hauteur visuelle correspondante : h = vizBase * m
            var hViz = vizBase * m;

            // Centrage
            // Centre SVG = (125, 90) (pour box 250x180)
            var startX = 60; // Marge gauche
            var centerY = 90;

            var startY = centerY + hViz / 2;

            var Ax = startX;
            var Ay = startY;
            var Bx = Ax + vizBase;
            var By = startY;
            var Cx = Bx;
            var Cy = startY - hViz;

            baseLine.setAttribute('x1', Ax);
            baseLine.setAttribute('y1', Ay);
            baseLine.setAttribute('x2', Bx);
            baseLine.setAttribute('y2', By);

            heightLine.setAttribute('x1', Bx);
            heightLine.setAttribute('y1', By);
            heightLine.setAttribute('x2', Cx);
            heightLine.setAttribute('y2', Cy);

            hypoLine.setAttribute('x1', Ax);
            hypoLine.setAttribute('y1', Ay);
            hypoLine.setAttribute('x2', Cx);
            hypoLine.setAttribute('y2', Cy);

            textBase.setAttribute('x', (Ax + Bx)/2);
            textBase.setAttribute('y', Ay + 20);

            textCalc.setAttribute('x', Bx + 10);
            textCalc.setAttribute('y', (By + Cy)/2 + 5);

            var mDisp = parseFloat(m.toFixed(2));
            var resDisp = parseFloat((5 * m).toFixed(2));
            textCalc.textContent = "5 x " + mDisp + " = " + resDisp;
        }}

        // Fonction de mise à jour complète
        const updateAll = () => {{
          computeScore();
          updateSlopeViz();
          sendSliderValues();
        }};

        // Envoyer les valeurs initiales (calculer d'abord le score)
        updateAll();

        // Mettre à jour lors des changements de sliders
        // On capture tous les événements : down (début), drag (pendant), up (fin)
        // Ajout de 'move' et 'hit' pour mieux capturer les clics directs sur la barre
        if (s_p) {{
          s_p.on('down', updateAll);
          s_p.on('drag', updateAll);
          s_p.on('up',   updateAll);
          s_p.on('move', updateAll);
        }}
        if (s_m) {{
          s_m.on('down', updateAll);
          s_m.on('drag', updateAll);
          s_m.on('up',   updateAll);
          s_m.on('move', updateAll);
        }}
      }})();
    </script>
  </body>
</html>"""
    # Utiliser une data URI directement pour éviter les problèmes de ressources locales
    import base64
    import warnings

    # Supprimer l'avertissement IPython.display.IFrame
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame instead")

        # Encoder la page HTML en base64 pour créer une data URI sécurisée
        page_bytes = page.encode('utf-8')
        page_b64 = base64.b64encode(page_bytes).decode('ascii')
        data_uri = f'data:text/html;base64,{page_b64}'

        # Utiliser HTML avec data URI (plus compatible que IFrame avec data URI)
        # Largeur augmentée pour inclure la slope box (+250px + 40px padding/gap)
        iframe_width = width + 40

        iframe_html = f'<div style="display:flex; justify-content:center;"><iframe src="{data_uri}" style="width:{iframe_width}px; height:{height + 90}px; border:none;"></iframe></div>'
        display(HTML(iframe_html))


### Validation


def function_validation_equation(errors, answers):
    m = geo.input_values['m']
    p = geo.input_values['p']
    ordonnee_M = answers['ordonnee_M']

    if not (isinstance(ordonnee_M, (int, float))):
        errors.append(
            "Les coordonnées de M doivent être des nombres. "
            "Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False

    if ordonnee_M != m * pointA[0] + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        return False

    return True


def function_validation_equation_B(errors, answers):
    m = geo.input_values['m']
    p = geo.input_values['p']
    ordonnee_M = answers['ordonnee_M']

    if not (isinstance(ordonnee_M, (int, float))):
        errors.append(
            "Les coordonnées de M doivent être des nombres. "
            "Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False

    if ordonnee_M != m * pointB[0] + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        return False

    return True


def function_validation_question_reduite_cartesienne(errors, answers):
    m = answers['m']
    p = answers['p']
    a = answers['a']
    b = answers['b']
    c = answers['c']

    expected_a = m
    expected_b = -1
    expected_c = p

    divA = a / expected_a
    divB = b / expected_b
    divC = c / expected_c

    if (a != 0 and b != 0 and c != 0) and (divA == divB == divC):
        return True

    errors.append(
        "Ce n'est pas la bonne réponse, regarde l'exemple pour voir comment trouver a, b et c à partir de m et p.")
    return False


A = (20, 30)
u = (20, 10)


def qcm_abc_cartesienne_c():
    create_qcm({
        'question': 'Que se passe t-il quand on augmente la valeur de c ?',
        'choices': [
            'La droite se déplace parallèlement',
            'La pente de la droite est modifiée.',
            'La droite ne bouge pas.'
        ],
        'answer_index': 0,
        'multiline': True
    })


def qcm_vecteur_colineaire():
    create_qcm({
        'question': '"Deux vecteurs sont colinéaires" signifie forcément : ',
        'choices': [
            "Qu'ils sont égaux",
            "Qu'ils ont le même sens",
            "Qu'ils ont la même direction",
            "Que leur déterminant est nul",
        ],
        'answers_indexes': [2,3],
        'multiline': True
    })


def qcm_determinant_nul():
    create_qcm({
        'question': 'Que peut-on affirmer sur le point M lorsque le déterminant est nul ?',
        'choices': [
            'M est à l\'origine. ',
            'Les coordonnées de M sont (80;60).',
            'Le point M appartient à la droite.'
        ],
        'answer_index': 2,
        'multiline': True
    })


def qcm_abc_cartesienne_b():
    create_qcm({
        'question': 'Que se passe t-il quand on augmente la valeur de b ?',
        'choices': [
            'La pente de la droite est modifiée.',
            'La droite se déplace parallèlement.',
            'La droite ne bouge pas.'
        ],
        'answer_index': 0,
        'multiline': True
    })


def qcm_abc_cartesienne_a():
    create_qcm({
        'question': 'Que se passe t-il quand on augmente la valeur de a ?',
        'choices': [
            'La pente de la droite est modifiée.',
            'La droite se déplace parallèlement.',
            'La droite ne bouge pas.',
            'L\'ordonnée à l\'origine est modifiée.'
        ],
        'answer_index': 0,
        'multiline': True
    })


def qcm_abc_cartesienne_ab():
    create_qcm({
        'question': 'Que se passe t-il lorsque a et b sont de même signe et tout deux non nuls ?',
        'choices': [
            'La pente de la droite est positive',
            'La pente de la droite est négative.',
            'L\'ordonnée à l\'origine est positive.',
        ],
        'answer_index': 1,
        'multiline': True
    })


def qcm_vecteurs_directeurs():
    create_qcm({
        'question': "Parmi ces vecteurs, lesquels sont des vecteurs directeurs de la droite en pointillé ?",
        'choices': [
            r'$\vec{u}$',
            r'$\vec{2u}$',
            r'$\vec{-u}$',
            r'$\vec{v}$'
        ],
        'answers_indexes': [0, 1, 2],
    })


def function_validation_question_w_directeur(errors, answers):
    global u_schema
    w = answers['w']
    u2 = (2 * u_schema[0], 2 * u_schema[1])
    u_neg = (-u_schema[0], -u_schema[1])

    if w == u_schema or w == u2 or w == u_neg:
        errors.append("Donne un vecteur qui n'est pas déjà dans les exemples.")
        return False

    determinant = u_schema[0] * w[1] - u_schema[1] * w[0]

    if determinant != 0:
        errors.append("Le vecteur w n'est pas un vecteur directeur de la droite.")
        return False

    return True


def function_validation_question_point_determinant_zero_M(errors, answers):
    M = answers['M']

    AM = (M[0] - A[0], M[1] - A[1])

    determinantM = u[0] * AM[1] - u[1] * AM[0]

    if determinantM != 0:
        errors.append("Le déterminant des vecteurs AM et u n'est pas égal à zéro.")

    return len(errors) == 0



def function_validation_question_points_determinant_zero_N_M(errors, answers):
    M = answers['M']
    N = answers['N']

    if M[0] == N[0] and M[1] == N[1]:
        errors.append("Donne 2 points différents.")
        return False

    AM = (M[0] - A[0], M[1] - A[1])
    AN = (N[0] - A[0], N[1] - A[1])

    determinantM = u[0] * AM[1] - u[1] * AM[0]
    determinantN = u[0] * AN[1] - u[1] * AN[0]

    if determinantM != 0:
        errors.append("Le déterminant des vecteurs AM et u n'est pas égal à zéro.")

    if determinantN != 0:
        errors.append("Le déterminant des vecteurs AN et u n'est pas égal à zéro.")

    return True


def function_validation_question_point_determinant_zero(errors, answers):
    M = answers['M']

    AM = (M[0] - A[0], M[1] - A[1])

    determinantM = u[0] * AM[1] - u[1] * AM[0]

    if determinantM != 0:
        errors.append("Le déterminant des vecteurs AM et u n'est pas égal à zéro.")

    return len(errors) == 0

def qcm_determinant():
    create_qcm({
        'question': r'Le déterminant de 2 vecteurs $\vec{u}(x_u, y_u)$ et $\vec{v}(x_v, y_v)$ est nul si :',
        'choices': [
            r'Il y a au moins une coordonnée nulle parmi $x_u$, $y_u$, $x_v$ ou $y_v$.',
            r'$x_u \times y_u - x_v \times y_v = 0$',
            r'$x_u \times y_v - y_u \times x_v = 0$',
            'Les deux vecteurs sont colinéaires.'
        ],
        'answers_indexes': [2, 3],
        'multiline': True
    })


def function_validation_score_vecteur_20(errors, answers):
    user_answer = answers['erreur_20']

    # Vérifications de base
    if not isinstance(user_answer, (int, float)):
        errors.append("Le pourcentage d'erreur doit être un nombre.")
        return False

    if user_answer < 0 or user_answer > 100:
        errors.append("Le pourcentage d'erreur doit être compris entre 0 et 100.")
        return False
    # Calcul des prédictions
    nb_erreurs = 6
    pourcentage_erreur = 30

    # Vérification si l'utilisateur a donné le nombre d'erreurs au lieu du pourcentage
    if user_answer == nb_erreurs:
        errors.append(
            f"Ce n'est pas la bonne valeur. Tu as donné le nombre d'erreurs ({nb_erreurs}) et non le pourcentage d'erreur.")
        return False
    # Vérification si l'utilisateur a donné la proportion pas en du pourcentage
    if user_answer == nb_erreurs / 20:
        errors.append(
            f"Tu es sur la bonne voie. Tu as bien donné la proportion mais on souhaite la réponse en pourcentage sans écrire le symbole %.")
        return False
    # Vérification de la réponse correcte
    if user_answer == pourcentage_erreur:
        if nb_erreurs == 0:
            pretty_print_success(
                "Bravo, c'est la bonne réponse. Il n'y a aucune erreur de classification sur ce schéma.")
        else:
            # Détails sur les erreurs pour le message
            pretty_print_success(
                f"Bravo, c'est la bonne réponse. Il y a deux images de 7 au dessus de la droite et une images de 2 au dessus, donc {nb_erreurs} erreurs soit {pourcentage_erreur}%.")
        return True
    else:
        errors.append(
            f"Ce n'est pas la bonne réponse. Compte le nombre d'erreurs, c'est à dire le nombre de points du mauvais côté de la droite, puis calcule le pourcentage d'erreur.")
        return False


def function_validation_question_vecteur_directeur_possible(errors, answers):
    u = answers['u']
    reponse = [4, 3]

    determinant = u[0] * reponse[1] - u[1] * reponse[0]

    if determinant != 0:
        errors.append("Le vecteur u n'est pas un vecteur directeur de la droite.")
        return False

    return True


def function_validation_lambda_P(errors, answers):
    user_lambda = answers['lambda_P']
    if not isinstance(user_lambda, (int, float)):
        errors.append("lambda_P doit être un nombre.")
        return False
    if user_lambda != 3:
        errors.append("Mauvaise réponse")
        return False
    return True


def function_validation_lecture_u(errors, answers):
    x_u = answers['x_u']
    y_u = answers['y_u']

    # réponse attendue
    x_u_expected = geo.input_values['ux']
    y_u_expected = geo.input_values['uy']

    if not (isinstance(x_u, (int, float)) and isinstance(y_u, (int, float))):
        errors.append(
            "Les coordonnées de u doivent être des nombres. "
            "Pour les nombres à virgule, utilisez un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False

    if x_u != x_u_expected or y_u != y_u_expected:
        errors.append("Les coordonnées de u ne sont pas correctes. Tu peux les lire sur la figure ci-dessus.")
        return False

    return True

validation_execution_tracer_points_droite_vecteur = MathadataValidate(success="")

validation_question_lecture_u = MathadataValidateVariables({
    'x_u': None,
    'y_u': None
},
    function_validation=function_validation_lecture_u,
    succes="")

validation_execution_tracer_points_droite_directeur = MathadataValidate(success="")
validation_execution_afficher_plusieurs_vecteurs = MathadataValidate(success="")
validation_execution_determinant_exercice = MathadataValidate(success="")
validation_question_score_droite_20 = MathadataValidateVariables({
    'erreur_20': None
},
    function_validation=function_validation_score_vecteur_20, success="")
validation_question_vecteur_directeur_possible = MathadataValidateVariables({
    'u': {
        'type': 'vecteur'
    }
}, tips=[{
    'seconds': 30,
    'trials': 1,
    'tip': 'Juste au-dessus une formule est proposée pour calculer un vecteur directeur à partir de a et b.'
}, {
    'seconds': 60,
    'trials': 2,
    'tip': 'Si la droite a pour équation ax + by + c = 0, alors un vecteur directeur possible est {-b a}.'
}],
    function_validation=function_validation_question_vecteur_directeur_possible,
    succes="Bravo, ce vecteur est bien un vecteur directeur de la droite.")
validation_question_w_directeur = MathadataValidateVariables({
    'w': {
        'type': 'vecteur'
    }
}, tips=[{
    'seconds': 30,
    'trials': 1,
    'tip': 'Rappelle-toi qu\'un vecteur directeur de la droite doit être colinéaire au vecteur u.'
}, {
    'seconds': 60,
    'trials': 2,
    'tip': 'Tu peux par exemple donner les coordonnés du vecteur 3u.'
}],
    function_validation=function_validation_question_w_directeur,
    succes="Bravo, ce vecteur w est bien un vecteur directeur de la droite. Il est colinéaire au vecteur u.")
validation_question_equation = MathadataValidateVariables({
    'ordonnee_M': None
},
    function_validation=function_validation_equation)
validation_question_equation_B = MathadataValidateVariables({
    'ordonnee_M': None
},
    function_validation=function_validation_equation_B)
validation_question_determinant_decouverte = MathadataValidateVariables({
    'determinant': 200
})
validation_question_reduite_cartesienne = MathadataValidateVariables({
    'm': 4,
    'p': 3,
    # Validation des autres paramètres avec la fonction custom
    'a': {'type': (int, float)},
    'b': {'type': (int, float)},
    'c': {'type': (int, float)},
},
    function_validation=function_validation_question_reduite_cartesienne)
validation_question_determinant_calcul = MathadataValidateVariables({
    'determinant': -50
}, tips=[{
    'seconds': 30,
    'trials': 1,
    'tip': 'Commencez par calculer les coordonnées du vecteur AM'
}, {
    'seconds': 60,
    'trials': 2,
    'tip': 'Le vecteur AM à comme coordonnées (5, 5). Quel calcul faut-il faire pour obtenir le déterminant ?'
}])


validation_question_points_determinant_zero_N_M = MathadataValidateVariables({
    'M': {
        'type': 'vecteur'
    },
    'N': {
        'type': 'vecteur'
    }
},
    function_validation=function_validation_question_points_determinant_zero_N_M,
    succes="Bravo, ces 2 points donnent des vecteurs dont le déterminant avec u vaut 0.")

validation_question_point_determinant_zero_M = MathadataValidateVariables({
    'M': {
        'type': 'vecteur'
    }
},
    function_validation=function_validation_question_point_determinant_zero_M,
    succes="Bravo, ce point donne un vecteur dont le déterminant avec u vaut 0.")

validation_exercice_lambda_P = MathadataValidateVariables(
    {'lambda_P': None},
    function_validation=function_validation_lambda_P,
    tips=[{
        'seconds': 10,
        'tip': 'Le curseur de la figure ne permet pas d\'atteindre P, il faut deviner la valeur de lambda_P.'
        }, {
        'seconds': 30,
        'trials': 1,
        'tip': 'Combien de fois tu peux tracer le vecteur u entre les points A et P ?'
        }],
    succes="")


validation_execution_lalambdada_exercice = MathadataValidate(success="")

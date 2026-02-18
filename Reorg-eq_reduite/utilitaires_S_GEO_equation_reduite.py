from utilitaires_common import *
from io import BytesIO
import base64
from sklearn.linear_model import LogisticRegression

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

new_steps_mnist_equ_red = {
    'intro': {
        'name': 'But du TP',
        'color': 'rgb(250,181,29)',
    },
    'data': {
        'name': 'Images numériques',
        'color': 'rgb(244,167,198)',
    },
    'carac': {
        'name': "Points caractéristiques",
        'color': 'rgb(241,132,18)',
    },
    'classif': {
        'name': 'Séparer avec une droite',
        'color': 'rgb(230,29,73)',
    },
    'apprendre': {
        'name': "Apprendre à faire moins d'erreurs",
        'color': 'rgb(62,178,136)',
    },
    'custom': {
        'name': 'Tes caractéristiques',
        'color': 'rgb(20,129,173)',
    },
}
## override stepbar config
## pas optimal car affiche quand même la base config
common.init_stepbar(new_steps_mnist_equ_red)

# Stepbar plus compacte pour ce notebook (largeur + tailles de police)
run_js(r"""
(function(){
  const styleId = "mathadata-style-stepbar-mnist-equ-red";
  const existing = document.getElementById(styleId);
  if (existing) existing.remove();

  const s = document.createElement("style");
  s.id = styleId;
  s.textContent = `
    #stepbar.mathadata-stepbar__container { width: 180px; }
    #stepbar .mathadata-stepbar__backdrop { width: 180px; padding: 16px 0; }
    #stepbar .mathadata-stepbar__labelHard { font-size: 14px; line-height: 1.15; }
    #stepbar .mathadata-stepbar__itemRow { padding: 0 10px; }
    #stepbar .mathadata-stepbar__item { width: 50px; height: 50px; flex: 0 0 50px; }
  `;
  document.head.appendChild(s);
})();
""")

def tracer_20_points_droite(slider_p=False, slider_m=False, show_eq=False):
    afficher_separation_line(show_slider_p=slider_p, show_slider_m=slider_m, show_equation=show_eq)


def tracer_20_points_droite_p():
    tracer_20_points_droite(slider_p=True, show_eq=True)


def tracer_20_points_droite_pm():
    tracer_20_points_droite(slider_p=True, slider_m=True, show_eq=True)


def qcm_ordonnee_origine():
    create_qcm({
        'question': "Quelle est l\'ordonnée à l'origine ?",
        'choices': ["L\'abscisse du point d'intersection de la droite avec l\'axe des x.",
                    "L\'abscisse du point d'intersection de la droite avec l\'axe des y.",
                    "L\'ordonnée du point d'intersection de la droite avec l\'axe des x.",
                    "L\'ordonnée du point d'intersection de la droite avec l\'axe des y."],
        'answer': "L\'ordonnée du point d'intersection de la droite avec l\'axe des y.",
    })


def qcm_train_test_proche():
    create_qcm({
        'question': "Pourquoi le pourcentage d’erreur sur les images de test est-il proche du pourcentage d’erreur sur les images d’entraînement ?",
        'choices': [
            "Car ce sont les mêmes images.",
            "Car ce sont des images du même type, mais différentes (elles n’ont pas été utilisées pendant l’entraînement).",
        ],
        'answer': "Car ce sont des images du même type, mais différentes (elles n’ont pas été utilisées pendant l’entraînement).",
    })


def qcm_pente():
    create_qcm({
        'question': 'Quelle droite a un coefficient directeur négatif ?',
        'choices': ['d1', 'd2'],
        'answer': 'd2',
    })


# --- Image mystère ---

def _mystere_placeholder():
    svg = """
    <svg xmlns='http://www.w3.org/2000/svg' width='140' height='140' viewBox='0 0 140 140'>
      <rect width='140' height='140' fill='black'/>
      <text x='70' y='85' font-family='Arial' font-size='80' text-anchor='middle' fill='white'>?</text>
    </svg>
    """.strip()
    b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    return f"data:image/svg+xml;base64,{b64}"


def _mystere_data():
    
    # Sélectionne des points connus dans le même ordre que la banque classique.
    # On parcourt d_train / r_train dans l'ordre naturel et on retient jusqu'à 20 exemples de chaque classe (2 puis 7) pour construire les dictionnaires known_points_a / known_points_b, ainsi qu'une séquence d'apparition pour l'animation.
    
    r = common.challenge.r_train
    d = common.challenge.d_train

    max_per_class = 20
    count_2 = 0
    count_7 = 0

    known_a = {}
    known_b = {}
    sequence = []  # liste de dicts {'group': 'A'/'B', 'key': 'A1'/'B3', ...}

    for i, label in enumerate(r):
        if label == 2 and count_2 < max_per_class:
            key = f"A{count_2 + 1}"
            coords = common.challenge.deux_caracteristiques(d[i])
            known_a[key] = [float(coords[0]), float(coords[1])]
            sequence.append({'group': 'A', 'key': key})
            count_2 += 1
        elif label == 7 and count_7 < max_per_class:
            key = f"B{count_7 + 1}"
            coords = common.challenge.deux_caracteristiques(d[i])
            known_b[key] = [float(coords[0]), float(coords[1])]
            sequence.append({'group': 'B', 'key': key})
            count_7 += 1

        if count_2 >= max_per_class and count_7 >= max_per_class:
            break

    return known_a, known_b, sequence


def _mystere_exo(
    show_zones=False,
    line_params=None,
    preplace_mystere=False,
    interactive=True,
    known_points_animate=True,
    preplace_known_points=True,
    hide_left_panel=False,
    html_title="Place le point C",
    checkpoint_enabled=True,
    show_status=True,
    exercise_validation=True,
    auto_pass_on_known_points_done=False,
):
    k2 = (40, 60)

    images = [_mystere_placeholder()]
    known_a, known_b, _ = _mystere_data()

    caption_html = f"<div style='font-weight:600;'>x<sub>C</sub> = {k2[0]} ; y<sub>C</sub> = {k2[1]}</div>"

    placer_mystere(
        html_title=html_title,
        images=images,
        expected_point=[k2[0], k2[1]],
        known_points_a=known_a,
        known_points_b=known_b,
        line_params=line_params,
        show_zones=show_zones,
        image_caption_html=caption_html,
        show_legend=True,
        force_origin=True,
        preplace_known_points=preplace_known_points,
        preplace_mystere=preplace_mystere,
        interactive=interactive,
        known_points_animate=known_points_animate,
        hide_left_panel=hide_left_panel,
        checkpoint_enabled=checkpoint_enabled,
        show_status=show_status,
        exercise_validation=exercise_validation,
        auto_pass_on_known_points_done=auto_pass_on_known_points_done,
    )

def mystere_qcm():
    create_qcm({
            'question': " À ton avis, l’image de caractéristique $C$ correspond à quel chiffre ?",
            'choices': ["2", "7"],
            'answer': "2",
        })


def points_connus_animation():
    known_a, known_b, order = _mystere_data()
    placer_caracteristiques(
        html_title="Points caractéristiques (exemples)",
        images=[],
        expected_points_a={},
        expected_points_b={},
        known_points_a=known_a,
        known_points_b=known_b,
        known_points_order=order,
        show_legend=True,
        force_origin=True,
        preplace_known_points=True,
        interactive=False,
        known_points_animate=True,
        checkpoint_enabled=False,
        show_status=False,
        exercise_validation=False,
        auto_pass_on_known_points_done=True,
        hide_left_panel=True,
        keep_aspect_ratio=False,
    )


def exercice_image_mystere():
    _mystere_exo(show_zones=False, interactive=True, known_points_animate=False, preplace_mystere=False)


def exercice_image_mystere_droite():
    line_params = {
        'm': common.challenge.droite_20_points['m'],
        'p': common.challenge.droite_20_points['p'],
    }
    _mystere_exo(
        show_zones=True,
        line_params=line_params,
        preplace_mystere=True,
        interactive=False,
        known_points_animate=False,
        preplace_known_points=True,
        checkpoint_enabled=False,
        show_status=False,
        exercise_validation=False,
        auto_pass_on_known_points_done=True,
    )


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


last_tracer_points_id = None


def tracer_points_droite(id_content=None, display_value="range", carac=None, initial_hidden=False, save=True,
                         side_box=True, disable_python_updates=False):
    if id_content is None:
        id_content = uuid.uuid4().hex
    global last_tracer_points_id
    last_tracer_points_id = id_content

    btn_m_minus = f'<button id="{id_content}-btn-m-minus" onclick="const el = document.getElementById(\'{id_content}-input-m\'); el.stepDown(); el.dispatchEvent(new Event(\'input\'))" style="width: 30px; height: 30px; cursor: pointer;">-</button>' if display_value == "range" else ""
    btn_m_plus = f'<button id="{id_content}-btn-m-plus" onclick="const el = document.getElementById(\'{id_content}-input-m\'); el.stepUp(); el.dispatchEvent(new Event(\'input\'))" style="width: 30px; height: 30px; cursor: pointer;">+</button>' if display_value == "range" else ""
    btn_p_minus = f'<button id="{id_content}-btn-p-minus" onclick="const el = document.getElementById(\'{id_content}-input-p\'); el.stepDown(); el.dispatchEvent(new Event(\'input\'))" style="width: 30px; height: 30px; cursor: pointer;">-</button>' if display_value == "range" else ""
    btn_p_plus = f'<button id="{id_content}-btn-p-plus" onclick="const el = document.getElementById(\'{id_content}-input-p\'); el.stepUp(); el.dispatchEvent(new Event(\'input\'))" style="width: 30px; height: 30px; cursor: pointer;">+</button>' if display_value == "range" else ""

    display(HTML(f'''
        <div id="{id_content}-container" style="{'visibility:hidden;' if initial_hidden else ''} max-width: 900px; margin: 0 auto;">
            <div id="{id_content}-score-container" style="text-align: center; font-weight: bold; font-size: 1.5rem;">Pourcentage d'erreur : <span id="{id_content}-score">...</span></div>

            <div style="display:flex; flex-direction:row; align-items:center; justify-content:center; gap:18px;">
                <div style="flex:1; min-width:320px;">
                    <canvas id="{id_content}-chart"></canvas>
                </div>
                <div id="{id_content}-slope-box-wrapper" style="display:{'flex' if side_box else 'none'}; align-items:center; justify-content:center;">
                    <div id="{id_content}-slope-box" style="width:250px; height:180px; border:0; border-radius:0; background:transparent; display:flex; align-items:center; justify-content:center; position:relative;">
                    </div>
                </div>
            </div>

            <div id="{id_content}-inputs" style="display: flex; gap: 1rem; justify-content: center; flex-direction: {'column' if display_value == "range" else 'row'};">
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <label for="{id_content}-input-m" id="{id_content}-label-m" style="color:#239E28;font-weight:bold;min-width:50px;white-space:nowrap;">m = </label>
                    {btn_m_minus}
                    <input type="{display_value}" {display_value == "range" and 'min="-5" max="5"'} value="0.5" step="0.1" id="{id_content}-input-m" style="color: #239E28">
                    {btn_m_plus}
                </div>
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                    <label for="{id_content}-input-p" id="{id_content}-label-p" style="color:#FF0000;font-weight:bold;min-width:50px;white-space:nowrap;">p = </label>
                    {btn_p_minus}
                    <input type="{display_value}" {display_value == "range" and 'min="-20" max="20"'} value="20" step="0.1" id="{id_content}-input-p">
                    {btn_p_plus}
                </div>
            </div>
        </div>
    '''))

    if carac is None:
        carac = common.challenge.deux_caracteristiques

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': display_value == "range",
        'save': save,
        'droite': {
            'mode': 'affine'
        },
        'inputs': {
            'm': True,
            'p': True,
        },
        'interception_point': True,
        'initial_values': {
            'm': 0.5,
            'p': 20,
        },
        'param_colors': {
            'm': '#239E28',
            'p': '#FF0000'
        },
        'compute_score': True,
        'disable_python_updates': disable_python_updates,
        'equation_fixed_position': True,
        'force_origin': True,
        'side_box': side_box,
    }

    run_js(
        f"mathadata.add_observer('{id_content}-container', () => window.mathadata.tracer_points('{id_content}', '{json.dumps(params, cls=NpEncoder)}'))")

    return id_content


def _error_for_m_p(m, p, carac, d_train, r_train, below, above):
    c_full = compute_c_train(carac, d_train)
    xs = c_full[:, 0]
    ys = c_full[:, 1]
    cond = (m * xs - ys + p) > 0
    preds = np.where(cond, below, above)
    err = 100 * np.mean(preds != r_train)
    return err if err <= 50 else 100 - err


def logistic_best_m_p(m_range, p_range, custom=False):
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques

    d_train = common.challenge.d_train
    r_train = common.challenge.r_train
    c_train = compute_c_train(carac, d_train)

    clf = LogisticRegression(max_iter=10000)
    clf.fit(c_train, r_train)

    w0, w1 = clf.coef_[0]
    b0 = clf.intercept_[0]
    if abs(w1) < 1e-9:
        w1 = 1e-9

    m = -float(w0) / float(w1)
    p = -float(b0) / float(w1)

    m_min, m_max = m_range
    p_min, p_max = p_range
    m = min(max(m, m_min), m_max)
    p = min(max(p, p_min), p_max)

    below = common.challenge.r_petite_caracteristique
    above = common.challenge.r_grande_caracteristique
    err = _error_for_m_p(m, p, carac, d_train, r_train, below, above)
    return {'m': float(m), 'p': float(p), 'error': float(err)}


def _pairs_to_best(best, m_range, p_range, steps=10, base=(0.5, 20.0)):
    m_min, m_max = m_range
    p_min, p_max = p_range
    pairs = []
    for i in range(1, steps + 1):
        t = i / (steps + 1)
        m = base[0] + t * (best['m'] - base[0])
        p = base[1] + t * (best['p'] - base[1])
        m = min(max(m, m_min), m_max)
        p = min(max(p, p_min), p_max)
        pairs.append((round(float(m), 1), round(float(p), 1)))
    pairs.append((round(float(best['m']), 1), round(float(best['p']), 1)))
    return pairs


def grid_search_pairs(target_error, m_range, p_range, custom=False, nb_faux=10):
    best = logistic_best_m_p(m_range, p_range, custom=custom)
    return _pairs_to_best(best, m_range, p_range, steps=nb_faux)


def grid_search_pairs_json(target_error, m_range, p_range, custom=False, nb_faux=10):
    best = logistic_best_m_p(m_range, p_range, custom=custom)
    pairs = _pairs_to_best(best, m_range, p_range, steps=nb_faux)
    
    # Recalcul des erreurs
    if custom:
        carac = common.challenge.deux_caracteristiques_custom
    else:
        carac = common.challenge.deux_caracteristiques
    d_train = common.challenge.d_train
    r_train = common.challenge.r_train
    below = common.challenge.r_petite_caracteristique
    above = common.challenge.r_grande_caracteristique
    
    errors = [_error_for_m_p(m, p, carac, d_train, r_train, below, above) for (m, p) in pairs]
    
    # S'assurer de prendre le vrai meilleur
    min_error_idx = np.argmin(errors)
    if min_error_idx != len(pairs) - 1:  # Si ce n'est pas le dernier (censé être best)
        best = {
            'm': round(float(pairs[min_error_idx][0]), 1),
            'p': round(float(pairs[min_error_idx][1]), 1),
            'error': round(float(errors[min_error_idx]), 2),
        }
        # Mettre le vrai best à la fin
        pairs[-1] = (best['m'], best['p'])
        errors[-1] = best['error']
    else:
        best = {
            'm': round(float(best['m']), 1),
            'p': round(float(best['p']), 1),
            'error': round(float(errors[-1]), 2),
        }
    
    errors = [round(float(e), 2) for e in errors]
    
    return json.dumps({
        'pairs': pairs,
        'errors': errors,
        'best': best,
        'target_error': target_error
    })


def grid_search_animate(id_content=None, readonly=True, nb_faux=10, create_graph=True, carac=None):
    global last_tracer_points_id
    target_error = 8
    m_range = (-5, 5)
    p_range = (-20, 20)
    custom = False
    if id_content is None and create_graph:
        id_content = tracer_points_droite(initial_hidden=True, carac=carac)
    elif id_content is None:
        id_content = last_tracer_points_id
    if id_content is None:
        print_error("Aucun graphe actif. Exécute d'abord tracer_points_droite().")
        return

    delay_ms = 1000

    pairs = grid_search_pairs(target_error, m_range, p_range, custom=custom, nb_faux=nb_faux)

    js_pairs = json.dumps(pairs)

    js = f"""
    (function() {{
      var id = "{id_content}";
      var pairs = {js_pairs};
      var idx = 0;
      if (!window.mathadata) window.mathadata = {{}};
      if (window.mathadata._gridTimer) {{
        clearInterval(window.mathadata._gridTimer);
        window.mathadata._gridTimer = null;
      }}

      function tryStart() {{
        var mInput = document.getElementById(id + "-input-m");
        var pInput = document.getElementById(id + "-input-p");
        var mMinus = document.getElementById(id + "-btn-m-minus");
        var mPlus  = document.getElementById(id + "-btn-m-plus");
        var pMinus = document.getElementById(id + "-btn-p-minus");
        var pPlus  = document.getElementById(id + "-btn-p-plus"); 
        var scoreEl = document.getElementById(id + "-score");
        var container = document.getElementById(id + "-container");
        if (!mInput || !pInput || !scoreEl) {{
          return false;
        }}
        if (container) {{
          container.style.visibility = 'visible';
        }}
        if ({'true' if readonly else 'false'}) {{
          mInput.readOnly = true;
          pInput.readOnly = true;
          mInput.style.pointerEvents = 'none';
          pInput.style.pointerEvents = 'none';
          mInput.style.opacity = '0.5';
          pInput.style.opacity = '0.5';
          [mMinus, mPlus, pMinus, pPlus].forEach(btn => {{
            if (btn) {{
                btn.disabled = true;
                btn.style.pointerEvents = 'none';
                btn.style.opacity = '0.3';
            }}
        }});
        }}

        function step() {{
          if (idx >= pairs.length) return;
          var pair = pairs[idx++];
          mInput.value = pair[0];
          pInput.value = pair[1];
          mInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
          pInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
          if (idx < pairs.length) {{
            setTimeout(step, {delay_ms});
          }}
        }}
        step();
        return true;
      }}

      // wait for DOM to be ready
      var attempts = 0;
      var timer = setInterval(function() {{
        attempts += 1;
        if (tryStart()) {{
          clearInterval(timer);
        }} else if (attempts > 50) {{
          clearInterval(timer);
          console.warn("grid_search_animate: inputs not found after waiting");
        }}
      }}, 100);
      window.mathadata._gridTimer = timer;
    }})();"""

    run_js(js)


def animate_pairs(id_content, pairs, readonly=True, delay_ms=1000):
    js_pairs = json.dumps(pairs)
    js = f"""
    (function() {{
      var id = "{id_content}";
      var pairs = {js_pairs};
      var idx = 0;
      if (!window.mathadata) window.mathadata = {{}};
      if (window.mathadata._gridTimer) {{
        clearInterval(window.mathadata._gridTimer);
        window.mathadata._gridTimer = null;
      }}

      function tryStart() {{
        var mInput = document.getElementById(id + "-input-m");
        var pInput = document.getElementById(id + "-input-p");
        var scoreEl = document.getElementById(id + "-score");
        var container = document.getElementById(id + "-container");
        if (!mInput || !pInput || !scoreEl) {{
          return false;
        }}
        if (container) {{
          container.style.visibility = 'visible';
        }}
        if ({'true' if readonly else 'false'}) {{
          mInput.readOnly = true;
          pInput.readOnly = true;
          mInput.style.pointerEvents = 'none';
          pInput.style.pointerEvents = 'none';
          mInput.style.opacity = '0.5';
          pInput.style.opacity = '0.5';
        }}

        function step() {{
          if (idx >= pairs.length) return;
          var pair = pairs[idx++];
          mInput.value = pair[0];
          pInput.value = pair[1];
          mInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
          pInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
          if (idx < pairs.length) {{
            setTimeout(step, {delay_ms});
          }}
        }}
        step();
        return true;
      }}

      var attempts = 0;
      var timer = setInterval(function() {{
        attempts += 1;
        if (tryStart()) {{
          clearInterval(timer);
        }} else if (attempts > 50) {{
          clearInterval(timer);
          console.warn("animate_pairs: inputs not found after waiting");
        }}
      }}, 100);
      window.mathadata._gridTimer = timer;
    }})();"""
    run_js(js)


def create_graph(figsize=(figw_full, figw_full)):
    fig, ax = plt.subplots(figsize=figsize)

    # Enlever les axes de droites et du haut
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Centrer les axes en (0,0)
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(("data", 0))

    # Afficher les flèches au bout des axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    # Nom des axex
    ax.set_xlabel('$x$', loc='right')
    ax.set_ylabel('$y$', loc='top', rotation='horizontal')

    return fig, ax


def tracer_droite(ax, m, p, x_min, x_max, color='black'):
    # Ajouter la droite
    x = np.linspace(x_min, x_max, 1000)
    y = m * x + p
    ax.plot(x, y, c=color)  # Ajout de la droite en noir

    # Display the equation of the line
    equation = f'$y = {m}x {"+" if p >= 0 else "-"} {abs(p)}$'
    ax.text(15, 3, equation, color=color, verticalalignment='top', horizontalalignment='left')

## Point de référence    
pointC1 = (20, 40)
pointC2 = (30, 10)
## Points à classer
pointM1 = (20, 30)
pointM2 = (30, 25)

def tracer_exercice_classification(display_point_coords=False, point_name="M1"):

    ## Droite de référence (mx + p)
    m = 0.5
    p = 20

    x = [pointC1[0]]
    y = [pointC1[1]]
    if point_name == "M2":
        x = [pointC2[0]]
        y = [pointC2[1]]

    y += [m * k1 + p for k1 in x]
    x += x

    _, ax = create_graph(figsize=(figw_full * 0.50, figw_full * 0.50))

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    mk2 = m * pointC1[0] + p
    if point_name == "M2":
        mk2 = m * pointC2[0] + p

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    # ax.set_yticks(np.arange(x_min, x_max, 5))
    ax.set_yticks([round(mk2, 2)])
    # remove the y axis ticks and labels
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels(['$y_M = ?$'])

    labels = [f'C({pointC1[0]}, {pointC1[1]})', f'M({pointM1[0]}, {round(mk2, 2)})' if display_point_coords else f'M(20, ?)']
    if point_name == "M2":
        labels = [f'C({pointC2[0]}, {pointC2[1]})',
                  f'M({pointM2[0]}, {round(mk2, 2)})' if display_point_coords else 'M(30, ?)']
    colors = ['C4', 'C3']
    for i in range(len(labels)):
        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i] / x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i] / x_max, linestyle='dotted', color='gray')

        ax.annotate(labels[i], (x[i] + 1, y[i]), va='center', color=colors[i])
        ax.scatter(x[i], y[i], marker='+', c=colors[i])

    tracer_droite(ax, m, p, x_min, x_max, color=colors[1])

    return ax


def exercice_calcul_au_dessus():
    tracer_exercice_classification()
    plt.show()
    try:
        plt.close()
    except AttributeError as e:
        # Workaround for matplotlib_pyodide/basthon backend: closing can fail if the DOM canvas was already destroyed.
        if "parentNode" not in str(e):
            raise


def qcm_dessus():
    create_qcm({
        'question': "Est-ce que l\'ordonnée du point M est plus petite ou plus grande que l\'ordonnée du point C ?",
        'choices': ['plus petite', 'plus grande'],
        'answer': 'plus petite',
    })

def qcm_dessous():
    create_qcm({
        'question': "Cette fois, l'ordonnée de M est-elle plus grande ou plus petit que celle de C ? En déduire le chiffre de l'image associée au point M",
        'choices': ["l'image est un 2, car le point C est au-dessus de la droite", "l'image est un 7, car le point C est en dessous de la droite"],
        'answer': "l'image est un 7, car le point C est en dessous de la droite",
    })


def qcm_dessus_dessous():
    create_qcm({
        'question': (
            "Pour savoir où se trouve un point C(x<sub>C</sub>, y<sub>C</sub>) par rapport à une droite d’équation y = mx + p :<br/><br/>"
            "Si y<sub>C</sub> &gt; m x<sub>C</sub> + p, alors le point C est..."
        ),
        'choices': [
            "Au-dessus de la droite.",
            "En-dessous de la droite.",
        ],
        'answer': "Au-dessus de la droite.",
        'multiline': True,
    })


def exercice_calcul_au_dessous():
    tracer_exercice_classification(point_name="M2")
    plt.show()
    try:
        plt.close()
    except AttributeError as e:
        # Workaround for matplotlib_pyodide/basthon backend: closing can fail if the DOM canvas was already destroyed.
        if "parentNode" not in str(e):
            raise


def affichage_zones_custom(a1, b1, a2, b2):
    common.challenge.affichage_2_cara(a1, b1, a2, b2, True)
    tracer_points_droite(display_value="number", carac=common.challenge.deux_caracteristiques_custom, save=False)


def afficher_customisation(enable_optimizer=False):
    display_id = uuid.uuid4().hex
    button_html = f'''
        <div style="display:flex; justify-content:center; align-items:center; margin: 10px 0;">
            <button id="{display_id}-grid-btn" class="mathadata-button mathadata-button--primary" style="max-width: 300px; width: 100%;" disabled title="Sélectionne des zones avant de lancer la recherche">
                Lancer la recherche de la meilleure droite
            </button>
        </div>
    ''' if enable_optimizer else ''
    
    # Utiliser mathadata.add_observer pour attendre que le bouton soit dans le DOM (AVANT display)
    if enable_optimizer:
        run_js(f'''
            mathadata.add_observer('{display_id}-grid-btn', () => {{
                if (!window.mathadata) window.mathadata = {{}};
                const base = {{m: 0.5, p: 20}};
                const msgEl = document.getElementById('{display_id}-grid-msg');
                if (msgEl) {{ msgEl.textContent = ''; msgEl.style.visibility = 'hidden'; }}

                function setLineVisible(visible) {{
                    const chart = window.mathadata.charts?.['{display_id}-chart'];
                    if (!chart) return;
                    const ds = chart.data?.datasets || [];
                    const lineIdx = ds.findIndex(d => d.type === 'line' && d.borderColor === 'black');
                    if (lineIdx >= 0) {{ ds[lineIdx].hidden = !visible; chart.update(); }}
                }}

                function setScoreVisible(visible) {{
                    const el = document.getElementById('{display_id}-score-container');
                    if (el) el.style.visibility = visible ? 'visible' : 'hidden';
                }}

	                function applyBase() {{
	                    const mInput = document.getElementById('{display_id}-input-m');
	                    const pInput = document.getElementById('{display_id}-input-p');
	                    if (!mInput || !pInput) return;
	                    mInput.value = base.m;
	                    pInput.value = base.p;
	                    mInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
	                    pInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
	                }}

	                function disableInputsOnce() {{
	                    const mInput = document.getElementById('{display_id}-input-m');
	                    const pInput = document.getElementById('{display_id}-input-p');
	                    if (!mInput || !pInput) return false;
	                    mInput.readOnly = true;
	                    pInput.readOnly = true;
	                    mInput.disabled = true;
	                    pInput.disabled = true;
	                    mInput.style.pointerEvents = 'none';
	                    pInput.style.pointerEvents = 'none';
	                    mInput.style.opacity = '0.5';
	                    pInput.style.opacity = '0.5';
	                    return true;
	                }}

	                function ensureInputsDisabled() {{
	                    let attempts = 0;
	                    const timer = setInterval(() => {{
	                        attempts += 1;
	                        if (disableInputsOnce() || attempts > 50) {{
	                            clearInterval(timer);
	                        }}
	                    }}, 100);
	                }}

	                // In optimizer mode, keep m/p inputs disabled at all times.
	                ensureInputsDisabled();

	                window.mathadata.on_custom_update = () => {{
	                    window.mathadata.run_python('update_custom()', (points) => {{
	                        mathadata.update_points('{display_id}', {{points}});
	                        document.getElementById('{display_id}-container').style.visibility = 'visible';
	                        applyBase();
	                        ensureInputsDisabled();
	                        setLineVisible(true);
	                        setScoreVisible(false);
	                        if (msgEl) {{ msgEl.textContent = ''; msgEl.style.visibility = 'hidden'; }}
                            const btn = document.getElementById('{display_id}-grid-btn');
                            if (btn) btn.disabled = false;
	                    }});
	                }}

                const btn = document.getElementById('{display_id}-grid-btn');
                if (btn) {{
                    btn.addEventListener('click', () => {{
                        if (window.mathadata._gridInFlight || window.mathadata._gridAnimating) return;
                        window.mathadata._gridInFlight = true;
                        btn.disabled = true;
                        if (msgEl) {{ msgEl.textContent = 'Calcul en cours...'; msgEl.style.visibility = 'visible'; }}
                        setLineVisible(true);
                        setScoreVisible(true);
                        applyBase();
                        document.getElementById('{display_id}-container').style.visibility = 'visible';
                        window.mathadata.run_python('grid_search_pairs_json(8, (-5, 5), (-20, 20), custom=True, nb_faux=10)', (pairsJson) => {{
                            const payload = (typeof pairsJson === 'string') ? JSON.parse(pairsJson) : pairsJson;
                            window.mathadata._grid_pairs = payload.pairs || [];
                            window.mathadata._grid_errors = payload.errors || [];
                            window.mathadata._grid_best = payload.best;
                            window.mathadata._grid_target = payload.target_error;
                            if (window.mathadata._grid_pairs.length > 0) {{
                                window.mathadata.animate_pairs_js('{display_id}');
                            }} else if (msgEl) {{
                                msgEl.textContent = 'Aucun couple trouvé.';
                                msgEl.style.visibility = 'visible';
                            }}
                            window.mathadata._gridInFlight = false;
                            btn.disabled = false;
                        }});
                    }});
                }}
            }});
        ''')
    
    display(HTML(f'''
        <div id="{display_id}"></div>
        {button_html}
        <div id="{display_id}-grid-msg" style="margin: 10px 0; font-weight: 600;"></div>
    '''))
    # Utilise display_custom_selection_2d si elle existe, sinon display_custom_selection
    if hasattr(common.challenge, 'display_custom_selection_2d'):
        common.challenge.display_custom_selection_2d(display_id)
    else:
        common.challenge.display_custom_selection(display_id)

    tracer_points_droite(id_content=display_id, display_value="number",
                         carac=common.challenge.deux_caracteristiques_custom,
                         initial_hidden=True, save=False,
                         disable_python_updates=enable_optimizer)

    if not enable_optimizer:
        run_js(f'''
            if (!window.mathadata) window.mathadata = {{}};
            window.mathadata.on_custom_update = () => {{
                window.mathadata.run_python('update_custom()', (points) => {{
                    mathadata.update_points('{display_id}', {{points}})
                    document.getElementById('{display_id}-container').style.visibility = 'visible';
                }});
            }}
        ''')
        return

    # Expose une animation JS (les couples seront calculés après update_custom)
    run_js(f'''
        if (!window.mathadata) window.mathadata = {{}};
        window.mathadata.animate_pairs_js = (id) => {{
            const pairs = window.mathadata._grid_pairs || [];
            const errors = window.mathadata._grid_errors || [];
            const mInput = document.getElementById(id + "-input-m");
            const pInput = document.getElementById(id + "-input-p");
            const scoreEl = document.getElementById(id + "-score");
            if (!mInput || !pInput) return;
            mInput.readOnly = true;
            pInput.readOnly = true;
            mInput.style.opacity = '0.5';
            pInput.style.opacity = '0.5';
            mInput.style.pointerEvents = 'none';
            pInput.style.pointerEvents = 'none';
            window.mathadata._gridAnimating = true;

            let idx = 0;
	            function finish() {{
	                const msgEl = document.getElementById(id + '-grid-msg');
	                const target = window.mathadata._grid_target;
	                const m = Number(mInput.value).toFixed(1);
	                const p = Number(pInput.value).toFixed(1);
                const eVal = errors.length ? errors[errors.length - 1] : null;
                if (msgEl && eVal !== null) {{
                    const eTxt = Number(eVal).toFixed(2);
                    msgEl.textContent = (eVal <= target)
                      ? `Couple minimisant l’erreur : m=${{m}}, p=${{p}} (erreur = ${{eTxt}}%)`
                      : `Couple minimisant l’erreur : m=${{m}}, p=${{p}} (erreur = ${{eTxt}}%). Objectif ${{target}}% non atteignable avec ces zones.`;
                    msgEl.style.visibility = 'visible';
                }}
                window.mathadata._gridAnimating = false;

                let newValues = {{}}
                newValues.a = Number(mInput.value)
                newValues.b = -1
                newValues.c = Number(pInput.value)
	           
	                mathadata.run_python(`set_input_values('${{JSON.stringify(newValues)}}')`)
	                
	            }}

            function step() {{
                if (idx >= pairs.length) return finish();
                const pair = pairs[idx];
                mInput.value = pair[0];
                pInput.value = pair[1];
                if (scoreEl && errors[idx] !== undefined) {{
                    scoreEl.textContent = `${{errors[idx].toFixed(2)}}%`;
                }}
                mInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                pInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                idx += 1;
                setTimeout(step, 1000);
            }}
            step();
        }};
    ''')


def exercice_association_deux_droites():
    exercice_association(
        questions_dict={
            'question1': ' Quelle droite a un coefficient directeur négatif ?',
            'question2': ' Quelle droite a un coefficient directeur positif ?',
            'question3': ' Quelle droite a un coefficient directeur nul ?',
        },
        answers_dict={
            'answer1': 'La droite d2.',
            'answer2': 'La droite d1.',
            'answer3': 'La droite d3.',
        },
        question_generale="Associez chaque droite à sa pente",
    )


def qcm_association_negatif():
    create_qcm({
        'question': "Quelle droite a un coefficient directeur négatif ?",
        'choices': ["La droite d1.",
                    "La droite d2.",
                    "La droite d3."],
        'answer': "La droite d2.",
    })


def qcm_association_positif():
    create_qcm({
        'question': "Quelle droite a un coefficient directeur positif ?",
        'choices': ["La droite d1.",
                    "La droite d2.",
                    "La droite d3."],
        'answer': "La droite d1.",
    })


def qcm_association_nul():
    create_qcm({
        'question': "Quelle droite a un coefficient directeur nul ?",
        'choices': ["La droite d1.",
                    "La droite d2.",
                    "La droite d3."],
        'answer': "La droite d3.",
    })


# JS

def calculer_score_droite(ensure_draw=True):
    test_r = None
    test_d = None
    if common.challenge.id == 'mnist':
        test_r = common.challenge.r_train_test
        test_d = common.challenge.d_train_test
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, ensure_draw=ensure_draw, test_d=test_d, test_r=test_r)

def verifier_score_droite():
    calculer_score_droite_geo(custom=False, validate=common.challenge.objectif_score_droite, error_msg=None, banque=False, success_msg="Super tu as trouvé une bonne droite de séparation !",
                              animation=False, ensure_draw=False, test_r = None, test_d = None)

def calculer_score_custom_droite(ensure_draw=True):
    test_r = None
    test_d = None
    if common.challenge.id == 'mnist':
        test_r = common.challenge.r_train_test
        test_d = common.challenge.d_train_test
    calculer_score_droite_geo(custom=True, validate=common.challenge.objectif_score_droite_custom,
                              error_msg="Continue à chercher 2 zones pour avoir moins de " + str(
                                  common.challenge.objectif_score_droite_custom) + "% d'erreur. Pense à changer les valeurs de m et p après avoir défini ta zone.", ensure_draw=ensure_draw, test_d=test_d, test_r=test_r)


# Variables globales pour les valeurs des sliders et pourcentage d'erreur
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


def afficher_separation_line(show_slider_p=False, show_slider_m=False,
                             p=1, slope=1,
                             show_equation=False,
                             width=466, height=400, jsx_version="1.4.0",
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
    js_show_p = 'true' if show_slider_p else 'false'
    js_show_m = 'true' if show_slider_m else 'false'
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
    if show_slider_m or show_slider_p:
        error_score = True

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

        # Calculate space needed for sliders
        slider_count = (1 if show_slider_p else 0) + (1 if show_slider_m else 0)
        if slider_count == 1:
            slider_y_min = -6
        elif slider_count == 2:
            slider_y_min = -10
        else:
            slider_y_min = 0

        view_y_min = min(data_y_min - y_padding, -2, slider_y_min)  # Include slider space

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
        slider_count = (1 if show_slider_p else 0) + (1 if show_slider_m else 0)
        if slider_count == 1:
            slider_y_min = -6
        elif slider_count == 2:
            slider_y_min = -10
        else:
            slider_y_min = -10

        view_x_min, view_x_max, view_y_max = -10, 50, 50
        view_y_min = min(-10, slider_y_min)

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
          height: {height}px; 
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
    <div class="title">
      Trouver une droite de séparation : y =
      <span style="color:{m_color}; font-weight:700;">m</span> x +
      <span style="color:{p_color}; font-weight:700;">p</span>
    </div>
    {score_div}

    <div class="main-container">
        <div id="{box_id}" class="jxgbox" style="width:{width}px; height:{height}px;"></div>

        <div id="slope-box-container" class="slope-box" style="display: {'flex' if show_slider_m else 'none'};">
            <svg id="slope-svg" width="100%" height="100%" viewBox="0 0 250 {height}" style="overflow: visible;">
                <text x="125" y="25" text-anchor="middle" font-weight="bold" font-size="16">Calcul de la pente m</text>
                <line id="slope-base" stroke="purple" stroke-width="4" stroke-linecap="round" />
                <line id="slope-height" stroke="orange" stroke-width="4" stroke-linecap="round" />
                <line id="slope-hypo" stroke="black" stroke-width="5" stroke-linecap="round" />
                <text id="text-base" text-anchor="middle" font-weight="bold" font-size="16" fill="purple">10</text>
                <text id="text-calc" text-anchor="start" font-weight="bold" font-size="16" fill="orange"></text>
                <foreignObject id="text-formula" x="0" y="{height - 60}" width="250" height="55"></foreignObject>
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
        // slider requests
        var wantP = {js_show_p};
        var wantM = {js_show_m};
        var sliderCount = (wantP ? 1 : 0) + (wantM ? 1 : 0);
        // YMIN depends on number of sliders and data range
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
        // computeScore pour le pourcentage d'erreur
        const computeScore = () => {{
              const n = {blue_pts}.length;

              // Points oranges incorrectement classés (au-dessus de la droite alors qu'ils devraient être en dessous)
              const errorsA = {orange_pts}.filter(([x, y]) => y >= currentM() * x + currentP()).length;

              // Points bleus incorrectement classés (en dessous de la droite alors qu'ils devraient être au-dessus)
              const errorsB = {blue_pts}.filter(([x, y]) => y < currentM() * x + currentP()).length;

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
        // slider horizontal extents - adapt to actual viewport
        var x_range = XMAX - XMIN;
        var x1 = XMIN + x_range * 0.15;  // 15% from left
        var x2 = XMAX - x_range * 0.15;  // 15% from right
        // helper to clamp slider y so it's visible
        function clampY(y) {{
          var minVisible = YMIN + 0.3;
          return y < minVisible ? minVisible : y;
        }}
        // create sliders and labels
        if (sliderCount === 1) {{
          var y_single = -4;

          // Fond blanc opaque pour toute la largeur
          var bg_p1 = board.create('point', [function(){{ return XMIN; }}, function(){{ return y_single + 1.5; }}], {{visible:false}});
          var bg_p2 = board.create('point', [function(){{ return XMAX; }}, function(){{ return y_single + 1.5; }}], {{visible:false}});
          var bg_p3 = board.create('point', [function(){{ return XMAX; }}, function(){{ return YMIN; }}], {{visible:false}});
          var bg_p4 = board.create('point', [function(){{ return XMIN; }}, function(){{ return YMIN; }}], {{visible:false}});
          board.create('polygon', [bg_p1, bg_p2, bg_p3, bg_p4], {{
            fillColor: '#FFFFFF', fillOpacity: 0.70, 
            borders: {{strokeColor: '#DEE2E6', strokeWidth: 2}},
            layer: 7, fixed: true, highlight: false
          }});

          if (wantP) {{
            // Label avec fond coloré
            board.create('text', [function(){{ return x1 - 1; }}, function(){{ return y_single; }}, "p"],
              {{fontSize:18, anchorX:'right', anchorY:'middle', color: 'black', layer:9, 
                fontWeight: 'bold', fixed: true, highlight: false}});

            s_p = board.create('slider', [[x1, y_single], [x2, y_single], [-20, p_fixed, 20]], {{
              withLabel: false, name:'', 
              strokeColor: '{p_color}', fillColor: '{p_color}', 
              strokeWidth: 4, size: 8,
              point1: {{strokeColor: '{p_color}', fillColor: '{p_color}', size: 6}},
              point2: {{strokeColor: '{p_color}', fillColor: '{p_color}', size: 6}},
              snapWidth: 1, layer: 8
            }});
            board.create('text', [
              function(){{ return (x1 + x2) / 2; }},
              function(){{ return y_single - 0.6; }},
              function(){{ return formatNumber(currentP()); }}
            ], {{fontSize:14, anchorX:'middle', anchorY:'top', color:'{p_color}', layer:9, fixed:true, highlight:false}});
          }} else if (wantM) {{
            // Label avec fond coloré  
            board.create('text', [function(){{ return x1 - 1; }}, function(){{ return y_single; }}, "m"],
              {{fontSize:18, anchorX:'right', anchorY:'middle', color: 'black', layer:9, 
                fontWeight: 'bold', fixed: true, highlight: false}});

            s_m = board.create('slider', [[x1, y_single], [x2, y_single], [-3, m_fixed, 3]], {{
              withLabel: false, name:'', 
              strokeColor: '{m_color}', fillColor: '{m_color}', 
              strokeWidth: 4, size: 8,
              point1: {{strokeColor: '{m_color}', fillColor: '{m_color}', size: 6}},
              point2: {{strokeColor: '{m_color}', fillColor: '{m_color}', size: 6}},
              snapWidth: 0.1, layer: 8
            }});
            board.create('text', [
              function(){{ return (x1 + x2) / 2; }},
              function(){{ return y_single - 0.6; }},
              function(){{ return formatNumber(currentM()); }}
            ], {{fontSize:14, anchorX:'middle', anchorY:'top', color:'{m_color}', layer:9, fixed:true, highlight:false}});
          }}
        }} else if (sliderCount === 2) {{
          var y_p = -8;
          var y_m = -4;

          // Fond blanc opaque pour toute la largeur
          var bg_p1 = board.create('point', [function(){{ return XMIN; }}, function(){{ return YMIN; }}], {{visible:false}});
          var bg_p2 = board.create('point', [function(){{ return XMAX; }}, function(){{ return YMIN; }}], {{visible:false}});
          var bg_p3 = board.create('point', [function(){{ return XMAX; }}, function(){{ return y_m + 1.5; }}], {{visible:false}});
          var bg_p4 = board.create('point', [function(){{ return XMIN; }}, function(){{ return y_m + 1.5; }}], {{visible:false}});
          board.create('polygon', [bg_p1, bg_p2, bg_p3, bg_p4], {{
            fillColor: '#FFFFFF', fillOpacity: 0.70, 
            borders: {{strokeColor: '#DEE2E6', strokeWidth: 2}},
            layer: 7, fixed: true, highlight: false
          }});

          // Labels avec fonds colorés
          board.create('text', [function(){{ return x1 - 1; }}, function(){{ return y_p; }}, "p"],
              {{fontSize:18, anchorX:'right', anchorY:'middle', color: 'black', layer:9, 
                fontWeight: 'bold', fixed: true, highlight: false}});
          board.create('text', [function(){{ return x1 - 1; }}, function(){{ return y_m; }}, "m"],
              {{fontSize:18, anchorX:'right', anchorY:'middle', color: 'black', layer:9, 
                fontWeight: 'bold', fixed: true, highlight: false}});

          // Sliders améliorés
          s_m = board.create('slider', [[x1, y_m], [x2, y_m], [-3, m_fixed, 3]], {{
            withLabel: false, name:'', 
            strokeColor: '{m_color}', fillColor: '{m_color}', 
            strokeWidth: 4, size: 8,
            point1: {{strokeColor: '{m_color}', fillColor: '{m_color}', size: 6}},
            point2: {{strokeColor: '{m_color}', fillColor: '{m_color}', size: 6}},
            snapWidth: 0.1, layer: 8
          }});
          board.create('text', [
            function(){{ return (x1 + x2) / 2; }},
            function(){{ return y_m - 0.6; }},
            function(){{ return formatNumber(currentM()); }}
          ], {{fontSize:14, anchorX:'middle', anchorY:'top', color:'{m_color}', layer:9, fixed:true, highlight:false}});
          s_p = board.create('slider', [[x1, y_p], [x2, y_p], [-20, p_fixed, 20]], {{
            withLabel: false, name:'', 
            strokeColor: '{p_color}', fillColor: '{p_color}', 
            strokeWidth: 4, size: 8,
            point1: {{strokeColor: '{p_color}', fillColor: '{p_color}', size: 6}},
            point2: {{strokeColor: '{p_color}', fillColor: '{p_color}', size: 6}},
            snapWidth: 1, layer: 8
          }});
          board.create('text', [
            function(){{ return (x1 + x2) / 2; }},
            function(){{ return y_p - 0.6; }},
            function(){{ return formatNumber(currentP()); }}
          ], {{fontSize:14, anchorX:'middle', anchorY:'top', color:'{p_color}', layer:9, fixed:true, highlight:false}});
        }}
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
        // Origin point O
        board.create('text', [-0.75, -1.5, 'O'], {{
        fontSize:16,
        anchorX:'middle',
        anchorY:'bottom',
        color:'#000000',
        layer:12,
        fixed:true,
        highlight:false
        }});
        // y-intercept on the y-axis (dynamic red point)
        var yInterceptPoint = board.create('point', [
          0,
          function(){{ return currentP(); }}
        ], {{
          name:'',
          strokeColor:'{p_color}',
          fillColor:'{p_color}',
          size:5,
          withLabel:false,
          fixed:true,
          highlight:false,
          layer:12
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

        board.create('text',[legend_x + rect_width + 0.5, legend_y_top, "Exemples d'images de 2"], {{
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

        board.create('text',[legend_x + rect_width + 0.5, legend_y_bottom, "Exemples d'images de 7"], {{
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
            var textFormula = document.getElementById('text-formula');

            var unit = 10; 
            var boardWidth = {width};
            var xRange = XMAX - XMIN;
            var pixelsPerUnit = boardWidth / xRange;
            var vizBase = unit * pixelsPerUnit;

            // Calcul vertical
            var hViz = vizBase * m;

            // Centrage
            var startX = (250 - vizBase) / 2; 
            var centerY = {height} / 2;
            var startY = centerY + hViz / 2;
            
            if (startY > {height} - 40) startY = {height} - 40;

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
            textBase.textContent = "10";

            textCalc.setAttribute('x', Bx + 10);
            textCalc.setAttribute('y', (By + Cy)/2 + 5);
            
            var mDisp = parseFloat(m.toFixed(2));
            var hDisp = parseFloat((m * 10).toFixed(2));
            textCalc.textContent = hDisp;

            if (textFormula) {{
                // Render the formula as a fraction (keep colors).
                textFormula.innerHTML = `
                  <div xmlns="http://www.w3.org/1999/xhtml"
                       style="width:250px; height:55px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:16px; line-height:1;">
                    <span style="color:{m_color};">m</span>
                    <span style="color:black;">&nbsp;=&nbsp;</span>
                    <span style="display:inline-flex; flex-direction:column; align-items:center; margin:0 4px;">
                      <span style="color:orange; padding:0 2px;">${{hDisp}}</span>
                      <span style="height:2px; width:100%; background:black; margin:2px 0;"></span>
                      <span style="color:purple; padding:0 2px;">10</span>
                    </span>
                    <span style="color:black;">&nbsp;=&nbsp;</span>
                    <span style="color:{m_color};">${{mDisp}}</span>
                  </div>
                `;
            }}
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
        if show_slider_m:
            iframe_width += 270  # 250 box + 20 gap

        iframe_html = f'<div style="display:flex; justify-content:center;"><iframe src="{data_uri}" style="width:{iframe_width}px; height:{height + 90}px; border:none;"></iframe></div>'
        display(HTML(iframe_html))


### Validation

def function_validation_calculer_score_droite(errors, answers):
    """Valide que le score d'erreur est inférieur ou égal à l'objectif"""
    if geo.input_values is None or 'a' not in geo.input_values or 'b' not in geo.input_values or 'c' not in geo.input_values:
        errors.append("Tu dois d'abord ajuster les paramètres de la droite dans le graphique ci-dessus.")
        return False

    score = compute_score(geo.input_values['a'], geo.input_values['b'], geo.input_values['c'], custom=False)
    objectif = common.challenge.objectif_score_droite

    if objectif < score < 100 - objectif:
        errors.append(
            f"Le pourcentage d'erreur est encore trop élevé. Continue à ajuster les paramètres m et p pour obtenir moins de {objectif}% d'erreur.")
        return False

    pretty_print_success(f"Bravo ! Tu as trouvé une droite avec {round(score, 2)}% d'erreur.")
    return True


validation_execution_tracer_20_points_droite = MathadataValidate(success="")
validation_execution_tracer_20_points_droite_p = MathadataValidate(success="")
validation_execution_tracer_20_points_droite_pm = MathadataValidate(success="")
validation_execution_calculer_score_droite = MathadataValidate(
    function_validation=function_validation_calculer_score_droite, success="")
validation_execution_point_droite = MathadataValidate(success="")
validation_execution_point_droite_dessous = MathadataValidate(success="")


def function_validation_equation(errors, answers):
    m = 0.5
    p = 20
    y_m = answers['y_M']

    if not (isinstance(y_m, (int, float))):
        errors.append(
            "Les coordonnées de M doivent être des nombres. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False

    if y_m != m * pointM1[0] + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        errors.append(f"m={m}, p={p}")
        return False

    return True


def function_validation_equation_b(errors, answers):
    m = 0.5
    p = 20
    y_m = answers['y_M']

    if not (isinstance(y_m, (int, float))):
        errors.append(
            "Les coordonnées de M doivent être des nombres. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False

    if y_m != m * pointM2[0] + p:
        errors.append("L'ordonnée de M n'est pas correcte.")
        return False

    return True


def function_validation_score_droite_20(errors, answers):
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
            f"Ce n'est pas la bonne réponse. Tu as donné le nombre d'erreurs (points mal classés) au lieu du pourcentage d'erreur. ")
        return False
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
                f"Bravo, c'est la bonne réponse. Il y a une image de 7 au dessus de la droite et cinq images de 2 en dessous, donc {nb_erreurs} erreurs pour 20 images soit {pourcentage_erreur}%.")
        return True
    else:
        errors.append(
            f"Ce n'est pas la bonne réponse. Compte le nombre d'erreurs, c'est à dire le nombre de points du mauvais côté de la droite, puis calcule le pourcentage d'erreur en divisant par le nombre total d'images.")
        return False


def function_validation_score_droite_p(errors, answers):
    user_answer = answers['p']
    if not (isinstance(user_answer, (int, float))):
        errors.append(
            "La valeur p doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    if user_answer != -1 and user_answer != 0:
        errors.append(
            "Le pourcentage d'erreur peut être plus petit. "
            "Utilise le curseur pour ajuster la valeur de p et réduire le pourcentage d'erreur.")
        return False
    pretty_print_success("Bravo, tu as trouvé la bonne valeur de p et la droite ayant cette ordonée à l'origine !")
    return True


def function_validation_score_droite_pm(errors, answers):
    user_answer_p = answers['p']
    user_answer_m = answers['m']
    if not (isinstance(user_answer_p, (int, float))):
        errors.append(
            "La valeur p doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    if not (isinstance(user_answer_m, (int, float))):
        errors.append(
            "La valeur m doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    err_return = ""
    if error_score != 5:
        err_return += "Le pourcentage d'erreur peut arriver à 5%. Essaie encore.\n"
    if user_answer_p != round(slider_p_value):
        err_return += f"Attention, la valeur de p n'est pas celle de la droite.\n"
    if user_answer_m != round(slider_m_value, 2):
        err_return += f"Attention, la valeur de m n'est pas celle de la droite.\n"
    if err_return != "":
        errors.append(err_return)
        return False
    pretty_print_success("Bravo, tu as trouvé les paramètres de la bonne droite !")
    return True


def function_validation_pente(errors, answers):
    user_answer_m1 = answers['m1']
    user_answer_m2 = answers['m2']
    if not (isinstance(user_answer_m1, (int, float))):
        errors.append(
            "La valeur m1 doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    if not (isinstance(user_answer_m2, (int, float))):
        errors.append(
            "La valeur m2 doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','. Exemple : 3.14 et non 3,14")
        return False
    cond1 = (user_answer_m1 != 1)
    cond2 = (user_answer_m2 != -0.5)
    if cond1 and cond2:
        errors.append(
            "Les deux réponses sont incorrectes. Essaie de faire les calculs étape par étape pour chaque coefficient directeur.")
        return False
    if cond1:
        errors.append("La pente de la première droite est incorrecte. Reessaie !")
        return False
    if cond2:
        errors.append(
            "La pente de la deuxième droite est incorrecte. Reessaie en faisant attention à l'ordre des calculs ! Le signe est important.")
        return False
    pretty_print_success("Bravo ! Tu as calculé correctement les pentes des deux droites.")
    return True


validation_question_equation = MathadataValidateVariables({
    'y_M': None},
    tips=[
        {
            'seconds': 20,
            'tip': 'Retrouvez l\'ordonnée de M en remplaçant x par x_M dans l\'équation de la droite.'
        },
        {
            'trials': 3,
            'tip': 'Retrouvez l\'ordonnée de M en remplaçant x par x_M = 20 dans l\'équation de la droite.'
        },
        {
            'seconds': 30,
            'trials': 5,
            'print_solution': False,  # Print "Voici la solution: test : 3"
            'validate': False  # Unlock the next cells
        }
    ],
    function_validation=function_validation_equation)
validation_question_equation_dessous = MathadataValidateVariables({
    'y_M': None
},
    tips=[
        {
            'seconds': 20,
            'tip': 'Retrouvez l\'ordonnée de M en remplaçant x par x_M dans l\'équation de la droite.'
        },
        {
            'trials': 3,
            'tip': 'Retrouvez l\'ordonnée de M en remplaçant x par x_M = 30 dans l\'équation de la droite.'
        },
        {
            'seconds': 30,
            'trials': 5,
            'print_solution': False,  # Print "Voici la solution: test : 3"
            'validate': False  # Unlock the next cells
        }
    ],
    function_validation=function_validation_equation_b)

validation_question_score_droite_20 = MathadataValidateVariables({
    'erreur_20': None
},
    tips=[
        {
            'seconds': 10,
            'trials': 1,
            'operator': 'OR',
            'tip': "Il y a 20 points en tout"
        }
    ],
    function_validation=function_validation_score_droite_20, success="")

validation_question_score_droite_p = MathadataValidateVariables({
    'p': None
},
    function_validation=function_validation_score_droite_p, success="")


def function_validation_image_mystere(errors, answers):
    def normalize(v):
        if isinstance(v, str):
            v = v.strip()
            if v.isdigit():
                return int(v)
        return v

    a = normalize(answers['classe_image_A'])
    if a not in [2, 7]:
        errors.append("La réponse doit être 2 ou 7.")
        return False
    if a != 2:
        errors.append("L'image A est mal classée. Observe les points voisins.")
        return False

    pretty_print_success("Bravo ! Tu as correctement identifié l'image.")
    return True


validation_question_image_mystere = MathadataValidateVariables({
    'classe_image_A': None,
},
    function_validation=function_validation_image_mystere,
    success="")


def get_possible_values_droite_pm():
    return [
        {'m': 0.5, 'p': 5},
        {'m': 0.5, 'p': 6},
        {'m': 0.4, 'p': 6},
        {'m': 0.4, 'p': 7}
    ]


validation_question_score_droite_pm = MathadataValidateVariables(get_names_and_values=get_possible_values_droite_pm,
                                                                 function_validation=function_validation_score_droite_pm,
                                                                 success="",
                                                                 tips=[
                                                                     {
                                                                         'seconds': 10,
                                                                         'tip': 'Les valeurs de m et p peuvent être lues directement sur l\'équation de la droite affichée en bas du graphique.'
                                                                     },
                                                                     {
                                                                         'trials': 2,
                                                                         'tip': 'Regardez l\'équation de la droite en bas du graphique : elle est de la forme y = m×x + p. Tu peux lire directement les valeurs de m (la pente) et p (l\'ordonnée à l\'origine).'
                                                                     },
                                                                     {
                                                                         'seconds': 30,
                                                                         'trials': 5,
                                                                         'print_solution': False,
                                                                         'validate': False
                                                                     }
                                                                 ])
validation_question_pente = MathadataValidateVariables({
    'm1': None,
    'm2': None
},
    tips=[
        {
            'seconds': 20,
            'tip': "Etapes de calcul: 1. Retrouve les coordonnées de deux points d'une droite. 2. Calcule les différences d'ordonnées et d'abscisses. 3. Calcule la pente avec la formule (différence d'ordonnées) / (différence d'abscisses)."
        },
        {
            'trials': 3,
            'tip': "Etapes de calcul pour le second graphique: 1. Les coordonnées de deux points de d2 : C(4; 11), D(10; 8). 2. différence d'ordonnées = 8-11 et différence d'abscisses = 10-4. 3. Calcule la pente avec la formule (différence d'ordonnées) / (différence d'abscisses)."
        },
        {
            'seconds': 30,
            'trials': 5,
            'print_solution': False,  # Print "Voici la solution: test : 3"
            'validate': False  # Unlock the next cells
        }
    ],
    function_validation=function_validation_pente, success="")
validation_question_association_droite = MathadataValidate(success="Bravo, toutes les associations sont correctes !")

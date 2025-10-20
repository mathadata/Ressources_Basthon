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


def tracer_20_points_droite():
    tracer_10_points_droite(dataset=common.challenge.dataset_20_points, labels=common.challenge.labels_20_points)


def tracer_10_points_droite(dataset=common.challenge.dataset_10_points, labels=common.challenge.labels_10_points):
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques,
                                                      d_train=dataset, r_train=labels)

    display_id = uuid.uuid4().hex
    display(HTML(f'''
        <canvas id="{display_id}-chart"></canvas>
    '''))

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
        f"setTimeout(() => window.mathadata.tracer_points('{display_id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_points_droite(id_content=None, display_value="range", carac=None, initial_hidden=False, save=True):
    if id_content is None:
        id_content = uuid.uuid4().hex

    display(HTML(f'''
        <div id="{id_content}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
            <div id="{id_content}-score-container" style="text-align: center; font-weight: bold; font-size: 2rem;">Pourcentage d'erreur : <span id="{id_content}-score">...</span></div>

            <canvas id="{id_content}-chart"></canvas>

            <div id="{id_content}-inputs" style="display: flex; gap: 1rem; justify-content: center; flex-direction: {'column' if display_value == "range" else 'row'};">
                <div>
                    <label for="{id_content}-input-m" id="{id_content}-label-m">m = </label>
                    <input type="{display_value}" {display_value == "range" and 'min="0" max="5"'} value="2" step="0.1" id="{id_content}-input-m">
                </div>
                <div>
                    <label for="{id_content}-input-p" id="{id_content}-label-p">p = </label>
                    <input type="{display_value}" {display_value == "range" and 'min="-10" max="10"'} value="0" step="0.1" id="{id_content}-input-p">
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
        'compute_score': True,
    }

    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id_content}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


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


def tracer_droite(ax, m, p, x_min, x_max, color='black', alt=False):
    # Ajouter la droite
    x = np.linspace(x_min, x_max, 1000)
    y = m * x + p
    ax.plot(x, y, c=color)  # Ajout de la droite en noir

    # Calculate a point along the line
    x_text = x_max - 2
    y_text = m * x_text + p - 4
    if y_text > x_max - 2:
        y_text = x_max - 2
        x_text = (y_text - p) / m + 4

    # Calculate the angle of the line
    angle = np.arctan(m) * 180 / np.pi

    # Display the equation of the line
    equation = f'$y = {m}x + {p}$'
    if alt:
        x_text = 20
        y_text = 15
    ax.text(x_text, y_text, equation, rotation=angle, color=color, verticalalignment='top', horizontalalignment='right')


pointA = (20, 40)
pointB = (30, 10)


def tracer_exercice_classification(display_m_coords=False, point_b=False):
    m = geo.input_values['m']
    p = geo.input_values['p']

    x = [pointA[0]]
    y = [pointA[1]]

    if point_b:
        x = [pointB[0]]
        y = [pointB[1]]

    y += [m * k1 + p for k1 in x]
    x += x

    fig, ax = create_graph(figsize=(figw_full * 0.50, figw_full * 0.50))

    # Définir les borne inf et sup des axes. On veut que le point (0,0) soit toujours sur le graphe
    x_min, x_max = min(0, np.min(x) - 2, np.min(y) - 2), max(0, np.max(x) + 2, np.max(y) + 2)
    x_max *= 1.2
    mk2 = m * pointA[0] + p
    if point_b:
        mk2 = m * pointB[0] + p

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((x_min, x_max))

    # Set the ticks on the x-axis at intervals of 5
    ax.set_xticks(np.arange(x_min, x_max, 5))

    # Set the ticks on the y-axis at intervals of 5
    # ax.set_yticks(np.arange(x_min, x_max, 5))
    ax.set_yticks([round(mk2, 2)])
    # remove the y axis ticks and labels
    ax.yaxis.set_ticks_position('none')
    ax.yaxis.set_ticklabels(['à calculer'])

    labels = [f'A({pointA[0]}, {pointA[1]})', f'M({pointA[0]}, {round(mk2, 2)})' if display_m_coords else 'M(20, ?)']
    if point_b:
        labels = [f'B({pointB[0]}, {pointB[1]})',
                  f'M({pointB[0]}, {round(mk2, 2)})' if display_m_coords else 'M(30, ?)']
    colors = ['C4', 'C3']
    for i in range(len(labels)):
        # Draw a dotted line from the point to the x-axis
        ax.axhline(y[i], xmin=0, xmax=x[i] / x_max, linestyle='dotted', color='gray')

        # Draw a dotted line from the point to the y-axis
        ax.axvline(x[i], ymin=0, ymax=y[i] / x_max, linestyle='dotted', color='gray')

        ax.annotate(labels[i], (x[i] + 1, y[i]), va='center', color=colors[i])
        ax.scatter(x[i], y[i], marker='+', c=colors[i])

    tracer_droite(ax, m, p, x_min, x_max, color=colors[1], alt=True)

    return ax


def exercice_calcul_au_dessus():
    tracer_exercice_classification()
    plt.show()
    plt.close()


def exercice_calcul_au_dessous():
    tracer_exercice_classification(point_b=True)
    plt.show()
    plt.close()


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

def calculer_score_droite():
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite)


def calculer_score_custom_droite():
    calculer_score_droite_geo(custom=True, validate=common.challenge.objectif_score_droite_custom,
                              error_msg="Continuez à chercher 2 zones pour avoir moins de " + str(
                                  common.challenge.objectif_score_droite_custom) + "% d'erreur. Pensez à changer les valeurs de m et p après avoir défini votre zone.")


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


validation_question_equation = MathadataValidateVariables({
    'ordonnee_M': None
}, function_validation=function_validation_equation)

validation_question_equation_B = MathadataValidateVariables({
    'ordonnee_M': None
}, function_validation=function_validation_equation_B)

import utilitaires_common as common
from utilitaires_common import *

# Pour accepter réponse élèves QCM
A = 'A'
B = 'B'
C = 'C'







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

common.challenge.deux_caracteristiques = common.challenge.deux_caracteristiques_moy_haut_moy_bas
def tracer_10_points_droite():
    data = common.challenge.dataset_10_points
    labels = common.challenge.labels_10_points

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques,
                                                      d_train=data, r_train=labels)

    id = uuid.uuid4().hex

    params = {
        'points': c_train_par_population,
        'droite': common.challenge.droite_10_points,
        'hover': True
    }
    params['droite']['avec_zones'] = True
    params['droite']['mode'] = 'cartesienne'

    run_js(
        f"mathadata.add_observer('{id}-chart', () => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'))")

    display(HTML(f'<canvas id="{id}-chart"></canvas>'))


def produit_scalaire_exercice():
    display(HTML(
        """<iframe scrolling="no" title="Produit scalaire et Classification" src="https://www.geogebra.org/material/iframe/id/gmtxwxat/width/3000/height/800/border/888888/sfsb/true/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/true/sdz/true/ctl/false" width="3000px" height="800px" style="border:0px;"> </iframe>"""))


def affichage_zones_custom_2(A1, B1, A2, B2, normal=False, trace=False):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    if trace:
        tracer_points_droite_vecteur(carac=common.challenge.deux_caracteristiques_custom, save=False, normal=normal,
                                     reglage_normal=normal, directeur=not normal, orthonormal=True)


def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_droite_vecteur(id_content=id, carac=common.challenge.deux_caracteristiques_custom,
                                 initial_hidden=True, save=False, normal=True, reglage_normal=True,orthonormal=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                mathadata.update_points('{id}', {{points}});

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                document.getElementById('{id}-container').style.visibility = 'visible';
            }})
        }}
    ''')


def calculer_score_droite(ensure_test=True):
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, ensure_test=ensure_test)


def calculer_score_droite_normal(ensure_test=True):
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite_vecteur_normal, banque=False,
                              animation=False, ensure_test=ensure_test)


def calculer_score_droite_normal_2custom(ensure_test=True):
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, banque=False,
                              ensure_test=ensure_test)


def calculer_score_custom_droite_2cara(ensure_test=True):
    calculer_score_droite_geo(custom=True, validate=11, banque=False, ensure_test=ensure_test,
                              error_msg="Continue à chercher 2 zones pour avoir moins de 11% d'erreur. "
                                        "N'oublie pas de mettre à jour les coordonnées du vecteur n et "
                                        "du point A après avoir défini ta sélection.")


def calculer_score_custom_droite(ensure_test=True):
    calculer_score_droite_geo(custom=True, validate=common.challenge.objectif_score_droite_custom, banque=False,
                              error_msg=f"Continue à chercher 2 zones pour avoir moins de "
                                        f"{common.challenge.objectif_score_droite_custom} % d'erreur. "
                                        f"N'oublie pas de mettre à jour les valeurs "
                                        f"de a, b et y après avoir défini ta sélection.",
                              ensure_test=ensure_test)


# Validation


def is_n_vers_le_haut(n):
    return (n[0] <= 0)


def function_validation_normal(errors, answers):
    n = answers['n']
    if not (isinstance(n, tuple) and len(n) == 2 and all(isinstance(x, (int, float)) for x in n)):
        errors.append(
            "Écris les coordonnées du vecteur entre parenthèses séparées par une virgule. "
            "Pour les valeurs non entières, utilise un point. Exemple : (3.5 , 5)")
        return False

    if n[0] * a + n[1] * b != 0:
        errors.append(
            "Ce n'est pas une réponse correcte. Un vecteur n orthogonal à un autre vecteur u (a, b) vérifie n.u=0. "
            "Une possibilité est le vecteur (-b,a).")
        return False

    return True


validation_execution_tracer_points_droite_vecteur = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_rappel = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_2 = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_3 = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, tu peux passer à la partie suivante.")
validation_score_droite_normal = MathadataValidate(success="Bien joué, tu peux passer à la partie suivante.")
validation_score_droite_normal_2custom = MathadataValidate(
    success="Bien joué, tu peux passer à la partie suivante.")
validation_question_normal = MathadataValidateVariables({
    'n': None
}, function_validation=function_validation_normal,
    success="C'est une bonne réponse. Le vecteur n est orthogonal au vecteur directeur.")

validation_execution_produit_scalaire_exercice = MathadataValidate(success="")

# Attention la réponse dépend du choix fait dans geogebra pour le vecteur normal
n_geogebra = (4, -8)
A_geogebra = (20, 30)
M1_geogebra = (40, 30)
M2_geogebra = (25, 35)
AM1_geogebra = (M1_geogebra[0] - A_geogebra[0], M1_geogebra[1] - A_geogebra[1])
AM2_geogebra = (M2_geogebra[0] - A_geogebra[0], M2_geogebra[1] - A_geogebra[1])


def function_validation_question_produit_scalaire(errors, answers):
    produit_scalaire = answers['produit_scalaire']
    if not isinstance(produit_scalaire, (int, float)):
        errors.append(
            "Le produit scalaire doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','")
        return False
    if produit_scalaire == Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire != function_calcul_produit_scalaire(n_geogebra, AM1_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False
    return True


validation_question_produit_scalaire = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire, tips=[
    {
        'trials': 1,
        'tip': 'Tu dois calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(40, 30)'
    },
    {
        'trials': 2,
        'seconds': 30,
        'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (40-20, 30-30))'
    },
    {
        'trials': 3,
        'seconds': 60,
        'tip': 'Pour calculer le produit scalaire, tu dois multiplier les coordonnées des deux vecteurs et additionner'
               ' les résultats. Nous avons n(4, -8) et AM(20, 0).'
    },
    {
        'trials': 4,
        'seconds': 120,
        'tip': 'Le produit scalaire est donné par 4*20 + (-8)*0'
    },
    {
        'trials': 5,
        'seconds': 120,
        'tip': 'Le produit scalaire est donné par 4*20 + (-8)*0 = 80'
    }
])

validation_execution_caracteristiques_custom = MathadataValidate(success="")
validation_execution_scatter_caracteristiques_ripou = MathadataValidate(success="")


def function_validation_question_produit_scalaire_2(errors, answers):
    produit_scalaire = answers['produit_scalaire']
    if not isinstance(produit_scalaire, (int, float)):
        errors.append(
            "Le produit scalaire doit être un nombre. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','")
        return False
    if produit_scalaire == Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire != function_calcul_produit_scalaire(n_geogebra, AM2_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False
    return True


validation_question_produit_scalaire_2 = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire_2, tips=[
    {
        'trials': 1,
        'tip': 'Tu dois calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
        'trials': 2,
        'seconds': 30,
        'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
        'trials': 3,
        'seconds': 60,
        'tip': 'Pour calculer le produit scalaire, tu dois multiplier les coordonnées des deux vecteurs et '
               'additionner les résultats. Nous avons n(4, -8) et AM(5, 5).'
    },
    {
        'trials': 4,
        'seconds': 120,
        'tip': 'Le produit scalaire est donné par 4*5 + (-8)*5'
    },
    {
        'trials': 5,
        'seconds': 120,
        'tip': 'Le produit scalaire est donné par  4*5 + (-8)*5 = -20'
    }
])


# ajout louis

def function_calcul_produit_scalaire(u, v):
    return u[0] * v[0] + u[1] * v[1]


def function_calcul_produit_vectoriel(u, v):
    return u[0] * v[1] - u[1] * v[0]


def function_sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


validation_question_produit_scalaire_louis = MathadataValidateVariables({
    'produit_scalaire': 10
}, tips=[
    {
        'trials': 1,
        'tip': 'Tu dois calculer le produit scalaire entre n(-2, 4) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
        'trials': 2,
        'seconds': 30,
        'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
        'trials': 3,
        'seconds': 60,
        'tip': 'Pour calculer le produit scalaire, tu dois multiplier les coordonnées des deux vecteurs et '
               'additionner les résultats. Nous avons n(-2, 4) et AM(5, 5).'
    },
    {
        'trials': 4,
        'seconds': 120,
        'tip': 'Le produit scalaire est donné par -2*5 + 4*5 = 10'
    }
])

xM = 40
yM = 20


def function_validation_question_decouverte_vecteur_normal(errors, answers):
    n = answers['n']
    if not check_coordinates(n, errors):
        return False

    a_expected = geo.input_values['a']
    b_expected = geo.input_values['b']

    # Vérifier si le vecteur est colinéaire (produit vectoriel = 0)
    if function_calcul_produit_vectoriel(n, (a_expected, b_expected)) != 0:
        # Cas 1 : L'ordre est inversé (b, a) au lieu de (a, b)
        if n[0] == b_expected and n[1] == a_expected:
            errors.append(
                "Tu y es presque ! C'est bien un vecteur normal, mais l'ordre des coordonnées n'est pas bon. "
                "Relis la question : on cherche n = (a, b).")
            return False

        # Cas 2 : Problème de signe mais bon ordre
        if abs(n[0]) == abs(a_expected) and abs(n[1]) == abs(b_expected):
            errors.append(
                "Tu y es presque mais il y a un problème de signe dans ta réponse. "
                "Relis la propriété juste au-dessus et lis l'équation de la droite sur le graphique.")
            return False

        # Cas 3 : L'ordre est inversé ET problème de signe
        if abs(n[0]) == abs(b_expected) and abs(n[1]) == abs(a_expected):
            errors.append(
                "Tu y es presque ! Il y a deux problèmes : l'ordre des coordonnées n'est pas bon et "
                "il y a une erreur de signe. On cherche n = (a, b).")
            return False

        # Cas 4 : Complètement faux
        errors.append(
            "Ce n'est pas une réponse correcte. Relis la propriété juste au-dessus et lis "
            "l'équation de la droite sur le graphique.")
        return False

    # Le vecteur est colinéaire mais pas exactement (a, b)
    if n[0] != a_expected or n[1] != b_expected:
        errors.append(
            "C'est bien un vecteur normal ! Mais nous cherchons le vecteur n = (a, b), "
            "avec a et b correspondant à l'équation de la droite.")
        return False

    return True


def _success_decouverte_vecteur_normal(answers):
    ux = geo.input_values['ux']
    uy = geo.input_values['uy']
    base = f"Bravo, le vecteur ({uy}, {-ux}) est bon ! C'est un vecteur normal possible. "
    suffix = f"Pour rappel le vecteur u est un vecteur directeur possible mais pas le seul. Le vecteur u que tu as choisi est ({ux}, {uy})."
    pretty_print_success(base + suffix)


validation_question_decouverte_vecteur_normal = MathadataValidateVariables({
    'n': None
}, function_validation=function_validation_question_decouverte_vecteur_normal,
    success="",  # on_success gère l'affichage complet
    on_success=_success_decouverte_vecteur_normal,
    tips=[{
        'trials': 1,
        'seconds': 30,
        'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. '
               'Lis les valeurs de a et b pour trouver un vecteur normal n.'
    },
        {
            'trials': 2,
            'seconds': 50,
            'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. '
                   'Un vecteur normal possible est n = (a, b).'
        }
    ])


def function_validation_produit_n_u(errors, answers):
    produit_scalaire_n_u = answers['produit_scalaire_n_u']
    if produit_scalaire_n_u is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if not isinstance(produit_scalaire_n_u, (int, float)):
        errors.append(
            "Le résultat doit être un nombre. Pour les nombres à virgule, utilise un point '.' et non une virgule ','")
        return False
    if produit_scalaire_n_u != 0:
        errors.append("La réponse est fausse. Relis la question.")
        return False
    return True


validation_question_vecteur_normal_non_unicite = MathadataValidateVariables({
    'n_1': None
}, function_validation=function_validation_produit_n_u,
    tips=[{
        'trials': 1,
        'seconds': 30,
        'tip': 'Tu peux utiliser le vecteur normal n que tu as trouvé précédemment et le modifier légèrement.'
    },
        {
            'trials': 2,
            'seconds': 50,
            'tip': 'Tu as déjà trouvé un vecteur normal. '
                   'Il donne la direction des vecteur orthogonaux au vecteur directeur u. '
        },
        {
            'trials': 3,
            'seconds': 50,
            'tip': 'Tu as déjà trouvé un vecteur normal n ( a , b ). '
                   'Un autre vecteur normal peut être ( k*a , k*b ) avec k un nombre non nul.'
        }
    ]
)

validation_question_produit_n_u = MathadataValidateVariables({
    'produit_scalaire_n_u': 0
}, function_validation=function_validation_produit_n_u,
    success="Bravo, c'est la bonne réponse. "
            "Le produit scalaire entre tout vecteur normal n et tout "
            "vecteur directeur u vaut bien 0 car ils sont orthogonaux ! ",
    tips=[
        {
            'trials': 1,
            'seconds': 3,
            'tip': 'Tu dois calculer le produit scalaire entre le vecteur n (que tu as donné à la cellule précédente) '
                   'et le vecteur u que tu as réglé dans le graphique.'
        },
        {
            'trials': 2,
            'seconds': 30,
            'tip': 'Tu dois calculer le produit scalaire entre le vecteur n '
                   '(celui que tu as donné à la cellule précédente) et le vecteur u que tu as réglé dans le graphique.'
                   ' Le produit scalaire est donné par n_x*u_x + n_y*u_y.'
        },
        {
            'seconds': 60,
            'trials': 5,
            'tip': 'Tu n\'as pas réussi cette question soit du fait d\'un bug de l\'activité soit du fait '
                   'd\'une erreur de ta part. Mais tu peux poursuivre l\'activité',
            'validate': True  # Unlock the next cells
        }
    ]
)


# Sens du vecteur normal : 
# Attention dépend du choix de caractéristiques !! 
def angle_vecteur(vecteur):
    angle_rad = np.arctan2(vecteur[1], vecteur[0])  # y, x
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360  # Pour que l'angle soit toujours entre 0 et 360°


# WIP
def function_validation_question_classe_direction_n(errors, answers):
    classe_sens_vecteur_normal = answers['classe_sens_vecteur_normal']
    classe_sens_oppose_vecteur_normal = answers['classe_sens_oppose_vecteur_normal']
    n = (geo.input_values['a'], geo.input_values['b'])
    if classe_sens_vecteur_normal is Ellipsis or classe_sens_oppose_vecteur_normal is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if angle_vecteur((12, 10)) <= angle_vecteur(n) < angle_vecteur((12, 10)) + 180:
        if classe_sens_vecteur_normal != common.challenge.r_grande_caracteristique:
            errors.append(
                "La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. "
                "Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != common.challenge.r_petite_caracteristique:
            errors.append(
                "La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. "
                "Relis la question et regarde le graphique.")
            return False
    else:
        if classe_sens_vecteur_normal != common.challenge.r_petite_caracteristique:
            errors.append(
                "La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != common.challenge.r_grande_caracteristique:
            errors.append(
                "La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
    return True


validation_question_classe_direction_n = MathadataValidateVariables({
    'classe_sens_vecteur_normal': None,
    'classe_sens_oppose_vecteur_normal': None
}, function_validation=function_validation_question_classe_direction_n,
    tips=[{
        'trials': 1,
        'seconds': 30,
        'tip': 'Ta réponse n\'est pas cohérente avec le graphique et le sens du vecteur normal n. '
               'Relis la question et regarde le graphique.'
    },
    ])

M_retourprobleme = common.challenge.M_retourprobleme

xM = M_retourprobleme[0]
yM = M_retourprobleme[1]


def function_validation_normal_2a(errors, answers):
    vec = answers['vecteur_AM']
    x_A = answers['x_A']
    y_A = answers['y_A']
    x_M = answers['x_M']
    y_M = answers['y_M']

    if check_coordinates(vec, errors) and (Ellipsis, Ellipsis, Ellipsis, Ellipsis) == (x_A, y_A, x_M, y_M):
        if vec != (xM - geo.input_values['xa'], yM - geo.input_values['ya']):
            errors.append("Les coordonnées du vecteur AM ne sont pas correctes. Reprends les calculs.")
            return False
        return True

    if not all(isinstance(coord, (int, float)) for coord in (x_A, y_A, x_M, y_M)):
        errors.append(
            "Les coordonnées des points doivent être des nombres. "
            "Pour les nombres à virgule, utilise un point '.' et non une virgule ','")
        return False
    if not check_coordinates(vec, errors):
        return False
    if (xM, yM) != (x_M, y_M):
        errors.append("Les coordonnées du point M ne sont pas correctes. Vérifie la position de M dans l'énoncé.")
        return False
    if (geo.input_values['xa'], geo.input_values['ya']) != (x_A, y_A):
        errors.append("Les coordonnées du point A ne sont pas correctes. Vérifie la position de A sur le graphique")
        return False
    if vec != (xM - geo.input_values['xa'], yM - geo.input_values['ya']):
        errors.append(
            "Ce n'est pas une réponse correcte. "
            "Retrouve la formule pour obtenir les coordonnées d'un vecteur à partir des coordonnées de deux points.")
        return False

    return len(errors) == 0


validation_question_normal_2a = MathadataValidateVariables({
    'vecteur_AM': None,
    'x_A': None,
    'y_A': None,
    'x_M': None,
    'y_M': None
}, function_validation=function_validation_normal_2a,
    tips=[{
        'trials': 2,
        'seconds': 60,
        'tip': 'Pour calculer un vecteur à partir des coordonnées de deux points, il faut soustraire les coordonnées '
               'du point de départ (A) de celles du point d\'arrivée (M). '
               'Par exemple, pour le vecteur AM, on fait (xM - xA, yM - yA).'
    }
    ])


def function_validation_normal_2b(errors, answers):
    valeur = answers['produit_scalaire_n_AM']
    vec = get_variable('vecteur_AM')

    if not isinstance(valeur, (int, float)):
        errors.append(
            "Le résultat doit être un nombre. Pour les nombres à virgule, utilise un point '.' et non une virgule ','")
        return False

    if valeur != function_calcul_produit_scalaire((geo.input_values['a'], geo.input_values['b']), vec):
        errors.append("La réponse n'est pas correcte. ")
        return False

    return True


validation_question_normal_2b = MathadataValidateVariables({
    'produit_scalaire_n_AM': None
}, function_validation=function_validation_normal_2b, tips=[{
    'trials': 1,
    'seconds': 60,
    'tip': 'Note sur un papier les coordonnées du vecteur n, '
           'les coordonnées du vecteur AM puis effectue le produit scalaire.'
}, {
    'trials': 2,
    'seconds': 60,
    'tip': 'Note sur un papier les coordonnées du vecteur n(a,b), '
           'les coordonnées du vecteur AM(e,f) puis effectue le produit scalaire ae+bf.'
}
])


def function_validation_normal_2c(errors, answers):
    produit_scalaire = get_variable('produit_scalaire_n_AM')
    reponse = answers['reponse']
    if reponse is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False

    if reponse not in (A, B, C):
        errors.append(" La réponse ne peut être que A, B ou C.")
        return False
    conditions = {
        A: produit_scalaire > 0,
        B: produit_scalaire < 0,
        C: produit_scalaire == 0
    }
    if conditions.get(reponse, False):
        return True
    errors.append("Ta réponse n'est pas correcte. Reprends l'énoncé")
    return False


validation_question_normal_2c = MathadataValidateVariables({
    'reponse': None,

}, function_validation=function_validation_normal_2c)


def function_validation_normal_2d(errors, answers):
    classe_de_M = answers['classe_de_M']

    # coeff de la droite de séparation
    c = -geo.input_values['a'] * geo.input_values['xa'] - geo.input_values['b'] * geo.input_values['ya']

    if geo.input_values['a'] * xM + geo.input_values['b'] * yM + c < 0:
        if geo.input_values['b'] < 0:
            vraie_classe_de_M = common.challenge.r_grande_caracteristique
        else:
            vraie_classe_de_M = common.challenge.r_petite_caracteristique
    else:
        if geo.input_values['b'] < 0:
            vraie_classe_de_M = common.challenge.r_petite_caracteristique
        else:
            vraie_classe_de_M = common.challenge.r_grande_caracteristique

    if classe_de_M is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if classe_de_M != vraie_classe_de_M:
        errors.append(
            "La réponse est incorecte. "
            "Relis au dessus comment déterminer la classe d'un point à l'aide d'un produit scalaire.")
        return False
    pretty_print_success(
        f"Bravo ! Tu es arrivé·e au bout du processus de Classification d'{data('un')} inconnu{e_fem()}.")
    return True


validation_question_normal_2d = MathadataValidateVariables({
    'classe_de_M': None
}, function_validation=function_validation_normal_2d, success="")

# Zone custom
A_2 = (7, 2)  # <- coordonnées du point A1
B_2 = (9, 25)  # <- coordonnées du point B1

A_1 = (14, 2)  # <- coordonnées du point A2
B_1 = (23, 10)  # <- coordonnées du point B2

validation_question_2cara_comprehension = MathadataValidateVariables({
    'reponse': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_bleus n'a pas la bonne valeur. Tu dois répondre par 2 ou 7."
            },
            {
                'value': 7,
                'if': "Il y a plus de pixels blanc (valeur élevée) dans la zone rouge pour le 2. "
                      "La moyenne du niveau de gris sera donc plus élevée pour le 2"
            },
        ]
    }
})

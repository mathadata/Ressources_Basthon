from IPython.display import display # Pour afficher des DataFrames avec display(df)
import pandas as pd
import os
import sys
# import mplcursors

# Pour accepter réponse élèves QCM
A = 'A'
B = 'B'
C = 'C'

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

if not sequence:
    from themes.geo.utilitaires import *
    import themes.geo.utilitaires as geo
else:
    from utilitaires_geo import *
    import utilitaires_geo as geo

if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))



def tracer_10_points_droite():
    data = common.challenge.dataset_10_points
    labels = common.challenge.labels_10_points

    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=common.challenge.deux_caracteristiques, d_train=data, r_train=labels)

    id = uuid.uuid4().hex
    display(HTML(f'<canvas id="{id}-chart"></canvas>'))
    
    params = {
        'points': c_train_par_population,
        'droite': common.challenge.droite_10_points,
        'hover': True
    }
    params['droite']['avec_zones'] = True
    params['droite']['mode'] = 'cartesienne'
    
    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")


def tracer_points_droite_vecteur(id=None, carac=None, initial_hidden=False, save=True, normal= None, directeur=False, reglage_normal=False):
    if id is None:
        id = uuid.uuid4().hex
    
    # Mise en place du conteneur pour le graphique
    display(HTML(f'''
        <!-- Conteneur pour afficher le taux d'erreur -->
        <div id="{id}-container" style="{'visibility:hidden;' if initial_hidden else ''}">
            <div id="{id}-score-container"
                style="
                text-align: center;
                font-weight: bold;
                font-size: 2rem;
                ">
                Erreur : <span id="{id}-score">...</span>
            </div>

            <!-- Zone canvas pour tracer le graphique -->
            <canvas id="{id}-chart"></canvas>

            <!-- Conteneur pour les champs d'entrée -->
            <div id="{id}-inputs"
                style="
                display: flex;
                gap: 2rem;
                justify-content: center;
                flex-direction: row;
                ">
                <!-- Cas « directeur » et pas en mode « reglage_normal » -->
                <div style="
                    display: {'flex' if (directeur and not reglage_normal) else 'none'};
                    flex-direction: row;
                    gap: 1.5rem;
                    ">
                    <!-- Paramètre ux -->
                    <div>
                        <label for="{id}-input-ux" id="{id}-label-ux">\u20D7u<sub>x</sub> = </label>
                        <input type="number"
                            id="{id}-input-ux"
                            value="5"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre uy -->
                    <div>
                        <label for="{id}-input-uy" id="{id}-label-uy">\u20D7u<sub>y</sub> = </label>
                        <input type="number"
                            id="{id}-input-uy"
                            value="10"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>

                <!-- Cas du mode « reglage_normal » -->
                <div style="
                    display: {'flex' if reglage_normal else 'none'};
                    flex-direction: row;
                    gap: 1.5rem;
                    ">
                    <!-- Paramètre a -->
                    <div>
                        <label for="{id}-input-nx" id="{id}-label-nx">\u20D7n<sub>x</sub> = </label>
                        <input type="number"
                            id="{id}-input-nx"
                            value="10"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre b -->
                    <div>
                        <label for="{id}-input-ny" id="{id}-label-ny">\u20D7n<sub>y</sub> = </label>
                        <input type="number"
                            id="{id}-input-ny"
                            value="-5"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>

                <!-- Paramètre x_A -->
                <div style="display: flex; flex-direction: row; gap: 1.5rem;">
                    <div>
                        <label for="{id}-input-xa" id="{id}-label-xa">x<sub>A</sub> = </label>
                        <input type="number"
                            id="{id}-input-xa"
                            value="50"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                    <!-- Paramètre y_A -->
                    <div>
                        <label for="{id}-input-ya" id="{id}-label-ya">y<sub>A</sub> = </label>
                        <input type="number"
                            id="{id}-input-ya"
                            value="50"
                            step="1"
                            style="width: 50px; height: 25px; font-size: 12px;">
                    </div>
                </div>
            </div>
        </div>
    '''))


    if normal is None:
        normal = False

    if carac is None:
        carac = common.challenge.deux_caracteristiques
    
    c_train_par_population = compute_c_train_by_class(fonction_caracteristique=carac)

    params = {
        'points': c_train_par_population,
        'custom': carac == common.challenge.deux_caracteristiques_custom,
        'hover': True,
        'displayValue': False,
        'save': save,
        'vecteurs': {
            'directeur': directeur,
            'normal': normal,
        },
        'droite': {
            'mode': 'cartesienne'
        },
        'inputs': {
            'xa': True,
            'ya': True,
        },
        'compute_score': True,
    }
    
    if reglage_normal:
        params['inputs']['nx'] = True
        params['inputs']['ny'] = True
    else:
        params['inputs']['ux'] = True
        params['inputs']['uy'] = True

    run_js(f"setTimeout(() => window.mathadata.tracer_points('{id}', '{json.dumps(params, cls=NpEncoder)}'), 500)")

def produit_scalaire_exercice():
    display(HTML("""<iframe scrolling="no" title="Produit scalaire et Classification" src="https://www.geogebra.org/material/iframe/id/gmtxwxat/width/3000/height/800/border/888888/sfsb/true/smb/false/stb/false/stbh/false/ai/false/asb/false/sri/false/rc/false/ld/true/sdz/true/ctl/false" width="3000px" height="800px" style="border:0px;"> </iframe>"""))

def affichage_zones_custom_2(A1, B1, A2, B2,normal=False,trace=False):
    common.challenge.affichage_2_cara(A1, B1, A2, B2, True)
    if trace:
        tracer_points_droite_vecteur(carac=common.challenge.deux_caracteristiques_custom, save=False, normal=normal, reglage_normal=normal,directeur= not normal)

def afficher_customisation():
    id = uuid.uuid4().hex
    display(HTML(f'''
        <div id="{id}"></div>
    '''))
    common.challenge.display_custom_selection_2d(id)

    tracer_points_droite_vecteur(id=id, carac=common.challenge.deux_caracteristiques_custom, initial_hidden=True, save=False, normal=True, reglage_normal=True)

    run_js(f'''
        window.mathadata.on_custom_update = () => {{
            window.mathadata.run_python('update_custom()', (points) => {{
                mathadata.update_points('{id}', {{points}});

                // AFFICHER LE GRAPH APRÈS LA SÉLECTION
                document.getElementById('{id}-container').style.visibility = 'visible';
            }})
        }}
    ''')

def calculer_score_droite():
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite)

def calculer_score_droite_normal():
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, banque=False)

def calculer_score_droite_normal_2custom():
    calculer_score_droite_geo(validate=common.challenge.objectif_score_droite, banque=False)

def calculer_score_custom_droite_2cara():
    calculer_score_droite_geo(custom=True, validate=11, banque=False, error_msg="Continuez à chercher 2 zones pour avoir moins de 11% d'erreur. N'oubliez pas de mettre à jour les valeurs de a, b et y après avoir défini votre zone.")

def calculer_score_custom_droite():
    calculer_score_droite_geo(custom=True, validate=6, banque=False, error_msg="Continuez à chercher 2 zones pour avoir moins de 6% d'erreur. N'oubliez pas de mettre à jour les valeurs de a, b et y après avoir défini votre zone.")
 
### Validation


def is_n_vers_le_haut(n):
    return (n[0]<=0)

def function_validation_normal(errors, answers):
    n = answers['n']
    if not (isinstance(n, tuple) and len(n) == 2 and all(isinstance(x, (int, float)) for x in n)):
        errors.append("Écrivez les coordonnées du vecteur entre parenthèses séparées par une virgule. Pour les valeurs non entières, utilisez un point. Exemple : (3.5 , 5)")
        return False

    if n[0]*a+n[1]*b!=0:
        errors.append("Ce n'est pas une réponse correcte. Un vecteur n orthogonal à un autre vecteur u (a, b) vérifie n.u=0. Une possibilité est le vecteur (-b,a).")
        return False

    return True 

validation_execution_tracer_points_droite_vecteur = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_2 = MathadataValidate(success="")
validation_execution_tracer_points_droite_vecteur_3 = MathadataValidate(success="")
validation_score_droite = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_score_droite_normal = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_score_droite_normal_2custom = MathadataValidate(success="Bien joué, vous pouvez passer à la partie suivante.")
validation_execution_point_droite = MathadataValidate(success="")
validation_question_normal = MathadataValidateVariables({
    'n': None
}, function_validation=function_validation_normal,success="C'est une bonne réponse. Le vecteur n est orthogonal au vecteur directeur.")

validation_execution_produit_scalaire_exercice = MathadataValidate(success="")


## Attention la réponse dépend du choix fait dans geogebra pour le vecteur normal
n_geogebra = (4, -8)
A_geogebra = (20, 30)
M1_geogebra = (40, 30)
M2_geogebra = (25, 35)
AM1_geogebra = (M1_geogebra[0] - A_geogebra[0], M1_geogebra[1] - A_geogebra[1])
AM2_geogebra = (M2_geogebra[0] - A_geogebra[0], M2_geogebra[1] - A_geogebra[1])


def function_validation_question_produit_scalaire(errors, answers):
    produit_scalaire = answers['produit_scalaire']
    if not isinstance(produit_scalaire, (int, float)):
        errors.append("Le produit scalaire doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if produit_scalaire==Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire!=function_calcul_produit_scalaire(n_geogebra,AM1_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False 
    return True

validation_question_produit_scalaire = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(40, 30)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (40-20, 30-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(20, 0).'
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
        errors.append("Le produit scalaire doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False
    if produit_scalaire==Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if produit_scalaire!=function_calcul_produit_scalaire(n_geogebra,AM2_geogebra):
        errors.append("Ce n'est pas la bonne valeur. Relis la définition du produit scalaire et vérifie tes calculs.")
        return False 
    return True

validation_question_produit_scalaire_2 = MathadataValidateVariables({
    'produit_scalaire': None
}, function_validation=function_validation_question_produit_scalaire_2, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4, -8) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(5, 5).'
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



validation_question_produit_scalaire_2 = MathadataValidateVariables({
    'produit_scalaire': -20
}, tips=[
    {
      'trials': 1,
      'tip': 'Vous devez calculer le produit scalaire entre n(4,-8) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(4, -8) et AM(5, 5).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par 4*5 + (-8)*5 = -20'
    }
  ])

# ajout louis

def function_calcul_produit_scalaire(u,v):
    return u[0]*v[0] + u[1]*v[1]

def function_calcul_produit_vectoriel(u,v):
    return u[0]*v[1] - u[1]*v[0]


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
      'tip': 'Vous devez calculer le produit scalaire entre n(-2, 4) et AM(?, ?) avec A(20, 30) et M(25, 35)'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Les coordonnées du vecteur AM sont (5, 5) (données par (25-20, 35-30))'
    },
    {
      'trials': 3,
      'seconds': 60,
      'tip': 'Pour calculer le produit scalaire, vous devez multiplier les coordonnées des deux vecteurs et additionner les résultats. Nous avons n(-2, 4) et AM(5, 5).'
    },
    {
      'trials': 4,
      'seconds': 120,
      'tip': 'Le produit scalaire est donné par -2*5 + 4*5 = 10'
    }
  ])

xM=40
yM=20

def function_validation_question_decouverte_vecteur_normal(errors, answers):
    n = answers['n']
    if not check_coordinates(n, errors):
        return False
    if function_calcul_produit_vectoriel(n,(geo.input_values['a'],geo.input_values['b'])) != 0:
        if abs(n[0])==abs( geo.input_values['a']) and abs(n[1])==abs(geo.input_values['b']):
            errors.append("Il y a une erreur de signe dans ta réponse. Relis la propriété juste au-dessus et lis l'équation de la droite sur le graphique.")
        else:
            errors.append("Ce n'est pas une réponse correcte. Relis la propriété juste au-dessus et lis l'équation de la droite sur le graphique.")
        return False
    if n[0] != geo.input_values['a'] or n[1] != geo.input_values['b']:
        errors.append("C'est bien un vecteur normal ! Mais nous cherchons le vecteur n = (a, b), avec a et b correspondant à l'équation de la droite." )
        return False
    return True
    
    

validation_question_decouverte_vecteur_normal= MathadataValidateVariables({
    'n': None,
    
}, function_validation=function_validation_question_decouverte_vecteur_normal,
tips= [{
      'trials': 1,
      'seconds': 30,
      'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. Lis les valeurs de a et b pour trouver un vecteur normal n.'
    },
    {
      'trials': 2,
      'seconds': 50,
      'tip': 'Sur le graphique l\'équation de la droite est donnée par ax + by + c = 0. Un vecteur normal possible est n = (a, b).'
    }
    ])

def function_validation_produit_n_u(errors, answers):
    produit_scalaire_n_u = answers['produit_scalaire_n_u']
    if produit_scalaire_n_u is Ellipsis:
         errors.append("Tu n'as pas remplacé les ...")
         return False
    if not isinstance(produit_scalaire_n_u, (int, float)):
            errors.append("Le résultat doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
            return False
    if produit_scalaire_n_u!=0:
        errors.append("La réponse est fausse. Relis la question.")
        return False
    return True

validation_question_produit_n_u = MathadataValidateVariables({
    'produit_scalaire_n_u': 0
}, function_validation=function_validation_produit_n_u, success="Bravo, c'est la bonne réponse. Le produit scalaire entre tout vecteur normal n et tout vecteur directeur u vaut bien 0 car ils sont orthogonaux ! ",
tips= [
    {
      'trials': 1,
      'seconds': 3,
      'tip': 'Tu dois calculer le produit scalaire entre le vecteur n (que tu as donné à la cellule précédente) et le vecteur u que tu as réglé dans le graphique.'
    },
    {
      'trials': 2,
      'seconds': 30,
      'tip': 'Tu dois calculer le produit scalaire entre le vecteur n (celui que tu as donné à la cellule précédente) et le vecteur u que tu as réglé dans le graphique. Le produit scalaire est donné par n_x*u_x< + n_y*u_y.'
    },
    {
      'seconds': 60,
      'trials': 5,
      'tip': 'Tu n\'as pas réussi cette question soit du fait d\'un bug de l\'activité soit du fait d\'une erreur de ta part. Mais tu peux poursuivre l\'activité',
      'validate': True # Unlock the next cells
    }
    ]
    )


# Sens du vecteur normal : 
# Attention dépend du choix de caractéristiques !! 
def angle_vecteur(vecteur):
    angle_rad = np.arctan2(vecteur[1], vecteur[0])  # y, x
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360  # Pour que l'angle soit toujours entre 0 et 360°

## WIP 
def function_validation_question_classe_direction_n(errors, answers):
    classe_sens_vecteur_normal = answers['classe_sens_vecteur_normal']
    classe_sens_oppose_vecteur_normal = answers['classe_sens_oppose_vecteur_normal']
    n=(geo.input_values['a'],geo.input_values['b'])
    if classe_sens_vecteur_normal is Ellipsis or classe_sens_oppose_vecteur_normal is Ellipsis:
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if angle_vecteur((12,10))<=angle_vecteur(n)<angle_vecteur((12,10))+180:
        if classe_sens_vecteur_normal != common.challenge.r_grande_caracteristique:
            errors.append("La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != common.challenge.r_petite_caracteristique:
            errors.append("La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
    else:
        if classe_sens_vecteur_normal != common.challenge.r_petite_caracteristique:
            errors.append("La réponse fournie pour \'classe_sens_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False
        if classe_sens_oppose_vecteur_normal != common.challenge.r_grande_caracteristique:
            errors.append("La réponse fournie pour \'classe_sens_oppose_vecteur normal\' est incorrecte. Relis la question et regarde le graphique.")
            return False    
    return True

validation_question_classe_direction_n = MathadataValidateVariables({
'classe_sens_vecteur_normal' : None,
'classe_sens_oppose_vecteur_normal' : None
}, function_validation=function_validation_question_classe_direction_n,
tips= [{
      'trials': 1,
      'seconds': 30,
      'tip': 'Ta réponse n\'est pas cohérente avec le graphique et le sens du vecteur normal n. Relis la question et regarde le graphique.'
    },
    ])


# point pour mnist
# M_retourprobleme=(40,20)

# point pour foetus   

M_retourprobleme = common.challenge.M_retourprobleme

xM = M_retourprobleme[0]
yM = M_retourprobleme[1]

def function_validation_normal_2a(errors, answers):
    vec = answers['vecteur_AM']
    x_A = answers['x_A']
    y_A = answers['y_A']
    x_M = answers['x_M']
    y_M = answers['y_M']
    
    if check_coordinates(vec, errors) and (Ellipsis,Ellipsis,Ellipsis,Ellipsis)==(x_A, y_A, x_M, y_M):
        if vec !=(xM-geo.input_values['xa'],yM-geo.input_values['ya']):
            errors.append("Les coordonnées du vecteur AM ne sont pas correctes. Reprends les calculs.")
            return False
        return True
      
    if not all(isinstance(coord, (int, float)) for coord in (x_A, y_A, x_M, y_M)):
        errors.append("Les coordonnées des points doivent être des nombres. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
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
        errors.append("Ce n'est pas une réponse correcte. Retrouve la formule pour obtenir les coordonnées d'un vecteur à partir des coordonnées de deux points.")
        return False

    return len(errors)==0     

validation_question_normal_2a = MathadataValidateVariables({
    'vecteur_AM': None,
    'x_A' : None,
    'y_A' : None,
    'x_M' : None,
    'y_M' : None
}, function_validation=function_validation_normal_2a,
tips= [{
      'trials': 2,
      'seconds': 60,
      'tip': 'Pour calculer un vecteur à partir des coordonnées de deux points, il faut soustraire les coordonnées du point de départ (A) de celles du point d\'arrivée (M). Par exemple, pour le vecteur AM, on fait (xM - xA, yM - yA).'
    }
    ])




def function_validation_normal_2b(errors, answers):
    valeur = answers['produit_scalaire_n_AM']
    vec = get_variable('vecteur_AM')
    
    if not isinstance(valeur, (int, float)):
        errors.append("Le résultat doit être un nombre. Pour les nombres à virgule, utilisez un point '.' et non une virgule ','")
        return False

    if valeur != function_calcul_produit_scalaire((geo.input_values['a'],geo.input_values['b']), vec):
        errors.append("La réponse n'est pas correcte. ")
        return False

    return True     

validation_question_normal_2b = MathadataValidateVariables({
    'produit_scalaire_n_AM': None
}, function_validation=function_validation_normal_2b,tips= [{
      'trials': 1,
      'seconds': 60,
      'tip': 'Note sur un papier les coordonnées du vecteur n, les coordonnées du vecteur AM puis effectue le produit scalaire.'
    },{
      'trials': 2,
      'seconds': 60,
      'tip': 'Note sur un papier les coordonnées du vecteur n(a,b), les coordonnées du vecteur AM(e,f) puis effectue le produit scalaire ae+bf.'
    }
    ])


def function_validation_normal_2c(errors, answers):
    produit_scalaire= get_variable('produit_scalaire_n_AM')
    reponse= answers['reponse']
    if reponse is Ellipsis :
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
    classe_de_M =answers['classe_de_M'] 

    # coeff de la droite de séparation
    c = -geo.input_values['a']*geo.input_values['xa'] - geo.input_values['b']*geo.input_values['ya']

    if geo.input_values['a']*xM + geo.input_values['b']*yM + c < 0:
        if geo.input_values['b'] < 0 :
            vraie_classe_de_M = common.challenge.r_grande_caracteristique
        else:
            vraie_classe_de_M = common.challenge.r_petite_caracteristique
    else:
        if geo.input_values['b'] < 0 :
            vraie_classe_de_M = common.challenge.r_petite_caracteristique
        else:
            vraie_classe_de_M = common.challenge.r_grande_caracteristique

    if classe_de_M is Ellipsis :
        errors.append("Tu n'as pas remplacé les ...")
        return False
    if classe_de_M!=vraie_classe_de_M:
        errors.append("La réponse est incorecte. Relis au dessus comment déterminer la classe d'un point à l'aide d'un produit scalaire.")
        return False
    print(f"Bravo ! Tu es arrivé·e au bout du processus de Classification d'{data('un')} inconnu{e_fem()}.")
    return True 

validation_question_normal_2d = MathadataValidateVariables({
  'classe_de_M': None
}, function_validation=function_validation_normal_2d,success="")




#Zone custom 
A_2 = (7, 2)       # <- coordonnées du point A1
B_2 = (9, 25)     # <- coordonnées du point B1


A_1 = (14, 2)     # <- coordonnées du point A2
B_1 = (23, 10)     # <- coordonnées du point B2

 
validation_question_2cara_comprehension = MathadataValidateVariables({
    'reponse': {
        'value': 2,
        'errors': [
            {
                'value': {
                    'in': [2, 7],
                },
                'else': "classe_points_bleus n'a pas la bonne valeur. Vous devez répondre par 2 ou 7."
            },
            {
                'value': 7,
                'if': "Il y a plus de pixels blanc (valeur élevée) dans la zone rouge pour le 2. La moyenne du niveau de gris sera donc plus élevée pour le 2"
                },
        ]
    }
})

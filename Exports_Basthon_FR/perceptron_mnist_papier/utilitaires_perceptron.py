import sys
import os
import __main__

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

try:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))
except FileNotFoundError: 
    pass

def affichage_poids(poids):
    if poids.shape != (10, 784):
        print_error("Impossible d'afficher les poids car vous avez rajouté des caractéristiques à vos images.")
        return
    
    fig, ax = plt.subplots(1, 10, figsize=(figw_full, 1))
    # Cachez les axes des subplots
    for j in range(10):
        ax[j].axis('off')
        ax[j].imshow(poids[j].reshape(28, 28))
    
    # Affichez les classes    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.2, wspace=0.05, hspace=0)
    plt.show()
    plt.close()

def _vote_neurone(c, poids, biais):
    return np.dot(c, poids) + biais

def _calcul_votes(c, tous_les_poids, tous_les_biais):
    return np.dot(c, tous_les_poids.T) + tous_les_biais

def _estimation(c, tous_les_poids, tous_les_biais):
    if (isinstance(c, list)):
        c = np.array(c)
    return np.argmax(_calcul_votes(c, tous_les_poids, tous_les_biais))

# version pour passer le validate : sans le print
def _apprentissage_perceptron_test(d_train, r_train, a = 1):
    if not has_variable("calculer_caracteristiques"):
        print_error("Vous devez définir la fonction `calculer_caracteristiques` pour pouvoir entraîner le réseau.")
        return
    
    calculer_caracteristiques = get_variable("calculer_caracteristiques")

    nb_caracteristiques = len(calculer_caracteristiques(d_train[0]))
    W = np.zeros((10, nb_caracteristiques))
    B = np.zeros(10)
    
    for i in range(len(d_train)):
        d = d_train[i]
        c = calculer_caracteristiques(d)
        r = r_train[i] # La vraie réponse

        r_est = _estimation(c, W, B)
        if r_est != r:
            W[r] = W[r] + a * c
            B[r] = B[r] + a
            
            W[r_est] = W[r_est] - a * c
            B[r_est] = B[r_est] - a

    return W, B

# version finale : avec le print
def _apprentissage_perceptron(d_train, r_train, a = 1):
    if not has_variable("calculer_caracteristiques"):
        print_error("Vous devez définir la fonction `calculer_caracteristiques` pour pouvoir entraîner le réseau.")
        return
    
    calculer_caracteristiques = get_variable("calculer_caracteristiques")

    nb_caracteristiques = len(calculer_caracteristiques(d_train[0]))
    W = np.zeros((10, nb_caracteristiques))
    B = np.zeros(10)

    print('Apprentissage des poids en cours, patience...')
    
    for i in range(len(d_train)):
        d = d_train[i]
        c = calculer_caracteristiques(d)
        r = r_train[i] # La vraie réponse

        r_est = _estimation(c, W, B)
        if r_est != r:
            W[r] = W[r] + a * c
            B[r] = B[r] + a
            
            W[r_est] = W[r_est] - a * c
            B[r_est] = B[r_est] - a

    return W, B

def initialiser_poids(nombre_poids, nombre_classes):
    poids = np.zeros((nombre_classes, nombre_poids))
    biais = np.zeros(nombre_classes)
    return poids, biais

def apprentissage_perceptron(d_train, r_train, a = 1):
    calculer_caracteristiques = get_variable("calculer_caracteristiques")
    estimation = get_variable("estimation")

    nb_caracteristiques = len(calculer_caracteristiques(d_train[0]))
    W = np.zeros((10, nb_caracteristiques))
    B = np.zeros(10)
    
    for i in range(len(d_train)):
        d = d_train[i]
        c = calculer_caracteristiques(d)
        r = r_train[i] # La vraie réponse
    
        r_est = estimation(c, W, B)

        if r_est != r:
            # Poids du neurone pour le bon chiffre
            w_r = W[r]
            
            # Poids du neurone pour le mauvais chiffre estimé
            w_r_est = W[r_est]
            
            for k in range(len(c)): # Boucle sur les caractéristiques pour faire les modifications de poids
                w_r[k] = w_r[k] + a * c[k]
                w_r_est[k] = w_r_est[k] - a * c[k]
                
            B[r] = B[r] + a
            B[r_est] = B[r_est] - a

    return W, B
    
def calculer_score_reseau(tous_les_poids, tous_les_biais):
    if not has_variable("calculer_caracteristiques"):
        print_error("Vous devez définir la fonction `calculer_caracteristiques` pour pouvoir calculer le score.")
        return
    
    calculer_caracteristiques = get_variable("calculer_caracteristiques")

    def algorithme(d):
        return _estimation(calculer_caracteristiques(d), tous_les_poids, tous_les_biais)

    calculer_score(algorithme) 

def soumettre(tous_les_poids, tous_les_biais):

    if not has_variable("token"):
        print_error("Vous devez entrer un token tout en haut du notebook et exécuter la cellule pour pouvoir soumettre votre réponse.")
        return

    if not has_variable("calculer_caracteristiques"):
        print_error("Vous devez définir la fonction `calculer_caracteristiques` pour pouvoir soumettre votre réponse.")
        return
    
    calculer_caracteristiques = get_variable("calculer_caracteristiques")
    
    c_test = np.array([calculer_caracteristiques(d) for d in common.challenge.d_test])
    preds = np.dot(c_test, tous_les_poids.T) + tous_les_biais
    preds = np.argmax(preds, axis=1)

    body = {
        'predictions': preds.tolist()
    }
    
    token = get_variable("token")
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    
    def cb(response):
        if response is None:
            print_error("Il y a eu une erreur lors de la soumission. Vérifie ta réponse et réessaie plus tard.")
            return
        score = response.get('score', None)
        if score is None:
            print_error("Il y a eu une erreur lors de la soumission. Vérifie ta réponse et réessaie plus tard.")
            return

        if common.session_score is None or score < common.session_score:
            common.session_score = score
            if common.highscore is None or common.session_score < common.highscore:
                common.highscore = common.session_score
            update_score()
            
        display(HTML(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); text-align: center; width: 300px; display: flex; flex-direction: column; align-items: center; width: 100%">
                <h2 style="font-size: 18px; color: #333; margin-bottom: 15px;">Résultats</h2>

                <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                    <p style="color: #777; margin-top: 1rem;">Erreur</p>
                    <p style="font-size: 24px; font-weight: bold; margin-top: 1rem;">{score * 100}%</p>
                </div>

                <div style="display: flex; justify-content: space-around; align-items: center; width: 100%; margin-top: 15px; margin-bottom: 25px;">
                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                        <p style="color: #777; margin: 1rem 0;">Classement {response.get('category')}</p>
                        <p style="font-size: 24px; font-weight: bold;">#{response.get('categoryRank')}</p>
                    </div>

                    <div style="flex: 1; display: flex; align-items: center; justify-content: center;">
                        <svg width="80px" height="80px" viewBox="0 0 120 120" id="Layer_1" version="1.1" xml:space="preserve" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
                            <style type="text/css">
                                .st0{{fill:#FFC54D;}}
                                .st1{{fill:#7A9FFF;}}
                                .st2{{fill:#EDB248;}}
                            </style>
                            <g>
                                <g>
                                    <path class="st0" d="M74.5,101.6h-5.3v-2c0-0.8-0.6-1.4-1.4-1.4H65l-1-7.3h2.2c0.8,0,1.4-0.6,1.4-1.4s-0.6-1.4-1.4-1.4h-2.6    l-0.3-2h-6.9l-0.3,2h-2.6c-0.8,0-1.4,0.6-1.4,1.4s0.6,1.4,1.4,1.4h2.2l-1,7.3h-2.8c-0.8,0-1.4,0.6-1.4,1.4v2h-5.3    c-2,0-3.6,1.6-3.6,3.6v4.5H78v-4.5C78.1,103.2,76.5,101.6,74.5,101.6z"/>
                                    <path class="st0" d="M75.7,30.1v6h25V51c-1.4,2.6-8.5,14-29,11.5l-0.7,6c2.3,0.3,4.5,0.4,6.6,0.4c22,0,28.7-15.3,29-16l0.2-0.6    V30.1H75.7z"/>
                                    <path class="st0" d="M48.9,68.4l-0.7-6c-20.5,2.6-27.7-8.9-29-11.5V36.1h25v-6h-31v22.2l0.2,0.6c0.3,0.7,7,16,29,16    C44.4,68.8,46.6,68.7,48.9,68.4z"/>
                                </g>
                                <g>
                                    <polygon class="st1" points="81.3,30.1 81.3,93.1 88.4,87.1 95.5,93.1 95.5,30.1   "/>
                                    <polygon class="st1" points="24.4,93.1 31.5,87.1 38.6,93.1 38.6,30.1 24.4,30.1   "/>
                                </g>
                                <path class="st0" d="M60,89.3L60,89.3c-16.4,0-29.7-13.3-29.7-29.7v-40h59.3v40C89.6,76,76.3,89.3,60,89.3z"/>
                                <path class="st2" d="M60.6,36.5l3.8,7.6c0.1,0.2,0.3,0.4,0.6,0.4l8.4,1.2c0.6,0.1,0.8,0.8,0.4,1.3l-6.1,5.9   c-0.2,0.2-0.3,0.4-0.2,0.7l1.4,8.4c0.1,0.6-0.5,1.1-1.1,0.8l-7.5-4c-0.2-0.1-0.5-0.1-0.7,0l-7.5,4c-0.5,0.3-1.2-0.2-1.1-0.8   l1.4-8.4c0-0.2,0-0.5-0.2-0.7L46.1,47c-0.4-0.4-0.2-1.2,0.4-1.3l8.4-1.2c0.2,0,0.4-0.2,0.6-0.4l3.8-7.6   C59.6,35.9,60.3,35.9,60.6,36.5z"/>
                            </g>
                        </svg>
                    </div>

                    <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                        <p style="color: #777; margin: 1rem 0;">Classement Global</p>
                        <p style="font-size: 24px; font-weight: bold;">#{response.get('rank')}</p>
                    </div>
                </div>

                <div>
                    <a href="https://mathadata.fr/fr/challenge/classement?token={token}" target="_blank" style="font-size: 14px; color: #007BFF; text-decoration: none;">Lien vers le classement complet</a>
                </div>
            </div>       
        """))

        validation_soumission()
    
    http_request(mathadata_endpoint + "/contest/submit", "POST", headers=headers, body=body, cb=cb)

def calculer_caracteristiques_contours(d):
    d_flat = d.flatten()
    c = np.concatenate((d_flat, np.absolute(d_flat[1:] - d_flat[:-1])))
    return c

# fonction vote d'un neurone, si l'élève ne la définit pas
def vote_neurone(c, w, b):
    v = 0 # On initialise à 0
    for i in range(len(c)): # Pour chaque pixel
        # TODO : Ajouter les votes du pixel i
        v = v + c[i] * w[i]
    
    # TODO : ajouter le biais
    v = v + b
    
    return v

### ----- CELLULES VALIDATION ----

def validation_token():
    token = get_variable("token")

    headers = {
        'Authorization': f'Bearer {token}',
    }
    
    def cb(res):
        if 'pseudo' not in res:
            print_error("Le token est invalide. Pour recevoir à nouveau votre token, rendez vous sur https://mathadata.fr/fr/challenge/renvoi_mail")
        else:
            print("Votre token est valide. Vous êtes inscrit sous le pseudo " + res['pseudo'])
            if 'highScore' in res:
                common.highscore = res['highScore']
                update_score()
            validation_breakpoint_token()

    http_request(mathadata_endpoint + "/contest/user", "GET", headers=headers, cb=cb) 

validation_breakpoint_token = MathadataValidate(success="")

validation_execution_calculer_caracteristiques = MathadataValidate(success="")

validation_execution_apprentissage = MathadataValidate(success="")

validation_question_vote_neurone = MathadataValidateVariables({
    'v': {
        'value': 320,
        'errors': [{
            'value': 300,
            'if': 'As tu bien pensé à ajouter le biais à la fin du calcul ?'
        }, {
            'value': 320,
            'else': 'Pour calculer v, multiplie les caractéristiques par les poids et ajoute le biais.'
        }]
    }
},
success="En effet, le vote du neurone est 0 * 5 + 200 * 1 + 100 * 0 + 50 * 2 + 20 = 320")

validation_estimation_reseau = MathadataValidateVariables({
    'r_est': {
        'value': 4,
        'errors': [{
            'value': 9,
            'if': 'Attention, indique bien l\'estimation du réseau et non le chiffre sur l\'image'
        },
        {
            'value': {
                'min': 0,
                'max': 9
            },
            'else': 'L\'estimation du réseau est toujours un chiffre entre 0 et 9.'
        },
        {
            'value': 4,
            'else': 'Pour déterminer l\'estimation, regarde le chiffre qui a le plus de votes.'
        }]
    }
},
success="En effet, le réseau de neurones a estimé que le chiffre sur l'image est 4 en lui donnant le plus de votes.")

validation_vote_neurone = MathadataValidateFunction(
    "vote_neurone",
    test_set=[(np.random.randint(0, 255, size=(784)), np.random.randint(0, 255, size=(784)), np.random.randint(0, 255),) for _ in range(10)],
    expected_function=_vote_neurone,
    #on_success=lambda _: setattr(__main__, 'vote_neurone', _vote_neurone),
)

validation_calcul_votes = MathadataValidateFunction(
    "calcul_votes",
    test_set=[(np.random.randint(0, 255, size=(784)), np.random.randint(0, 255, size=(10, 784)), np.random.randint(0, 255, size=(10)),) for _ in range(10)],
    expected_function=_calcul_votes,
    #on_success=lambda _: setattr(__main__, 'calcul_votes', _calcul_votes),
)

validation_estimation = MathadataValidateFunction(
    "estimation",
    test_set=[(np.random.randint(0, 255, size=(784)), np.random.randint(0, 255, size=(10, 784)), np.random.randint(0, 255, size=(10)),) for _ in range(10)],
    expected_function=_estimation,
    on_success=lambda _: setattr(__main__, 'estimation', _estimation),
)

validation_apprentissage_perceptron = MathadataValidateFunction(
    'apprentissage_perceptron',
    test_set=lambda: [(common.challenge.d_train[100*i:100 + 100*i], common.challenge.r_train[100*i:100+100*i]) for i in range(10)],
    expected_function=_apprentissage_perceptron_test,
    on_success=lambda _: setattr(__main__, 'apprentissage_perceptron', _apprentissage_perceptron),
)

validation_soumission = MathadataValidate(success="")

validation_calculer_caracteristiques_contours = MathadataValidateFunction(
    "calculer_caracteristiques",
    test_set=lambda: common.challenge.d_train[:10],
    expected_function=calculer_caracteristiques_contours,
    # on_success=lambda _: setattr(__main__, 'calculer_caracteristiques', calculer_caracteristiques_contours),
)

def function_validation_calculer_caracteristiques(errors, answers):
    if not has_variable("calculer_caracteristiques"):
        errors.append("La fonction `calculer_caracteristiques` n'est pas définie.")
        return False
    
    calculer_caracteristiques = get_variable("calculer_caracteristiques")
    for i in range(10):
        c = calculer_caracteristiques(common.challenge.d_train[i])
        if not isinstance(c, list):
            errors.append("La fonction `calculer_caracteristiques` doit retourner une liste.")
            return False
        
        for j in range(len(c)):
            if not isinstance(c[j], (int, float)) and not np.issubdtype(type(c[j]), np.number):
                errors.append("La liste retournée par `calculer_caracteristiques` doit contenir uniquement des nombres. Votre liste contient un élément de type " + str(type(c[j])) + " : " + str(c[j]))
                return False
    
    return True

validation_calculer_caracteristiques_custom = MathadataValidate(function_validation=function_validation_calculer_caracteristiques, success="Votre fonction renvoit bien une liste de nombres. Vous pouvez tester vos caractéristiques en lançant l'apprentissage dans la cellule suivante.")
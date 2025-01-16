import sys
import os

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

t = 30 

def classification_expected(x, t):
    return common.challenge.r_low_feature if x <= t else common.challenge.r_high_feature

### ----- CELLULES VALIDATION ----

def on_caracteristique_success(answers):
    display_bank(carac=common.challenge.feature)

validate_fonction_caracteristique = MathadataValidateFunction(
    'feature',
    test_set=lambda: common.challenge.d_train[0:100].tolist(),
    expected=lambda: compute_c_train(common.challenge.feature, common.challenge.d_train[0:100]),
    on_success=on_caracteristique_success
)

def classification_test_set():
    # Compute characteristics for the first 100 images
    c_train = compute_c_train(common.challenge.feature, common.challenge.d_train[0:100])
    # Create a list of repeated values (10 times 30, 10 times 31, etc.)
    repeated_values = [value for value in range(30, 40) for _ in range(10)]
    
    # Combine the two lists into a list of arguments
    args_list = [(c_train[i], repeated_values[i]) for i in range(100)]
    return args_list

validate_fonction_classification = MathadataValidateFunction(
    'classification',
    test_set=classification_test_set,
    expected=lambda: [classification_expected(x, t) for x, t in classification_test_set()]
)

def on_algorithme_success(answers):
    display_bank(carac=common.challenge.feature, showPredictions=True)

validate_fonction_algorithme = MathadataValidateFunction(
    'algorithm',
    test_set=lambda: common.challenge.d_train[0:100].tolist(),
    expected=lambda: [classification_expected(common.challenge.feature(d), get_variable('t')) for d in common.challenge.d_train[0:100]],
    on_success=on_algorithme_success
)

def on_calcul_score_success(answers):
    if not has_variable('algorithm'):
        print_error("The `algorithme` function has not been defined in order to compute the percentage error.")
        
    calculer_score(algorithme=get_variable('algorithm'), feature=common.challenge.feature)

validate_fonction_calcul_score = MathadataValidateFunction(
    'compute_error',
    test_set=lambda: [(common.challenge.d_train[0:100], common.challenge.r_train[0:100])],
    expected=lambda: [erreur_train(common.challenge.d_train[0:100], common.challenge.r_train[0:100], get_variable('t'), classification_expected, common.challenge.feature) * 100],
    on_success=on_calcul_score_success
)

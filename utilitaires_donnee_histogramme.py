import sys
import os

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

notebook_id = 5

common.start_analytics_session(notebook_id)

if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))


def afficher_histogramme(div_id, seuil=None, caracteristique=None):
    if not caracteristique:
        if not check_is_defined('caracteristique'):
            print_error("La fonction caracteristique n'est pas définie.")
            return
        caracteristique = get_variable('caracteristique')
    c_train = compute_c_train(caracteristique, common.challenge.d_train)
    data = {}
    for i in range(len(c_train)):
        c = c_train[i]
        k = int(c)
        if k not in data:
            data[k] = [0,0]
        
        if common.challenge.r_train[i] == common.challenge.classes[0]:
            data[k][0] += 1
        else:
            data[k][1] += 1

    js = f"window.mathadata.displayHisto('{json.dumps(data)}', '{div_id}'"
    if seuil is not None:
        js += f", {seuil}"
    js += ")"
    run_js(js)


def calculer_score_zone_custom(): 
    global e_train
    x = get_variable('x')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    caracteristique = caracteristique_zone

    if not check_is_defined(x) or not check_is_defined(r_petite_caracteristique) or not check_is_defined(r_grande_caracteristique):
        return

    def algorithme(d):
        k = caracteristique(d)
        if k < x:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
    
    def cb(score):
        validation_score_zone_custom()

    calculer_score(algorithme, method="moyenne custom", parameters=f"x={x}", cb=cb) 

def calculer_score_hist_seuil():
    x = get_variable('x')
    r_petite_caracteristique = get_variable('r_petite_caracteristique')
    r_grande_caracteristique = get_variable('r_grande_caracteristique')
    caracteristique = get_variable('caracteristique')

    if not check_is_defined(x) or not check_is_defined(r_petite_caracteristique) or not check_is_defined(r_grande_caracteristique):
        return
    
    if x > 35 or x < 33:
        print_error("Trouve un seuil x qui donne un score inférieur à 31% pour continuer.")
        return
    
    def algorithme(d):
        k = caracteristique(d)
        if k < x:
            return r_petite_caracteristique
        else:
            return r_grande_caracteristique
    
    def cb(score):
        validation_question_hist_seuil()

    calculer_score(algorithme, method="moyenne ref hist", parameters=f"x={x}", cb=cb) 

def calculer_score_carac_2_seuil_optimise():
    if not validation_question_carac_2_seuil_optimise():
        return

    t = get_variable('t')
    caracteristique = common.challenge.caracteristique2
    classification = get_variable('classification')

    def algorithme(d):
        x = caracteristique(d)
        return classification(x, t)
    
    calculer_score(algorithme, method="carac 2 seuil optim", parameters=f"t={t}") 

def calculer_score_carac_custom_seuil_optimise():
    if not validation_question_carac_custom_seuil_optimise():
        return

    t = get_variable('t')
    caracteristique = common.challenge.caracteristique_custom
    classification = get_variable('classification')

    def algorithme(d):
        x = caracteristique(d)
        return classification(x, t)
    
    calculer_score(algorithme, method="carac custom seuil optim", parameters=f"t={t}") 

run_js('''
window.mathadata.displayHisto = (data, div_id, seuil) => {
    if (typeof data === 'string')
        data = JSON.parse(data)
    const config = {
        type: 'bar',
        data: {
            labels: Object.keys(data),
            datasets: [
                {
                    label: "Nombre d'images de 2",
                    data: Object.values(data).map(v => v[0]),
                    backgroundColor: 'blue',
                    borderColor: 'blue',
                    borderWidth: 1
                },
                {
                    label: "Nombre d'images de 7",
                    data: Object.values(data).map(v => v[1]),
                    backgroundColor: 'orange',
                    borderColor: 'orange',
                    borderWidth: 1
                },
            ]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        },
    };
    
    if (seuil !== undefined) {
        // Ensure the label for the vertical line is correctly formatted.
        config.data.datasets.push({
            type: 'scatter',
            data: [{x: seuil, y: 0}, {x: seuil, y: 100}],
            showLine: true,
            label: `x = ${seuil}`,
        });
    }
    window.mathadata.create_chart(div_id, config)
}
''')

### ----- CELLULES VALIDATION ----

validation_question_carac_2_seuil_optimise = MathadataValidateVariables(
    get_names_and_values=get_validate_seuil_optimized,
    success="Bravo, c'est la bonne réponse ! Ton seuil est maintenant optimal"
)
validation_question_carac_custom_seuil_optimise = MathadataValidateVariables(
    get_names_and_values=get_validate_seuil_optimized,
    success="Bravo, c'est la bonne réponse ! Ton seuil est maintenant optimal"
)
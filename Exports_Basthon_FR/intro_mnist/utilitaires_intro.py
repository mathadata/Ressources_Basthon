import sys
import os
from matplotlib.ticker import AutoMinorLocator

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utilitaires_common import *
import utilitaires_common as common

if not sequence:
    # For dev environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("strings", os.path.join(current_dir, "strings.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))

def afficher_customisation():
    display(HTML('''
        <div id="custom"></div>
        <canvas id="graph_custom"></canvas>
    '''))
    common.challenge.display_custom_selection('custom')

def update_custom():
    return json.dumps(get_erreur_plot(common.challenge.caracteristique_custom), cls=NpEncoder)


# JS

run_js('''
window.mathadata.on_custom_update = () => {
    window.mathadata.run_python('update_custom()', (res) => {
        window.mathadata.tracer_erreur('graph_custom', res[0], res[1])
    })
}    
''')

# VALIDATION

validation_execution_afficher_customisation = MathadataValidate(success="")
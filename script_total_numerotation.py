import subprocess
import os

files =[
"Exports_Basthon_FR/P_GEO_droite_produit_scalaire_mnist/export_P_GEO_droite_produit_scalaire_mnist.ipynb",

"Exports_Basthon_FR/S_GEO_equation_de_droite_mnist/export_S_GEO_equation_de_droite_mnist.ipynb",

"Exports_Basthon_FR/S_GEO_milieu_distance_mnist/export_S_GEO_milieu_distance_mnist.ipynb",

"Exports_Basthon_FR/S_STATS_moyenne_histogramme_mnist/export_S_STATS_moyenne_histogramme_mnist.ipynb",

"Exports_Basthon_FR/S_STATS_moyenne_histogramme_foetus/export_S_STATS_moyenne_histogramme_foetus.ipynb",



]

script_path = "/Users/loca/Documents/Notebook/Ressources_Basthon/script_numerotation.py"

for file_path in files:
    abs_file_path = os.path.abspath(file_path)
    subprocess.run(["python3", script_path, abs_file_path])



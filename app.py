import pandas as pd
import tkinter as tk
from tkinter import messagebox
import joblib

# Charger ton pipeline (modèle + preprocessing)
model = joblib.load("model_fraude_bancaire.pkl")  # <-- Mets ici le bon chemin

# Fonction de prédiction
def predict():
    try:
        # Récupérer les entrées utilisateur
        type_input = type_var.get()
        amount = float(entry_amount.get())
        oldbalanceOrg = float(entry_oldbalanceOrg.get())
        newbalanceOrig = float(entry_newbalanceOrig.get())
        oldbalanceDest = float(entry_oldbalanceDest.get())
        newbalanceDest = float(entry_newbalanceDest.get())

        # Créer l'input sous forme de tableau
        columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        features = pd.DataFrame([[type_input, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]], columns=columns)

        # Prédiction avec le pipeline
        prediction = model.predict(features)

        # Afficher le résultat
        if prediction[0] == 1:
            messagebox.showwarning("Résultat", "🚨 Fraude détectée !")
        else:
            messagebox.showinfo("Résultat", "✅ Pas de fraude détectée.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur : {str(e)}")

# Créer la fenêtre principale
root = tk.Tk()
root.title("Détection de Fraude Bancaire")
root.geometry("2000x1000")  # largeur x hauteur en pixels


# Widgets
tk.Label(root, text="Type de transaction").pack()
type_var = tk.StringVar()
entry_type = tk.Entry(root, textvariable=type_var)
entry_type.pack()

tk.Label(root, text="Montant").pack()
entry_amount = tk.Entry(root)
entry_amount.pack()

tk.Label(root, text="Ancien solde origine").pack()
entry_oldbalanceOrg = tk.Entry(root)
entry_oldbalanceOrg.pack()

tk.Label(root, text="Nouveau solde origine").pack()
entry_newbalanceOrig = tk.Entry(root)
entry_newbalanceOrig.pack()

tk.Label(root, text="Ancien solde destination").pack()
entry_oldbalanceDest = tk.Entry(root)
entry_oldbalanceDest.pack()

tk.Label(root, text="Nouveau solde destination").pack()
entry_newbalanceDest = tk.Entry(root)
entry_newbalanceDest.pack()

# Bouton de prédiction
tk.Button(root, text="Prédire", command=predict).pack(pady=10)

# Boucle principale
root.mainloop()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.offsetbox import AnchoredText
from __future__ import print_function
from rdkit import Chem
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from rdkit.Chem import rdMolDescriptors as Descriptor
from sklearn.neural_network import MLPRegressor


#Reading groups of molecules from an SDF file
suppl = Chem.SDMolSupplier('ISSCAN_v1a_774_10Dec04.sdf')

#Keeps track of the molecules (key of the dictionary) and their target and feature_array
molecule_dict = {}

#Reading molecules from SDF file
for mol in suppl:
    if mol is None: 
        continue
    #Accessing the chemical name of the molecule
    chemical_name = mol.GetProp('ChemName').rstrip()
    #Accessing the TD50 mouse of the molecule
    TD50 = mol.GetProp('TD50_Mouse')
    #If TD50 value exists for the molecule, adds the molecule and it's TD50 value to the dictionary, 
    if TD50 != 'ND' and TD50 != 'NP':
        molecule_dict[chemical_name] = {'target':np.log10(float(TD50)), 'feature_array':()


# define the mass of a 'H' atom
H_mass = 1.008
# all possible atoms in the dataset
atom_symbols = ["Br","C","Cl","F","H","I","N","O","P","S"]

 
def get_feature_array(mol_obj):
    """
    Input: RDKit molecule object
    Returns: Ordered tuple containing the counts of each atom symbol and mass for the current molecule
    Ordered tuple format:
    ("Br","C","Cl","F","H","I","N","O","P","S", mass_molecule)
    
    Example:
    For '1,2-Dichloropropane', your function should return the following:
    (0, 3, 2, 0, 6, 0, 0, 0, 0, 0, 112.98700000000001)
    """
    
    #Keep track of the orderes tuple containing the features of the molecule
    feature_list = []
    
    #Keep track of all the atoms in the molecule
    atoms = []
    
    #Accesses and saves all the atoms of the molecule in the 'atoms' list
    for i in range(mol_obj.GetNumAtoms()):
        atom = mol_obj.GetAtomWithIdx(i).GetSymbol()
        atoms.append(atom)
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            [atoms.append('H') for i in range(num_Hs)]
    
    #Counts the amount of each type of atom found in the molecule and adds it to the feature_list
    for atom in atom_symbols:
        count = 0
        for mol_atom in atoms:
            if atom == mol_atom:
                count += 1
        feature_list.append(count) 

    #Calculates the mass of the molecule 
    mass = 0
    for i in range(mol_obj.GetNumAtoms()):
        mass += mol_obj.GetAtomWithIdx(i).GetMass()
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            mass += H_mass*num_Hs        
    feature_list.append(mass)
    
    return tuple(feature_list)


#Overwrites the feature_array key of the second-level dictionary of the molecule
for mol in suppl:
    if mol is None: 
        continue
    #Defines the chemical name of the molecule
    chemical_name = mol.GetProp('ChemName').rstrip()
    
    #Try's to update the feature_array key of the second-level dictionary of the molecule
    try:
        molecule_dict[chemical_name]["feature_array"] = get_feature_array(mol)
    #If the molecule doesn't exist in the dictionary, it continues to the next molecule of the SDF file
    except KeyError:
        continue


atom_symbols = ["Br","C","Cl","F","H","I","N","O","P","S"]

def get_feature_array(mol_obj):
    """
    Input: RDKit molecule object
    
    Returns an ordered tuple containing:
    1. positions 1-10 are the counts of atom symbols in the molecule
    2. position 11 is the mass of the molecule
    3. position 12 is the number of rings in the molecule
    4. positions 13-22 are the counts of atom symbols found within rings (i.e., aromatic atoms) of the molecule
    5. position 23 is the total mass of aromatic atoms in the molecule
    
    Ordered tuple format:
    ("Br","C","Cl","F","H","I","N","O","P","S", mass_molecule, number of rings,  
    "Br","C","Cl","F","H","I","N","O","P","S", mass_aromatic atoms)
    
    Example:
    For '1-Amino-2-Methylanthraquinone', your function should return the following:
    (0, 15, 0, 0, 11, 0, 1, 2, 0, 0, 237.25799999999995, 3, 0, 12, 0, 0, 6, 0, 0, 0, 0, 0, 150.17999999999998)
    """
    
    
    #Keep track of the orderes tuple containing the features of the molecule
    feature_list = []
    
    #Keep track of all the atoms in the molecule
    atoms = []
    
    #Accesses and saves all the atoms of the molecule in the 'atoms' list
    for i in range(mol_obj.GetNumAtoms()):
        atom = mol_obj.GetAtomWithIdx(i).GetSymbol()
        atoms.append(atom)
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            [atoms.append('H') for i in range(num_Hs)]
            
    #Counts the amount of each type of atom found in the molecule and adds it to the feature_list
    for atom in atom_symbols:
        count = 0
        for mol_atom in atoms:
            if atom == mol_atom:
                count += 1
        feature_list.append(count) 
    
    #Calculates the mass of the molecule 
    mass = 0
    for i in range(mol_obj.GetNumAtoms()):
        mass += mol_obj.GetAtomWithIdx(i).GetMass()
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            mass += H_mass*num_Hs
    feature_list.append(mass)
    
    #Counts the number of rings in the molecule
    ring_count = mol_obj.GetRingInfo().NumRings()
    feature_list.append(ring_count)
    
    #Counts the number of aromatic atoms in the molecule
    H_count = 0
    for atom1 in atom_symbols:
        count = 0  
        for atom in mol_obj.GetAtoms():
            if atom.GetSymbol() == atom1 and atom.GetIsAromatic():
                count += 1
                num_Hs = atom.GetTotalNumHs()
                H_count += num_Hs
        feature_list.append(count)
    feature_list[16] = H_count
    
    #Calculates the total mass of the aromatic atoms in the molecule
    mass_aromatic_atoms = 0
    for atom in mol_obj.GetAtoms():
        if atom.GetIsAromatic():
            mass_aromatic_atoms += atom.GetMass()
            num_Hs = atom.GetTotalNumHs()
            mass_aromatic_atoms += H_mass * num_Hs
    feature_list.append(mass_aromatic_atoms)

    return tuple(feature_list)


#Overwrites the feature_array key of the second-level dictionary of the molecule
for mol in suppl:
    if mol is None: 
        continue
    #Defines the chemical name of the molecule
    chemical_name = mol.GetProp('ChemName').rstrip()
    
    #Try's to update the feature_array key of the second-level dictionary of the molecule
    try:
        molecule_dict[chemical_name]["feature_array"] = get_feature_array(mol)
    #If the molecule doesn't exist in the dictionary, it continues to the next molecule of the SDF file
    except KeyError:
        continue





def create_scatterplot(x_vals, y_vals, title, x_label, y_label, log_scale=False):
    """
    Input: x_vals, y_vals, title, x_label, y_label, log_scale = True/False
    
    Output:
    Creates a scatterplot with the given x- and y-values, title, x- and y-labels, 
    calculates a Spearman correlation, and includes a dashed line to represent the plot's diagonal
    
    log_scale:
    log_scale default = False
    If log_scale set to True, x- and y-axes will be log scaled.
    
    """
    
    #Creates scatterplot with given x- and y-vals
    plt.plot(x_vals, y_vals, 'bo')

    #Sets x- and y-axes to log scale if log_scale = True
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
    
    #Calculates Spearman correlation
    spearman_calc = stats.spearmanr(x_vals, b=y_vals)
    first_num = round(spearman_calc[0], 2) 
    second_num = spearman_calc[1]
    
    #Creates plot's diagonal
    x = plt.xlim()
    y = plt.ylim()
    plt.plot(range(int(x[0]), int(y[1])), 'k--')

    #Plots the x- and y-labels, title, and spearman correlation 
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.figtext(0.5, 0.2, f"Spearman = {first_num}({format(second_num, '.3g')})")
    plt.title(title)
             
    #Prints the graph
    plt.show()
    plt.close()


#Tests the scatterplot function
x_vals,y_vals = [],[]
for key in molecule_dict.keys():
    x_vals.append(sum(molecule_dict[key]["feature_array"][:len(atom_symbols)]))
    y_vals.append(molecule_dict[key]["feature_array"][10])

create_scatterplot(x_vals,y_vals,"Molecular weight vs. number of atoms","Number of atoms in molecule",\
                   "Molecular weight (g/mol)", True)


#Keeps track of the keys in the dictionary
data = []

#Saves all keys of the dictionary in 'data'
for key in molecule_dict.keys():
    data.append(key)

#sorts the 'data' list
data = sorted(data)


#Splits data into training and testing datasets
molecules_train, molecules_test = model_selection.train_test_split(data, test_size = 0.2)


from sklearn import tree

#Keeps track of the training dataset input 
X_train= []
#Keeps track of the training dataset output 
y_train= []

#Splits training dataset into input (feature_array) and output (target)
for molecule in molecules_train:
    X_train.append(molecule_dict[molecule]['feature_array'])
    y_train.append(molecule_dict[molecule]['target'])

#Train a descision tree regressor implementation to predict TD50 values:
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

#Keeps track of the testing dataset input 
X_test= []
#Keeps track of the testing dataset output 
y_test= []

#Splits testing dataset into input (feature_array) and output (target)
for molecule in molecules_test:
    X_test.append(molecule_dict[molecule]['feature_array'])
    y_test.append(molecule_dict[molecule]['target'])

#Creates a set of TD50 predictions from your learned model using the testing dataset.
y_pred = clf.predict(X_test)


#Calculates the mean squared error of predicted and true TD50 values
MSE = round(mean_squared_error(y_test, y_pred), 3)

#Creates a scatterplot comparing the predicted and true TD50  values for the test dataset
create_scatterplot(y_pred, y_test, f"TD50 prediction using a decision tree \nMSE={MSE}", \
                   "Decision tree regressor predictions", "True value", log_scale=True)


#Calculates the importance for each feature of the molecule
importances = list(clf.feature_importances_)

#Ordered list of features of the molecule
features = ["num_Br","num_C","num_Cl","num_F","num_H","num_I","num_N","num_O","num_P","num_S","total_mass",\
            "num_rings","num_aro_Br","num_aro_C","num_aro_Cl","num_aro_F","num_aro_H","num_aro_I",\
            "num_aro_N","num_aro_O","num_aro_P","num_aro_S","total_mass_aro"]

#Keeps track of features that have an importance greater than 0 
curr_importances = []
curr_features = []

#Saves features that have an importance greater than 0 into a list
for i, importance in enumerate(importances):
    if importance > 0:
        curr_importances.append(importance)
        curr_features.append(features[i])

#Sorts feature importances from most to least improtant
curr_importances, curr_features = zip(*sorted(zip(curr_importances, curr_features), reverse=True))


def create_barplot(x_labels, bar_heights):
    """
    Creates barplot that displays feature importance of your regressor as a Matplotlib barplot, where the:
        y-axis is the feature importance measure (bar_heights: contains importance of each feature)
        x-axis is the features used in the model (containing x-tick labels given by x_labels)
    """
    
    #Creates the bars on the x-axis
    N = np.asarray(range(len(x_labels)))+0.5
    
    #Plots the bars, labels, and title of the barplot
    plt.bar(N, bar_heights, 1.0, color  ="g", edgecolor = "k")
    plt.xticks(N, x_labels, rotation='vertical')
    plt.title("Feature importance for learned decision tree")
    plt.xlabel("Feature")
    plt.ylabel("Importance")



#Creates barpot with features and importances
create_barplot(curr_features,curr_importances)


atom_symbols = ["Br","C","Cl","F","H","I","N","O","P","S"]

def get_feature_array(mol_obj):
    """
    Input: RDKit molecule object
    
    Returns an ordered tuple containing:
    1. positions 1-10 are the counts of atom symbols in the molecule
    2. position 11 is the mass of the molecule
    3. position 12 is the number of rings in the molecule
    4. positions 13-22 are the counts of atom symbols found within rings (i.e., aromatic atoms) of the molecule
    5. position 23 is the total mass of aromatic atoms in the molecule
    6. position 24 is the tota number of H-bond donors and acceptors
    
    Ordered tuple format:
    ("Br","C","Cl","F","H","I","N","O","P","S", mass_molecule, number of rings,  
    "Br","C","Cl","F","H","I","N","O","P","S", mass_aromatic atoms)
    
    Example:
    For '1-Amino-2-Methylanthraquinone', your function should return the following:
    (0, 15, 0, 0, 11, 0, 1, 2, 0, 0, 237.25799999999995, 3, 0, 12, 0, 0, 6, 0, 0, 0, 0, 0, 150.17999999999998)
    """
    
    
    #Keep track of the orderes tuple containing the features of the molecule
    feature_list = []
    
    #Keep track of all the atoms in the molecule
    atoms = []
    
    #Accesses and saves all the atoms of the molecule in the 'atoms' list
    for i in range(mol_obj.GetNumAtoms()):
        atom = mol_obj.GetAtomWithIdx(i).GetSymbol()
        atoms.append(atom)
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            [atoms.append('H') for i in range(num_Hs)]
            
    #Counts the amount of each type of atom found in the molecule and adds it to the feature_list
    for atom in atom_symbols:
        count = 0
        for mol_atom in atoms:
            if atom == mol_atom:
                count += 1
        feature_list.append(count) 
    
    #Calculates the mass of the molecule 
    mass = 0
    for i in range(mol_obj.GetNumAtoms()):
        mass += mol_obj.GetAtomWithIdx(i).GetMass()
        num_Hs = mol_obj.GetAtomWithIdx(i).GetTotalNumHs()
        if num_Hs:
            mass += H_mass*num_Hs
    feature_list.append(mass)
    
    #Counts the number of rings in the molecule
    ring_count = mol_obj.GetRingInfo().NumRings()
    feature_list.append(ring_count)
    
    #Counts the number of aromatic atoms in the molecule
    H_count = 0
    for atom1 in atom_symbols:
        count = 0  
        for atom in mol_obj.GetAtoms():
            if atom.GetSymbol() == atom1 and atom.GetIsAromatic():
                count += 1
                num_Hs = atom.GetTotalNumHs()
                H_count += num_Hs
        feature_list.append(count)
    feature_list[16] = H_count
    
    #Calculates the total mass of the aromatic atoms in the molecule
    mass_aromatic_atoms = 0
    for atom in mol_obj.GetAtoms():
            if atom.GetIsAromatic():
                mass_aromatic_atoms += atom.GetMass()
                num_Hs = atom.GetTotalNumHs()
                mass_aromatic_atoms += H_mass * num_Hs
    feature_list.append(mass_aromatic_atoms)

    #Counting number of H-bond donors and acceptors
    HBAs = Descriptor.CalcNumHBA(mol_obj)
    HBDs = Descriptor.CalcNumHBD(mol_obj)
    feature_list.append(HBAs + HBDs)
    
    return tuple(feature_list)


#Overwrites the feature_array key of the second-level dictionary of the molecule
for mol in suppl:
    if mol is None: 
        continue
    #Defines the chemical name of the molecule
    chemical_name = mol.GetProp('ChemName').rstrip()
    
    #Try's to update the feature_array key of the second-level dictionary of the molecule
    try:
        molecule_dict[chemical_name]["feature_array"] = get_feature_array(mol)
    #If the molecule doesn't exist in the dictionary, it continues to the next molecule of the SDF file
    except KeyError:
        continue


#Keeps track of the keys in the dictionary
data2 = []

#Saves all keys of the dictionary in 'data'
for key in molecule_dict.keys():
    data2.append(key)

#sorts the 'data' list
data2 = sorted(data)

#Splits data into training and testing datasets
molecules_train2, molecules_test2 = model_selection.train_test_split(data2, test_size = 0.2)

#Keeps track of the training dataset input 
X_train2= []
#Keeps track of the training dataset output 
y_train2= []

#Splits training dataset into input (feature_array) and output (target)
for molecule in molecules_train2:
    X_train2.append(molecule_dict[molecule]['feature_array'])
    y_train2.append(molecule_dict[molecule]['target'])

#Train a descision tree regressor implementation to predict TD50 values:
clf2 = tree.DecisionTreeRegressor()
clf2 = clf2.fit(X_train2, y_train2)

#Keeps track of the testing dataset input 
X_test2= []
#Keeps track of the testing dataset output 
y_test2= []

#Splits testing dataset into input (feature_array) and output (target)
for molecule in molecules_test2:
    X_test2.append(molecule_dict[molecule]['feature_array'])
    y_test2.append(molecule_dict[molecule]['target'])

#Creates a set of TD50 predictions from your learned model using the testing dataset.
y_pred2 = clf2.predict(X_test2)

#Calculates the mean squared error of predicted and true TD50 values
MSE = mean_squared_error(y_test2, y_pred2)

#Creates a scatterplot comparing the predicted and true TD50  values for the test dataset
create_scatterplot(y_pred2, y_test2, f"TD50 prediction using a decision tree \nMSE={MSE}", \
                   "Decision tree regressor predictions", "True value", True)



#Calculates the importance for each feature of the molecule
importances2 = list(clf2.feature_importances_)

#Ordered list of features of the molecule
features = ["num_Br","num_C","num_Cl","num_F","num_H","num_I","num_N","num_O","num_P","num_S","total_mass",\
            "num_rings","num_aro_Br","num_aro_C","num_aro_Cl","num_aro_F","num_aro_H","num_aro_I",\
            "num_aro_N","num_aro_O","num_aro_P","num_aro_S","total_mass_aro", "H_bonds"]

#Keeps track of features that have an importance greater than 0 
curr_importances2 = []
curr_features2 = []

#Saves features that have an importance greater than 0 into a list
for i, importance in enumerate(importances2):
    if importance > 0:
        curr_importances2.append(importance)
        curr_features2.append(features[i])

#Sorts feature importances from most to least improtant
curr_importances2, curr_features2 = zip(*sorted(zip(curr_importances2, curr_features2), reverse=True))

#Creates barpot with features and importances
create_barplot(curr_features2,curr_importances2)


#Train a MLPRegressor implementation to predict TD50 values:
clf3 = MLPRegressor()
clf3 = clf3.fit(X_train2, y_train2)

#Creates a set of TD50 predictions from your learned model using the testing dataset.
y_pred3 = clf3.predict(X_test2)

#Calculates the mean squared error of predicted and true TD50 values
MSE = round(mean_squared_error(y_test2, y_pred3), 3)

#Creates a scatterplot comparing the predicted and true TD50  values for the test dataset
create_scatterplot(y_pred3, y_test2, f"TD50 prediction using a decision tree \nMSE={MSE}", \
                   "Decision tree regressor predictions", "True value", log_scale=True)




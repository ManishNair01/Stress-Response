import os
import sys
import pickle

projectabspathname = os.path.abspath('stress_model.pkl')
print(projectabspathname)
projectname = 'predict.ipynb'
projectpickle = open(str(projectabspathname),'wb')
pickle.dump(projectname, projectpickle)
projectpickle.close()
"""
Script to retrieve transferability experiments setting
(i.e. dataframe path, target classes, and task type)
"""

from local_data.constants import *


def get_experiment_setting(experiment):

    # Transferability for classification
    if experiment == "02_MESSIDOR":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "02_MESSIDOR.csv",
                   "task": "classification",
                   "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
                               "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
                               "proliferative diabetic retinopathy": 4}}
    elif experiment == "25_REFUGE":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv",
                   "task": "classification",
                   "targets": {"no glaucoma": 0, "glaucoma": 1}}
    elif experiment == "13_FIVES":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "age related macular degeneration": 1, "diabetic retinopathy": 2,
                               "glaucoma": 3}}
    elif experiment == "08_ODIR200x3":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    elif experiment == "36_ACRIMA":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "36_ACRIMA.csv",
                   "task": "classification",
                   "targets": {"no glaucoma": 0, "glaucoma": 1}}
    elif experiment == "CAT_MYA_2":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CAT_MYA_2.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "pathologic myopia": 1, "cataract": 2}}
    elif experiment == "05_20x3":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "05_20x3.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "retinitis pigmentosa": 1, "macular hole": 2}}
    elif experiment == "MHL_RP_2":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "MHL_RP_2.csv",
                   "task": "classification",
                   "targets": {"normal": 0, "retinitis pigmentosa": 1, "macular hole": 2}}
    elif experiment == "37_DeepDRiD_train_eval":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_train_eval.csv",
                   "task": "classification",
                   "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
                               "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
                               "proliferative diabetic retinopathy": 4}}
    elif experiment == "37_DeepDRiD_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_test.csv",
                   "task": "classification",
                   "targets": {"no diabetic retinopathy": 0, "mild diabetic retinopathy": 1,
                               "moderate diabetic retinopathy": 2, "severe diabetic retinopathy": 3,
                               "proliferative diabetic retinopathy": 4}}
    elif experiment == "38_MMAC23A_train":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "38_MMAC23A_train.csv",
                   "task": "classification",
                   "targets": {"myopic maculopathy grade cero": 0, "myopic maculopathy grade one": 1,
                               "myopic maculopathy grade two": 2, "myopic maculopathy grade three": 3,
                               "myopic maculopathy grade four": 4}}
    elif experiment == "38_MMAC23A_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "38_MMAC23A_test.csv",
                   "task": "classification",
                   "targets": {"myopic maculopathy grade cero": 0, "myopic maculopathy grade one": 1,
                               "myopic maculopathy grade two": 2, "myopic maculopathy grade three": 3,
                               "myopic maculopathy grade four": 4}}
    elif experiment == "38_MMAC23B_test":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "38_MMAC23B_test.csv",
                   "task": "classification",
                   "targets": {"myopic maculopathy grade cero": 0, "myopic maculopathy grade one": 1,
                               "myopic maculopathy grade two": 2, "myopic maculopathy grade three": 3,
                               "myopic maculopathy grade four": 4}}
    elif experiment == "CGI_HRDC_Task1":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CGI_HRDC_Task1.csv",
                   "task": "classification",
                   "targets": {"no hypertensive": 0, "hypertensive": 1}}
    elif experiment == "CGI_HRDC_Task2":
        setting = {"dataframe": PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "CGI_HRDC_Task2.csv",
                   "task": "classification",
                   "targets": {"no hypertensive retinopathy": 0, "hypertensive retinopathy": 1}}

    else:
        setting = None
        print("Experiment not prepared...")

    return setting


"""
This script contains the code used to obtain the
assembly data dataframes with text and categorical
labels, partitioned for train/evaluate.

Also, we prepare here the dataframes for prompts
(zero-shot) and transferability experiments.
"""

import os
import json
import glob

import pandas as pd
import numpy as np

from local_data.constants import *

if not os.path.exists(PATH_DATAFRAME_PRETRAIN):
    os.mkdir(PATH_DATAFRAME_PRETRAIN)
if not os.path.exists(PATH_DATAFRAME_TRANSFERABILITY):
    os.mkdir(PATH_DATAFRAME_TRANSFERABILITY)
if not os.path.exists(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION):
    os.mkdir(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION)
if not os.path.exists(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION):
    os.mkdir(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION)


def adequate_01_eyepacs():
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    path_dataset = "01_EYEPACS/"

    partitions = ["train", "test", "val"]
    data = []
    for iPartition in partitions:
        print(iPartition)
        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + iPartition + ".csv")

        for iFile in range(dataframe.shape[0]):
            print(iFile, end="\r")
            image_path = path_dataset + "documents/" + dataframe["image_id"][iFile].split("/")[-1].replace(".jpg", ".jpeg")
            categories, atributes = [], []

            categories.append(labels_dr[dataframe["dr"][iFile]])
            if os.path.isfile(PATH_DATASETS + image_path):
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "01_EYEPACS.csv")


def adequate_02_messidor():
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    labels_dme = {0: "no diabetic macular edema", 1: "diabetic macular edema"}
    labels_gradable = {0: "noisy", 1: "clean"}

    path_dataset = "02_MESSIDOR/"

    # The link to MESSIDOR2 labels is at:
    # https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades?select=messidor_data.csv
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "messidor_data.csv")

    data = []
    for iFile in range(dataframe.shape[0]):
        image_path = path_dataset + "documents/" + dataframe["image_id"][iFile].replace(".jpg", ".JPG")
        categories, atributes = [], []

        # Noisy/Clean
        atributes.append(labels_gradable[dataframe["adjudicated_gradable"][iFile]])
        if dataframe["adjudicated_gradable"][iFile] == 1:
            categories.append(labels_dr[dataframe["adjudicated_dr_grade"][iFile]])
            categories.append(labels_dme[dataframe["adjudicated_dme"][iFile]])

        if os.path.isfile(PATH_DATASETS + image_path):
            if len(categories) > 0:
                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "02_MESSIDOR.csv")


def adequate_03_idrid():
    path_dataset = "03_IDRID/"
    data = []

    # A.Segmentation
    subpath = "A.%20Segmentation/A. Segmentation/"
    subpath_images = "1. Original Images/a. Training Set/"
    subpath_gt = "2. All Segmentation Groundtruths/a. Training Set/"
    annotations_paths = ["1. Microaneurysms", "2. Haemorrhages", "3. Hard Exudates", "4. Soft Exudates"]
    annotations_abbreviations = ["MA", "HE", "EX", "SE"]
    annotations_categories = ["microaneurysms", "haemorrhages", "hard exudates", "soft exudates"]

    files_segmentation = os.listdir(PATH_DATASETS + path_dataset + subpath + subpath_images)

    for iFile in files_segmentation:
        image_path = path_dataset + subpath + subpath_images + iFile

        categories = []
        atributes = []
        for i in range(len(annotations_categories)):
            annotation_path = PATH_DATASETS + path_dataset + subpath + subpath_gt + annotations_paths[i] + "/"\
                              + iFile.replace(".jpg","_" + annotations_abbreviations[i] + ".tif")
            if os.path.isfile(annotation_path):
                categories.append(annotations_categories[i])

        data.append({"image": image_path,
                     "atributes": atributes,
                     "categories": categories})

    # B.Grading
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    labels_dme = {0: "no referable diabetic macular edema", 1: "non clinically significant diabetic macular edema",
                  2: "diabetic macular edema"}

    subpath = "B.%20Disease%20Grading/B. Disease Grading/"
    subpath_images = "1. Original Images/a. Training Set/"
    dataframe = "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subpath + dataframe)
    for iFile in range(dataframe.shape[0]):
        image_path = path_dataset + subpath + subpath_images + dataframe["Image name"][iFile] + ".jpg"
        categories = []
        atributes = []

        categories.append(labels_dr[dataframe["Retinopathy grade"][iFile]])
        categories.append(labels_dme[dataframe["Risk of macular edema "][iFile]])

        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": atributes,
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "03_IDRID.csv")


def adequate_03_idrid_segmentation():
    path_dataset = "03_IDRID/"
    data = []

    # A.Segmentation
    subpath = "A.%20Segmentation/A. Segmentation/"
    subpath_images = "1. Original Images/"
    subpath_masks = "2. All Segmentation Groundtruths/"
    partitions = ["a. Training Set/", "b. Testing Set/"]
    partitions_names = ["train", "test"]

    categories_paths = ["1. Microaneurysms/", "2. Haemorrhages/", "3. Hard Exudates/", "4. Soft Exudates/"]
    annotations_abbreviations = ["MA", "HE", "EX", "SE"]
    annotations_categories = ["microaneurysms", "haemorrhages", "hard_exudates", "soft_exudates"]

    for iPartition in range(len(partitions)):
        for iCategory in range(len(categories_paths)):
            data = []
            files = os.listdir(PATH_DATASETS + path_dataset + subpath + subpath_masks + partitions[iPartition] + categories_paths[iCategory])

            for iFile in files:
                image_path = path_dataset + subpath + subpath_images +\
                             partitions[iPartition] + iFile.replace(".tif", ".jpg").replace("_" + annotations_abbreviations[iCategory], "")
                mask_path = path_dataset + subpath + subpath_masks + partitions[iPartition] + categories_paths[iCategory] + iFile
                data.append({"image": image_path,
                             "mask": mask_path})

            df_out = pd.DataFrame(data)
            df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION + "03_IDRID_" +
                          annotations_categories[iCategory] + '_' + partitions_names[iPartition] + '.csv')


def adequate_04_rfmid():
    template_diseases = {'DR': 'diabetic retinopathy', 'ARMD': 'age-related macular degeneration',
                         'MH': 'media haze', 'DN': 'drusens', 'MYA': 'myopia', 'BRVO': 'branch retinal vein occlusion',
                         'TSLN': 'tessellation', 'ERM': 'epiretinal membrane', 'LS': 'laser scar',
                         'MS': 'macular scar', 'CSR': 'central serous retinopathy', 'ODC': 'optic disc cupping',
                         'CRVO': 'central retinal vein occlusion', 'TV': 'tortuous vessels', 'AH': 'asteroid hyalosis',
                         'ODP': 'optic disc pallor', 'ODE': 'optic disc edema', 'ST': 'shunt',
                         'AION': 'anterior ischemic optic neuropathy', 'PT': 'parafoveal telangiectasia',
                         'RT': 'retinal traction', 'RS': 'retinitis', 'CRS': 'chorioretinitis', 'EDN': 'edudates',
                         'RPEC': 'retinal pigment epithelium changes', 'MHL': 'macular hole',
                         'RP': 'retinitis pigmentosa', 'CWS':'cotton wool spots', 'CB': 'colobomas',
                         'ODPM': 'optic disc pit maculopathy', 'PRH': 'preretinal haemorrhage',
                         'MNF': 'myelinated nerve fibers', 'HR': 'haemorrhagic retinopathy',
                         'CRAO': 'central retinal artery occlusion', 'TD': 'tilted disc',
                         'CME': 'cystoid macular edema', 'PTCR': 'post traumatic choroidal rupture',
                         'CF': 'choroidal folds', 'VH': 'vitreous haemorrhage', 'MCA': 'macroaneurysm',
                         'VS': 'vasculitis', 'BRAO': 'branch retinal artery occlusion', 'PLQ': 'plaque',
                         'HPED': 'haemorrhagic pigment epithelial detachment', 'CL': 'collaterals'}

    path_dataset = "04_RFMid/"
    partitions = ["Training", "Validation", "Testing"]
    letters = ["a", "b", "c"]
    data = []
    for iPartition in range(len(partitions)):
        subpath_images = "1. Original Images/" + letters[iPartition] + ". " + partitions[iPartition] + " Set/"
        subpath_dataframe = "2. Groundtruths/"  + letters[iPartition] +  ". RFMiD_" + partitions[iPartition] + "_Labels.csv"

        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subpath_dataframe)
        for iFile in range(dataframe.shape[0]):
            image_path = path_dataset + subpath_images + str(dataframe["ID"][iFile]) + ".png"
            categories, atributes = [], []

            if dataframe["Disease_Risk"][iFile] == 1:
                categories.append("a disease")
                ids = np.argwhere(np.array(dataframe)[iFile, 2:])

                for i in list(ids):
                    dis_abreviation = dataframe.columns.to_list()[i[0]+2]
                    categories.append(template_diseases[dis_abreviation])
            else:
                categories.append("no disease")
                categories.append("healthy")

            if os.path.isfile(PATH_DATASETS + image_path):

                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "04_RFMid.csv")


def adequate_05_1000x39():

    categories_template = {'0.0.Normal': 'normal', '0.1.Tessellated fundus': 'tessellation',
                           '0.2.Large optic cup': 'large optic cup', '0.3.DR1': 'mild diabetic retinopathy',
                           '1.0.DR2': 'moderate diabetic retinopathy', '1.1.DR3': 'severe diabetic retinopathy',
                           '2.0.BRVO': 'branch retinal vein occlusion', '2.1.CRVO': 'central retinal vein occlusion',
                           '3.RAO': 'central retinal artery occlusion', '4.Rhegmatogenous RD': 'retina detachment',
                           '5.0.CSCR':'central serous retinopathy', '5.1.VKH disease': 'Vogt-Koyanagi syndrome',
                           '6.Maculopathy': 'maculopathy', '7.ERM': 'epiretinal membrane', '8.MH': 'macular hole',
                           '9.Pathological myopia': 'pathologic myopia', '10.0.Possible glaucoma': 'glaucoma',
                           '10.1.Optic atrophy': 'optic atrophy',
                           '11.Severe hypertensive retinopathy': 'severe hypertensive retinopathy',
                           '12.Disc swelling and elevation': 'disc swelling and elevation',
                           '13.Dragged Disc': 'dragged disk',
                           '14.Congenital disc abnormality': 'congenital disk abnormality',
                           '15.0.Retinitis pigmentosa':'retinitis pigmentosa',
                           '15.1.Bietti crystalline dystrophy': 'Bietti crystalline dystrophy',
                           '16.Peripheral retinal degeneration and break': 'peripheral retinal degeneration and break',
                           '17.Myelinated nerve fiber': 'myelinated nerve fibers',
                           '18.Vitreous particles': 'vitreous haemorrhage', '19.Fundus neoplasm': 'neoplasm',
                           '20.Massive hard exudates': 'hard exudates',
                           '21.Yellow-white spots-flecks': 'yellow-white spots flecks',
                           '22.Cotton-wool spots':'cotton wool spots', '23.Vessel tortuosity':'tortuous vessels',
                           '24.Chorioretinal atrophy-coloboma': 'colobomas',
                           '25.Preretinal hemorrhage': 'preretinal haemorrhage', '26.Fibrosis': 'fibrosis',
                           '27.Laser Spots': 'laser scar', '28.Silicon oil in eye': 'silicon oil',
                           '29.0.Blur fundus without PDR': 'no proliferative diabetic retinopathy',
                           '29.1.Blur fundus with suspected PDR': 'proliferative diabetic retinopathy'}

    categories_test = ["0.0.Normal", "15.0.Retinitis pigmentosa", "8.MH"]

    path_dataset = "05_1000x39/"
    data = []

    categories = os.listdir(PATH_DATASETS + path_dataset)
    [categories.remove(iCategory) for iCategory in categories_test]

    for iCategory in categories:
        images = os.listdir(PATH_DATASETS + path_dataset + iCategory + "/")

        for iImage in images:
            data.append({"image": path_dataset + iCategory + "/" + iImage,
                         "atributes": [],
                         "categories": [categories_template[iCategory]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "05_1000x39.csv")

    data = []
    for iCategory in categories_test:
        images = os.listdir(PATH_DATASETS + path_dataset + iCategory + "/")
        images = images[0:20]

        for iImage in images:
            data.append({"image": path_dataset + iCategory + "/" + iImage,
                         "atributes": [],
                         "categories": [categories_template[iCategory]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "05_20x4.csv")


def adequate_06_DEN():
    path_dataset = "06_DEN/"
    data = []

    partitions = ["DeepEyeNet_train.json", "DeepEyeNet_test.json", "DeepEyeNet_valid.json"]
    for iPartition in partitions:
        f = open(PATH_DATASETS + path_dataset + iPartition)
        meta = json.load(f)

        for iSample in meta:
            image_path = path_dataset + list(iSample.keys())[0]
            categories, atributes = [], []

            info = iSample[list(iSample.keys())[0]]
            categories.extend(info["keywords"].split(", "))
            categories.extend(info["clinical-description"].split(". "))

            if "" in categories:
                categories.remove("")

            if os.path.isfile(PATH_DATASETS + image_path):

                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "06_DEN.csv")


def adequate_07_lag():
    path_dataset = "07_LAG/"
    categories_paths = ["non_glaucoma", "suspicious_glaucoma"]
    categories = ["no glaucoma", "glaucoma"]

    data = []
    for i in range(len(categories_paths)):
        images = os.listdir(PATH_DATASETS + path_dataset + categories_paths[i] + "/image/")

        for iImage in images:
            data.append({"image": path_dataset + categories_paths[i] + "/image/" + iImage,
                         "atributes": [],
                         "categories": [categories[i]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "07_LAG.csv")


def adequate_08_odir5k():
    path_dataset = "08_ODIR-5K/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "full_df.csv")

    # Train and Incremental test division
    mask_cat = np.logical_and(dataframe["C"].values == 1, dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 1)
    mask_myo = np.logical_and(dataframe["M"].values == 1, dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 1)
    mask_normal = np.logical_and(dataframe["N"].values == 1, dataframe[["D", "G", "C", "A", "H", "M", "O"]].values.sum(-1) == 0)
    mask_normal[np.argwhere(mask_normal == 1)[200:]] = False
    mask_test = np.logical_or(mask_cat, mask_myo)
    mask_test = np.logical_or(mask_test, mask_normal)

    dataframe_train = dataframe[np.logical_not(mask_test)]
    dataframe_test = dataframe[mask_test]

    # Train subset
    data = []
    for iFile in range(dataframe_train.shape[0]):
        id = dataframe_train["ID"].values[iFile]

        for iEye in ["Right", "Left"]:
            image_path = path_dataset + "preprocessed_images/" + str(id) + "_" + (iEye).lower() + ".jpg"
            categories = []
            description = dataframe_train[(iEye + "-Diagnostic Keywords")].values[iFile]
            if "myop" not in description and "cataract" not in description:
                categories.extend(description.split("，"))
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "08_ODIR.csv")

    # Test subset
    data = []
    counter_n, counter_m, counter_c = 1, 1, 1
    for iFile in range(dataframe_test.shape[0]):
        id = dataframe_test["ID"].values[iFile]

        for iEye in ["Right", "Left"]:
            image_path = path_dataset + "preprocessed_images/" + str(id) + "_" + (iEye).lower() + ".jpg"
            description = dataframe_test[(iEye + "-Diagnostic Keywords")].values[iFile]
            if "myop" in description and counter_m<=200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["pathologic myopia"]})
                    counter_m += 1
            if "cataract" in description and counter_c<=200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["cataract"]})
                    counter_c += 1
            if "normal" in description and counter_n<=200:
                if os.path.isfile(PATH_DATASETS + image_path):
                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": ["normal"]})
                    counter_n += 1

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "08_ODIR200x3.csv")


def adequate_09_papila():
    path_dataset = "09_PAPILA/"
    subpath_images = "FundusImages/"
    dataframes = [pd.read_excel(PATH_DATASETS + path_dataset + "ClinicalData/patient_data_od.xlsx"),
                  pd.read_excel(PATH_DATASETS + path_dataset + "ClinicalData/patient_data_os.xlsx")]
    labels_glaucoma = {0: "normal", 1: "glaucoma", 2: "glaucoma"}

    data = []
    for iFile in range(dataframes[0].shape[0]-2):
        id = dataframes[0]["Unnamed: 0"][iFile+2][1:]

        image_path = path_dataset + subpath_images + "RET" + id + "OD" + ".jpg"
        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [labels_glaucoma[dataframes[0]["Diagnosis"][iFile+2]]]})

        image_path = path_dataset + subpath_images + "RET" + id + "OS" + ".jpg"
        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [labels_glaucoma[dataframes[1]["Diagnosis"][iFile+2]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "09_PAPILA.csv")


def adequate_10_paraguay():
    path_dataset = "10_PARAGUAY/"
    data = []

    subpaths = ["1. No DR signs", "2. Mild (or early) NPDR", "3. Moderate NPDR",
                "4. Severe NPDR", "5. Very Severe NPDR", "6. PDR", "7. Advanced PDR"]
    categories = ["no diabetic retinopathy", "mild diabetic retinopathy",
                  "moderate diabetic retinopathy", "severe diabetic retinopathy",
                  "severe diabetic retinopathy", "proliferative diabetic retinopathy",
                  "proliferative diabetic retinopathy"
                  ]

    for iPath in range(len(subpaths)):
        images = os.listdir(PATH_DATASETS + path_dataset + subpaths[iPath] + "/")

        for iImage in images:
            data.append({"image": path_dataset + subpaths[iPath] + "/" + iImage,
                         "atributes": [],
                         "categories": [categories[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "10_PARAGUAY.csv")


def adequate_11_stare():
    path_dataset = "11_STARE/"
    data = []
    metadata = "all-mg-codes.txt"

    for line in open(PATH_DATASETS + path_dataset + metadata):
        categories, atributes = [], []
        columns = line.strip().split("\t")

        image_path = path_dataset + "documents/" + columns[0] + ".ppm"
        description = columns[-1].split("\n")[0].lower().split("        ")[-1].replace("\"", "")

        categories.extend(description.replace("\t", "").replace(" and ", " or ").replace("?", "").split(" or "))

        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "11_STARE.csv")


def adequate_12_aria():
    path_dataset = "12_ARIA/"
    categories_subpath = ["aria_a_markups", "aria_c_markups", "aria_d_markups"]
    categories = ["age-related macular degeneration", "normal", "diabetic retinopathy"]
    data = []

    for i in range(len(categories)):
        for iFile in os.listdir(PATH_DATASETS + path_dataset + categories_subpath[i] + "/"):
            if iFile != "Thumbs.db":

                data.append({"image": path_dataset + categories_subpath[i] + "/" + iFile,
                             "atributes": [],
                             "categories": [categories[i]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "12_ARIA.csv")


def adequate_13_fives():
    path_dataset = "13_FIVES/"
    images_subpath = ["train/Original/", "test/Original/"]
    labels_dme = {"A": "age related macular degeneration",
                  "D": "diabetic retinopathy",
                  "G": "glaucoma",
                  "N": "normal"}
    data = []
    for iSubpath in images_subpath:
        files = os.listdir(PATH_DATASETS + path_dataset + iSubpath)
        for iFile in files:
            if iFile != "Thumbs.db":
                category__code = iFile.split(".")[0].split("_")[-1]
                data.append({"image": path_dataset + iSubpath + iFile,
                             "atributes": [],
                             "categories": [labels_dme[category__code]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "13_FIVES.csv")


def adequate_14_agar300():
    path_dataset = "14_AGAR300/"
    finding = ["microaneurysms", "diabetic retinopathy"]
    images_subpath = "img/"

    data = []
    files = os.listdir(PATH_DATASETS + path_dataset + images_subpath)
    for iFile in files:
        data.append({"image": path_dataset + images_subpath + iFile,
                     "atributes": [],
                     "categories": finding})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "14_AGAR300.csv")


def adequate_15_aptos():
    path_dataset = "15_APTOS/"
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train.csv")
    images_subpath = "train_images/"

    data = []
    for iFile in range(dataframe.shape[0]):
        image_path = path_dataset + images_subpath + dataframe["id_code"][iFile] + ".png"
        categories, atributes = [], []

        categories.append(labels_dr[dataframe["diagnosis"][iFile]])
        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": atributes,
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "15_APTOS.csv")


def adequate_16_fundoct():
    path_dataset = "16_FUND-OCT/"
    data = []
    dict_macula = {'acute CSR': 'acute central serous retinopathy', 'chronic CSR': 'chronic central serous retinopathy',
                   'ci-DME': 'cystoid macular edema', 'geographic_AMD': 'geographical age-related macular degeneration',
                   'Healthy': 'normal', 'neovascular_AMD': 'neovascular age-related macular degeneration'}
    # Glaucoma/NoGlaucoma
    subpath = "Dataset/OD/"

    data = []
    files = glob.glob(PATH_DATASETS + path_dataset + subpath + "*/*/*/*Color*")
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": [iFile.replace(PATH_DATASETS, "").split("/")[3].lower().replace("healthy", "normal")]})

    # Macula-related
    subpath = "Dataset/Macula/"
    files = glob.glob(PATH_DATASETS + path_dataset + subpath + "*/*/*/*Color*")
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": [
                         dict_macula[iFile.replace(PATH_DATASETS, "").split("/")[3]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "16_FUND-OCT.csv")


def adequate_17_diaretdb1():
    import xml.etree.ElementTree as ET

    path_dataset = "17_DiaRetDB1/"
    subpath_images = "documents/"
    subpath_annotations = "groundtruth/"

    files = os.listdir(PATH_DATASETS + path_dataset + subpath_images)
    data = []
    for iFile in files:
        categories = []
        for annotator in ["_01.xml","_02.xml","_03.xml","_04.xml"]:
            annotation_id = iFile.replace(".png", annotator)
            tree = ET.parse(PATH_DATASETS + path_dataset + subpath_annotations + annotation_id)
            root = tree.getroot()
            for item in root.findall('./markinglist/marking/'):
                if item.tag == 'markingtype':
                    categories.append(item.text.lower().replace("_", " ").replace("irma", "intraretinal microvascular abnormalities"))
        categories = list(np.unique(categories))
        categories.remove("disc")
        if len(categories) == 0:
            continue

        data.append({"image": path_dataset + subpath_images + iFile.replace(PATH_DATASETS, ""),
                     "atributes": [],
                     "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "17_DiaRetDB1.csv")


def adequate_18_drions_db():
    path_dataset = "18_DRIONS-DB/"
    images_subpath = "documents/"

    data = []
    files = os.listdir(PATH_DATASETS + path_dataset + images_subpath)
    for iFile in files:
        data.append({"image": path_dataset + images_subpath + iFile,
                     "atributes": [],
                     "categories": ["no cataract", "a disease"]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "18_DRIONS-DB.csv")


def adequate_19_drishtigs1():
    path_dataset = "19_Drishti-GS1/"
    dataframe = pd.read_excel(PATH_DATASETS + path_dataset + "Drishti-GS1_diagnosis.xlsx", skiprows=3)[1:]
    subpath_images = ["Drishti-GS1_files/Training/Images/", "Drishti-GS1_files/Test/Images/"]
    data = []
    for iPartition in subpath_images:
        for iFile in range(dataframe.shape[0]):
            id = dataframe["Drishti-GS File"].values[iFile][:-1]
            finding = dataframe["Total"].values[iFile].lower()
            image_path = path_dataset + iPartition + id + ".png"

            if os.path.isfile(PATH_DATASETS + image_path):

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [finding.replace("glaucomatous", "glaucoma")]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "19_Drishti-GS1.csv")


def adequate_20_e_ophta():
    path_dataset = "20_E-ophta/"
    labels = {"EX": "exudates", "healthy": "healthy", "MA": "microaneurysms"}
    subpath_images = ["e_optha_EX/EX/", "e_optha_EX/healthy/", "e_optha_MA/MA/", "e_optha_EX/healthy/"]

    data = []
    for iSub in subpath_images:
        finding = labels[iSub.split("/")[-2]].replace("healthy", "normal")
        for root, dirs, files in os.walk(PATH_DATASETS + path_dataset + iSub):
            for filename in files:
                if filename != "Thumbs.db":
                    print(os.path.join(root, filename))

                    data.append({"image": os.path.join(root, filename).replace(PATH_DATASETS, ""),
                                 "atributes": [],
                                 "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "20_E-ophta.csv")


def adequate_21_g1020():
    path_dataset = "21_G1020/"
    image_subpath = "Images/"
    labels = {0: "normal", 1: "glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "G1020.csv")

    data = []
    for iFile in range(dataframe.shape[0]):
        id = dataframe["imageID"].values[iFile]
        finding = labels[dataframe["binaryLabels"].values[iFile]]
        image_path = path_dataset + image_subpath + id

        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "21_G1020.csv")


def adequate_22_heimed():
    path_dataset = "22_HEI-MED/"

    data = []
    return 1


def adequate_23_hrf():
    path_dataset = "23_HRF/"
    data = []

    # Disease
    image_subpath = "documents/"
    labels = {"dr": "diabetic retinopathy", "g": "glaucoma", "h": "normal"}
    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    for iFile in files:
        data.append({"image": path_dataset + image_subpath + iFile,
                     "atributes": [],
                     "categories": [labels[iFile.split("_")[-1].split(".")[0]]]})

    # Noise
    image_subpath = "Noise/"
    labels = {"bad": "noisy", "good": "clean"}
    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    for iFile in files:
        data.append({"image": path_dataset + image_subpath + iFile,
                     "atributes": [labels[iFile.split("_")[-1].split(".")[0]]],
                     "categories": []})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "23_HRF.csv")


def adequate_24_origa():
    path_dataset = "24_ORIGA/"
    image_subpath = "Images/"
    labels = {0: "no glaucoma", 1: "glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "OrigaList.csv")

    data = []
    for iFile in range(dataframe.shape[0]):
        id = dataframe["Filename"].values[iFile]
        finding = labels[dataframe["Glaucoma"].values[iFile]]
        image_path = path_dataset + image_subpath + id

        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "24_ORIGA.csv")


def adequate_25_refuge():
    path_dataset = "25_REFUGE/"
    data = []

    # Disease
    image_subpath = "train/Images/" # We only have labels for train subset
    labels = {"g": "glaucoma", "n": "no glaucoma"}
    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    for iFile in files:
        data.append({"image": path_dataset + image_subpath + iFile,
                     "atributes": [],
                     "categories": [labels[iFile[0]]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "25_REFUGE.csv")


def adequate_26_roc():
    path_dataset = "26_ROC/"

    files = glob.glob(PATH_DATASETS + path_dataset + "*/*/*.jpg")
    data = []
    for iFile in files:
        data.append({"image": iFile.replace(PATH_DATASETS,""),
                     "atributes": [],
                     "categories": ["microaneurysms"]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "26_ROC.csv")


def adequate_27_brset():
    path_dataset = "27_BRSET/"
    image_subpath = "fundus_photos/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "labels.csv")

    anatomical_dict = {"1": "normal", "2": "abnormal", 'bv': ""}
    dr_dict = {"0": "no diabetic retinopathy",
               "1": "mild diabetic retinopathy",
               "2": "moderate diabetic retinopathy",
               "3": "severe diabetic retinopathy.",
               "4": "proliferative diabetic retinopathy"}
    findings = ["macular_edema", "scar", "nevus", "amd", "vascular_occlusion", "hypertensive_retinopathy",
                "drusens", "hemorrhage", "retinal_detachment", "myopic_fundus", "increased_cup_disc", "other"]
    find_names = ["macular edema", "scar", "nevus", "age-related macular degeneration", "vascular occlusion",
                  "hypertensive retinopathy", "drusens", "hemorrhage", "retina detachment",
                  "myopia", "increased cup disc", "a disease"]

    data = []
    for iFile in range(dataframe.shape[0]):
        categories, atributes = [], []
        id = dataframe["image_id"].values[iFile] + ".jpg"

        # optic_disc
        categories.append(anatomical_dict[dataframe["optic_disc"].values[iFile]] + " optic disc")
        # vessels
        categories.append(anatomical_dict[str(dataframe["vessels"].values[iFile])] + " vessels")
        # macula
        categories.append(anatomical_dict[str(dataframe["macula"].values[iFile])] + " macula")
        # DR_ICDR
        categories.append(dr_dict[str(dataframe["DR_ICDR"].values[iFile])])
        # Noise
        if dataframe["focus"].values[iFile] == 2 or dataframe["iluminaton"].values[iFile] == 2 \
                or dataframe["image_field"].values[iFile] == 2 or dataframe["image_field"].values[iFile] == 2 :
            atributes.append("Noisy")
        # findings
        for i in range(len(findings)):
            if dataframe[findings[i]].values[iFile] == 1:
                categories.append(find_names[i])

        image_path = path_dataset + image_subpath + id
        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "27_BRSET.csv")


def adequate_28_OIA():

    path_dataset = "28_OIA-DDR/"
    data = []
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy", 5: ""}

    subpath_grading = "DR_grading/"
    subpath_segmentation = "lesion_segmentation/"
    lesions_path = ["EX/", "HE/", "MA/", "SE/"]
    lesions = ["hard exudates", "haemorrhages", "microaneurysms", "soft exudates"]
    partitions = ["train", "test", "valid"]

    for iPartition in partitions:
        dataframe = pd.read_table(PATH_DATASETS + path_dataset + subpath_grading + iPartition + ".txt", delimiter=" ",
                                  header=None)
        files = list(dataframe[0].values)

        for iFile in range(len(files)):
            categories, atributes = [], []

            image_path = path_dataset + subpath_grading + iPartition + "/" + files[iFile]
            categories.append(labels_dr[dataframe[1].values[iFile]])

            if dataframe[1].values[iFile] == 5:
                atributes.append("noisy")

            for i in range(len(lesions_path)):
                for iiPartition in partitions:
                    if os.path.isfile(PATH_DATASETS + path_dataset + subpath_segmentation + iiPartition + "/" + "label/" + lesions_path[i] + files[iFile].replace(".jpg", ".tif")):
                        categories.append(lesions[i])

            if os.path.isfile(PATH_DATASETS + image_path):

                data.append({"image": image_path,
                             "atributes": atributes,
                             "categories": categories})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "28_OIA-DDR.csv")


def adequate_29_airogs():

    path_dataset = "29_AIROGS/"
    image_subpath = "documents/" # We only have labels for train subset
    labels = {"RG": "glaucoma", "NRG": "no glaucoma"}
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "train_labels.csv")

    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    data = []
    for iFile in files[2:]:
        print(iFile)
        id = dataframe["challenge_id"] == iFile.split(".")[0]

        finding = labels[dataframe[id]["class"].values[0]]
        image_path = path_dataset + image_subpath + iFile

        if os.path.isfile(PATH_DATASETS + image_path):

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "29_AIROGS.csv")


def adequate_30_sustech():

    path_dataset = "30_SUSTech-SYSU/"
    image_subpath = "originalImages/"  # We only have labels for train subset
    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy", 5: ""}

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "Labels.csv")

    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    data = []
    for iFile in files:
        print(iFile)
        id = dataframe["Fundus_images"] == iFile

        if np.argwhere(id.values).__len__() > 0:

            finding = labels_dr[dataframe[id]["DR_grade_American_Academy_of_Ophthalmology"].values[0]]
            image_path = path_dataset + image_subpath + iFile

            findings = [finding]
            if os.path.isfile(PATH_DATASETS + image_path):
                if os.path.isfile(PATH_DATASETS + path_dataset + "exudatesLabels/" + iFile.split(".")[0] + ".xml"):
                    findings.append("exudates")

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": findings})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "30_SUSTech-SYSU.csv")


def adequate_31_jichi():

    path_dataset = "31_JICHI/"
    image_subpath = "documents/"  # We only have labels for train subset
    labels_dr = {"ndr": ["no diabetic retinopathy"],
                 "sdr": ["microaneurysm", "retinal hemorrhage", "hard exudate", "retinal edema",
                         "more than three small soft exudates"],
                 "ppdr": ["soft exudate", "varicose veins", "intraretinal microvascular abnormality",
                          "non-perfusion area over one disc area"],
                 "pdr": ["neovascularization", "preretinal haemorrhage", "fibrovascular proliferativemembrane",
                         "tractionalretinaldetachment"],
                 }

    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + "list.csv")

    files = os.listdir(PATH_DATASETS + path_dataset + image_subpath)

    data = []
    for iFile in files:
        print(iFile)
        id = dataframe["Image"] == iFile

        if np.argwhere(id.values).__len__() > 0:

            finding = labels_dr[dataframe[id]["Davis_grading_of_one_figure"].values[0]]
            image_path = path_dataset + image_subpath + iFile

            if os.path.isfile(PATH_DATASETS + image_path):

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": finding})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "31_JICHI.csv")


def adequate_32_chaksu():

    path_dataset = "32_CHAKSU/"
    subpaths = ["Train/", "Test/"]
    data = []
    scanners = ["Bosch", "Forus", "Remidio"]
    dataframe_id = "Glaucoma_Decision_Comparison_[SCAN]_majority.csv"
    path_images = "1.0_Original_Fundus_Images/"
    labels = {"NORMAL": "no glaucoma", "GLAUCOMA SUSPECT": "glaucoma"}
    formats = {'Bosch': '.JPG', 'Forus': '.png', 'Remidio': '.jpg'}

    for iSubpath in subpaths:
        for iScanner in scanners:

            dataframe = pd.read_csv(PATH_DATASETS + path_dataset + iSubpath + dataframe_id.replace("[SCAN]", iScanner))
            files = dataframe["Images"].values.tolist()

            for iFile in files:
                image_path = path_dataset + iSubpath + path_images + iScanner + "/" + iFile.split(".")[0] + formats[iScanner]

                if os.path.isfile(PATH_DATASETS + image_path):
                    if "Majority Decision" in  list(dataframe.keys()):
                        ky = "Majority Decision"
                    else:
                        ky = "Glaucoma Decision"

                    finding = labels[dataframe[dataframe["Images"] == iFile][ky].values[0]]

                    data.append({"image": image_path,
                                 "atributes": [],
                                 "categories": [finding]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "32_CHAKSU.csv")


def adequate_33_dr():

    path_dataset = "33_DR1-2/"
    subpaths = ["Cotton-wool Spots", "Deep Hemorrhages", "Drusen", "Hard Exudates", "Normal Images", "Red Lesions",
                "Superficial Hemorrhages"]
    transfer = {"Cotton-wool Spots": "cotton wool spots", "Deep Hemorrhages": "deep haemorrhages",
                "Drusen": "drusens", "Hard Exudates": "hard exudates", "Normal Images": "normal",
                "Red Lesions": "red small dots", "Superficial Hemorrhages": "superficial haemorrhages"}
    data = []
    for iPath in subpaths:
        files = os.listdir(PATH_DATASETS + path_dataset + iPath + "/")

        for iFile in files:
            image_path = path_dataset + iPath + "/" + iFile

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [transfer[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "33_DR1-2.csv")


def adequate_34_cataract():

    path_dataset = "34_Cataract/"
    subpaths = ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]
    transfer = {"1_normal": "normal", "2_cataract": "cataract",
                "2_glaucoma": "glaucoma", "3_retina_disease": "retinitis"}
    data = []
    for iPath in subpaths:
        files = os.listdir(PATH_DATASETS + path_dataset + iPath + "/")

        for iFile in files:
            image_path = path_dataset + iPath + "/" + iFile

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [transfer[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "34_Cataract.csv")


def adequate_35_scardat():

    path_dataset = "35_ScarDat/"
    subpaths = ["train/", "val/", "test/"]
    subsubsubpaths = ["positive/", "negative/"]

    transfer = {"positive/": "laser scar", "negative/": "no laser scar"}
    data = []
    for iPath in subpaths:
        for iiPath in subsubsubpaths:

            files = os.listdir(PATH_DATASETS + path_dataset + iPath + iiPath)

            for iFile in files:
                image_path = path_dataset + iPath + iiPath + iFile

                data.append({"image": image_path,
                             "atributes": [],
                             "categories": [transfer[iiPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_PRETRAIN + "35_ScarDat.csv")


def adequate_36_acrima():

    path_dataset = "36_ACRIMA/"
    subpaths = ["G/", "noG/"]
    transfer = {"G/": "glaucoma", "noG/": "no glaucoma"}

    data = []
    for iPath in subpaths:

        files = os.listdir(PATH_DATASETS + path_dataset + iPath )

        for iFile in files:
            image_path = path_dataset + iPath + iFile

            data.append({"image": image_path,
                         "atributes": [],
                         "categories": [transfer[iPath]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "36_ACRIMA.csv")


def adequate_37_DeepDRiD():

    path_dataset = "37_DeepDRiD/"
    subpath = "regular_fundus_images/"
    paritions_path = ["regular-fundus-training/", "regular-fundus-validation/"]
    paritions = ["train", "val"]

    labels_dr = {0: "no diabetic retinopathy", 1: "mild diabetic retinopathy", 2: "moderate diabetic retinopathy",
                 3: "severe diabetic retinopathy", 4: "proliferative diabetic retinopathy", 5: ""}
    data = []
    for i in range(0, len(paritions)):

        dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subpath + paritions_path[i] + paritions_path[i][:-1] + ".csv")
        for ii in range(dataframe.shape[0]):

            iFile = dataframe["image_path"][ii].replace("\\", "/").replace(paritions_path[i][:-1] , "Images")
            dr = np.nanmean(dataframe[["left_eye_DR_Level", "right_eye_DR_Level"]].values[ii,:])

            if os.path.isfile(PATH_DATASETS + path_dataset + subpath + paritions_path[i]  + iFile):

                data.append({"image": path_dataset + subpath + paritions_path[i]  + iFile,
                             "atributes": [],
                             "categories": [labels_dr[int(dr)]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_train_eval.csv")

    subsubpath = "regular_fundus_images/Online-Challenge1&2-Evaluation/"
    dataframe = pd.read_csv(PATH_DATASETS + path_dataset + subsubpath + "Challenge1_labels.csv")

    data = []
    for ii in range(dataframe.shape[0]):
        iFile = "Images/" + dataframe["image_id"][ii].split("_")[0] + "/" + dataframe["image_id"][ii] + ".jpg"
        dr = dataframe["DR_Levels"][ii]

        if os.path.isfile(PATH_DATASETS + path_dataset + subsubpath + iFile):
            data.append({"image": path_dataset + subsubpath + iFile,
                         "atributes": [],
                         "categories": [labels_dr[int(dr)]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_CLASSIFICATION + "37_DeepDRiD_test.csv")


def adequate_mmac_segmentation():
    path_dataset = "101_MMAC23/"
    data = []

    # A.Segmentation
    subpath = "2. Segmentation of Myopic Maculopathy Plus Lesions/"
    subpath_images = "1. Images/"
    subpath_masks = "2. Groundtruths/"

    categories_paths = ["1. Lacquer Cracks/", "2. Choroidal Neovascularization/", "3. Fuchs Spot/"]
    annotations_abbreviations = ["LC", "CN", "FS"]

    subpath_partitions = "1. Training Set/"

    for iCategory in range(len(categories_paths)):
        data = []
        files = os.listdir(PATH_DATASETS + path_dataset + subpath + categories_paths[iCategory] + subpath_images + subpath_partitions)

        for iFile in files:
            image_path = path_dataset + subpath + categories_paths[iCategory] + subpath_images + subpath_partitions + iFile
            mask_path = image_path.replace(subpath_images, subpath_masks)
            data.append({"image": image_path,
                         "mask": mask_path})

        df_out = pd.DataFrame(data)
        df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION + "mmac_" + annotations_abbreviations[iCategory] + "_segmentation.csv")


def adequate_CGI_HRDC():

    # TASK 1
    path_dataset = "102_CGI-HRDC/"
    subpath = "1-Hypertensive Classification/"
    subpath_images = "1-Images/1-Training Set/"
    dataframe_train = "2-Groundtruths/HRDC Hypertensive Classification Training Labels.csv"

    dataframe = pd.read_csv(
        PATH_DATASETS + path_dataset + subpath + dataframe_train)

    labels_dr = {0: "no hypertensive",
                 1: "hypertensive"}
    data = []
    for i in range(dataframe.shape[0]):

        iFile = dataframe["Image"][i]
        dr = dataframe["Hypertensive"][i]

        if os.path.isfile(PATH_DATASETS + path_dataset + subpath +subpath_images + iFile):
            data.append({"image": path_dataset + subpath + subpath_images + iFile,
                         "atributes": [],
                         "categories": [labels_dr[int(dr)]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION + "CGI_HRDC_Task1.csv")

    # TASK 2
    path_dataset = "102_CGI-HRDC/"
    subpath = "2-Hypertensive Retinopathy Classification/"
    subpath_images = "1-Images/1-Training Set/"
    dataframe_train = "2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv"

    dataframe = pd.read_csv(
        PATH_DATASETS + path_dataset + subpath + dataframe_train)

    labels_dr = {0: "no hypertensive retinopathy",
                 1: "hypertensive retinopathy"}
    data = []
    for i in range(dataframe.shape[0]):

        iFile = dataframe["Image"][i]
        dr = dataframe["Hypertensive Retinopathy"][i]

        if os.path.isfile(PATH_DATASETS + path_dataset + subpath +subpath_images + iFile):
            data.append({"image": path_dataset + subpath + subpath_images + iFile,
                         "atributes": [],
                         "categories": [labels_dr[int(dr)]]})

    df_out = pd.DataFrame(data)
    df_out.to_csv(PATH_DATAFRAME_TRANSFERABILITY_SEGMENTATION + "CGI_HRDC_Task2.csv")

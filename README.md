# Digitized-vs-Image-ECG-classification

**Abstract:** This code supplements the article: Automatic classification of healthy and disease conditions from images or digital standard 12-lead electrocardiograms \
**Authors:** Vadim Gliner, Noam Keidar, Vladimir Makarov, Arutyun I. Avetisyan, Assaf Schuster & Yael Yaniv \
**Published at:** Scientific Reports volume 10, Article number: 16331 (2020) 

### Instructions for use
1. Download the project from git and unzip
2. Download the digitized ECG database from: (https://technionmail-my.sharepoint.com/:u:/r/personal/vadimgl_campus_technion_ac_il/Documents/Digitized_emergency.zipx?csf=1&web=1&e=aKE0hW)
3. Download the Images ECG database from: (https://technionmail-my.sharepoint.com/:u:/g/personal/vadimgl_campus_technion_ac_il/EeTTvjs0gDFKn5lG837ICtwB_y109zWoFW6xTSfJWNXr3A?e=3ObKRF)
4. Unzip databases from paragraphs 2,3 and put the files under the folder *Database* in project's directory. Make sure that you have in this folder the following files:
* diagnosis_digitized0.hdf5
* diagnosis_digitized1.hdf5
* diagnosis_digitized2.hdf5
* Digitized_emergency.p
* ECG_paper.jpg
* Unified_rendered_db.hdf5
If you want to retrain, do steps 5,6,7. Otherwise, skip to 8.
5. Adjust the batch size in functions *RunNetDigitizedToMultiClassBinary* 
(line 161 at Main_script.py) & *RunNoamsECG_ImageClassification* (lines 317,319 at Main_script.py) to fit the memory of your GPU
6. Take out the checkpoints from *checkpoints* folder
7. Run "Main_script.py"
8. Just for classification, without retraining, Run  "Main_script.py" without taking out the checkpoints file from "checkpoints" folder
9. Enjoy. For questions: (vadim.gliner@gmail.com)

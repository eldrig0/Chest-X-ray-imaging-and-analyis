# Chest-X-ray-imaging-and-analyis

Background.

Computer Aided diagnosis is currently not availabe in Bhutan. The radiologist are often very busy and also the number of qualified radiologist is at minimum with only four Radiologist in JDNRH (Thimphu hospital) which is responsible for serving more than 80,000 people.
The number of patient coming to get their medial imaging (X-rays, MRI, CT-scans) and for disease diagnosis is increasing (exact figures in finding) which delays the time patients to get their disease diagnois reports.

So to help better and faster Diagnosis, we are going to use Computer Vision and related technologies to help the radiologists in Bhutan. There are many research papers showing the possibilitites of Computer Vision in Medical Imaging and Computer Aided Diagnosis.

The major problem/challenge in this project is the lack of data that can be used to fuel the algorithm. The data we are hoping to use is a quality data with annotations and is based on Bhutanese population, however to gets things moving I used a dataset that is provided by NIH (National Institute of Health, USA) who compiled the dataset of scans from more than 30,000 patients, including many with advanced lung disease.
The dataset given in the link. https://nihcc.app.box.com/v/ChestXray-NIHCC
The discription for the dataset is given in https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community.

Although the dataset contains more than 100,000 scans of chest ray with 14 labelled diseases and i used only 2000 images and one disease which makes it a binary classification task.


Steps:
1. The annotations are analysed and is the analysis is in annotations_analysis.py
2. The labels and the images of the disease is extracted and reshaped.
3. The labels are then converted to one hot matrix
4. Build the model in tensorflow and etc.

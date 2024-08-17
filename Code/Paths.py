import os

class ProjectPath:
        
        def __init__(self):

            self.ROOT_DIR="E:\\vsc2.0\\GitHub\\Grape_instance_segementation_and_masking"

            self.CodePath=os.path.join(self.ROOT_DIR,"Code")

            self. DatasetPath=os.path.join(self.ROOT_DIR,"PreprocessedDataset")
            self.DS_ImagesPath=os.path.join(self.DatasetPath,"Images")
            self.DS_NPZMasksPath=os.path.join(self.DatasetPath,"NPZFiles")
            self.DS_TextFilesPath=os.path.join(self.DatasetPath,"TextFiles")
            self.DS_JsonFilePath=os.path.join(self.DatasetPath,"MRCNN_DS.json")
            self.DS_YoloTextFilesPath=os.path.join(self.DatasetPath,"YOLOTextFiles")
            self.DS_YoloDS_Path=os.path.join(self.DatasetPath,"YOLODS")


            self.SegmentatedImages_Path=os.path.join(self.ROOT_DIR,"Segmentated_Images")
            self.SG_GroundTruthImages_Path=os.path.join(self.SegmentatedImages_Path,"GroundTruthImages")
            self.SG_GroundTruthImages_WithBoxes_Path=os.path.join(self.SegmentatedImages_Path,"GroundTruthImagesWithBoxes")
            self.SG_YoloPredictedImages_Path=os.path.join(self.SegmentatedImages_Path,"YOLOPredictedImages")
            self.SG_YoloPredictedImages_WithBoxes_Path=os.path.join(self.SegmentatedImages_Path,"YOLOPredictedImagesWithBoxes")
            self.SG_YoloPredictedImages_WithBoxes_ConfidenceThreshold_0point5_Path=os.path.join(self.SegmentatedImages_Path,"YOLOPredictedImagesWithBoxesConfidence_0.5")
            self.SG_MRCNNPredictedImages_Path=os.path.join(self.SegmentatedImages_Path,"MRCNNPredictedImages")
            self.SG_MRCNNPredictedImages_WithBoxes_Path=os.path.join(self.SegmentatedImages_Path,"MRCNNPredictedImagesWithBoxes")

            self.MRCNN_Training_Logs_Path=os.path.join(self.ROOT_DIR,"MRCNNTrainingLogs")
            self.MRCNN_Trained_Weights_Path=os.path.join(self.MRCNN_Training_Logs_Path,"MRCNNTrainedWeights")

            self.Yolo_Training_Logs_Path=os.path.join(self.ROOT_DIR,"YoloTrainingLogs")


            self.ResultCurvesPath=os.path.join(self.ROOT_DIR,"ResultCurves")

        def Change_ROOT_DIR(self,path):
            self.ROOT_DIR=path


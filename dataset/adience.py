from dataset import Dataset
import sqlite3
import dlib
import cv2
import os
import numpy as np
import pandas as pd
from loggers import Log
import csv
import scipy
from scipy.special import softmax
from scipy.stats import norm
import tensorflow as tf


class AdienceDataset(Dataset):
    """Class that abstracts Aflw dataset.
    """

    def __init__(self,config):
        # self.conn = sqlite3.connect("/home/mtk/dataset/aflw-files/aflw/data/aflw.sqlite")
        super(AdienceDataset, self).__init__(config)

    """ method that resizes image to the same resolution
    image which have width and height equal or less than
    values specified by max_size.
        e.g
        img = np.zeros((200,300))
        img = resize_down_image(img,(100,100))
        img.shape # (66,100)

    """

    def clean_metadata(self):
        # Definitions
        files_dir = os.path.join(self.config.dataset_dir, "files")
        files_list = os.listdir(files_dir)
        csv_dir = os.path.join(self.config.dataset_dir, "cleaned.csv")
        # Clear the csv file
        open(csv_dir, "w").close()

        all_lines = []
        for path in files_list:
            # Join the paths to get the file
            path = os.path.join(files_dir, path)
            lines = []
            
            # Open the .txt files and read their contents 
            with open(path, 'r') as fd:
                for line in fd.readlines():
                    line = line.split('\t')
                    # Figuer out the indexs for the attributes i need
                    if line[3] != "None":
                        age = line[3].split(",")
                        name = "coarse_tilt_aligned_face." + line[2] + "." + line[1]
                        if len(age) == 1:
                            line = [line[0] + ","+ name  + "," + age[0] + '\n']
                        else:
                            line = [line[0] + "," + name + "," + age[0] +" "+ age[1] + '\n']
                        lines += line
            # Remove the first line
            lines.__delitem__(0)
            all_lines += lines
            
        # make a .csv file and write out to it
        with open(csv_dir, "a") as csv_fd:
            for line in all_lines:
                csv_fd.write(line)

    
    def encode_age(self, start, end=None):
        MAX_AGE = 128
    
        if end == None:
            return np.array([1] * start + [0] * (MAX_AGE-start))

        age_range = range(start, end+1)
        encoded = []
        # Encode and 
        for num in age_range:
            encoded.append(np.array([1] * num + [0] * (MAX_AGE-num)))
        # Get mean and std
        mean = scipy.mean(age_range)
        std = scipy.std(age_range)
        # Calculate norm
        rv = norm(loc=mean, scale=std)
        # Get PDF for each value
        pdf_values = [rv.pdf(n) for n in age_range]
        # Calclate softmax
        weights = softmax(pdf_values)#.reshape(len(age_range), 1)
        # Multiply with encoding
        array = np.matmul(weights ,encoded)
        # take the sum along the 0 axis
        # array = array.sum(0)
        # Convert to a Tensor
        array = tf.convert_to_tensor(array, dtype=tf.float32)
        return array

    def resize_down_image(self,img,max_img_shape):
        img_h,img_w = img.shape[0:2]
        w, h = img_w,img_h
        if max_img_shape[0]<h:
            w = (max_img_shape[0]/float(h))  * w
            h = max_img_shape[0]
        if max_img_shape[1]<w:
            h = (max_img_shape[1]/float(w)) * h
            w = max_img_shape[1]
        if h == img_h:
            return img,1
        else:
            scale = img_h/h
            img = scipy.misc.imresize(img, (int(w),int(h))).astype(np.float32)/255
            return img,scale

    def selective_search(self,img,min_size=(2200),max_img_size=(24,24),debug=False):
        cand_rects = []
        img,scale = self.resize_down_image(img,max_img_size)
        dlib.find_candidate_object_locations(img,cand_rects,min_size=min_size)
        rects = [(int(crect.left() * scale),
             int(crect.top()* scale),
             int(crect.right()* scale),
             int(crect.bottom()* scale),
            ) for crect in cand_rects]
        for rect in rects:
            cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0),2)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    """loads dataset from a given filename"""
    def load_dataset(self):
        if self.config.label == "detection":
            if not self.contain_dataset_files():
                self.meet_convention()
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading pickle files")
            Log.DEBUG_OUT =False
            self.train_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"train.pkl"))
            self.test_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"test.pkl"))
            if os.path.exists(os.path.join(self.config.dataset_dir,"validation.pkl")):
                self.validation_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"validation.pkl"))
            else:
                self.validation_dataset = None
                frameinfo = getframeinfo(currentframe())
                Log.WARNING("Unable to find validation dataset",file_name=__name__,line_number=frameinfo.lineno)
            self.train_dataset = self.fix_labeling_issue(self.train_dataset)
            self.test_dataset = self.fix_labeling_issue(self.test_dataset)
            self.validation_dataset = self.fix_labeling_issue(self.validation_dataset)
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded train, test and validation dataset")
            Log.DEBUG_OUT =False
            test_indexes = np.arange(len(self.test_dataset))
            np.random.shuffle(test_indexes)
            validation_indexes = np.arange(len(self.validation_dataset))
            np.random.shuffle(validation_indexes)

            self.test_dataset = self.test_dataset.iloc[test_indexes].reset_index(drop=True)
            self.validation_dataset = self.validation_dataset.iloc[validation_indexes].reset_index(drop=True)

            self.test_dataset = self.test_dataset[:1000]
            self.validation_dataset = self.validation_dataset[:100]
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading test images")
            Log.DEBUG_OUT =False
            self.test_dataset_images = self.load_images(self.test_dataset).astype(np.float32)/255
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading validation images")
            Log.DEBUG_OUT =False
            self.validation_dataset_images = self.load_images(self.validation_dataset).astype(np.float32)/255
            self.test_detection = self.test_dataset["is_face"].values
            self.dataset_loaded = True
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded all dataset and images")
            Log.DEBUG_OUT =False

        elif (self.config.label == "pose"):
            if not self.contain_dataset_files():
                self.meet_convention()
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading pickle files")
            Log.DEBUG_OUT =False
            self.train_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"train.pkl"))
            self.test_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"test.pkl"))
            if os.path.exists(os.path.join(self.config.dataset_dir,"validation.pkl")):
                self.validation_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"validation.pkl"))
            else:
                self.validation_dataset = None
                frameinfo = getframeinfo(currentframe())
                Log.WARNING("Unable to find validation dataset",file_name=__name__,line_number=frameinfo.lineno)
            self.train_dataset = self.fix_labeling_issue(self.train_dataset)
            self.test_dataset = self.fix_labeling_issue(self.test_dataset)
            self.validation_dataset = self.fix_labeling_issue(self.validation_dataset)
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded train, test and validation dataset")
            Log.DEBUG_OUT =False
            test_indexes = np.arange(len(self.test_dataset))
            np.random.shuffle(test_indexes)
            validation_indexes = np.arange(len(self.validation_dataset))
            np.random.shuffle(validation_indexes)

            self.test_dataset = self.test_dataset.iloc[test_indexes].reset_index(drop=True)
            self.validation_dataset = self.validation_dataset.iloc[validation_indexes].reset_index(drop=True)

            self.test_dataset = self.test_dataset[:1000]
            self.validation_dataset = self.validation_dataset[:100]
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading test images")
            Log.DEBUG_OUT =False
            self.test_dataset_images = self.load_images(self.test_dataset).astype(np.float32)/255
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading validation images")
            Log.DEBUG_OUT =False
            self.validation_dataset_images = self.load_images(self.validation_dataset).astype(np.float32)/255
            self.test_detection = self.test_dataset["is_face"].values
            self.dataset_loaded = True
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded all dataset and images")
            Log.DEBUG_OUT =False
        
        elif (self.config.label == "age"):
            if not self.contain_dataset_files():
                self.clean_metadata()
                self.meet_convention()
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading pickle files")
            Log.DEBUG_OUT =False
            self.train_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"train.pkl"))
            self.test_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"test.pkl"))
            if os.path.exists(os.path.join(self.config.dataset_dir,"validation.pkl")):
                self.validation_dataset = self.get_meta(os.path.join(self.config.dataset_dir,"validation.pkl"))
            else:
                self.validation_dataset = None
                frameinfo = getframeinfo(currentframe())
                Log.WARNING("Unable to find validation dataset",file_name=name ,line_number=frameinfo.lineno)
            self.train_dataset = self.fix_labeling_issue(self.train_dataset)
            self.test_dataset = self.fix_labeling_issue(self.test_dataset)
            self.validation_dataset = self.fix_labeling_issue(self.validation_dataset)
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded train, test and validation dataset")
            Log.DEBUG_OUT =False
            test_indexes = np.arange(len(self.test_dataset))
            np.random.shuffle(test_indexes)
            validation_indexes = np.arange(len(self.validation_dataset))
            np.random.shuffle(validation_indexes)

            self.test_dataset = self.test_dataset.iloc[test_indexes].reset_index(drop=True)
            self.validation_dataset = self.validation_dataset.iloc[validation_indexes].reset_index(drop=True)

            # self.test_dataset = self.test_dataset[:1000]
            # self.validation_dataset = self.validation_dataset[:100]

            Log.DEBUG_OUT = True
            Log.DEBUG("Loading test images")
            Log.DEBUG_OUT =False
            self.test_dataset_images = self.load_images(self.test_dataset).astype(np.float32)/255
            Log.DEBUG_OUT = True
            Log.DEBUG("Loading validation images")
            Log.DEBUG_OUT =False
            self.validation_dataset_images = self.load_images(self.validation_dataset).astype(np.float32)/255
            self.test_detection = self.test_dataset["age"].values
            self.dataset_loaded = True
            Log.DEBUG_OUT = True
            Log.DEBUG("Loaded all dataset and images")
            Log.DEBUG_OUT =False

        else:
            raise NotImplementedError("Not implemented for labels:"+str(self.labels))

    def generator(self, batch_size):
        raise NotImplementedError("Not implmented!")

    def detection_data_genenerator(self, batch_size):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                X = X.reshape(-1, self.config.image_shape[0], self.config.image_shape[1], self.config.image_shape[2])
                detection = self.get_column(current_dataframe, "is_face").astype(np.uint8)
                detection = np.eye(2)[detection]
                yield X,detection

    def smile_data_generator(self,batch_size=32):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = current_images.astype(np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])
                smile = self.get_column(current_dataframe,"Smiling").astype(np.uint8)
                smile = np.eye(2)[smile]
                yield X,smile

    def age_data_generator(self, batch_size=32):
        while True:
            indexes = np.arange(len(self.train_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.train_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = np.array(current_images.tolist(), dtype=np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])

                # age = self.get_column(current_dataframe,"age").astype(str)
                age = np.array(self.get_column(current_dataframe,"age").tolist())
                # print(X)
                # Maybe do a list return instead of a string
                yield X,age
    def age_val_generator(self, batch_size=32):
        while True:
            indexes = np.arange(len(self.validation_dataset))
            np.random.shuffle(indexes)
            for i in range(0,len(indexes)-batch_size,batch_size):
                current_indexes = indexes[i:i+batch_size]
                current_dataframe = self.validation_dataset.iloc[current_indexes].reset_index(drop=True)
                current_images = self.load_images(current_dataframe)
                X = np.array(current_images.tolist(), dtype=np.float32)/255
                X = X.reshape(-1,self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2])

                # age = self.get_column(current_dataframe,"age").astype(str)
                age = np.array(self.get_column(current_dataframe,"age").tolist())

                # Maybe do a list return instead of a string
                yield X,age

    def load_images(self,dataframe):
        output_images = np.zeros((len(dataframe),self.config.image_shape[0],self.config.image_shape[1],self.config.image_shape[2]))
        imgs_folder = os.path.join(self.config.dataset_dir, "faces")
        for index,row in dataframe.iterrows():
            file_location = os.path.join(imgs_folder, row["file_location"])
            # file_location = row["file_location"]
            img = cv2.imread(file_location)
            if img is None:
                print("Unable to read image from ",file_location)
                continue
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # img = cv2.resize(img,(self.config.image_shape[0],self.config.image_shape[1]))
            # img = np.array(Image.fromarray(img).resize(self.config.image_shape[0], self.config.image_shape[1])).astype(np.float32)/255
            # img = scipy.misc.imresize(img, (self.config.image_shape[0], self.config.image_shape[1])).astype(np.float32)/255
            output_images[index] = cv2.resize(img, self.config.image_shape[:2])
            # output_images[index] = img
        return output_images

    def meet_convention(self):
        if self.contain_dataset_files():
            return
        elif os.path.exists(os.path.join(self.config.dataset_dir,"all.pkl")):
            dataframe = pd.read_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))
            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
        else:
            if self.config.label.lower()=="age":
                # self.clean_metadata()
                dataframe = self.load_age_dataset()
            else:
                dataframe = self.load_face_non_face_dataset()

            train,test,validation = self.split_train_test_validation(dataframe)
            train.to_pickle(os.path.join(self.config.dataset_dir,"train.pkl"))
            test.to_pickle(os.path.join(self.config.dataset_dir,"test.pkl"))
            validation.to_pickle(os.path.join(self.config.dataset_dir,"validation.pkl"))
            dataframe.to_pickle(os.path.join(self.config.dataset_dir,"all.pkl"))

    def load_age_dataset(self):
        file_locations = []
        age_lable = []
        csv_dir = os.path.join(self.config.dataset_dir, "cleaned.csv")
        with open(csv_dir, 'r') as fd:
            for line in fd.readlines():
                line = line.split(',')
                file_locations += [os.path.join(line[0], line[1])]
                line[2] = line[2].strip()
                splited = line[2].split(' ')
                
                if len(splited)>1:
                    splited = [int(splited[0].split('(')[1]),
                            int(splited[-1].split(')')[0])]
                    encoded = self.encode_age(splited[0], splited[1])
                    age_lable += [encoded]
                else:
                    encoded = self.encode_age(int(line[2]))
                    age_lable += [encoded]
        
        output_df = pd.DataFrame(columns=["file_location","age"])
        output_df["file_location"] = file_locations
        output_df["age"] = age_lable
        return output_df

    def load_face_non_face_dataset(self):
        output_file_locations = []
        output_is_face = []
        for img_path in os.listdir(os.path.join(self.config.dataset_dir)):
            output_file_locations+=[os.path.join(self.config.dataset_dir,img_path)]
            output_is_face+=[1]
        for img_path in os.listdir(os.path.join(self.config.dataset_dir)):
            output_file_locations+=[os.path.join(self.config.dataset_dir,img_path)]
            output_is_face+=[0]
        output_df = pd.DataFrame(columns=["file_location","is_face"])
        output_df["file_location"] = output_file_locations
        output_df["is_face"] = output_is_face
        return output_df

    def fix_labeling_issue(self,dataset):
        return dataset

    def rect_intersection(self,rect1,rect2):
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea

    def rect_union(self,rect1,rect2):

        assert rect1.shape == (4,) , "rect1 shape should be (4,) and it is "+str(rect1.shape)
        assert rect2.shape == (4,) , "rect2 shape should be (4,) and it is "+str(rect2.shape)

        width1 = np.abs(rect1[0]-rect1[2])
        height1 = np.abs(rect1[1]-rect1[3])

        width2 = np.abs(rect2[0]-rect2[2])
        height2 = np.abs(rect2[1]-rect2[3])
        area1 = width1 * height1
        area2 = width2 * height2

        return area1+area2 - self.rect_intersection(rect1,rect2)

    def bb_intersection_over_union(self,boxA, boxB):
        intr = self.rect_intersection(boxA,boxB)
        if(intr<=0):
            return 0
        runion = rect_union(boxA,boxB)
        if(runion<=0):
            return 0
        iou = intr / float(runion)
        return iou

    def get_dataset_name(self):
        return "aflw"

class Rect(object):
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def area(self):
        return self.w * self.h
    def intersection(self,rect):
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(self.x, rect.x));
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]));
        overlapArea = x_overlap * y_overlap;
        return overlapArea
    def union(self,rect):
        assert rect1.shape == (4,) , "rect1 shape should be (4,) and it is "+str(rect1.shape)
        assert rect2.shape == (4,) , "rect2 shape should be (4,) and it is "+str(rect2.shape)

        width1 = np.abs(rect1[0]-rect1[2])
        height1 = np.abs(rect1[1]-rect1[3])

        width2 = np.abs(rect2[0]-rect2[2])
        height2 = np.abs(rect2[1]-rect2[3])
        area1 = width1 * height1
        area2 = width2 * height2

        return area1+area2 - self.rect_intersection(rect1,rect2)
    def iou(self,rect):
        pass
    def __str__(self):
        return "("+str(self.x)+","+self.y+") (" +str(self.w)+","+self.h+")"

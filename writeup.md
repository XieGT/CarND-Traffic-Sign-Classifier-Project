# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/md1.png "Visualization"
[image2]: ./examples/md2.png "Traffic Signs"
[]: 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.
From the following python codes I get the size of data set
 
 ```
 n_train = len(X_train)
 n_validation = len(X_valid)
 n_test = len(X_test)
 image_shape = X_train[0].shape
 n_classes = y_train.max() - y_train.min() + 1
 ```
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

(1) Use pandas to read in the index of the different traffic sign plot with 8 rows and 6 columns, 48 figures in total.

![alt text][image1]

(2) Use histogram show the distribution of each traffic sign in the training set
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The images has been normalized in the range of [-1,1], with the mean value of 0. The Python codes are as following:


```python
def img_norm(image):
    """
    Normalize the image 
    Returns an image normalized to the range of -1.0 to 1.0.
    """
    return image/128.0 - 1.0
```




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The ultimate model of mine consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x15 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x15 			|
| Convolution 5x5	   | 1x1 stride, valid padding, outputs 10x10x30 |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride,  outputs 5x5x30         |
| Flatten | 750 |
| Fully connected | input 750, outputs 400 |
| RELU |	|
| Fully connected | input 400, outputs 120 |
|      RELU       |
|      Dropout    |    keep 75% of the weights                                          |
| Fully connected | input 120, outputs 43 |
| Softmax |	|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

(1) Optimizer is chosen to be Adam Optimizer, better than the original gradient optimizer.

(2) Batch size is 128, since we use amazon cloud, this makes no difference.

(3) The number of epochs is 10, tried 15 which only improved a little bit.

(4) The learning rate is 0.001, from previous experice.


(5) I added a 0.75 dropout rate in the training to avoid overfitting. compared to original plan imporved a lot.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

1. I used the Lenet structure which introduced in the class.
2. the depth of first CNN layer chosen to be 15 which meant to capture more features. Similarly the second CNN layer has the depth of 30. 
3. the Adam Optimizer is adopted after compared with the original Gradient optimizer.
4. the dropout processdure is introduced to avoid the overfitting.
5. the validation accuracy is 0.94 > 0.93 which is the requirement.





### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image2] 

the roundabout sign might be difficult to classify, because it is a little bit small. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        				  |
|:---------------------:|:---------------------------------------------:|
|      Turn left     |   Turn left    	|
|        Yield       |      Yield     	|
|       Priority     |  	Priority		|
|         Stop       |     Stop			|
|		Roundabout	 |   Roundabout		|
|       No Entry     |    No Entry   	|



The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on validaton set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the "Output Top 5 Softmax Probabilities For Each Image Found on the Web" section. The code is as follows:

```python
top_5_pro = tf.nn.top_k(tf.nn.softmax(logits), k=5)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_5_res = sess.run(top_5_pro, feed_dict={x: X_data, keep_prob:1.}) 
    print(top_5_res)
```

 

(1) This image is the Turn left sign, the top five softmax probabilities are

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
|      1      | 34, Turn left  		|
| 0     		 | 30, Beware of snow 	|
|      0      | 35, Ahead only		|
|      0      | 11, Right-of-way at the next intersection	|
|      0      | 13, Yield |


(2) This image is the Yield sign, the top five softmax probabilities are

| Probability |      Prediction      |
| :---------: | :------------------: |
|      0.99   |  13, Yield         |
|      0      |   3, Speed limit (60km/h)     |
|      0      |  15, No vehicles     |
|      0      |  28, Children crossing |
|      0      |   9, No passing  |

(3) This image is the Priority road, the top five softmax probabilities are

| Probability |        Prediction         |
| :---------: | :-----------------------: |
|    0.74     |     12, Priority road     |
|    0.23     |     32, End of all speed and passing limit        |
|      0.01   | 38, Keep right |
|      0      | 13, Yield    |
|      0      | 17, No entry        |

(4) This image is the Stop sign, the top five softmax probabilities are

| Probability |      Prediction       |
| :---------: | :-------------------: |
|      1      | 14, Stop        		  	|
|      0      | 17, No entry         	|
|      0      |  5, Speed limit(80km/h)	|
|      0      |  3, Speed limit(60km/h)	|
|      0      |  1, Speed limit(30km/h)	|

(5) This image is the Roundabout sign, the top five softmax probabilities are

| Probability |      Prediction      |
| :---------: | :------------------: |
|      1      | 40, Roundabout         |
|      0      | 33, Turn right ahead |
|      0      |  1, Speed limit(30km/h)|         
|      0      | 38, Keep right      |
|      0      | 11, Right-of-way at the next intersection |

(6) This image is the No entry sign, the top five softmax probabilities are

| Probability |      Prediction      |
| :---------: | :------------------: |
|      0.99   | 17, Roundabout         |
|      0      | 12, Priority road |
|      0      | 14, Stop|         
|      0      | 30, Beware of ice/snow      |
|      0      | 10, No passing for vehicles over 3.5 metric tons |
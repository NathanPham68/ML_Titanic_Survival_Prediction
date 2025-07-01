# Titanic_Survival_Prediction
## I) Introduction
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

## II) Loading the data
![image](https://github.com/user-attachments/assets/f8bf2446-130c-4869-9a1f-92ba5ae7f2ac)

![image](https://github.com/user-attachments/assets/c153f3ac-0cce-4abf-b60a-420afd9d41dd)

Notes:
* Training set has 'Survived'
* Testing set does not have 'Survived'
* sibsp: # of siblings / spouses aboard the Titanic
* parch: # of parents / children aboard the Titanic
* ticket: Ticket number
* cabin: Cabin number

![image](https://github.com/user-attachments/assets/87ef7d84-77a6-4f50-9e34-23719398f2a4)

The first column is the passenger ID. The ID for a passenger is just a number to identify this passenger in this dataset. So this column is not really a part of the information we should care about. We can drop this column or make it the index for this dataset. Let's make it the index for the dataset just to demonstrate the use of df.set_index method.

![image](https://github.com/user-attachments/assets/1eeeeba5-5df1-41bf-8953-0e1d63c65658)

![image](https://github.com/user-attachments/assets/758773b2-47b4-4e7f-8f9f-2a37319a5f3d)

## III) Feature Classification: Categorical vs Numerical
This helps us select the appropriate plots for visualization.

Which features are categorical?
* Categorical Features: nominal, ordinal, ratio, interval
* To classify the samples into sets of simila samples

Which features are numerical?
* Numerical Features: discrete, continuous or timeseries
* These values change from sample to sample

![image](https://github.com/user-attachments/assets/4b60cf5d-6f90-4856-91b3-ab65a3d4b6ac)

![image](https://github.com/user-attachments/assets/d60158f1-aad6-483b-947c-b731c3a816dc)

* Categorical: Survived, Sex, Embarked, Pclass (ordinal), SibSp, Parch

  Embarked: Port of Embarkation - C = Cherbourg, Q = Queenstown, S = Southampton

* Numerical: (continuous) Age, Fare, (discrete)
* Mix types of data: Ticket, Cabin
* Contain Error / Typo: Name
* Blank or Null: Cabin > Age > Embarked
* Various Data Type: String, Int, Float

According to the data dictionary, we know that if a passenger is marked as 1, he/ she survived. Clearly the number 1 or 0 is a flag for the person's survivorship. Yet the data type of the column is int64, which is a numerical type. We can change that with the following command.

![image](https://github.com/user-attachments/assets/9a79add7-e3d7-4756-9c28-23d097dcab61)

![image](https://github.com/user-attachments/assets/f1027a6a-7fe7-4858-a33a-72277cb97a0e)

### Distribution of Numerical feature values across the samples

![image](https://github.com/user-attachments/assets/66ec6cbe-ec12-4cce-93a3-a34d58b9407c)

### Distribution of Categorical features

![image](https://github.com/user-attachments/assets/098ff144-9cc3-4e63-a503-20c4779b2e82)

## IV) Exploratory Data Analysis (EDA)
### Correlating categorical features
* Categorical: Survived, Sex, Embarked, Pclass (ordinal), SibSp, Parch

* Target variable: 'Survived'

![image](https://github.com/user-attachments/assets/455e9d1b-2933-4af9-8319-b73ac25e649e)

Only 38% survived the disaster. So the training data suffers from data imbalance but it is not serve which is why I will not consider techniques like sampling to tackle the imbalance.

![image](https://github.com/user-attachments/assets/924ca601-62a1-4f33-b15a-46ce6a033023)

![image](https://github.com/user-attachments/assets/213ccb65-9244-41b1-b7e4-1826a8db9029)

![image](https://github.com/user-attachments/assets/4ed45942-955b-49da-903b-05d213cf7fcd)

Remaining Categorical Feature Columns

![image](https://github.com/user-attachments/assets/c8033f7d-e601-4209-9470-168253911856)

Observation: Survival Rate
* Fig 1: Female survival rate > male
* Fig 2: Most people embarked on Southampton, and also had the highest people not survived
* Fig 3: 1st class higher survival rate
* Fig 4: People going with 0 SibSp are mostly not survived the number of passenger with 1-2 family members has a better chance of survival
* Fig 5: People going with 0 Parch are mostly not survived

### EDA for Numerical Features
Numberical features: (continuous) Age, Fare

![image](https://github.com/user-attachments/assets/8f326542-0c2d-4baf-8520-c8d59e4194b5)

* Majority passengers were from 18-40 ages
* Children had more chance to survive than other ages

![image](https://github.com/user-attachments/assets/f07a67fe-a6f1-4055-b970-dc18548b9b50)

![image](https://github.com/user-attachments/assets/d74f2d14-d395-4de4-9a84-c3fc64d7bc5e)

![image](https://github.com/user-attachments/assets/134b997a-4655-478b-b5bb-fde557c54ac9)

![image](https://github.com/user-attachments/assets/a4d1ddd0-03e8-4641-a16d-c29adc1ff5ae)

![image](https://github.com/user-attachments/assets/bb30ace0-db46-43cb-8209-cd4afe0b79a9)

![image](https://github.com/user-attachments/assets/c0b5699e-bd32-4669-90b1-f2a28387a54b)

Distribution of fare
* Fare does not follow a normal distribution and has a huge spike at the price range (0 - $100)
* The distribution is skewed to the left with 75% of the fare paid under 31 and a max paid fare of 512.

Quartile plot
* Passenger with Luxury & Expensive Fare will have more chance to survive.

## V) Feature engineering & Data wrangling
### Name
* Regular expression

![image](https://github.com/user-attachments/assets/215d022e-57c7-452c-90e3-7eac55ee985e)

![image](https://github.com/user-attachments/assets/87d5f833-7dfc-46c5-86ac-9d35c026093a)

![image](https://github.com/user-attachments/assets/918253a2-df33-4bfd-ae63-7479be42fd1d)

![image](https://github.com/user-attachments/assets/2c57644b-2502-4cb6-9e6f-d9fac78f0c76)

![image](https://github.com/user-attachments/assets/d14d7cbf-c079-4af4-9c4f-1b43b9d44ea4)

![image](https://github.com/user-attachments/assets/9cacc93f-8eca-4a6a-987b-1c46f97abbcd)

![image](https://github.com/user-attachments/assets/08187aa9-7a7b-4988-90ca-47651e3d4ce4)

![image](https://github.com/user-attachments/assets/fe54dc9e-3074-4c83-ab26-13dcf09c2a3b)

![image](https://github.com/user-attachments/assets/96a54ccf-693b-44da-819c-feb4c559c761)

![image](https://github.com/user-attachments/assets/8db68c3f-6156-42ee-9c5e-61cb381e1e5f)

![image](https://github.com/user-attachments/assets/64d6a5b2-5456-4cbe-8a61-1eb8c7d86a17)

### Family
* SibSp, Parch

![image](https://github.com/user-attachments/assets/51663052-c86b-479b-aac0-821eeee30c71)

![image](https://github.com/user-attachments/assets/6e4b43fe-3e8c-46eb-822c-e6e2ed29cb08)

![image](https://github.com/user-attachments/assets/bdccefee-f71a-4bb7-aed5-f11a0f029631)

![image](https://github.com/user-attachments/assets/dafc6fce-3238-4278-bf6d-7e20f0b4ed30)

![image](https://github.com/user-attachments/assets/e1e10f8a-ab03-4664-be5a-402c6882fc3a)

![image](https://github.com/user-attachments/assets/e962bb66-4d4c-4727-a3cb-26088ce75210)

### Data wrangling

![image](https://github.com/user-attachments/assets/d9c184f3-49ff-4b1c-bd29-73edfcef30db)

#### Filling missing values
* Filling missing values with median of whole dataset

![image](https://github.com/user-attachments/assets/997ad95b-d61c-4d1b-88ef-fdfe581eaded)

![image](https://github.com/user-attachments/assets/daaba20e-5c5e-4814-be7d-cba01855b1b6)

![image](https://github.com/user-attachments/assets/e1ae68f9-5273-4f46-bd32-a265f93d544b)

![image](https://github.com/user-attachments/assets/ae6fc8dd-0889-4830-a763-31d9eb92fb05)

![image](https://github.com/user-attachments/assets/2bb4d5f1-3611-4803-b869-ae3f5c5e4a3b)

![image](https://github.com/user-attachments/assets/18f2cba9-cd2a-44f6-a5e0-cc0595f0f0d3)

## VI) Model training

![image](https://github.com/user-attachments/assets/1d7e1db5-faa2-47e2-bf83-cca35ec74793)

![image](https://github.com/user-attachments/assets/83164853-a84f-44f5-a0b1-f9a250e3534d)

![image](https://github.com/user-attachments/assets/9950d8fd-61b2-4ecf-971a-0e01b3332690)

![image](https://github.com/user-attachments/assets/e54a46e2-c7df-4850-bdef-ca348f682903)

### Cross-validation

![image](https://github.com/user-attachments/assets/5a050214-881f-478a-9fdc-0c0942898507)

### Baseline model comparison

![image](https://github.com/user-attachments/assets/9272bee6-d580-49f0-8c32-0c7182b0a5a8)

![image](https://github.com/user-attachments/assets/c7f8721c-10e3-493e-af65-2dd573f37447)

![image](https://github.com/user-attachments/assets/9f17e295-b145-440b-8b1c-8d15623df0d5)

![image](https://github.com/user-attachments/assets/ec9a83c8-f471-416e-afd4-36fc8934d75a)
